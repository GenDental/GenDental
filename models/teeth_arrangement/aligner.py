import torch
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
import pytorch_lightning as pl
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import Transform3d, rotation_6d_to_matrix
from vector_quantize_pytorch import FSQ, LFQ
from contextlib import contextmanager
import math
import torch.nn.functional as F
from utils import instantiate_from_config
from torch_ema import ExponentialMovingAverage as EMA
from utils import instantiate_from_config, instantiate_non_trainable_model
from utils import write_pointcloud
import torch.nn.init as init

def remove_prefix_from_state_dict(state_dict, prefix="GPT_Transformer."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # 去掉前缀
        else:
            new_key = key  # 保留原样
        new_state_dict[new_key] = value
    return new_state_dict



def kabsch_algorithm_torch(A, B):
    """
    A, B: shape (N, 3) or (N, D), torch.Tensor
    return: R (D, D) rotation matrix
    """
    assert A.shape == B.shape

    A_mean = A - torch.mean(A, dim=0, keepdim=True)
    B_mean = B - torch.mean(B, dim=0, keepdim=True)

    H = A_mean.T @ B_mean   

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.T
    R = V @ U.T

    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    return R

def rotation_matrix_to_angle_torch(R):
    """
    R: (3, 3) torch.Tensor
    return: 角度（单位：度）
    """
    trace = torch.trace(R)
    cos_theta = (trace - 1.0) / 2.0

    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angle = torch.acos(cos_theta)

    return angle * 180.0 / math.pi

def rigid_register_kabsch(A: torch.Tensor, B: torch.Tensor):
    """
    A, B: (N, 3) torch tensors
    Returns:
        A_registered: (N, 3)
        R: (3, 3)
        t: (1, 3)
    """

    assert A.shape == B.shape
    assert A.shape[1] == 3

    centroid_A = A.mean(dim=0, keepdim=True)  # (1, 3)
    centroid_B = B.mean(dim=0, keepdim=True)  # (1, 3)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered  # (3, 3)

    U, S, Vh = torch.linalg.svd(H)

    R = Vh.T @ U.T

    if torch.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T

    t = centroid_B - centroid_A @ R.T  # (1, 3)

    A_registered = A @ R.T + t

    return A_registered, R, t

def icp_register(A, B, max_iter=50):
    """
    A: (Na, 3)
    B: (Nb, 3)
    返回:
        A_aligned, R_total, t_total
    """

    A_curr = A.clone()
    R_total = torch.eye(3, device=A.device)
    t_total = torch.zeros(1, 3, device=A.device)

    for _ in range(max_iter):
        dists = torch.cdist(A_curr, B)  # (Na, Nb)
        nn_idx = dists.argmin(dim=1)    # (Na,)
        B_match = B[nn_idx]             # (Na, 3)

        A_reg, R, t = rigid_register_kabsch(A_curr, B_match)

        A_curr = A_reg
        R_total = R @ R_total
        t_total = t @ R_total.T + t_total

    return A_curr, R_total, t_total

class Aligner(pl.LightningModule):
    def __init__(self, aligner_config, optimizer_config=None, scheduler_config=None, use_ema=True):
        super().__init__()
        self.aligner = instantiate_from_config(aligner_config)
        self.use_ema = use_ema
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
    
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate_from_config(self.optimizer_config, params=trainable_params, lr=self.learning_rate)
        scheduler = instantiate_from_config(self.scheduler_config,optimizer=optimizer,max_epochs=self.max_epochs)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
    }



    def forward(self, batch):
        index,before_points,after_points,masks = batch
        bs = before_points.shape[0]
        after_points = rearrange(after_points,'b n p c -> (b n) p c')
        centroid = torch.mean(before_points,dim=-2,keepdim=False)

        outputs1 = self.aligner(centroid, before_points).to(torch.float32).cuda()
        predicted_centroid1 = outputs1[:,:,:3]
        dofs1 = rearrange(outputs1[:,:,3:],'b n c -> (b n) c')
        criterion = nn.MSELoss(reduction='none')
        rot_matrix1 = rotation_6d_to_matrix(dofs1)
        predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
        predicted_points = torch.bmm(predicted_points,rot_matrix1)
        predicted_points = predicted_points + rearrange(predicted_centroid1,'b n c->(b n) c').unsqueeze(1)
        loss = 0.1*(criterion(predicted_points,after_points).sum(dim=(1,2)) * masks.flatten()).sum()
        ADD = criterion(predicted_points,after_points).sum(dim=-1).sqrt().mean(dim=-1)
        ADD = 30* (ADD * masks.flatten()).sum() / masks.sum()

        return loss, ADD
    
    def training_step(self, batch, batch_idx):
        loss, add = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, add = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
            f"{split}_add": add.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def on_test_start(self):
        self.global_index = 0
        self.all_ADD = []
        self.all_indices = []
        self.all_angles = []
        self.all_CSA = []
        self.all_PA_ADD = []
    
    def test_step(self, batch, batch_idx):
        index,before_points,after_points,before_normals,after_normals,masks = batch
        bs = before_points.shape[0]
        after_centroid = torch.mean(after_points,dim=-2,keepdim=False)
        after_points = rearrange(after_points,'b n p c -> (b n) p c')
        centroid = torch.mean(before_points,dim=-2,keepdim=False)

        outputs1 = self.aligner(centroid, before_points).to(torch.float32).cuda()
        predicted_centroid1 = outputs1[:,:,:3]
        dofs1 = rearrange(outputs1[:,:,3:],'b n c -> (b n) c')
        criterion = nn.MSELoss(reduction='none')
        rot_matrix1 = rotation_6d_to_matrix(dofs1)
        predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
        predicted_points = torch.bmm(predicted_points,rot_matrix1)
        predicted_points = predicted_points + rearrange(predicted_centroid1,'b n c->(b n) c').unsqueeze(1)
        loss = 0.1*(criterion(predicted_points,after_points).sum(dim=(1,2)) * masks.flatten()).sum()

        ADD = (predicted_points - after_points).norm(dim=-1).mean(dim=-1)
        ADD = (30*rearrange(ADD,'(b n) -> b n', b=bs)*masks).sum(dim=-1) / masks.sum(dim=-1)
        self.all_ADD.append(ADD)
        self.all_indices.append(index)


        # ME_rot
        flattend_mask = masks.flatten()
        for i in range(predicted_points.shape[0]):
            if flattend_mask[i]:
                gt_mesh_vertices = after_points[i]
                predicted_mesh_vertices = predicted_points[i]
                matrix = kabsch_algorithm_torch(gt_mesh_vertices,predicted_mesh_vertices)
                angle = abs(rotation_matrix_to_angle_torch(matrix))
                self.all_angles.append(angle.item())
        
        # CSA
        before_points = rearrange(before_points,'b n p c -> (b n) p c')
        flattend_mask = masks.flatten()
        centroid_dis = rearrange(predicted_centroid1 - centroid,'b n c -> (b n) c')
        gt_centroid_dis = rearrange(after_centroid - centroid,'b n c -> (b n) c')
        for i in range(predicted_points.shape[0]):
            if flattend_mask[i]:
                scale = 30
                gt_mesh_vertices = after_points[i]
                predicted_mesh_vertices = predicted_points[i]
                before_mesh_vertices = before_points[i]
                predicted_trans = centroid_dis[i]
                gt_trans = gt_centroid_dis[i]
                length = 512
                matrix1 = kabsch_algorithm_torch(before_mesh_vertices[:length]*scale,gt_mesh_vertices[:length]*scale)
                matrix2 = kabsch_algorithm_torch(before_mesh_vertices[:length]*scale,predicted_mesh_vertices[:length]*scale)
                T1 = torch.eye(4).to(after_points.device)
                T1[:3,:3] = matrix1
                T1[:3,3] = gt_trans*scale
                T2 = torch.eye(4).to(after_points.device)
                T2[:3,:3] = matrix2
                T2[:3,3] = predicted_trans*scale
                T1 = T1.view((1,-1))
                T2 = T2.view((1,-1))
                cos_sim = (F.cosine_similarity(T1, T2)+1)/2
                self.all_CSA.append(cos_sim.item())


    def on_test_end(self):
        all_ADD = torch.cat(self.all_ADD)         # [N] (整个数据集的样本数)
        all_indices = torch.cat(self.all_indices) # [N]
        add = all_ADD.mean()
        print("Average ADD:", add.item())
        angles = torch.tensor(self.all_angles)
        average_angle = angles.mean()
        print("Average Rotation Angle (degrees):", average_angle.item())
        cos_sim = torch.tensor(self.all_CSA)
        average_cos_sim = cos_sim.mean()
        print("Average Cosine Similarity:", average_cos_sim.item())

        pa_add = torch.tensor(self.all_PA_ADD)
        average_pa_add = pa_add.mean()
        print("Average PAADD:", average_pa_add.item())
        



