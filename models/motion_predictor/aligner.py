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

    # 去中心化
    A_mean = A - torch.mean(A, dim=0, keepdim=True)
    B_mean = B - torch.mean(B, dim=0, keepdim=True)

    # 计算协方差矩阵 H
    H = A_mean.T @ B_mean   # 等价于 torch.matmul(A_mean.T, B_mean)

    # SVD 分解
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.T

    # 计算旋转矩阵
    R = V @ U.T

    # 处理反射问题（确保 det(R) = +1）
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

    # 数值稳定性裁剪，防止 acos 出现 nan
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angle = torch.acos(cos_theta)

    return angle * 180.0 / math.pi

class Aligner(pl.LightningModule):
    def __init__(self, aligner_config, optimizer_config=None, scheduler_config=None, use_ema=True):
        super().__init__()
        # self.quantizer = instantiate_non_trainable_model(quantizer_config)
        self.aligner = instantiate_from_config(aligner_config)
        self.use_ema = use_ema
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        # if self.use_ema:
        #     self.ema_model = EMA(self.transformer.parameters(), decay=0.9999)
    
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
        index, before_pts, after_pts, masks, matrices, inv_matrices = batch
        bs = before_pts.shape[0]
        centroid = torch.mean(before_pts,dim=-2,keepdim=False)
        predicted_params = self.aligner(centroid, before_pts, after_pts).to(torch.float32).cuda()
        l = 20
        n = 512

        before_points = repeat(before_pts,'b p n c -> (b l p) n c', l=l)
        # before_centroid = repeat(before_centroid,'b p c -> (b l p) c', l=l)
        ADD_mask = repeat(masks,'b p -> (b l p n)', l=l, n=512)
        masks = repeat(masks,'b p -> (b l p)', l=l)   

        gt_matrices = rearrange(matrices[:,:l],'b l p c1 c2-> (b l p) c2 c1')
        gt_transform = Transform3d(matrix=gt_matrices)
        gt_points = gt_transform.transform_points(before_points)

        rotation_6d = predicted_params[:,:,:,:6]
        transition =  predicted_params[:,:,:,6:]
        rotation_6d = rearrange(rotation_6d,'b l p c -> (b l p) c')
        rot_matrix = rotation_6d_to_matrix(rotation_6d)
        transition = rearrange(transition,'b l p c -> (b l p) c')

        predicted_matrices = torch.zeros_like(gt_matrices).to(gt_matrices.device)
        predicted_matrices[:,:3,:3] = rot_matrix
        predicted_matrices[:,:3,3] = transition
        predicted_matrices[:,3,3] = 1
        predicted_matrices = rearrange(predicted_matrices,'b c1 c2 -> b c2 c1')

        predicted_transform = Transform3d(matrix=predicted_matrices)
        predicted_points = predicted_transform.transform_points(before_points)
        
        criterion = nn.MSELoss(reduction='none')

        predicted_reshaped = rearrange(predicted_points,'(b l p) n c -> b l p n c', b=bs, l=l)
        gt_reshaped = rearrange(gt_points,'(b l p) n c -> b l p n c', b=bs, l=l)
        ADD = 30 * torch.norm(predicted_reshaped - gt_reshaped, dim=-1).sum() / ADD_mask.sum()

        rec_loss = criterion(predicted_points, gt_points)
        rec_loss = (rec_loss * masks.reshape(-1,1,1)).sum(dim=(-1,-2)).mean()
        loss = 100 * rec_loss 

        return loss, ADD
    
    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['ema_state'] = self.ema_model.state_dict()

    # def on_load_checkpoint(self, checkpoint):
    #     if 'ema_state' in checkpoint:
    #         self.ema_model.load_state_dict(checkpoint['ema_state'])
    
    # @contextmanager
    # def ema_scope(self, context=None):
    #     if self.use_ema:
    #         self.ema_model.store(self.gpt_transformer.parameters())
    #         self.ema_model.copy_to(self.gpt_transformer.parameters())
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema:
    #             self.ema_model.restore(self.gpt_transformer.parameters())
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")
    
    def training_step(self, batch, batch_idx):
        loss, ADD = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
            f"{split}_ADD": ADD.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, ADD = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
            f"{split}_ADD": ADD.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def on_test_start(self):
        self.global_index = 0
        self.all_loss = []
        self.all_indices = []
        self.all_adds = []
        self.all_angles = []
        self.all_CSA = []
    
    def test_step(self, batch, batch_idx):
        # index, before_pts, after_pts, before_normals, after_normals,  masks, matrices, inv_matrices = batch
        # bs = before_pts.shape[0]
        # centroid = torch.mean(before_pts,dim=-2,keepdim=False)
        # predicted_params = self.aligner(centroid, before_pts, after_pts).to(torch.float32).cuda()
        # l = 21

        # before_points = repeat(before_pts,'b p n c -> (b l p) n c', l=l)
        # # before_centroid = repeat(before_centroid,'b p c -> (b l p) c', l=l)
        # masks = repeat(masks,'b p -> (b l p)', l=l)   

        # gt_matrices = rearrange(matrices,'b l p c1 c2-> (b l p) c2 c1')
        # gt_transform = Transform3d(matrix=gt_matrices)
        # gt_points = gt_transform.transform_points(before_points)

        # rotation_6d = predicted_params[:,:,:,:6]
        # transition =  predicted_params[:,:,:,6:]
        # rotation_6d = rearrange(rotation_6d,'b l p c -> (b l p) c')
        # rot_matrix = rotation_6d_to_matrix(rotation_6d)
        # transition = rearrange(transition,'b l p c -> (b l p) c')

        # predicted_matrices = torch.zeros_like(gt_matrices).to(gt_matrices.device)
        # predicted_matrices[:,:3,:3] = rot_matrix
        # predicted_matrices[:,:3,3] = transition
        # predicted_matrices[:,3,3] = 1
        # predicted_matrices = rearrange(predicted_matrices,'b c1 c2 -> b c2 c1')

        # predicted_transform = Transform3d(matrix=predicted_matrices)
        # predicted_points = predicted_transform.transform_points(before_points)
        
        # criterion = nn.MSELoss(reduction='none')

        
        # rec_loss = criterion(predicted_points, gt_points)
        # rec_loss = (rec_loss * masks.reshape(-1,1,1)).sum(dim=(-1,-2))
        # rec_loss = rearrange(rec_loss,'(b l p) -> b l p', b=bs, l=l).mean(dim=-1).mean(dim=-1)
        # self.all_loss.append(rec_loss)
        # self.all_indices.append(index)
        # loss, ADD = self.forward(batch)
        # split = 'test'
        # loss_dict = {
        #     f"{split}_total_loss": loss.detach(),
        #     f"{split}_ADD": ADD.detach(),
        # }
        # self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        # self.all_adds.append(ADD.detach())
        index, before_pts, after_pts, masks, matrices, inv_matrices = batch
        bs = before_pts.shape[0]
        centroid = torch.mean(before_pts,dim=-2,keepdim=False)
        predicted_params = self.aligner(centroid, before_pts, after_pts).to(torch.float32).cuda()
        l = 20
        n = 512

        before_points = repeat(before_pts,'b p n c -> (b l p) n c', l=l)
        # before_centroid = repeat(before_centroid,'b p c -> (b l p) c', l=l)
        ADD_mask = repeat(masks,'b p -> (b l p n)', l=l, n=512)
        masks = repeat(masks,'b p -> (b l p)', l=l)   

        gt_matrices = rearrange(matrices[:,:l],'b l p c1 c2-> (b l p) c2 c1')
        gt_transform = Transform3d(matrix=gt_matrices)
        gt_points = gt_transform.transform_points(before_points)

        rotation_6d = predicted_params[:,:,:,:6]
        transition =  predicted_params[:,:,:,6:]
        rotation_6d = rearrange(rotation_6d,'b l p c -> (b l p) c')
        rot_matrix = rotation_6d_to_matrix(rotation_6d)
        transition = rearrange(transition,'b l p c -> (b l p) c')

        predicted_matrices = torch.zeros_like(gt_matrices).to(gt_matrices.device)
        predicted_matrices[:,:3,:3] = rot_matrix
        predicted_matrices[:,:3,3] = transition
        predicted_matrices[:,3,3] = 1
        predicted_matrices = rearrange(predicted_matrices,'b c1 c2 -> b c2 c1')

        predicted_transform = Transform3d(matrix=predicted_matrices)
        predicted_points = predicted_transform.transform_points(before_points)
        
        criterion = nn.MSELoss(reduction='none')

        predicted_reshaped = rearrange(predicted_points,'(b l p) n c -> b l p n c', b=bs, l=l)
        gt_reshaped = rearrange(gt_points,'(b l p) n c -> b l p n c', b=bs, l=l)
        ADD = 30 * torch.norm(predicted_reshaped - gt_reshaped, dim=-1).sum() / ADD_mask.sum()

        rec_loss = criterion(predicted_points, gt_points)
        rec_loss = (rec_loss * masks.reshape(-1,1,1)).sum(dim=(-1,-2)).mean()
        loss = 100 * rec_loss 
        self.all_adds.append(ADD.detach())

        flattend_mask = masks.flatten()
        for i in range(predicted_points.shape[0]):
            if flattend_mask[i]:
                gt_mesh_vertices = gt_points[i]
                predicted_mesh_vertices = predicted_points[i]
                matrix = kabsch_algorithm_torch(gt_mesh_vertices,predicted_mesh_vertices)
                angle = abs(rotation_matrix_to_angle_torch(matrix))
                self.all_angles.append(angle.item())
        
        # print(centroid.shape)
        # exit()

        # centroid_dis = rearrange(predicted_centroid1 - centroid,'b n c -> (b n) c')
        # gt_centroid_dis = rearrange(after_centroid - centroid,'b n c -> (b n) c')
        # for i in range(predicted_points.shape[0]):
        #     if flattend_mask[i]:
        #         scale = 30
        #         gt_mesh_vertices = after_points[i]
        #         predicted_mesh_vertices = predicted_points[i]
        #         before_mesh_vertices = before_points[i]
        #         predicted_trans = centroid_dis[i]
        #         gt_trans = gt_centroid_dis[i]
        #         length = 512
        #         matrix1 = kabsch_algorithm_torch(before_mesh_vertices[:length]*scale,gt_mesh_vertices[:length]*scale)
        #         matrix2 = kabsch_algorithm_torch(before_mesh_vertices[:length]*scale,predicted_mesh_vertices[:length]*scale)
        #         T1 = torch.eye(4).to(after_points.device)
        #         T1[:3,:3] = matrix1
        #         T1[:3,3] = gt_trans*scale
        #         T2 = torch.eye(4).to(after_points.device)
        #         T2[:3,:3] = matrix2
        #         T2[:3,3] = predicted_trans*scale
        #         T1 = T1.view((1,-1))
        #         T2 = T2.view((1,-1))
        #         cos_sim = (F.cosine_similarity(T1, T2)+1)/2
        #         self.all_CSA.append(cos_sim.item())

        
        return loss

    def on_test_end(self):
        # all_loss = torch.cat(self.all_loss)         # [N] (整个数据集的样本数)
        # all_indices = torch.cat(self.all_indices) # [N]

        # # 取全局 top10
        # topk_values, topk_indices = torch.topk(all_loss, k=20)
        # top10_indices = all_indices[topk_indices]
        # print("Top10 loss values:", topk_values)
        # print("Corresponding dataset indices:", top10_indices)
        all_adds = torch.tensor(self.all_adds)
        mean_add = torch.mean(all_adds)
        print("Mean ADD:", mean_add.item())
        mean_angle = torch.tensor(self.all_angles).mean()
        print("Mean Angle:", mean_angle.item())
        # mean_CSA = sum(self.all_CSA) / len(self.all_CSA)


