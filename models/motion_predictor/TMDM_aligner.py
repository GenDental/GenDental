import torch
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
import pytorch_lightning as pl
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import (
    Transform3d,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    rotation_6d_to_matrix,
)
from vector_quantize_pytorch import FSQ, LFQ
from contextlib import contextmanager
from models.motion_predictor.aligner import kabsch_algorithm_torch, rotation_matrix_to_angle_torch
import math
from utils import instantiate_from_config
from torch_ema import ExponentialMovingAverage as EMA
from utils import instantiate_from_config, instantiate_non_trainable_model
from utils import write_pointcloud
import torch.nn.init as init
from models.motion_predictor.shape_pcn_model import get_model
from models.motion_predictor.tooth_motion_diffusion_loss import cal_ADD

def matrices_to_6d(matrices):
    angles = matrix_to_euler_angles(matrices[:,:3,:3], convention='XYZ')
    outputs = torch.cat([angles,matrices[:,:3,3]],dim=-1)
    return outputs


def cal_ME_rot(rtv_step, rtv_step_gt, before_pts, masks):
    """Compute the same Kabsch-based rotation error as the motion evaluator."""
    n, b, l, _ = rtv_step.shape
    before_points = repeat(
        before_pts, 'b n p c -> (b l n) p c', l=l
    )

    pred_params = rearrange(rtv_step, 'n b l c -> (b l n) c')
    gt_params = rearrange(rtv_step_gt, 'b n l c -> (b l n) c')

    pred_matrices = torch.zeros(
        (b * l * n, 4, 4),
        device=before_pts.device,
        dtype=before_pts.dtype,
    )
    pred_matrices[:, :3, :3] = euler_angles_to_matrix(
        pred_params[:, :3], convention='XYZ'
    )
    pred_matrices[:, :3, 3] = pred_params[:, 3:]
    pred_matrices[:, 3, 3] = 1

    gt_matrices = torch.zeros_like(pred_matrices)
    gt_matrices[:, :3, :3] = euler_angles_to_matrix(
        gt_params[:, :3], convention='XYZ'
    )
    gt_matrices[:, :3, 3] = gt_params[:, 3:]
    gt_matrices[:, 3, 3] = 1

    predicted_points = Transform3d(
        matrix=pred_matrices.transpose(1, 2)
    ).transform_points(before_points)
    gt_points = Transform3d(
        matrix=gt_matrices.transpose(1, 2)
    ).transform_points(before_points)

    valid = repeat(masks, 'b n -> (b l n)', l=l).bool()
    angles = []
    for gt_tooth, predicted_tooth in zip(
        gt_points[valid], predicted_points[valid]
    ):
        relative_rotation = kabsch_algorithm_torch(
            gt_tooth, predicted_tooth
        )
        angles.append(
            rotation_matrix_to_angle_torch(relative_rotation)
        )
    return angles


class Aligner(pl.LightningModule):
    def __init__(self, diffuser_config, optimizer_config=None, scheduler_config=None, use_ema=True):
        super().__init__()
        self.aligner = instantiate_from_config(diffuser_config)
        self.shape_model = get_model(256, 512, 32)
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
        index, before_pts,after_pts,mask,matrices,inv_matrices = batch
        l = 20
        b,_,p,_,_ = matrices.shape
        rearranged_matrices = rearrange(matrices[:,:l],'b l p c1 c2-> (b l p) c1 c2')
        x0_input = matrices_to_6d(rearranged_matrices)
        x0_input = rearrange(x0_input,'(b l p) c -> b p l c',b=b,l=l,p=p)
        before_shape_code, _, _ = self.shape_model(before_pts)
        after_shape_code, _, _ = self.shape_model(after_pts)
        shape_code = torch.cat([before_shape_code,after_shape_code],dim=-1)
        shape_code = shape_code.to(before_pts.device)
        loss = self.aligner.get_loss(x0_input, before_pts, shape_code, matrices, mask)
        

        return loss
    
    @torch.no_grad()
    def sample(self, batch):
        index, before_pts,after_pts,mask,matrices,inv_matrices = batch
        self.aligner.eval()
        self.shape_model.eval()
        l = 20
        b,_,p,_,_ = matrices.shape
        rearranged_matrices = rearrange(matrices[:,:l],'b l p c1 c2-> (b l p) c1 c2')
        x0_input = matrices_to_6d(rearranged_matrices)
        x0_input = rearrange(x0_input,'(b l p) c -> b p l c',b=b,l=l,p=p)
        before_shape_code, _, _ = self.shape_model(before_pts)
        after_shape_code, _, _ = self.shape_model(after_pts)
        shape_code = torch.cat([before_shape_code,after_shape_code],dim=-1)
        shape_code = shape_code.to(before_pts.device)
        rtv_sampled = self.aligner.get_target(shape_code)
        ADD = cal_ADD(rtv_sampled,x0_input,before_pts,mask)
        angles = cal_ME_rot(rtv_sampled, x0_input, before_pts, mask)


        return ADD, angles
    
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
        loss = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        # ADD = self.sample(batch)
        split = 'val'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
            # f"{split}_ADD": ADD.detach()
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def on_test_start(self):
        self.global_index = 0
        self.all_loss = []
        self.all_indices = []
        self.all_adds = []
        self.all_angles = []
    
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
        ADD, angles = self.sample(batch)
        # split = 'test'
        # loss_dict = {
        #     f"{split}_total_loss": loss.detach(),
        #     f"{split}_ADD": ADD.detach(),
        # }
        # self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        self.all_adds.append(ADD.detach())
        self.all_angles.extend(angle.detach().cpu() for angle in angles)
        return ADD

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
        mean_angle = torch.stack(self.all_angles).mean()
        print("Mean Angle:", mean_angle.item())


