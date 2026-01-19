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
from models.stage_two.smooth import postprocess_smooth_se3
import torch.nn.init as init

def remove_prefix_from_state_dict(state_dict, prefix="GPT_Transformer."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def rotation_matrix_to_angle(R):
    # R: (..., 3, 3)
    cos_theta = ((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1+1e-7, 1-1e-7)
    return torch.acos(cos_theta)

def smoothness_loss(predicted_matrices, masks, alpha=1.0, beta=1.0):
    R = predicted_matrices[..., :3, :3]   # (b, l, p, 3, 3)
    t = predicted_matrices[..., :3, 3]    # (b, l, p, 3)

    trans_diff = (t[:, 1:] - t[:, :-1]).norm(dim=-1)  # (b, l-1, p)
    loss_trans = trans_diff.mean(dim=1) 

    R_rel = torch.matmul(R[:, :-1].transpose(-1, -2), R[:, 1:])  # (b, l-1, p, 3, 3)
    rot_diff = rotation_matrix_to_angle(R_rel)  # (b, l-1, p)
    loss_rot = rot_diff.mean(dim=1)

    loss = alpha * loss_trans + beta * loss_rot

    loss = (loss * masks).mean()
    return loss

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp


def smooth_predicted_matrices_batch(predicted_matrices, window=5, poly=2):
    b, l, p, _, _ = predicted_matrices.shape
    device = predicted_matrices.device

    smoothed = np.tile(np.eye(4), (b, l, p, 1, 1)) 

    times = np.arange(l) 

    for bi in range(b):
        for pi in range(p):
            seq = predicted_matrices[bi, :, pi].cpu().numpy()  # (l, 4, 4)


            t = seq[:, :3, 3]  # (l, 3)
            if l >= window: 
                t_smooth = savgol_filter(t, window_length=window, polyorder=poly, axis=0)
            else:  
                t_smooth = t

            Rm = seq[:, :3, :3]  # (l, 3, 3)
            rots = R.from_matrix(Rm)

            slerp = Slerp(times, rots)
            R_smooth = slerp(times).as_matrix()  # (l, 3, 3)

            smoothed[bi, :, pi, :3, :3] = R_smooth
            smoothed[bi, :, pi, :3, 3] = t_smooth

    return torch.from_numpy(smoothed).to(device).float()

class MotionTransfer(pl.LightningModule):
    def __init__(self, transformer_config, optimizer_config=None, scheduler_config=None, use_ema=True):
        super().__init__()
        # self.quantizer = instantiate_non_trainable_model(quantizer_config)
        self.transformer = instantiate_from_config(transformer_config)
        self.use_ema = use_ema
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.sos_token = nn.Parameter(torch.zeros(1024))
        if self.use_ema:
            self.ema_model = EMA(self.transformer.parameters(), decay=0.9999)
    
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

    def on_fit_start(self):
        if self.use_ema:
            self.ema_model.to(self.device)


    def forward(self, batch):
        index, before_pts, after_pts, masks, matrices, inv_matrices = batch
        before_centroid = before_pts.mean(dim=-2)
        after_centroid = after_pts.mean(dim=-2)
        predicted_params = self.transformer(before_pts, before_centroid, after_pts, masks)
        predicted_params = rearrange(predicted_params, 'b p (l c) -> b l p c', c=9)
        b, l, p, c = predicted_params.shape
        

        after_points = repeat(after_pts,'b p n c -> (b l p) n c', l=l)
        after_centroid = repeat(after_centroid,'b p c -> (b l p) c', l=l)
        repeated_masks = repeat(masks,'b p -> (b l p)', l=l)   

        gt_matrices = rearrange(inv_matrices,'b l p c1 c2-> (b l p) c2 c1')
        gt_transform = Transform3d(matrix=gt_matrices)
        gt_points = gt_transform.transform_points(after_points)

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
        predicted_points = predicted_transform.transform_points(after_points)

        predicted_matrices = rearrange(predicted_matrices,'(b l p) c1 c2 -> b l p c1 c2', b=b, l=l, p=p)
        loss_smooth = smoothness_loss(predicted_matrices, masks, alpha=1.0, beta=1.0)

        criterion = nn.MSELoss(reduction='none')

        
        rec_loss = criterion(predicted_points, gt_points)
        rec_loss = (rec_loss * repeated_masks.reshape(-1,1,1)).sum(dim=(-1,-2)).mean()
        loss = 100 * rec_loss + 600 * loss_smooth

        return loss, 100*rec_loss
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state'] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'ema_state' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state'])
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_model.store(self.gpt_transformer.parameters())
            self.ema_model.copy_to(self.gpt_transformer.parameters())
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_model.restore(self.gpt_transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def training_step(self, batch, batch_idx):
        loss, rec_loss = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}_total_loss": rec_loss.detach(),
            f"{split}_smooth_loss": (loss-rec_loss).detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        if self.use_ema:
            self.ema_model.update()

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, rec_loss = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}_total_loss": rec_loss.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def on_test_start(self):
        self.num = 0
    
    def test_step(self, batch, batch_idx):
        index, before_pts, after_pts, before_normals, after_normals,  masks, matrices, inv_matrices = batch
        before_centroid = before_pts.mean(dim=-2)
        after_centroid = after_pts.mean(dim=-2)
        predicted_params = self.transformer(before_pts, before_centroid, after_pts, masks)
        predicted_params = rearrange(predicted_params, 'b p (l c) -> b l p c', c=9)
        b, l, p, c = predicted_params.shape
        

        after_points = repeat(after_pts,'b p n c -> (b l p) n c', l=l)
        after_centroid = repeat(after_centroid,'b p c -> (b l p) c', l=l)   

        rotation_6d = predicted_params[:,:,:,:6]
        transition =  predicted_params[:,:,:,6:]
        rot_matrix = rotation_6d_to_matrix(rotation_6d)


        predicted_matrices = torch.zeros(b,l,p,4,4).to(before_pts.device)
        predicted_matrices[...,:3,:3] = rot_matrix
        predicted_matrices[...,:3,3] = transition
        predicted_matrices[...,3,3] = 1
        grouped_predicted_matrices = rearrange(predicted_matrices,'b l p c1 c2 -> (b l p) c2 c1')

        predicted_transform = Transform3d(matrix=grouped_predicted_matrices)
        generated_points = predicted_transform.transform_points(after_points)
        generated_points = rearrange(generated_points,'(b l p) n c -> b l p n c',b=b, p=p, l=l)
        for i in range(b):
            out_before_points = generated_points[i][0].cpu().numpy()
            style_points = before_pts[i].cpu().numpy()
            out_after_points = after_pts[i].cpu().numpy()
            out_matrices = predicted_matrices[i].cpu().numpy()
            output_dir = '/data3/leics/dataset/teeth/synthetic_after_process_npz'
            out_mask = masks[i].cpu().numpy()
            import os 
            os.makedirs(output_dir,exist_ok=True)
            np.savez(os.path.join(output_dir, f'{index[i]}.npz'), before_pts=out_before_points, after_pts=out_after_points, style_pts=style_points, mask=out_mask,inv_matrices=out_matrices)



