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
from models.stage_two.smooth import postprocess_smooth_se3

def remove_prefix_from_state_dict(state_dict, prefix="GPT_Transformer."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # 去掉前缀
        else:
            new_key = key  # 保留原样
        new_state_dict[new_key] = value
    return new_state_dict

class MotionTransferSampler(pl.LightningModule):
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
        before_pts, after_pts, before_normals, after_normals,  masks, inv_matrices = batch
        before_centroid = before_pts.mean(dim=-2)
        after_centroid = after_pts.mean(dim=-2)
        predicted_params = self.transformer(before_pts, before_centroid, after_pts, masks)
        predicted_params = rearrange(predicted_params, 'b p (l c) -> b l p c', c=9)
        b, l, p, c = predicted_params.shape
        

        after_points = repeat(after_pts,'b p n c -> (b l p) n c', l=l)
        after_centroid = repeat(after_centroid,'b p c -> (b l p) c', l=l)
        masks = repeat(masks,'b p -> (b l p)', l=l)   

        gt_matrices = rearrange(inv_matrices,'b l p c1 c2-> (b l p) c2 c1')
        gt_transform = Transform3d(matrix=gt_matrices)
        gt_points = gt_transform.transform_points(after_points)

        rotation_6d = predicted_params[:,:,:,:6]
        transition =  predicted_params[:,:,:,6:]
        rotation_6d = rearrange(rotation_6d,'b l p c -> (b l p) c')
        rot_matrix = rotation_6d_to_matrix(rotation_6d)
        transition = rearrange(transition,'b l p c -> (b l p) c')


        # predicted_matrices = torch.zeros_like(gt_matrices).to(gt_matrices.device)
        # predicted_matrices[:,:3,:3] = rot_matrix
        # predicted_matrices[:,:3,3] = transition
        # predicted_matrices[:,3,3] = 1
        # predicted_matrices = rearrange(predicted_matrices,'b c1 c2 -> b c2 c1')

        # predicted_transform = Transform3d(matrix=predicted_matrices)
        # predicted_points = predicted_transform.transform_points(after_points)
        predicted_points = torch.bmm(after_points - after_centroid.unsqueeze(-2), rot_matrix) + (transition + after_centroid).unsqueeze(-2)
        criterion = nn.MSELoss(reduction='none')

        
        rec_loss = criterion(predicted_points, gt_points)
        rec_loss = (rec_loss * masks.reshape(-1,1,1)).sum(dim=(-1,-2)).mean()
        loss = 100 * rec_loss 

        return loss
    
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
        loss = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
            f"{split}_lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        if self.use_ema:
            self.ema_model.update()

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}_total_loss": loss.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss
    
    def on_test_start(self):
        self.num = 0
    
    def test_step(self, batch, batch_idx):
        index, before_pts, after_pts, masks = batch
        before_centroid = before_pts.mean(dim=-2)
        after_centroid = after_pts.mean(dim=-2)
        predicted_params = self.transformer(before_pts, before_centroid, after_pts, masks)
        predicted_params = rearrange(predicted_params, 'b p (l c) -> b l p c', c=9)
        b, l, p, c = predicted_params.shape
        

        after_points = repeat(after_pts,'b p n c -> (b l p) n c', l=l)
        after_centroid = repeat(after_centroid,'b p c -> (b l p) c', l=l)
        # masks = repeat(masks,'b p -> (b l p)', l=l)   

        rotation_6d = predicted_params[:,:,:,:6]
        transition =  predicted_params[:,:,:,6:]
        # rotation_6d = rearrange(rotation_6d,'b l p c -> (b l p) c')
        rot_matrix = rotation_6d_to_matrix(rotation_6d)
        # transition = rearrange(transition,'b l p c -> (b l p) c')


        predicted_matrices = torch.zeros(b,l,p,4,4).to(before_pts.device)
        predicted_matrices[...,:3,:3] = rot_matrix
        predicted_matrices[...,:3,3] = transition
        predicted_matrices[...,3,3] = 1
        grouped_predicted_matrices = rearrange(predicted_matrices,'b l p c1 c2 -> (b l p) c2 c1')

        predicted_transform = Transform3d(matrix=grouped_predicted_matrices)
        generated_points = predicted_transform.transform_points(after_points)
        generated_points = rearrange(generated_points,'(b l p) n c -> b l p n c',b=b, p=p, l=l)
        for j in range(10):
            for i in range(b):
                out_before_points = generated_points[i][0].cpu().numpy()
                style_points = before_pts[i].cpu().numpy()
                out_after_points = after_pts[i].cpu().numpy()
                out_matrices = predicted_matrices[i].cpu().numpy()
                l = out_matrices.shape[0]
                eye = np.eye(4)  # shape = (4,4)

                # 扩展并重复
                matrices = np.tile(eye, (21, 32, 1, 1))  # shape = (21,32,4,4)
                for j in range(l-1):
                    for rearranged_index in range(32):
                        matrices[j][rearranged_index] = out_matrices[j+1][rearranged_index] @ np.linalg.inv(out_matrices[0][rearranged_index])
                for rearranged_index in range(32):
                    matrices[l-1][rearranged_index] = np.linalg.inv(out_matrices[0][rearranged_index])
                output_dir = '/data3/leics/dataset/teeth/pure_synthetic_motion_dataset_nonsmoothed10x'
                out_mask = masks[i].cpu().numpy()
                import os 
                os.makedirs(output_dir,exist_ok=True)
                np.savez(os.path.join(output_dir, f'{self.num}.npz'), before_pts=out_before_points, after_pts=out_after_points, style_pts=style_points, mask=out_mask, matrices=matrices, inv_matrices=out_matrices)
                self.num += 1



        # predicted_points = generated_points
        # criterion = nn.MSELoss(reduction='none')

        # # predicted_points = rearrange(predicted_points,'(b l p) n c -> b l p n c', p=p, l=l)
        # for i in range(l):
        #     points = []
        #     for j in range(32):
        #         if masks[0,j]:
        #             points.append(predicted_points[0,i,j])
        #     points = torch.cat(points,dim=0)
        #     write_pointcloud(points.cpu().numpy(), f'/data3/leics/dataset/teeth/synthetic_tmp/{i}.ply')
        # points = []
        # for j in range(32):
        #     if masks[0,j]:
        #         points.append(after_pts[0,j])
        # points = torch.cat(points,dim=0)
        # write_pointcloud(points.cpu().numpy(), f'/data3/leics/dataset/teeth/synthetic_tmp/21.ply')    
        # exit()


