import torch
import os
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
import pytorch_lightning as pl
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from pytorch3d.loss import chamfer_distance
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
            new_key = key[len(prefix):] 
        else:
            new_key = key 
        new_state_dict[new_key] = value
    return new_state_dict

class ToothGPT(pl.LightningModule):
    def __init__(self, transformer_config, optimizer_config=None, scheduler_config=None, use_ema=True):
        super().__init__()
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
        index, before_pts, after_pts, masks = batch
        center = after_pts.mean(dim=-2)
        rec_pts, predicted_masks, commit_loss = self.transformer(after_pts, center)
        # codes = rearrange(codes,'n2 b n1 c -> b (n1 n2) c')

        gt_masks = masks.flatten().long()

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        rec_pts = rec_pts[:,:-1]

        predicted_masks = predicted_masks[:,:-1].reshape(-1,2)
        loss3 = criterion(predicted_masks,gt_masks)

        result_masks = (masks).to(torch.float).flatten()
        gt_points = rearrange(after_pts,'b n p c -> (b n) p c')
        rec = rearrange(rec_pts,'b n p c -> (b n) p c')
        loss1 = chamfer_distance(rec,gt_points,norm=2,batch_reduction=None,point_reduction='sum')[0]
        loss2 = chamfer_distance(rec,gt_points,norm=1,batch_reduction=None,point_reduction='sum')[0]

        loss1 = (loss1  * result_masks).sum() / result_masks.sum()
        loss2 = (loss2  * result_masks).sum() / result_masks.sum()
        loss = loss1 + loss2 + loss3 + commit_loss.mean()

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
        index, before_pts, after_pts, before_normals, after_normals,  masks = batch
        center = after_pts.mean(dim=-2)
        outputroot = '/data3/leics/dataset/teeth/old_gpt_segment_lower'
        os.makedirs(outputroot,exist_ok=True)
        if self.num > 2000:
            exit()
        for _ in range(100):
            rand_center = center + torch.rand_like(center) * 0.02
            eps = 1e-6
            rand_pts = torch.from_numpy(np.random.random(after_pts.shape) * eps).cuda().float()
            for i in range(16):
                rec_pts, predicted_masks,commit_loss = self.transformer(rand_pts, rand_center)
                rand_pts[:,i] = rec_pts[:,i]
            gt_masks = masks.flatten().long()

            criterion = nn.CrossEntropyLoss()
            rec_pts = rand_pts

            predicted_masks = predicted_masks[:,:-1].reshape(-1,2)
            loss3 = criterion(predicted_masks,gt_masks)

            result_masks = (masks).to(torch.float).flatten()
            gt_points = rearrange(after_pts,'b n p c -> (b n) p c')
            predicted_masks = rearrange(predicted_masks, '(b n) c -> b n c', b=before_pts.shape[0])
            merged = False
            if merged:
                for i in range(rec_pts.shape[0]):
                    points = []
                    for j in range(16):
                        if predicted_masks[i,j,1] > predicted_masks[i,j,0]:
                            points.append(rec_pts[i][j].cpu().numpy())
                    points = np.concatenate(points,axis=0)
                    write_pointcloud(points,f'{outputroot}/{self.num}.ply')
                    self.num += 1
            else:
                for i in range(rec_pts.shape[0]):
                    for j in range(16):
                        if predicted_masks[i,j,1] > predicted_masks[i,j,0]:
                            write_pointcloud(rec_pts[i][j].cpu().numpy(),f'{outputroot}/{self.num}_{j}.ply')
                    self.num += 1


