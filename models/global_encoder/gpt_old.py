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
import torch.nn.functional as F
from utils import instantiate_from_config
from torch_ema import ExponentialMovingAverage as EMA
from models.transformer import *
from utils import instantiate_from_config, instantiate_non_trainable_model
from utils import write_pointcloud
import torch.nn.init as init
from typing import Dict

def remove_prefix_from_state_dict(state_dict, prefix="GPT_Transformer."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # 去掉前缀
        else:
            new_key = key  # 保留原样
        new_state_dict[new_key] = value
    return new_state_dict

class oldGPT(pl.LightningModule):
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
            "interval": "epoch",
            "frequency": 1,
        }
    }

    def on_fit_start(self):
        if self.use_ema:
            self.ema_model.to(self.device)


    def forward(self, batch):
        index, before_pts, after_pts, before_normals, after_normals,  masks = batch
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
        predicted_centers = rec_pts.mean(dim=-2)
        target_centers = after_pts.mean(dim=-2)

        center_error = F.smooth_l1_loss(
            predicted_centers,
            target_centers,
            reduction="none",
        ).mean(dim=-1)

        center_error = center_error.flatten()

        center_loss = (
            (center_error * result_masks).sum()
            / result_masks.sum().clamp_min(1.0)
        )
        

        loss1 = (loss1  * result_masks).sum() / result_masks.sum()
        loss2 = (loss2  * result_masks).sum() / result_masks.sum()
        loss = loss1 + loss2 + loss3 + commit_loss.mean() + center_loss

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
    
    # def on_test_start(self):
    #     self.num = 0
    
    # def test_step(self, batch, batch_idx):
    #     index, before_pts, after_pts, before_normals, after_normals,  masks = batch
    #     center = after_pts.mean(dim=-2)
    #     outputroot = '/data3/leics/dataset/teeth/old_gpt_segment_lower'
    #     os.makedirs(outputroot,exist_ok=True)
    #     if self.num > 2000:
    #         exit()
    #     for _ in range(100):
    #         rand_center = center + torch.rand_like(center) * 0.02
    #         eps = 1e-6
    #         rand_pts = torch.from_numpy(np.random.random(after_pts.shape) * eps).cuda().float()
    #         for i in range(16):
    #             rec_pts, predicted_masks,commit_loss = self.transformer(rand_pts, rand_center)
    #             rand_pts[:,i] = rec_pts[:,i]
    #         gt_masks = masks.flatten().long()

    #         criterion = nn.CrossEntropyLoss()
    #         rec_pts = rand_pts

    #         predicted_masks = predicted_masks[:,:-1].reshape(-1,2)
    #         loss3 = criterion(predicted_masks,gt_masks)

    #         result_masks = (masks).to(torch.float).flatten()
    #         gt_points = rearrange(after_pts,'b n p c -> (b n) p c')
    #         predicted_masks = rearrange(predicted_masks, '(b n) c -> b n c', b=before_pts.shape[0])
    #         merged = False
    #         if merged:
    #             for i in range(rec_pts.shape[0]):
    #                 points = []
    #                 for j in range(16):
    #                     if predicted_masks[i,j,1] > predicted_masks[i,j,0]:
    #                         points.append(rec_pts[i][j].cpu().numpy())
    #                 points = np.concatenate(points,axis=0)
    #                 write_pointcloud(points,f'{outputroot}/{self.num}.ply')
    #                 self.num += 1
    #         else:
    #             for i in range(rec_pts.shape[0]):
    #                 for j in range(16):
    #                     if predicted_masks[i,j,1] > predicted_masks[i,j,0]:
    #                         write_pointcloud(rec_pts[i][j].cpu().numpy(),f'{outputroot}/{self.num}_{j}.ply')
    #                 self.num += 1



    def on_test_start(self) -> None:
        """
        Initialize autoregressive point-cloud generation.

        Missing teeth in the dataset are represented by tiny random points
        rather than exact zeros. Generation follows the same convention when
        an absent tooth is fed back as context.
        """
        self.num = 0

        self.test_output_dir = "/data3/leics/dataset/teeth/tmp2/"
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Maximum number of samples saved by each process.
        self.max_test_samples = 2000

        # The current point decoder is deterministic, so repeated generation
        # normally produces the same result.
        self.num_test_generations = 1

        # True: merge all valid teeth in one sample into a single PLY.
        # False: save each valid tooth as a separate PLY.
        self.save_merged = True

        # Keep this consistent with the dataset representation of missing teeth.
        self.missing_point_eps = 1e-6

    @torch.inference_mode()
    def autoregressive_generate(
        self,
        batch_size: int,
        num_teeth: int,
        num_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        GPT-style next-token generation.

        The Transformer internally prepends SOS. At step i, output position i
        predicts tooth i.

        Two point tensors are maintained:

        1. generated_points:
           Stores the raw point prediction for every tooth position. A False
           mask never clears or overwrites this raw prediction.

        2. context_points:
           Stores the sequence fed back into the Transformer. For a valid
           predicted tooth, the raw prediction is used. For an invalid tooth,
           tiny random points are used, matching the dataset convention:

               missing tooth -> random points in [0, 1e-6)

        Only generated_points at valid mask positions are saved.
        """
        # Future placeholders and missing teeth follow the dataset convention.
        context_points = (
            torch.rand(
                batch_size,
                num_teeth,
                num_points,
                3,
                device=device,
                dtype=dtype,
            )
            * self.missing_point_eps
        )
        context_centers = context_points.mean(dim=-2)

        # Raw model predictions. These are never cleared by the mask.
        generated_points = torch.zeros_like(context_points)

        generated_masks = torch.zeros(
            batch_size,
            num_teeth,
            device=device,
            dtype=torch.bool,
        )

        generated_mask_logits = torch.zeros(
            batch_size,
            num_teeth,
            2,
            device=device,
            dtype=dtype,
        )

        for tooth_index in range(num_teeth):
            all_predicted_points, all_predicted_masks, _ = self.transformer(
                context_points,
                context_centers,
            )

            expected_output_length = num_teeth + 1

            if all_predicted_points.shape[1] != expected_output_length:
                raise RuntimeError(
                    "Unexpected point prediction sequence length: "
                    f"expected {expected_output_length}, "
                    f"received {all_predicted_points.shape[1]}."
                )

            if all_predicted_masks.shape[1] != expected_output_length:
                raise RuntimeError(
                    "Unexpected mask prediction sequence length: "
                    f"expected {expected_output_length}, "
                    f"received {all_predicted_masks.shape[1]}."
                )

            # Position i predicts tooth i.
            next_points = all_predicted_points[:, tooth_index]
            next_mask_logits = all_predicted_masks[:, tooth_index]
            next_mask = next_mask_logits.argmax(dim=-1).bool()

            # Always retain the raw point output, even when next_mask is False.
            generated_points[:, tooth_index] = next_points
            generated_masks[:, tooth_index] = next_mask
            generated_mask_logits[:, tooth_index] = next_mask_logits

            # Match the training dataset:
            # valid tooth   -> feed back its generated geometry
            # missing tooth -> feed back tiny random points, not exact zeros
            missing_points = (
                torch.rand_like(next_points) * self.missing_point_eps
            )
            feedback_points = torch.where(
                next_mask[:, None, None],
                next_points,
                missing_points,
            )

            context_points[:, tooth_index] = feedback_points
            context_centers[:, tooth_index] = feedback_points.mean(dim=-2)

        return {
            "points": generated_points,
            "masks": generated_masks,
            "mask_logits": generated_mask_logits,
            "context_points": context_points,
            "context_centers": context_centers,
        }

    def test_step(
        self,
        batch,
        batch_idx: int,
    ) -> None:
        (
            index,
            before_pts,
            after_pts,
            before_normals,
            after_normals,
            masks,
        ) = batch

        if self.num >= self.max_test_samples:
            return

        if after_pts.ndim != 4:
            raise ValueError(
                "after_pts must have shape [B, N, P, 3], "
                f"but received {tuple(after_pts.shape)}."
            )

        batch_size, num_teeth, num_points, coordinate_dim = after_pts.shape

        if coordinate_dim != 3:
            raise ValueError(
                f"Expected point coordinate dimension 3, got {coordinate_dim}."
            )

        expected_num_teeth = self.transformer.num_groups - 1

        if num_teeth != expected_num_teeth:
            raise ValueError(
                "The number of teeth does not match old_transformer: "
                f"batch contains {num_teeth}, "
                f"but transformer expects {expected_num_teeth}."
            )

        for generation_index in range(self.num_test_generations):
            generated = self.autoregressive_generate(
                batch_size=batch_size,
                num_teeth=num_teeth,
                num_points=num_points,
                device=after_pts.device,
                dtype=after_pts.dtype,
            )

            generated_points = generated["points"]
            generated_masks = generated["masks"]
            generated_mask_logits = generated["mask_logits"]

            generated_mask_probabilities = torch.softmax(
                generated_mask_logits.float(),
                dim=-1,
            )[..., 1]

            for batch_index in range(batch_size):
                if self.num >= self.max_test_samples:
                    return

                sample_id = self.num
                rank = self.global_rank

                file_prefix = (
                    f"rank{rank}_"
                    f"sample{sample_id:06d}_"
                    f"generation{generation_index:03d}"
                )

                valid_mask = generated_masks[batch_index]

                if self.save_merged:
                    # Save only valid teeth, but use their untouched raw point
                    # predictions from generated_points.
                    if valid_mask.any():
                        merged_points = generated_points[
                            batch_index,
                            valid_mask,
                        ].reshape(-1, 3)

                        merged_path = os.path.join(
                            self.test_output_dir,
                            f"{file_prefix}.ply",
                        )

                        write_pointcloud(
                            merged_points.detach().cpu().numpy(),
                            merged_path,
                        )
                    else:
                        print(
                            f"[rank {rank}] sample {sample_id}: "
                            "no valid teeth predicted; no PLY was saved."
                        )

                else:
                    # Save only positions whose predicted mask is valid.
                    for tooth_index in range(num_teeth):
                        if not valid_mask[tooth_index]:
                            continue

                        tooth_points = generated_points[
                            batch_index,
                            tooth_index,
                        ]

                        tooth_probability = generated_mask_probabilities[
                            batch_index,
                            tooth_index,
                        ].item()

                        tooth_path = os.path.join(
                            self.test_output_dir,
                            (
                                f"{file_prefix}_"
                                f"tooth{tooth_index:02d}_"
                                f"prob{tooth_probability:.3f}.ply"
                            ),
                        )

                        write_pointcloud(
                            tooth_points.detach().cpu().numpy(),
                            tooth_path,
                        )

                self.num += 1