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
from utils import instantiate_from_config, instantiate_non_trainable_model
from utils import write_pointcloud
import torch.nn.init as init
from typing import Dict, Optional

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
    def __init__(
        self,
        transformer_config,
        optimizer_config=None,
        scheduler_config=None,
        use_ema=True,
        generation_num_candidates: int = 4,
        test_output_dir: str = "./gpt_samples",
        num_test_generations: int = 1,
        max_test_samples: int = 2000,
        save_merged: bool = True,
        missing_point_eps: float = 1e-6,
    ):
        super().__init__()
        self.transformer = instantiate_from_config(transformer_config)
        self.use_ema = bool(use_ema)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.generation_num_candidates = int(generation_num_candidates)
        if self.generation_num_candidates <= 0:
            raise ValueError("generation_num_candidates must be positive.")
        self.test_output_dir = str(test_output_dir)
        self.num_test_generations = int(num_test_generations)
        self.max_test_samples = int(max_test_samples)
        self.save_merged = bool(save_merged)
        self.missing_point_eps = float(missing_point_eps)
        if self.missing_point_eps <= 0:
            raise ValueError("missing_point_eps must be positive.")
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


    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 4:
            index, before_pts, after_pts, masks = batch
        elif len(batch) == 6:
            (
                index,
                before_pts,
                after_pts,
                before_normals,
                after_normals,
                masks,
            ) = batch
            del before_normals, after_normals
        else:
            raise ValueError(
                "oldGPT expects a 4-item point batch or a 6-item batch "
                f"with normals, received {len(batch)} items."
            )
        del index, before_pts
        return after_pts, masks.bool()

    def forward(self, batch):
        after_pts, masks = self._unpack_batch(batch)
        center = after_pts.mean(dim=-2)
        batch_size = after_pts.shape[0]

        if self.training:
            generation_noise = None
            num_candidates = self.generation_num_candidates
        else:
            noise_dim = (
                self.transformer.generator_blocks.generation_noise_dim
            )
            generation_noise = after_pts.new_zeros(batch_size, noise_dim)
            num_candidates = 1

        rec_pts, predicted_masks, commit_loss = self.transformer(
            after_pts,
            center,
            generation_noise=generation_noise,
            num_candidates=num_candidates,
        )
        if rec_pts.ndim == 4:
            rec_pts = rec_pts.unsqueeze(0)
        if rec_pts.ndim != 5:
            raise ValueError(
                "Expected reconstructed points with shape "
                "[K, B, N+1, P, 3]."
            )

        rec_candidates = rec_pts[:, :, :-1]
        candidate_count, _, num_teeth, num_points, _ = (
            rec_candidates.shape
        )
        expanded_gt = after_pts.unsqueeze(0).expand(
            candidate_count, -1, -1, -1, -1
        )
        flat_rec = rec_candidates.reshape(
            candidate_count * batch_size * num_teeth,
            num_points,
            3,
        )
        flat_gt = expanded_gt.reshape_as(flat_rec)

        chamfer_l2 = chamfer_distance(
            flat_rec,
            flat_gt,
            norm=2,
            batch_reduction=None,
            point_reduction="sum",
        )[0].reshape(candidate_count, batch_size, num_teeth)
        chamfer_l1 = chamfer_distance(
            flat_rec,
            flat_gt,
            norm=1,
            batch_reduction=None,
            point_reduction="sum",
        )[0].reshape(candidate_count, batch_size, num_teeth)

        predicted_centers = rec_candidates.mean(dim=-2)
        target_centers = center.unsqueeze(0)
        center_error = F.smooth_l1_loss(
            predicted_centers,
            target_centers.expand_as(predicted_centers),
            reduction="none",
        ).mean(dim=-1)

        valid = masks.to(rec_candidates.dtype).unsqueeze(0)
        valid_per_sample = valid.sum(dim=-1).clamp_min(1.0)
        candidate_scores = (
            ((chamfer_l2 + chamfer_l1 + center_error) * valid).sum(dim=-1)
            / valid_per_sample
        )
        best_candidate = candidate_scores.argmin(dim=0)
        batch_indices = torch.arange(batch_size, device=after_pts.device)

        selected_l2 = chamfer_l2[best_candidate, batch_indices]
        selected_l1 = chamfer_l1[best_candidate, batch_indices]
        selected_center = center_error[best_candidate, batch_indices]
        selected_points = rec_candidates[best_candidate, batch_indices]

        flat_valid = masks.to(selected_l2.dtype)
        valid_count = flat_valid.sum().clamp_min(1.0)
        loss_l2 = (selected_l2 * flat_valid).sum() / valid_count
        loss_l1 = (selected_l1 * flat_valid).sum() / valid_count
        center_loss = (
            (selected_center * flat_valid).sum() / valid_count
        )

        predicted_masks = predicted_masks[:, :-1].reshape(-1, 2)
        mask_loss = F.cross_entropy(
            predicted_masks,
            masks.reshape(-1).long(),
        )
        commit_loss = commit_loss.mean()
        loss = loss_l2 + loss_l1 + mask_loss + commit_loss + center_loss

        if candidate_count > 1:
            candidate_spread = (
                rec_candidates
                - rec_candidates.mean(dim=0, keepdim=True)
            ).square().mean().sqrt()
        else:
            candidate_spread = loss.new_zeros(())

        return {
            "loss": loss,
            "chamfer_l2": loss_l2,
            "chamfer_l1": loss_l1,
            "center_loss": center_loss,
            "mask_loss": mask_loss,
            "commit_loss": commit_loss,
            "candidate_spread": candidate_spread,
            "reconstruction": selected_points,
        }

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            checkpoint["ema_state"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and "ema_state" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state"])

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_model.store(self.transformer.parameters())
            self.ema_model.copy_to(self.transformer.parameters())
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_model.restore(self.transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        if self.use_ema:
            self.ema_model.update()

    def on_validation_start(self):
        if self.use_ema:
            self.ema_model.store(self.transformer.parameters())
            self.ema_model.copy_to(self.transformer.parameters())

    def on_validation_end(self):
        if self.use_ema:
            self.ema_model.restore(self.transformer.parameters())

    def _log_losses(self, output, stage):
        values = {
            f"{stage}_total_loss": output["loss"],
            f"{stage}_chamfer_l2": output["chamfer_l2"],
            f"{stage}_chamfer_l1": output["chamfer_l1"],
            f"{stage}_center_loss": output["center_loss"],
            f"{stage}_mask_loss": output["mask_loss"],
            f"{stage}_commit_loss": output["commit_loss"],
            f"{stage}_candidate_spread": output["candidate_spread"],
        }
        self.log_dict(
            values,
            prog_bar=True,
            logger=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=output["reconstruction"].shape[0],
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        output = self.forward(batch)
        self._log_losses(output, "train")
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output = self.forward(batch)
        self._log_losses(output, "val")
        return output["loss"]
    
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
        self.num = 0
        os.makedirs(self.test_output_dir, exist_ok=True)

    @torch.inference_mode()
    def autoregressive_generate(
        self,
        batch_size: int,
        num_teeth: int,
        num_points: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
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
                generator=generator,
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
        noise_dim = self.transformer.generator_blocks.generation_noise_dim
        generation_noise = torch.randn(
            batch_size,
            noise_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        for tooth_index in range(num_teeth):
            all_predicted_points, all_predicted_masks, _ = self.transformer(
                context_points,
                context_centers,
                generation_noise=generation_noise,
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
            missing_points = torch.rand(
                next_points.shape,
                device=next_points.device,
                dtype=next_points.dtype,
                generator=generator,
            ) * self.missing_point_eps
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

    @torch.inference_mode()
    def generate(
        self,
        output_dir: Optional[str] = None,
        num_samples: int = 1,
        batch_size: int = 1,
        save_merged: Optional[bool] = None,
        seed: int = 3407,
    ) -> None:
        """Generate structured samples without constructing a DataLoader."""
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive.")

        output_dir = output_dir or self.test_output_dir
        save_merged = (
            self.save_merged if save_merged is None else bool(save_merged)
        )
        os.makedirs(output_dir, exist_ok=True)

        parameter = next(self.transformer.parameters())
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(int(seed))
        num_teeth = self.transformer.num_teeth
        num_points = self.transformer.group_size

        saved = 0
        with self.ema_scope():
            while saved < num_samples:
                current_batch = min(batch_size, num_samples - saved)
                generated = self.autoregressive_generate(
                    batch_size=current_batch,
                    num_teeth=num_teeth,
                    num_points=num_points,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    generator=generator,
                )

                for batch_index in range(current_batch):
                    sample_index = saved + batch_index
                    points = generated["points"][batch_index]
                    mask = generated["masks"][batch_index]
                    structured_points = points.clone()
                    missing_count = int((~mask).sum().item())
                    if missing_count:
                        structured_points[~mask] = torch.rand(
                            missing_count,
                            num_points,
                            3,
                            device=points.device,
                            dtype=points.dtype,
                            generator=generator,
                        ) * self.missing_point_eps

                    np.savez(
                        os.path.join(output_dir, f"{sample_index}.npz"),
                        after_pts=structured_points.float().cpu().numpy(),
                        mask=mask.cpu().numpy(),
                    )

                    if save_merged:
                        if mask.any():
                            write_pointcloud(
                                points[mask].reshape(-1, 3).float().cpu().numpy(),
                                os.path.join(output_dir, f"{sample_index}.ply"),
                            )
                    else:
                        for tooth_index in range(num_teeth):
                            if not mask[tooth_index]:
                                continue
                            write_pointcloud(
                                points[tooth_index].float().cpu().numpy(),
                                os.path.join(
                                    output_dir,
                                    f"{sample_index}_{tooth_index}.ply",
                                ),
                            )
                saved += current_batch

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