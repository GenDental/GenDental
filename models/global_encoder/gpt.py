import os
from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from utils import instantiate_from_config, write_pointcloud

try:
    from torch_ema import ExponentialMovingAverage as EMA
except ImportError:
    EMA = None


class LatentGPT(pl.LightningModule):
    """Lightning wrapper for continuous latent VAE + GPT prior."""

    def __init__(
        self,
        transformer_config,
        optimizer_config=None,
        scheduler_config=None,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        posterior_reconstruction_weight: float = 1.0,
        prior_reconstruction_weight: float = 1.0,
        latent_prior_kl_weight: float = 0.1,
        posterior_standard_kl_weight: float = 1e-4,
        center_loss_weight: float = 1.0,
        center_nll_weight: Optional[float] = None,
        mask_weight: float = 1.0,
        chamfer_l2_weight: float = 1.0,
        chamfer_l1_weight: float = 1.0,
        kl_warmup_epochs: int = 0,
        latent_prior_free_bits: float = 0.0,
        posterior_standard_free_bits: float = 0.0,
        mask_class_weights: Optional[Sequence[float]] = None,
        test_output_dir: str = "/data3/leics/dataset/teeth/latent_gpt_samples",
        num_test_generations: int = 8,
        max_test_samples: int = 2000,
        save_merged: bool = True,
        latent_temperature: float = 1.0,
        center_temperature: float = 0.5,
        mask_temperature: float = 0.8,
    ) -> None:
        super().__init__()

        self.transformer = instantiate_from_config(transformer_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.posterior_reconstruction_weight = float(
            posterior_reconstruction_weight
        )
        self.prior_reconstruction_weight = float(prior_reconstruction_weight)
        self.latent_prior_kl_weight = float(latent_prior_kl_weight)
        self.posterior_standard_kl_weight = float(
            posterior_standard_kl_weight
        )
        # Backward-compatible alias for older YAML files. The value now
        # weights a non-negative Smooth L1 center loss, not Gaussian NLL.
        if center_nll_weight is not None:
            center_loss_weight = center_nll_weight
        self.center_loss_weight = float(center_loss_weight)
        self.mask_weight = float(mask_weight)
        self.chamfer_l2_weight = float(chamfer_l2_weight)
        self.chamfer_l1_weight = float(chamfer_l1_weight)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.latent_prior_free_bits = float(latent_prior_free_bits)
        self.posterior_standard_free_bits = float(
            posterior_standard_free_bits
        )
        if self.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be non-negative.")
        if self.latent_prior_free_bits < 0.0:
            raise ValueError("latent_prior_free_bits must be non-negative.")
        if self.posterior_standard_free_bits < 0.0:
            raise ValueError(
                "posterior_standard_free_bits must be non-negative."
            )

        if mask_class_weights is None:
            class_weights = None
        else:
            if len(mask_class_weights) != 2:
                raise ValueError(
                    "mask_class_weights must contain [missing, present]."
                )
            class_weights = torch.tensor(
                list(mask_class_weights),
                dtype=torch.float32,
            )
            if (class_weights <= 0.0).any():
                raise ValueError("mask_class_weights must be positive.")
        self.register_buffer(
            "mask_class_weights_tensor",
            class_weights,
            persistent=True,
        )

        loss_weights = {
            "posterior_reconstruction_weight": self.posterior_reconstruction_weight,
            "prior_reconstruction_weight": self.prior_reconstruction_weight,
            "latent_prior_kl_weight": self.latent_prior_kl_weight,
            "posterior_standard_kl_weight": self.posterior_standard_kl_weight,
            "center_loss_weight": self.center_loss_weight,
            "mask_weight": self.mask_weight,
            "chamfer_l2_weight": self.chamfer_l2_weight,
            "chamfer_l1_weight": self.chamfer_l1_weight,
        }
        negative_weights = {
            name: value for name, value in loss_weights.items() if value < 0.0
        }
        if negative_weights:
            raise ValueError(
                "All loss weights must be non-negative, but received: "
                f"{negative_weights}"
            )

        self.test_output_dir = test_output_dir
        self.num_test_generations = int(num_test_generations)
        self.max_test_samples = int(max_test_samples)
        self.save_merged = bool(save_merged)
        self.latent_temperature = float(latent_temperature)
        self.center_temperature = float(center_temperature)
        self.mask_temperature = float(mask_temperature)

        self.use_ema = bool(use_ema)
        if self.use_ema:
            if EMA is None:
                raise ImportError(
                    "torch_ema is required when use_ema=True."
                )
            self.ema_model = EMA(
                self.transformer.parameters(),
                decay=ema_decay,
            )
        else:
            self.ema_model = None

    def configure_optimizers(self):
        trainable_params = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        optimizer = instantiate_from_config(
            self.optimizer_config,
            params=trainable_params,
            lr=self.learning_rate,
        )

        if self.scheduler_config is None:
            return optimizer

        scheduler = instantiate_from_config(
            self.scheduler_config,
            optimizer=optimizer,
            max_epochs=self.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:
        if self.ema_model is not None:
            self.ema_model.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.ema_model is not None:
            self.ema_model.update()

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.ema_model is not None:
            checkpoint["ema_state"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint) -> None:
        if self.ema_model is not None and "ema_state" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state"])

    @contextmanager
    def ema_scope(self):
        if self.ema_model is None:
            yield
            return

        self.ema_model.store(self.transformer.parameters())
        self.ema_model.copy_to(self.transformer.parameters())
        try:
            yield
        finally:
            self.ema_model.restore(self.transformer.parameters())

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        # Boolean indexing avoids the unsafe NaN * 0 pattern. For values with
        # shape [B, N, D], values[masks] has shape [num_valid, D].
        valid_masks = masks.bool()
        if not valid_masks.any():
            return values.sum() * 0.0
        return values[valid_masks].float().mean()

    @staticmethod
    def _diagonal_gaussian_kl(
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mu: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q || p), returned per latent dimension in FP32."""
        q_mu = q_mu.float()
        q_logvar = q_logvar.float().clamp(-12.0, 8.0)
        p_mu = p_mu.float()
        p_logvar = p_logvar.float().clamp(-12.0, 8.0)

        # exp(q_logvar - p_logvar) is more stable than computing two
        # exponentials and dividing them.
        variance_ratio = torch.exp(q_logvar - p_logvar)
        squared_mean_term = (q_mu - p_mu).pow(2) * torch.exp(-p_logvar)
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + variance_ratio
            + squared_mean_term
            - 1.0
        )
        return kl.clamp_min(0.0)

    @staticmethod
    def _standard_gaussian_kl(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        mu = mu.float()
        logvar = logvar.float().clamp(-12.0, 8.0)
        kl = 0.5 * (
            torch.exp(logvar) + mu.pow(2) - 1.0 - logvar
        )
        return kl.clamp_min(0.0)

    @staticmethod
    def _strict_nonnegative(
        name: str,
        value: torch.Tensor,
        tolerance: float = 1e-6,
    ) -> torch.Tensor:
        """Keep mathematically non-negative losses non-negative.

        Tiny negative values can occur from floating-point roundoff. They are
        clamped to zero. A materially negative value indicates an implementation
        bug and raises immediately rather than being silently hidden.
        """
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Loss component {name} contains NaN or Inf: {value.detach()}"
            )

        minimum = value.detach().float().min().item()
        if minimum < -tolerance:
            raise FloatingPointError(
                f"Loss component {name} must be non-negative, "
                f"but its minimum is {minimum}."
            )
        return value.clamp_min(0.0)

    @staticmethod
    def _chamfer_on_valid_teeth(
        predicted_points: torch.Tensor,
        target_points: torch.Tensor,
        masks: torch.Tensor,
        norm: int,
    ) -> torch.Tensor:
        valid_masks = masks.bool()
        if not valid_masks.any():
            return predicted_points.sum() * 0.0

        # Pytorch3D Chamfer is much safer in FP32 than under BF16/FP16
        # autocast, especially during the first unstable optimization steps.
        predicted_valid = predicted_points[valid_masks].float()
        target_valid = target_points[valid_masks].float()
        return chamfer_distance(
            predicted_valid,
            target_valid,
            norm=norm,
            batch_reduction="mean",
            point_reduction="mean",
        )[0]

    def _kl_warmup_factor(self) -> float:
        if self.kl_warmup_epochs == 0:
            return 1.0
        return min(1.0, float(self.current_epoch) / self.kl_warmup_epochs)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
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
                'LatentGPT expects a 4-item point batch or a 6-item batch '
                f'with normals, received {len(batch)} items.'
            )
        del index, before_pts

        masks = masks.bool()
        centers = after_pts.mean(dim=2)

        output = self.transformer(
            neighborhood=after_pts,
            center=centers,
            input_masks=masks,
            # Validation must be deterministic so checkpoint ranking does not
            # depend on a fresh posterior sample every epoch.
            deterministic_posterior=not self.training,
        )

        zero_reconstruction_loss = (
            output["posterior_reconstruction"].sum() * 0.0
        )
        posterior_chamfer_l2 = zero_reconstruction_loss
        posterior_chamfer_l1 = zero_reconstruction_loss
        if self.chamfer_l2_weight > 0.0:
            posterior_chamfer_l2 = self._chamfer_on_valid_teeth(
                output["posterior_reconstruction"],
                after_pts,
                masks,
                norm=2,
            )
        if self.chamfer_l1_weight > 0.0:
            posterior_chamfer_l1 = self._chamfer_on_valid_teeth(
                output["posterior_reconstruction"],
                after_pts,
                masks,
                norm=1,
            )
        posterior_reconstruction_loss = (
            self.chamfer_l2_weight * posterior_chamfer_l2
            + self.chamfer_l1_weight * posterior_chamfer_l1
        )

        prior_chamfer_l2 = zero_reconstruction_loss
        prior_chamfer_l1 = zero_reconstruction_loss
        if self.chamfer_l2_weight > 0.0:
            prior_chamfer_l2 = self._chamfer_on_valid_teeth(
                output["prior_reconstruction"],
                after_pts,
                masks,
                norm=2,
            )
        if self.chamfer_l1_weight > 0.0:
            prior_chamfer_l1 = self._chamfer_on_valid_teeth(
                output["prior_reconstruction"],
                after_pts,
                masks,
                norm=1,
            )
        prior_reconstruction_loss = (
            self.chamfer_l2_weight * prior_chamfer_l2
            + self.chamfer_l1_weight * prior_chamfer_l1
        )

        latent_prior_kl_per_dim = self._diagonal_gaussian_kl(
            output["posterior_mu"],
            output["posterior_logvar"],
            output["prior_mu"],
            output["prior_logvar"],
        )
        latent_prior_kl_raw = self._masked_mean(
            latent_prior_kl_per_dim,
            masks,
        )
        latent_prior_kl = self._masked_mean(
            latent_prior_kl_per_dim.clamp_min(self.latent_prior_free_bits),
            masks,
        )

        posterior_standard_kl_per_dim = self._standard_gaussian_kl(
            output["posterior_mu"],
            output["posterior_logvar"],
        )
        posterior_standard_kl_raw = self._masked_mean(
            posterior_standard_kl_per_dim,
            masks,
        )
        posterior_standard_kl = self._masked_mean(
            posterior_standard_kl_per_dim.clamp_min(
                self.posterior_standard_free_bits
            ),
            masks,
        )

        if masks.any():
            center_loss = F.smooth_l1_loss(
                output["center_mu_normalized"][masks].float(),
                output["target_centers_normalized"][masks].float(),
                reduction="mean",
            )
        else:
            center_loss = output["center_mu_normalized"].sum() * 0.0

        mask_loss = F.cross_entropy(
            output["mask_logits"].reshape(-1, 2),
            masks.reshape(-1).long(),
            weight=self.mask_class_weights_tensor,
        )

        kl_warmup_factor = self._kl_warmup_factor()

        total_loss = (
            self.posterior_reconstruction_weight
            * posterior_reconstruction_loss
            + self.prior_reconstruction_weight
            * prior_reconstruction_loss
            + kl_warmup_factor
            * self.latent_prior_kl_weight
            * latent_prior_kl
            + kl_warmup_factor
            * self.posterior_standard_kl_weight
            * posterior_standard_kl
            + self.center_loss_weight * center_loss
            + self.mask_weight * mask_loss
        )

        # Every reported component is mathematically non-negative. Do not
        # silently accept a negative component: clamp only tiny roundoff and
        # raise for a real implementation error.
        posterior_reconstruction_loss = self._strict_nonnegative(
            "posterior_reconstruction_loss", posterior_reconstruction_loss
        )
        prior_reconstruction_loss = self._strict_nonnegative(
            "prior_reconstruction_loss", prior_reconstruction_loss
        )
        latent_prior_kl = self._strict_nonnegative(
            "latent_prior_kl", latent_prior_kl
        )
        posterior_standard_kl = self._strict_nonnegative(
            "posterior_standard_kl", posterior_standard_kl
        )
        center_loss = self._strict_nonnegative(
            "center_loss", center_loss
        )
        mask_loss = self._strict_nonnegative("mask_loss", mask_loss)

        total_loss = self._strict_nonnegative("total_loss", total_loss)

        return {
            "loss": total_loss,
            "posterior_reconstruction_loss": posterior_reconstruction_loss,
            "prior_reconstruction_loss": prior_reconstruction_loss,
            "posterior_chamfer_l2": posterior_chamfer_l2,
            "posterior_chamfer_l1": posterior_chamfer_l1,
            "prior_chamfer_l2": prior_chamfer_l2,
            "prior_chamfer_l1": prior_chamfer_l1,
            "latent_prior_kl": latent_prior_kl,
            "latent_prior_kl_raw": latent_prior_kl_raw,
            "posterior_standard_kl": posterior_standard_kl,
            "posterior_standard_kl_raw": posterior_standard_kl_raw,
            "kl_warmup_factor": total_loss.new_tensor(kl_warmup_factor),
            "center_loss": center_loss,
            "mask_loss": mask_loss,
        }

    def _shared_step(self, batch, stage: str):
        output = self.forward(batch)
        batch_size = batch[2].shape[0]

        log_values = {
            f"{stage}_total_loss": output["loss"],
            f"{stage}_posterior_rec": output[
                "posterior_reconstruction_loss"
            ],
            f"{stage}_prior_rec": output["prior_reconstruction_loss"],
            f"{stage}_posterior_cd_l2": output["posterior_chamfer_l2"],
            f"{stage}_prior_cd_l2": output["prior_chamfer_l2"],
            f"{stage}_latent_prior_kl": output["latent_prior_kl"],
            f"{stage}_latent_prior_kl_raw": output["latent_prior_kl_raw"],
            f"{stage}_posterior_standard_kl": output[
                "posterior_standard_kl"
            ],
            f"{stage}_kl_warmup": output["kl_warmup_factor"],
            f"{stage}_center_loss": output["center_loss"],
            f"{stage}_mask_loss": output["mask_loss"],
        }

        self.log_dict(
            log_values,
            prog_bar=True,
            logger=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_test_start(self) -> None:
        self.num = 0
        os.makedirs(self.test_output_dir, exist_ok=True)

    @torch.inference_mode()
    def generate(
        self,
        output_dir: Optional[str] = None,
        num_samples: int = 1,
        batch_size: int = 1,
        save_merged: Optional[bool] = None,
        latent_temperature: Optional[float] = None,
        center_temperature: Optional[float] = None,
        mask_temperature: Optional[float] = None,
        seed: int = 3407,
    ) -> None:
        """Generate and save samples without constructing a DataLoader."""
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError('num_samples and batch_size must be positive.')

        output_dir = output_dir or self.test_output_dir
        save_merged = (
            self.save_merged if save_merged is None else bool(save_merged)
        )
        latent_temperature = (
            self.latent_temperature
            if latent_temperature is None
            else float(latent_temperature)
        )
        center_temperature = (
            self.center_temperature
            if center_temperature is None
            else float(center_temperature)
        )
        mask_temperature = (
            self.mask_temperature
            if mask_temperature is None
            else float(mask_temperature)
        )
        os.makedirs(output_dir, exist_ok=True)

        parameter = next(self.parameters())
        saved = 0
        with self.ema_scope():
            while saved < num_samples:
                current_batch = min(batch_size, num_samples - saved)
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(seed + saved)
                generated = self.transformer.generate(
                    batch_size=current_batch,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    latent_temperature=latent_temperature,
                    center_temperature=center_temperature,
                    mask_temperature=mask_temperature,
                    sample_masks=True,
                    generator=generator,
                )

                for batch_index in range(current_batch):
                    sample_index = saved + batch_index
                    valid_mask = generated['masks'][batch_index]
                    if save_merged:
                        structured_points = generated['points'][
                            batch_index
                        ].clone()
                        missing_count = int((~valid_mask).sum().item())
                        if missing_count:
                            missing_points = torch.rand(
                                missing_count,
                                structured_points.shape[1],
                                3,
                                device=structured_points.device,
                                dtype=structured_points.dtype,
                                generator=generator,
                            ) * 1e-6
                            structured_points[~valid_mask] = missing_points
                        np.savez(
                            os.path.join(
                                output_dir, f'{sample_index}.npz'
                            ),
                            after_pts=structured_points.float().cpu().numpy(),
                            mask=valid_mask.cpu().numpy(),
                        )
                        if valid_mask.any():
                            points = generated['points'][
                                batch_index, valid_mask
                            ].reshape(-1, 3)
                            write_pointcloud(
                                points.float().cpu().numpy(),
                                os.path.join(
                                    output_dir, f'{sample_index}.ply'
                                ),
                            )
                        continue

                    for tooth_index in range(self.transformer.num_teeth):
                        if not valid_mask[tooth_index]:
                            continue
                        points = generated['points'][
                            batch_index, tooth_index
                        ]
                        write_pointcloud(
                            points.float().cpu().numpy(),
                            os.path.join(
                                output_dir,
                                f'{sample_index}_{tooth_index}.ply',
                            ),
                        )
                saved += current_batch

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        after_pts = batch[2]
        batch_size = after_pts.shape[0]

        if self.num >= self.max_test_samples:
            return

        with self.ema_scope():
            for generation_index in range(self.num_test_generations):
                seed = (
                    3407
                    + self.global_rank * 1_000_000
                    + batch_idx * 10_000
                    + generation_index
                )
                generator = torch.Generator(device=after_pts.device)
                generator.manual_seed(seed)

                generated = self.transformer.generate(
                    batch_size=batch_size,
                    device=after_pts.device,
                    dtype=after_pts.dtype,
                    latent_temperature=self.latent_temperature,
                    center_temperature=self.center_temperature,
                    mask_temperature=self.mask_temperature,
                    sample_masks=True,
                    generator=generator,
                )

                generated_points = generated["points"]
                generated_masks = generated["masks"]
                mask_probabilities = torch.softmax(
                    generated["mask_logits"].float(),
                    dim=-1,
                )[..., 1]

                for sample_index in range(batch_size):
                    if self.num >= self.max_test_samples:
                        return

                    prefix = (
                        f"rank{self.global_rank}_"
                        f"sample{self.num:06d}_"
                        f"generation{generation_index:03d}"
                    )

                    valid_mask = generated_masks[sample_index]
                    if self.save_merged:
                        if valid_mask.any():
                            merged_points = generated_points[
                                sample_index,
                                valid_mask,
                            ].reshape(-1, 3)
                            num = generation_index * batch_size + sample_index
                            write_pointcloud(
                                merged_points.detach().cpu().numpy(),
                                os.path.join(
                                    self.test_output_dir,
                                    f"{num}.ply",
                                ),
                            )
                    else:
                        num = generation_index * batch_size + sample_index
                        for tooth_index in range(
                            self.transformer.num_teeth
                        ):
                            if not valid_mask[tooth_index]:
                                continue
                            probability = mask_probabilities[
                                sample_index,
                                tooth_index,
                            ].item()
                            write_pointcloud(
                                generated_points[
                                    sample_index,
                                    tooth_index,
                                ].detach().cpu().numpy(),
                                os.path.join(
                                    self.test_output_dir,
                                    (
                                        f"{num}_"
                                        f"{tooth_index}.ply"
                                    ),
                                ),
                            )

                    self.num += 1
