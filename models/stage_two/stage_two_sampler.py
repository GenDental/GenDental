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
from contextlib import contextmanager, nullcontext
import math
from utils import instantiate_from_config
from torch_ema import ExponentialMovingAverage as EMA
from utils import instantiate_from_config, instantiate_non_trainable_model
from utils import write_pointcloud
import torch.nn.init as init
from models.stage_two.smooth import postprocess_smooth_se3
from pathlib import Path
from typing import List, Tuple
from utils import read_pointcloud

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
    def __init__(
        self, transformer_config, optimizer_config=None,
        scheduler_config=None, use_ema=True, task_mode='motion',
    ):
        super().__init__()
        if task_mode not in {'target', 'motion'}:
            raise ValueError(
                "task_mode must be either 'target' or 'motion', "
                f"got {task_mode!r}."
            )
        self.task_mode = task_mode
        if 'params' not in transformer_config:
            transformer_config['params'] = {}
        transformer_config['params']['num_steps'] = (
            1 if task_mode == 'target' else 21
        )
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

        
        squared_error = criterion(predicted_points, gt_points)
        valid_teeth = masks.sum().clamp_min(1.0)
        rec_loss = (
            squared_error * masks.reshape(-1, 1, 1)
        ).sum() / valid_teeth
        loss = 100 * rec_loss 

        return loss
    
    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            checkpoint['ema_state'] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema and 'ema_state' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state'])
    
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

    @staticmethod
    def _sort_key(path: Path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

    @staticmethod
    def _load_style(
        path: Path,
        num_teeth: int,
        num_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Load style geometry without using its mask as supervision."""
        with np.load(path) as sample:
            if 'before_pts' in sample:
                style_points = sample['before_pts']
            elif 'style_pts' in sample:
                style_points = sample['style_pts']
            else:
                raise KeyError(
                    f'{path} must contain before_pts or style_pts.'
                )

        expected_shape = (num_teeth, num_points, 3)
        if style_points.shape != expected_shape:
            raise ValueError(
                f'{path}: expected style shape {expected_shape}, '
                f'got {style_points.shape}.'
            )
        style_points = style_points.astype(np.float32, copy=True)
        if not np.isfinite(style_points).all():
            raise ValueError(f'{path}: style points contain NaN or Inf.')

        # The style mask must not decide which teeth are generated. However,
        # zero-padded style teeth are unsafe inputs for the point encoder.
        # Replace only degenerate geometry with tiny noise, without reading or
        # applying the style mask.
        tooth_extent = np.ptp(style_points, axis=1).max(axis=1)
        degenerate = tooth_extent < 1e-8
        degenerate_count = int(degenerate.sum())
        if degenerate_count:
            style_points[degenerate] = (
                rng.random((degenerate_count, num_points, 3)) * 1e-6
            )
        return style_points

    @staticmethod
    def _load_after(
        path: Path,
        num_teeth: int,
        num_points: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load Stage-I geometry and its authoritative tooth mask."""
        if path.suffix.lower() != '.npz':
            raise ValueError(
                f'{path}: Stage-II sampling requires a Stage-I NPZ so '
                'the generated mask is available; PLY input is ambiguous.'
            )

        with np.load(path) as sample:
            if 'after_pts' in sample:
                points = sample['after_pts']
            elif 'points' in sample:
                points = sample['points']
            else:
                raise KeyError(
                    f'{path} must contain after_pts or points.'
                )
            if 'mask' not in sample:
                raise KeyError(
                    f'{path} must contain the Stage-I generated mask.'
                )
            masks = sample['mask']

        expected_shape = (num_teeth, num_points, 3)
        if points.shape != expected_shape:
            raise ValueError(
                f'{path}: expected point shape {expected_shape}, '
                f'got {points.shape}.'
            )
        if masks.shape != (num_teeth,):
            raise ValueError(
                f'{path}: expected mask shape {(num_teeth,)}, '
                f'got {masks.shape}.'
            )

        points = points.astype(np.float32, copy=True)
        if not np.isfinite(points).all():
            raise ValueError(f'{path}: Stage-I points contain NaN or Inf.')
        masks = masks.astype(bool, copy=False)
        missing_count = int((~masks).sum())
        if missing_count:
            points[~masks] = (
                rng.random((missing_count, num_points, 3)) * 1e-6
            )
        return points, masks

    @classmethod
    def _paired_inputs(
        cls,
        style_dir: Path,
        data_dir: Path,
        rng: np.random.Generator,
    ) -> List[Tuple[str, Path, Path]]:
        style_files = sorted(
            style_dir.glob('*.npz'),
            key=cls._sort_key,
        )
        data_files = {}
        for suffix in ('*.npz', '*.ply'):
            for path in data_dir.glob(suffix):
                data_files.setdefault(path.stem, path)

        if not style_files:
            raise FileNotFoundError(
                f'No style NPZ files found in: {style_dir}'
            )
        if not data_files:
            raise FileNotFoundError(
                f'No data PLY or NPZ files found in: {data_dir}'
            )

        ordered_data = sorted(data_files.values(), key=cls._sort_key)
        return [
            (
                data_file.stem,
                style_files[int(rng.integers(len(style_files)))],
                data_file,
            )
            for data_file in ordered_data
        ]

    def _generate_batch(
        self,
        style_points: torch.Tensor,
        after_points: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_mode == 'target':
            return self._generate_target_batch(
                style_points, after_points, masks
            )
        return self._generate_motion_batch(
            style_points, after_points, masks
        )

    def _generate_target_batch(
        self,
        style_points: torch.Tensor,
        after_points: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        style_centers = style_points.mean(dim=-2)
        after_centers = after_points.mean(dim=-2)
        predicted = self.transformer(
            style_points, style_centers, after_points, masks
        )
        rotations = rotation_6d_to_matrix(
            rearrange(predicted[:, :, :6], 'b n c -> (b n) c')
        )
        translations = rearrange(
            predicted[:, :, 6:], 'b n c -> (b n) c'
        )
        grouped_after = rearrange(
            after_points, 'b n p c -> (b n) p c'
        )
        grouped_centers = rearrange(
            after_centers, 'b n c -> (b n) c'
        )
        generated = torch.bmm(
            grouped_after - grouped_centers.unsqueeze(1), rotations
        ) + (translations + grouped_centers).unsqueeze(1)
        generated = rearrange(
            generated, '(b n) p c -> b n p c', n=32
        )

        batch_size, num_teeth = style_points.shape[:2]
        matrices = torch.zeros(
            batch_size, 1, num_teeth, 4, 4,
            device=style_points.device, dtype=style_points.dtype,
        )
        matrices[:, 0, :, :3, :3] = rearrange(
            rotations, '(b n) c1 c2 -> b n c1 c2',
            b=batch_size, n=num_teeth,
        )
        matrices[:, 0, :, :3, 3] = rearrange(
            translations, '(b n) c -> b n c',
            b=batch_size, n=num_teeth,
        )
        matrices[:, 0, :, 3, 3] = 1
        return generated, matrices

    def _generate_motion_batch(
        self,
        style_points: torch.Tensor,
        after_points: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        style_centers = style_points.mean(dim=-2)
        predicted = self.transformer(
            style_points,
            style_centers,
            after_points,
            masks,
        )
        predicted = rearrange(
            predicted, 'b p (l c) -> b l p c', c=9
        )
        batch_size, num_steps, num_teeth, _ = predicted.shape
        rotations = rotation_6d_to_matrix(predicted[..., :6])
        translations = predicted[..., 6:]

        matrices = torch.zeros(
            batch_size,
            num_steps,
            num_teeth,
            4,
            4,
            device=style_points.device,
            dtype=style_points.dtype,
        )
        matrices[..., :3, :3] = rotations
        matrices[..., :3, 3] = translations
        matrices[..., 3, 3] = 1

        repeated_after = repeat(
            after_points,
            'b p n c -> (b l p) n c',
            l=num_steps,
        )
        transforms = Transform3d(
            matrix=rearrange(
                matrices,
                'b l p c1 c2 -> (b l p) c2 c1',
            )
        )
        generated = transforms.transform_points(repeated_after)
        generated = rearrange(
            generated,
            '(b l p) n c -> b l p n c',
            b=batch_size,
            l=num_steps,
            p=num_teeth,
        )
        return generated[:, 0], matrices

    @torch.inference_mode()
    def generate(
        self,
        style_dir: str,
        data_dir: str,
        output_dir: str = 'stage_two_samples',
        batch_size: int = 1,
        num_teeth: int = 32,
        num_points: int = 512,
        seed: int = 3407,
    ) -> None:
        """Generate Stage II samples directly from two input directories."""
        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        style_path = Path(style_dir).expanduser().resolve()
        data_path = Path(data_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        if not style_path.is_dir():
            raise FileNotFoundError(f'Style directory not found: {style_path}')
        if not data_path.is_dir():
            raise FileNotFoundError(f'Data directory not found: {data_path}')
        output_path.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        pairs = self._paired_inputs(style_path, data_path, rng)
        parameter = next(self.parameters())

        with nullcontext():  # Use the same raw weights as validation.
            for start in range(0, len(pairs), batch_size):
                current_pairs = pairs[start:start + batch_size]
                styles, afters, masks = [], [], []
                for _, style_file, data_file in current_pairs:
                    style = self._load_style(
                        style_file, num_teeth, num_points, rng
                    )
                    after, stage_one_mask = self._load_after(
                        data_file, num_teeth, num_points, rng
                    )
                    styles.append(style)
                    afters.append(after)
                    masks.append(stage_one_mask)

                style_tensor = torch.as_tensor(
                    np.stack(styles),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                after_tensor = torch.as_tensor(
                    np.stack(afters),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                mask_tensor = torch.as_tensor(
                    np.stack(masks),
                    device=parameter.device,
                    dtype=torch.bool,
                )
                generated, predicted_matrices = self._generate_batch(
                    style_tensor, after_tensor, mask_tensor
                )
                if not torch.isfinite(generated).all():
                    stems = [pair[0] for pair in current_pairs]
                    raise FloatingPointError(
                        'Stage-II generated NaN or Inf for samples '
                        f'{stems}. Refusing to write invalid NPZ files.'
                    )


                for index, (stem, style_file, _) in enumerate(current_pairs):
                    if self.task_mode == 'target':
                        np.savez(
                            output_path / f'{stem}.npz',
                            before_pts=generated[index].float().cpu().numpy(),
                            after_pts=after_tensor[index].float().cpu().numpy(),
                            mask=mask_tensor[index].cpu().numpy(),
                        )
                        continue

                    out_matrices = predicted_matrices[index].cpu().numpy()
                    relative_matrices = np.tile(
                        np.eye(4, dtype=out_matrices.dtype),
                        (out_matrices.shape[0], num_teeth, 1, 1),
                    )
                    first_inverse = np.linalg.inv(out_matrices[0])
                    for step in range(out_matrices.shape[0] - 1):
                        relative_matrices[step] = (
                            out_matrices[step + 1] @ first_inverse
                        )
                    relative_matrices[-1] = first_inverse

                    np.savez(
                        output_path / f'{stem}.npz',
                        before_pts=generated[index].float().cpu().numpy(),
                        after_pts=after_tensor[index].float().cpu().numpy(),
                        style_pts=style_tensor[index].float().cpu().numpy(),
                        mask=mask_tensor[index].cpu().numpy(),
                        style_id=np.asarray(style_file.stem),
                        matrices=relative_matrices,
                        inv_matrices=out_matrices,
                    )
    
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
