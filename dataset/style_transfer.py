
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import copy
import open3d as o3d
from einops import rearrange
from utils import read_pointcloud
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch3d.transforms import matrix_to_rotation_6d

class TargetDataset(data.Dataset):
    # Initial/final-state samples used by the original style branch.
    def __init__(self, data_path, index_file, with_normals=False):
        super().__init__()
        self.data_path = data_path
        self.indexes = np.load(index_file)
        self.indexes = self.indexes[self.indexes != 145]
        self.with_normals = with_normals

    def __getitem__(self, idx):
        index = self.indexes[idx]
        sample = np.load(os.path.join(self.data_path, f'{index}.npz'))
        before_pts = sample['before_pts'].copy()
        after_pts = sample['after_pts'].copy()
        mask = sample['mask']

        if self.with_normals:
            before_normals = sample['before_normals']
            after_normals = sample['after_normals']
        else:
            before_normals = np.zeros_like(before_pts)
            after_normals = np.zeros_like(after_pts)

        eps = 1e-6
        for i in range(32):
            if not mask[i]:
                before_pts[i] = after_pts[i] = (
                    np.random.random(after_pts[i].shape) * eps
                )

        return (
            index,
            torch.tensor(before_pts, dtype=torch.float32),
            torch.tensor(after_pts, dtype=torch.float32),
            torch.tensor(before_normals, dtype=torch.float32),
            torch.tensor(after_normals, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.indexes)


class MotionDataset(data.Dataset):
    def __init__(self, data_path, index_file):
        super().__init__()
        self.data_path = data_path
        self.indexes = np.load(index_file)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        data = np.load(os.path.join(self.data_path, f'{index}.npz'))
        before_pts = data['before_pts']
        after_pts = data['after_pts']
        mask = data['mask']
        # before_normals = data['before_normals']
        # after_normals = data['after_normals']
        matrices = data['matrices']
        inv_matrices = data['inv_matrices']
        eps = 1e-6
        for i in range(32):
            if not mask[i]:
                before_pts[i] = after_pts[i] = np.random.random(after_pts[i].shape) * eps
        before_pts = torch.tensor(before_pts,dtype=torch.float32)
        after_pts = torch.tensor(after_pts,dtype=torch.float32)
        # before_normals = torch.tensor(before_normals,dtype=torch.float32)
        # after_normals = torch.tensor(after_normals,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.float32)
        matrices = torch.tensor(matrices,dtype=torch.float32)
        inv_matrices = torch.tensor(inv_matrices,dtype=torch.float32)
        # return index, before_pts,after_pts,before_normals,after_normals,mask,matrices,inv_matrices
        # before_pts [32,512,3]
        # matrices [L,32,4,4]
        # mask [32]
        return index, before_pts,after_pts,mask,matrices,inv_matrices
    
    def __len__(self):
        return len(self.indexes)

class MotionDataManager(pl.LightningDataModule):
    def __init__(self, data_path, index_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size = batch_size

    def train_dataloader(self):
        file = os.path.join(self.index_path,'train.npy')
        train_dataset = MotionDataset(self.data_path,file)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        file = os.path.join(self.index_path,'val.npy')
        val_dataset = MotionDataset(self.data_path,file)
        return DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True)
    
    def test_dataloader(self):
        file = os.path.join(self.index_path,'test.npy')
        test_dataset = MotionDataset(self.data_path,file)
        return DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)


class StageTwoDataManager(pl.LightningDataModule):
    # Select the unchanged target or motion batch format by task_mode.
    def __init__(
        self, data_path, index_path, batch_size, task_mode='motion',
        with_normals=False,
    ):
        super().__init__()
        if task_mode not in {'target', 'motion'}:
            raise ValueError(
                "task_mode must be either 'target' or 'motion', "
                f"got {task_mode!r}."
            )
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size = batch_size
        self.task_mode = task_mode
        self.with_normals = with_normals

    def _dataset(self, split):
        # The original target branch used test.npy for validation, while the
        # current motion branch uses a separate val.npy.
        filename = (
            'test.npy'
            if self.task_mode == 'target' and split == 'val'
            else f'{split}.npy'
        )
        index_file = os.path.join(self.index_path, filename)
        if self.task_mode == 'target':
            return TargetDataset(
                self.data_path, index_file, self.with_normals
            )
        return MotionDataset(self.data_path, index_file)

    def train_dataloader(self):
        return DataLoader(
            self._dataset('train'),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.task_mode == 'motion',
        )

    def val_dataloader(self):
        return DataLoader(
            self._dataset('val'),
            batch_size=self.batch_size,
            drop_last=self.task_mode == 'motion',
        )

    def test_dataloader(self):
        return DataLoader(
            self._dataset('test'),
            batch_size=self.batch_size,
            drop_last=self.task_mode == 'motion',
        )
