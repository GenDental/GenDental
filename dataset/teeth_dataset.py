
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

class TeethDataset(data.Dataset):
    def __init__(self, data_path, index_file, with_normals=False):
        super().__init__()
        self.data_path = data_path
        self.indexes = np.load(index_file)
        self.indexes = self.indexes[self.indexes != 145] 
        self.with_normals = with_normals

    def __getitem__(self, idx):
        index = self.indexes[idx]
        try:
            sample = np.load(os.path.join(self.data_path, f'{index}.npz'))
        except Exception as error:
            raise RuntimeError(f'Error loading {index}.npz') from error
            
        before_pts = sample['before_pts'].copy()
        after_pts = sample['after_pts'].copy()
        mask = sample['mask'].astype(bool, copy=False)
        eps = 1e-6
        for i in range(32):
            if not mask[i]:
                before_pts[i] = after_pts[i] = np.random.random(after_pts[i].shape) * eps
        if not np.isfinite(before_pts[mask]).all():
            raise ValueError(
                f'{index}.npz contains NaN or Inf in an unmasked before tooth'
            )
        if not np.isfinite(after_pts[mask]).all():
            raise ValueError(
                f'{index}.npz contains NaN or Inf in an unmasked after tooth'
            )
        before_pts = torch.tensor(before_pts,dtype=torch.float32)
        after_pts = torch.tensor(after_pts,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.float32)
        if not self.with_normals:
            return index, before_pts, after_pts, mask

        missing_keys = {
            key for key in ('before_normals', 'after_normals')
            if key not in sample
        }
        if missing_keys:
            raise KeyError(
                f'{index}.npz is missing normal arrays: '
                f'{sorted(missing_keys)}. Set with_normals=false when the '
                'model does not use normals.'
            )
        before_normals = torch.tensor(
            sample['before_normals'], dtype=torch.float32
        )
        after_normals = torch.tensor(
            sample['after_normals'], dtype=torch.float32
        )
        return (
            index,
            before_pts,
            after_pts,
            before_normals,
            after_normals,
            mask,
        )
    
    def __len__(self):
        return len(self.indexes) 

class TeethDataManager(pl.LightningDataModule):
    def __init__(self, data_path, index_path, batch_size, with_normals=True):
        super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size = batch_size
        self.with_normals = with_normals

    def train_dataloader(self):
        file = os.path.join(self.index_path,'train0.npy')
        train_dataset = TeethDataset(self.data_path,file,self.with_normals)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        file = os.path.join(self.index_path,'val.npy')
        val_dataset = TeethDataset(self.data_path,file,self.with_normals)
        return DataLoader(val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        file = os.path.join(self.index_path,'test.npy')
        test_dataset = TeethDataset(self.data_path,file,self.with_normals)
        return DataLoader(test_dataset, batch_size=self.batch_size)
