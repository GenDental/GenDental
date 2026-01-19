
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
style_set = [i for i in range(1603,2493)]
# style_set = [1883]

class SyntheticDataset(data.Dataset):
    def __init__(self, style_path, data_path, index_file):
        super().__init__()
        self.data_path = data_path
        self.style_path = style_path
        self.indexes = np.load(index_file)
        # self.indexes = [i for i in range(250)]

    def __getitem__(self, idx):
        index = self.indexes[idx]
        style_index = random.choice(style_set)
        style_data = np.load(os.path.join(self.style_path, f'{style_index}.npz'))
        before_pts = style_data['before_pts']
        style_mask = style_data['mask']
        # mask = style_data['mask']
        mask = np.zeros(32)
        eps = 1e-6

        after_pts = np.random.random(before_pts.shape) * eps
        for i in range(32):
            pcd_path = os.path.join(self.data_path, f'{index}_{i}.ply')
            if os.path.exists(pcd_path):
                mask[i] = 1
                after_pts[i] = read_pointcloud(pcd_path)

        for i in range(32):
            if not style_mask[i]:
                before_pts[i] = np.random.random(after_pts[i].shape) * eps
        
        before_pts = torch.tensor(before_pts,dtype=torch.float32)
        after_pts = torch.tensor(after_pts,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.float32)
        return index,before_pts,after_pts,mask
    
    def __len__(self):
        return len(self.indexes)

class SyntheticDataManager(pl.LightningDataModule):
    def __init__(self, data_path, style_path, index_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.style_path = style_path
        self.batch_size = batch_size

    def train_dataloader(self):
        file = os.path.join(self.index_path,'train.npy')
        train_dataset = SyntheticDataset(self.style_path,self.data_path,file)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        file = os.path.join(self.index_path,'val.npy')
        val_dataset = SyntheticDataset(self.style_path,self.data_path,file)
        return DataLoader(val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        file = os.path.join(self.index_path,'test.npy')
        test_dataset = SyntheticDataset(self.style_path,self.data_path,file)
        return DataLoader(test_dataset, batch_size=self.batch_size)