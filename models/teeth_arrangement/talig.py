import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
import numpy as np
import math
from functools import partial
from models.teeth_arrangement.pointnet import PointNet
import torch

class TAligNet(nn.Module):
    def __init__(self, dim):
        super(TAligNet, self).__init__()  # 调用父类的初始化方法
        self.encoders = nn.ModuleList([PointNet(final_dim=512) for _ in range(32)])

        self.regressor = nn.Sequential(
            nn.Linear(1632, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 9),
            nn.Tanh()
        )
        self.global_encoder = PointNet(final_dim = dim)
        self.bn = nn.BatchNorm1d(1024)
        
        
    def forward(self,centroid, points):
        '''
        centroid: [bs, 32, 3]
        points: [bs, 32, 512, 3]
        '''
        bs = centroid.shape[0]
        n = centroid.shape[1]
        encodings = []
        for i in range(n):
            encoding = self.encoders[i](points[:,i].permute(0,2,1))
            encodings.append(encoding)
        embedding = torch.stack(encodings,dim=1)

        global_embedding = self.global_encoder(points.view(bs,-1,3).permute(0,2,1)).unsqueeze(1).repeat(1,n,1)

        center_emb = centroid.view(bs,-1).unsqueeze(1).repeat(1,n,1)
        embedding = torch.cat([embedding, global_embedding,center_emb],dim=-1)
        dofs = self.regressor(embedding)

        return dofs