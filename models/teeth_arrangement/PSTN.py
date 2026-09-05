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
from models.teeth_arrangement.pointnet import PointNet, PointNetEncoder, PointNetPlusEncoder
import torch

class PSTN(nn.Module):
    def __init__(self,args):
        super(PSTN, self).__init__()
        # self.p1_local_encoder = PointNet(final_dim=1024)
        self.p1_local_encoders = nn.ModuleList([PointNet(final_dim=1024) for i in range(32)])
        # self.p1_local_encoder = PointNet(final_dim=1024)
        self.p1_global_encoder = PointNet(final_dim=1024)
        self.p2_global_encoder = PointNetPlusEncoder(512,final_dim=512)
        # self.p2_global_encoder = PointNet(final_dim=512)
        # self.U = PointNet(channel=2563,first_dim=1024,second_dim=512,final_dim=512)
        self.regressor = nn.Sequential(
                nn.Linear(4099, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 9),
                nn.Tanh()
        )
        self.ln = nn.LayerNorm(4099)
        
        
    def forward(self,centroid, points):
        '''
        points [bs,32,128,3]
        '''
        bs = centroid.shape[0]
        n = centroid.shape[1]
        encodings = []
        for i in range(n):
            encoding = self.p1_local_encoders[i](points[:,i].permute(0,2,1))
            # encoding = self.p1_local_encoder(points[:,i].permute(0,2,1))
            encodings.append(encoding)
        p1_local_embedding = torch.stack(encodings,dim=1)  #[bs,32,1024]

        global_points = points.view(bs,-1,3)
        p1_global_embedding = self.p1_global_encoder(global_points.permute(0,2,1)).unsqueeze(1).repeat(1,n,1)

        p2_global_embedding = self.p2_global_encoder(global_points.permute(0,2,1)).unsqueeze(1).repeat(1,n,1)
        embedding = torch.cat([points.view(bs,n,-1),p1_local_embedding,p1_global_embedding,p2_global_embedding,centroid],dim=-1) #[bs,32,2947]
        # embedding = torch.cat([p1_local_embedding,p1_global_embedding,p2_global_embedding,centroid],dim=-1)
        # parameter = self.U(embedding.transpose(1,2))
        dofs = self.regressor(embedding)
        return dofs