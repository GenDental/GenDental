import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
import numpy as np
from models.teeth_arrangement.GGNN import GGNN
import math
from functools import partial
from models.teeth_arrangement.pointnet import PointNet
import torch

class TANet(nn.Module):
    def __init__(self,dim):

        """
        TANet类的初始化方法
        参数:
            args: 配置参数对象
        """
        super(TANet, self).__init__()  # 调用父类的初始化方法
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
        self.global_encoder = PointNet(final_dim = 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.norm_layer = nn.LayerNorm(512)
        self.propagation_model = GGNN()
        A = torch.zeros(34,34*8)
        for i in range(32):  
            if i!=15 and i!=31: 
                A[i,i+1] = 1
                A[i+1,i] = 1
            if i<=15:   
                A[i,34+15-i] = 1
                A[15-i,34+i] = 1

                A[i,68+32] = 1
                A[32,68+i] = 1
            else:
                A[i,34+47-i] = 1
                A[47-i,34+i] = 1

                A[i,68+33] = 1
                A[33,68+i] = 1
        A[32,102+33] = 1
        A[33,102+32] = 1
        A[:,136:] = A[:,:136]
        self.A = nn.parameter.Parameter(A,requires_grad=False)
        
        
    def forward(self,centroid, points):
        '''
        centroid: [bs, 32, 3]
        points: [bs, 32, 512, 3]
        '''
        bs = centroid.shape[0]
        n = centroid.shape[1]
        encodings = []
        # centered_points = points - centroid.unsqueeze(2).repeat(1,1,points.shape[2],1)
        for i in range(n):
            # encoding = self.encoder(centered_points[:,i].permute(0,2,1))
            # encoding = self.encoders[i](centered_points[:,i].permute(0,2,1))
            encoding = self.encoders[i](points[:,i].permute(0,2,1))
            encodings.append(encoding)
        for _ in range(2):
            encodings.append(torch.zeros_like(encodings[0]).cuda())
        embedding = torch.stack(encodings,dim=1)  #[bs,16,768]
        add_feature = self.propagation_model(embedding,self.A.unsqueeze(0).repeat(bs,1,1))
        trans_feature = embedding[:,:32] + add_feature[:,:32]
        trans_feature = self.norm_layer(trans_feature)


        global_embedding = self.global_encoder(points.view(bs,-1,3).permute(0,2,1)).unsqueeze(1).repeat(1,n,1)

        center_emb = centroid.view(bs,-1).unsqueeze(1).repeat(1,n,1)
        embedding = torch.cat([global_embedding,trans_feature,center_emb],dim=-1)
        # embedding [bs,32,C]
        dofs = self.regressor(embedding)
        # [bs,32,9]
        # [bs,20,32,9]

        return dofs