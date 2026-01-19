import torch.nn as nn 
from loguru import logger 
from models.single_encoder.pvcnn2 import create_pointnet2_sa_components 
from models import *
from timm.models.layers import trunc_normal_
import torch
    

class StyleEncoder(nn.Module):
    force_att = 0 # add attention to all layers  
    def __init__(self, zdim, input_dim):
        super().__init__()
        sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 64]]],
        [[32, 1, 8], [128, 0.4, 32, [64, zdim]]]
        ]
        self.sa_blocks = sa_blocks
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp_mean = nn.Linear(128,zdim) 
        self.mlp_var = nn.Linear(128,zdim)
        self.pe = nn.Parameter(torch.zeros(zdim))
        trunc_normal_(self.pe,std=0.02)
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]
    
    def forward(self,x):
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
        style_mean = torch.mean(features,1)
        style_var = torch.var(features,1)
        style_feature = self.mlp_mean(style_mean) + self.mlp_var(style_var) + self.pe.expand(x.shape[0],-1)
        return style_feature

class ShapeEncoder(nn.Module):
    sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 128]]]
        ]
    force_att = 0 # add attention to all layers  
    def __init__(self,zdim, input_dim):
        super().__init__()
        sa_blocks = self.sa_blocks 
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features, zdim) 
        self.zdim = zdim 
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]

    def forward(self, x):
        """
        Args: 
            x: B,N,3 
        Returns: 
            mu, sigma: B,D
        """
        # output = {} 
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
       
        features = features.max(-1)[0]
        features = self.mlp(features)

        return features

