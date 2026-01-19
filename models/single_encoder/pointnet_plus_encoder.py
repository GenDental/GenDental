# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch.nn as nn 
from loguru import logger 
from models.single_encoder.pvcnn2 import create_pointnet2_sa_components, create_pointnet2_fp_modules  
from models import *
from timm.models.layers import trunc_normal_
import torch

# implement the global encoder for VAE model 

class PointNetEncoder(nn.Module):
    force_att = 0 # add attention to all layers  
    # def __init__(self, zdim, input_dim, extra_feature_channels=0, args={}):
    def __init__(self, encoder_dims, sa_blocks):
        super().__init__()
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=3, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features, encoder_dims) 
        self.layers = nn.ModuleList(layers) 


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

class PointNetPlusEncoder(nn.Module):
    force_att = 0 # add attention to all layers  
    # def __init__(self, zdim, input_dim, extra_feature_channels=0, args={}):
    sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 128]]]
        ]
    def __init__(self, encoder_dims):
        super().__init__()
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(self.sa_blocks, 
            extra_feature_channels=0, input_dim=3, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features, encoder_dims) 
        self.layers = nn.ModuleList(layers) 


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
    
    
class StyleEncoder(nn.Module):
    force_att = 0 # add attention to all layers  
    def __init__(self, style_dims):
        super().__init__()
        sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 128]]]
        ]
        self.sa_blocks = sa_blocks
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=3, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features,style_dims)
        self.layers = nn.ModuleList(layers) 

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
    
    def forward(self,x):
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        
        features = features.max(-1)[0]
        features = self.mlp(features)
        # features: B,D,N; xyz: B,3,N
        return features



