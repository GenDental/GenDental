import torch
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from pytorch3d.loss import chamfer_distance
from models import *
from vector_quantize_pytorch import ResidualVQ
import torch.nn.functional as F
from models.single_encoder.pointnet_plus_encoder import PointNetPlusEncoder
import math

class MLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        dtype = torch.float32
        self.dense_h_to_4h = nn.Linear(
            hidden_size,
            hidden_size * 4,
            bias=False,
            dtype=dtype,
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            hidden_size * 2,
            hidden_size,
            bias=False,
            dtype=dtype,
        )

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # xq.shape = [batch_size, seq_len, dim]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.freqs_cis = precompute_freqs_cis(embed_dim,num_groups)
        self.mlp = MLP(embed_dim)

    def forward(self, x, attn_mask):

        x = self.ln_1(x)
        # a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        q,k = apply_rotary_emb(x,x,self.freqs_cis.to(x.device))
        a, _ = self.attn(q, k, x, need_weights=False, attn_mask=attn_mask, is_causal=True)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

class GPT_extractor(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers,trans_dim, group_size, num_groups, pretrained=False
    ):
        super(GPT_extractor, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        # start of sequence token

        self.layers1 = nn.ModuleList()
        for _ in range(num_layers):
            self.layers1.append(Block(embed_dim, num_heads, num_groups))
        # self.ln_f1 = nn.LayerNorm(embed_dim)
        self.ln_f1 = nn.GroupNorm(8, embed_dim)

    def forward(self, h, ab_pos, attn_mask, classify=False):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, length, C = h.shape
        for layer in self.layers1:
            h = layer(h + ab_pos, attn_mask)
        h = self.ln_f1(h.transpose(1, 2)).transpose(1, 2)
        return h

class Encoder_large(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Encoder_small(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class TokenEncoder(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.local_encoder = Encoder_large(config.encoder_channel)
    
    def forward(self,points):
        '''
        points: [b,n,p,c]
        '''
        return self.local_encoder()

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb

class GPT_generator(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, trans_dim, group_size
    ):
        super(GPT_generator, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.increase_dim = nn.Sequential(
            nn.Linear(self.trans_dim, 3*(self.group_size)),
            nn.LeakyReLU(),
            nn.Linear(3*(self.group_size), 3*(self.group_size)),
            nn.LeakyReLU(),
            nn.Linear(3*(self.group_size), 3*(self.group_size)),
        )
        # fp_layers, _ = create_pointnet2_fp_modules(fp_blocks, 128, [3, 32],has_temb=0)
        # self.fp_layers = nn.ModuleList(fp_layers)
        self.eos_predictor = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LeakyReLU(),
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.LeakyReLU(),
            nn.Linear(self.trans_dim, 2),
            # nn.Sigmoid()
        )

    def forward(self, h):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        batch, l, C = h.shape

        # input = h[:,2:]
        # for layer in self.fp_layers:
        #     input = layer(input)
        # rebuild_points = input.reshape(batch,-1,3)
        rebuild_points = self.increase_dim(h).reshape(batch,-1,self.group_size,3)
        # eos_prob = -(torch.log(self.eos_predictor(h[torch.arange(batch),length])) + torch.log((1-self.eos_predictor(h[torch.arange(batch),length-1]))))4
        predicted_masks = self.eos_predictor(h)
        return rebuild_points, predicted_masks
    

class ToothTransformer(nn.Module):
    def __init__(self, trans_dim,
                 depth,
                 drop_path_rate,
                 num_heads,
                 group_size,
                 encoder_dims,
                 decoder_depth,
                 num_groups,
                 encoder_config,
                 style_dims,
                 codebook_size,
                 num_quantizers,
                 codebook_dim,):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.num_quantizers = num_quantizers

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)
        self.encoders = nn.ModuleList([PointNetPlusEncoder(self.encoder_dims) for i in range(num_groups-1)])

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 10000)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            num_groups=num_groups,
            pretrained=True,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        # do not perform additional mask on the first (self.keep_attend) tokens
        self.num_groups = num_groups
        self.sos_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dims))
        self.ab_sos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.residual_vq = ResidualVQ(
            dim = self.encoder_dims,
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            num_quantizers = self.num_quantizers,
            shared_codebook= True,
            kmeans_init = True,   # set to True
            kmeans_iters = 10          
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        nn.init.normal_(self.ab_sos)
        nn.init.normal_(self.sos_token)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

    def forward(self, neighborhood, center, classify=False):
        # neighborhood: B G N D -> B 32 512 3
        # group_input_tokens = self.encoder(neighborhood)  # B 32 C
        group_input_tokens = torch.stack([self.encoders[i](neighborhood[:,i,:,:]) for i in range(neighborhood.size(1))],dim=1) # B 32 C
        group_input_tokens, indices, commit_loss = self.residual_vq(group_input_tokens)
        sos_token = self.sos_token.expand(group_input_tokens.size(0),-1,-1) # B 1 C
        group_input_tokens = torch.cat([sos_token,group_input_tokens],dim=1) # B 33 C

        # absolute_position
        ab_sos = self.ab_sos.expand(group_input_tokens.size(0), -1, -1)
        pos_absolute = self.pos_embed(center)
        pos_absolute = torch.cat([ab_sos, pos_absolute], dim=1)
        
        attn_mask = torch.full(
            (self.num_groups, self.num_groups), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        encoded_features = self.blocks(
            group_input_tokens, pos_absolute, attn_mask, classify=classify)
        # encoded_features = encoded_features.detach()
        generated_points, predicted_masks = self.generator_blocks(
            encoded_features)
        
        return generated_points, predicted_masks, commit_loss