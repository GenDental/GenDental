import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from models.single_encoder.pointnet_plus_encoder import PointNetPlusEncoder


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"trans_dim={dim} must be divisible by num_heads={num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = float(attn_dropout)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(
                causal_mask,
                torch.finfo(scores.dtype).min,
            )
            attention = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            attention = F.dropout(
                attention,
                p=self.attn_dropout,
                training=self.training,
            )
            attended = torch.matmul(attention, value)

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(batch_size, seq_len, self.dim)
        attended = self.out_proj(attended)
        return self.out_dropout(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 8.0 / 3.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = 256 * math.ceil(hidden_dim / 256)

        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attention(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 8.0 / 3.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        drop_path_values = torch.linspace(
            0.0,
            drop_path_rate,
            depth,
        ).tolist()

        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path_values[layer_index],
                )
                for layer_index in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or input_dim
        depth = max(int(depth), 1)

        layers = []
        current_dim = input_dim
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ToothPosteriorEncoder(nn.Module):
    """Encode each local tooth point cloud into a Gaussian posterior."""

    def __init__(
        self,
        encoder_dims: int,
        latent_dim: int,
        num_teeth: int,
        share_encoder: bool = False,
        encoder_sa_blocks=None,
        encoder_tooth_chunk_size: int = 4,
    ) -> None:
        super().__init__()

        self.num_teeth = num_teeth
        self.share_encoder = share_encoder
        self.encoder_tooth_chunk_size = int(encoder_tooth_chunk_size)
        if self.encoder_tooth_chunk_size <= 0:
            raise ValueError("encoder_tooth_chunk_size must be positive.")

        if share_encoder:
            self.encoder = PointNetPlusEncoder(
                encoder_dims,
                sa_blocks=encoder_sa_blocks,
            )
            self.encoders = None
        else:
            self.encoder = None
            self.encoders = nn.ModuleList(
                [
                    PointNetPlusEncoder(
                        encoder_dims,
                        sa_blocks=encoder_sa_blocks,
                    )
                    for _ in range(num_teeth)
                ]
            )

        # A shared geometric encoder should not have to infer the anatomical
        # slot from shape alone. Slot conditioning preserves tooth identity
        # while letting all slots benefit from the same encoder parameters.
        self.tooth_slot_embedding = nn.Parameter(
            torch.zeros(1, num_teeth, encoder_dims)
        )
        trunc_normal_(self.tooth_slot_embedding, std=0.02)
        self.feature_norm = nn.LayerNorm(encoder_dims)
        self.posterior_head = nn.Linear(encoder_dims, 2 * latent_dim)

    def forward(
        self,
        local_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if local_points.ndim != 4:
            raise ValueError(
                "local_points must have shape [B, N, P, 3]."
            )
        if local_points.shape[1] != self.num_teeth:
            raise ValueError(
                f"Expected {self.num_teeth} teeth, "
                f"received {local_points.shape[1]}."
            )

        features = []
        if self.share_encoder:
            # Batch a few tooth slots at a time. Flattening all B*N teeth in a
            # single PointNet++ call is fast but can create a very large peak
            # activation footprint for practical batch sizes.
            batch_size, _, num_points, coordinate_dim = local_points.shape
            for start in range(0, self.num_teeth, self.encoder_tooth_chunk_size):
                end = min(start + self.encoder_tooth_chunk_size, self.num_teeth)
                points = local_points[:, start:end].reshape(
                    batch_size * (end - start),
                    num_points,
                    coordinate_dim,
                )
                feature = self.encoder(points)
                feature = feature.reshape(batch_size, end - start, -1)
                features.append(feature)
            features = torch.cat(features, dim=1)
        else:
            for tooth_index in range(self.num_teeth):
                points = local_points[:, tooth_index]
                feature = self.encoders[tooth_index](points)
                if feature.ndim > 2:
                    feature = feature.reshape(feature.shape[0], -1)
                features.append(feature)
            features = torch.stack(features, dim=1)

        features = self.feature_norm(features + self.tooth_slot_embedding)
        posterior_params = self.posterior_head(features)
        posterior_mu, posterior_logvar = posterior_params.chunk(2, dim=-1)
        return posterior_mu, posterior_logvar


class ToothPointDecoder(nn.Module):
    """Decode one latent per tooth into a zero-mean local point cloud."""

    def __init__(
        self,
        latent_dim: int,
        group_size: int,
        hidden_dim: int,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.decoder = MLPHead(
            input_dim=latent_dim,
            output_dim=group_size * 3,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_teeth, _ = latents.shape
        local_points = self.decoder(self.latent_norm(latents))
        local_points = local_points.reshape(
            batch_size,
            num_teeth,
            self.group_size,
            3,
        )
        local_points = local_points - local_points.mean(
            dim=2,
            keepdim=True,
        )
        return local_points


class LatentGPTTransformer(nn.Module):
    """
    Continuous latent VAE + causal GPT prior for tooth point-cloud generation.

    Training alignment:
        input  = [SOS, z_0, z_1, ..., z_{N-1}]
        output = [prior(z_0), prior(z_1), ..., prior(z_{N-1}), extra]

    The outer module supervises only output[:, :N]. At output position i,
    causal attention can see only SOS and teeth before i, so there is no
    target-latent or target-center leakage.
    """

    def __init__(
        self,
        trans_dim: int,
        depth: int,
        drop_path_rate: float,
        num_heads: int,
        group_size: int,
        encoder_dims: int,
        decoder_depth: int,
        num_groups: int,
        encoder_config=None,
        style_dims: Optional[int] = None,
        codebook_size: Optional[int] = None,
        num_quantizers: Optional[int] = None,
        codebook_dim: Optional[int] = None,
        latent_dim: int = 256,
        share_encoder: bool = False,
        encoder_sa_blocks=None,
        encoder_tooth_chunk_size: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.05,
        mlp_ratio: float = 8.0 / 3.0,
        center_scale: float = 1.0,
        posterior_logvar_min: float = -8.0,
        posterior_logvar_max: float = 4.0,
        prior_logvar_min: float = -6.0,
        prior_logvar_max: float = 2.0,
        decoder_hidden_dim: Optional[int] = None,
        direct_point_residual_scale: float = 0.1,
        missing_point_noise_scale: float = 1e-6,
    ) -> None:
        super().__init__()

        # Kept only so the existing YAML can be migrated incrementally.
        del encoder_config, style_dims, codebook_size, num_quantizers, codebook_dim

        if num_groups < 2:
            raise ValueError("num_groups must be SOS + at least one tooth.")
        if center_scale <= 0:
            raise ValueError("center_scale must be positive.")

        self.trans_dim = trans_dim
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.num_groups = num_groups
        self.num_teeth = num_groups - 1
        self.center_scale = float(center_scale)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.prior_logvar_min = float(prior_logvar_min)
        self.prior_logvar_max = float(prior_logvar_max)
        self.direct_point_residual_scale = float(
            direct_point_residual_scale
        )
        if self.direct_point_residual_scale < 0:
            raise ValueError(
                "direct_point_residual_scale must be non-negative."
            )
        self.missing_point_noise_scale = float(missing_point_noise_scale)
        if self.missing_point_noise_scale <= 0:
            raise ValueError("missing_point_noise_scale must be positive.")

        self.posterior_encoder = ToothPosteriorEncoder(
            encoder_dims=encoder_dims,
            latent_dim=latent_dim,
            num_teeth=self.num_teeth,
            share_encoder=share_encoder,
            encoder_sa_blocks=encoder_sa_blocks,
            encoder_tooth_chunk_size=encoder_tooth_chunk_size,
        )

        decoder_hidden_dim = decoder_hidden_dim or max(trans_dim, latent_dim * 2)
        self.point_decoder = ToothPointDecoder(
            latent_dim=latent_dim,
            group_size=group_size,
            hidden_dim=decoder_hidden_dim,
            depth=decoder_depth,
            dropout=dropout,
        )

        self.latent_to_token = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, trans_dim),
        )
        self.center_embedding = nn.Sequential(
            nn.Linear(3, max(trans_dim // 2, 64)),
            nn.SiLU(),
            nn.Linear(max(trans_dim // 2, 64), trans_dim),
            nn.LayerNorm(trans_dim),
        )

        self.sos_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.missing_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.sequence_position = nn.Parameter(
            torch.zeros(1, num_groups, trans_dim)
        )
        self.input_norm = nn.LayerNorm(trans_dim)

        self.transformer = CausalTransformer(
            dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        # Conditional prior p(z_i | z_<i, centers_<i, masks_<i)
        self.prior_mu_head = MLPHead(
            input_dim=trans_dim,
            output_dim=latent_dim,
            hidden_dim=trans_dim,
            depth=2,
            dropout=dropout,
        )
        self.prior_logvar_head = MLPHead(
            input_dim=trans_dim,
            output_dim=latent_dim,
            hidden_dim=trans_dim,
            depth=2,
            dropout=dropout,
        )
        # Let the causal GPT state directly correct local tooth geometry while
        # the sampled prior latent continues to provide stochastic variation.
        self.context_point_residual_head = MLPHead(
            input_dim=trans_dim,
            output_dim=group_size * 3,
            hidden_dim=decoder_hidden_dim,
            depth=decoder_depth,
            dropout=dropout,
        )

        # Deterministic center prediction in normalized coordinates.
        # Center uncertainty is intentionally not learned: an unconstrained
        # Gaussian center NLL can become negative by shrinking the predicted
        # variance. Geometry diversity is provided by the sampled tooth latent.
        self.center_mu_head = MLPHead(
            input_dim=trans_dim,
            output_dim=3,
            hidden_dim=trans_dim,
            depth=2,
            dropout=dropout,
        )
        # Predict a residual around a learned anatomical center template. This
        # removes the burden of regressing every absolute center from scratch,
        # especially for the first autoregressive tooth predicted from SOS.
        self.center_template_normalized = nn.Parameter(
            torch.zeros(1, self.num_teeth, 3)
        )
        self.mask_head = MLPHead(
            input_dim=trans_dim,
            output_dim=2,
            hidden_dim=trans_dim,
            depth=2,
            dropout=dropout,
        )

        self.apply(self._init_weights)
        trunc_normal_(self.sos_token, std=0.02)
        trunc_normal_(self.missing_token, std=0.02)
        trunc_normal_(self.sequence_position, std=0.02)
        self._initialize_logvar_biases()
        self._initialize_context_point_residual()
        self._rescale_residual_projections(depth)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _initialize_logvar_biases(self) -> None:
        # Start with moderate posterior/prior uncertainty instead of unit
        # variance. This reduces the magnitude of stochastic latent samples
        # during the first optimization steps.
        if self.posterior_encoder.posterior_head.bias is not None:
            with torch.no_grad():
                self.posterior_encoder.posterior_head.bias[
                    self.latent_dim :
                ].fill_(-2.0)

        prior_last = self.prior_logvar_head.net[-1]
        if isinstance(prior_last, nn.Linear) and prior_last.bias is not None:
            nn.init.constant_(prior_last.bias, -2.0)

    def _initialize_context_point_residual(self) -> None:
        # Start exactly from the latent decoder behavior. The direct GPT path
        # is learned gradually instead of perturbing geometry at initialization.
        last_layer = self.context_point_residual_head.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def _rescale_residual_projections(self, depth: int) -> None:
        residual_std = 0.02 / math.sqrt(2.0 * depth)
        for block in self.transformer.blocks:
            trunc_normal_(block.attention.out_proj.weight, std=residual_std)
            trunc_normal_(block.mlp.down_proj.weight, std=residual_std)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if deterministic:
            return mu

        # Keep exp and sampling in FP32 even under BF16/FP16 autocast.
        mu_fp32 = mu.float()
        logvar_fp32 = logvar.float()
        std_fp32 = torch.exp(0.5 * logvar_fp32)
        noise_fp32 = torch.randn(
            mu_fp32.shape,
            device=mu.device,
            dtype=torch.float32,
            generator=generator,
        )
        sampled = (
            mu_fp32
            + float(temperature) * std_fp32 * noise_fp32
        )
        return sampled.to(dtype=mu.dtype)

    def _validate_inputs(
        self,
        neighborhood: torch.Tensor,
        center: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if neighborhood.ndim != 4:
            raise ValueError(
                "neighborhood must have shape [B, N, P, 3]."
            )

        batch_size, num_teeth, num_points, coordinate_dim = neighborhood.shape
        if num_teeth != self.num_teeth:
            raise ValueError(
                f"Expected {self.num_teeth} teeth, received {num_teeth}."
            )
        if num_points != self.group_size:
            raise ValueError(
                f"Expected {self.group_size} points, received {num_points}."
            )
        if coordinate_dim != 3:
            raise ValueError("The final point dimension must be 3.")

        if center is None:
            center = neighborhood.mean(dim=2)
        if center.shape != (batch_size, num_teeth, 3):
            raise ValueError(
                f"center must have shape [B, N, 3], got {center.shape}."
            )

        if input_masks is None:
            input_masks = torch.ones(
                batch_size,
                num_teeth,
                device=neighborhood.device,
                dtype=torch.bool,
            )
        else:
            if input_masks.shape != (batch_size, num_teeth):
                raise ValueError(
                    f"input_masks must have shape [B, N], got {input_masks.shape}."
                )
            input_masks = input_masks.bool()

        return center, input_masks

    def encode_posterior(
        self,
        neighborhood: torch.Tensor,
        center: torch.Tensor,
        input_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_points = neighborhood - center[:, :, None, :]

        # The dataset represents a missing tooth using tiny random points,
        # not an exactly degenerate all-zero cloud. Preserve that behavior: an
        # all-zero cloud can make PointNet++ grouping/attention numerically
        # degenerate in some implementations.
        missing_local_points = (
            torch.rand_like(local_points) - 0.5
        ) * (2.0 * self.missing_point_noise_scale)
        missing_local_points = (
            missing_local_points
            - missing_local_points.mean(dim=2, keepdim=True)
        )
        local_points = torch.where(
            input_masks[:, :, None, None],
            local_points,
            missing_local_points,
        )

        posterior_mu, posterior_logvar = self.posterior_encoder(local_points)

        valid_latent_mask = input_masks[:, :, None]
        invalid_valid_mu = (~torch.isfinite(posterior_mu)) & valid_latent_mask
        invalid_valid_logvar = (
            ~torch.isfinite(posterior_logvar)
        ) & valid_latent_mask
        if invalid_valid_mu.any() or invalid_valid_logvar.any():
            raise FloatingPointError(
                "PointNet posterior encoder produced NaN/Inf for a valid "
                "tooth. Check point coordinates and PointNet internals."
            )

        # A missing tooth is excluded from all posterior losses. Replace any
        # non-finite values produced for such degenerate positions before
        # masking, otherwise NaN can survive later arithmetic.
        posterior_mu = torch.nan_to_num(
            posterior_mu, nan=0.0, posinf=0.0, neginf=0.0
        )
        posterior_logvar = torch.nan_to_num(
            posterior_logvar, nan=0.0, posinf=0.0, neginf=0.0
        )
        posterior_logvar = posterior_logvar.clamp(
            self.posterior_logvar_min,
            self.posterior_logvar_max,
        )

        # Invalid positions must be explicitly finite before KL computation.
        # Masking a NaN later by multiplication does not work because NaN*0
        # is still NaN.
        posterior_mu = torch.where(
            input_masks[:, :, None],
            posterior_mu,
            torch.zeros_like(posterior_mu),
        )
        posterior_logvar = torch.where(
            input_masks[:, :, None],
            posterior_logvar,
            torch.zeros_like(posterior_logvar),
        )
        posterior_z = self.reparameterize(
            posterior_mu,
            posterior_logvar,
            deterministic=deterministic,
        )

        posterior_z = torch.where(
            input_masks[:, :, None],
            posterior_z,
            torch.zeros_like(posterior_z),
        )
        return posterior_mu, posterior_logvar, posterior_z

    def _build_sequence(
        self,
        latents: torch.Tensor,
        centers: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]

        latent_tokens = self.latent_to_token(latents)
        normalized_centers = centers / self.center_scale
        center_tokens = self.center_embedding(normalized_centers)
        tooth_tokens = latent_tokens + center_tokens

        missing_tokens = self.missing_token.expand(
            batch_size,
            self.num_teeth,
            -1,
        )
        tooth_tokens = torch.where(
            masks[:, :, None],
            tooth_tokens,
            missing_tokens,
        )

        sos_token = self.sos_token.expand(batch_size, -1, -1)
        sequence_tokens = torch.cat([sos_token, tooth_tokens], dim=1)
        sequence_tokens = (
            sequence_tokens
            + self.sequence_position[:, : sequence_tokens.shape[1]]
        )
        return self.input_norm(sequence_tokens)

    def predict_prior(
        self,
        history_latents: torch.Tensor,
        history_centers: torch.Tensor,
        history_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sequence = self._build_sequence(
            history_latents,
            history_centers,
            history_masks,
        )
        hidden_states = self.transformer(sequence)

        prior_mu = self.prior_mu_head(hidden_states)
        prior_logvar = self.prior_logvar_head(hidden_states).clamp(
            self.prior_logvar_min,
            self.prior_logvar_max,
        )
        center_residual_normalized = self.center_mu_head(hidden_states)
        center_template_with_extra_token = F.pad(
            self.center_template_normalized,
            (0, 0, 0, 1),
        )
        center_mu_normalized = (
            center_residual_normalized + center_template_with_extra_token
        )
        mask_logits = self.mask_head(hidden_states)

        return {
            "hidden_states": hidden_states,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "center_mu_normalized": center_mu_normalized,
            "mask_logits": mask_logits,
        }

    def decode_prior_local_points(
        self,
        prior_latent: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent shape and add a direct causal-GPT point residual."""
        if prior_latent.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "prior_latent and hidden_states must share [B, N]."
            )
        base_points = self.point_decoder(prior_latent)
        batch_size, num_teeth = prior_latent.shape[:2]
        residual = self.context_point_residual_head(hidden_states).reshape(
            batch_size,
            num_teeth,
            self.group_size,
            3,
        )
        residual = torch.tanh(residual)
        residual = residual - residual.mean(dim=2, keepdim=True)
        local_points = (
            base_points + self.direct_point_residual_scale * residual
        )
        return local_points - local_points.mean(dim=2, keepdim=True)

    def forward(
        self,
        neighborhood: torch.Tensor,
        center: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
        deterministic_posterior: bool = False,
        classify: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del classify

        center, input_masks = self._validate_inputs(
            neighborhood,
            center,
            input_masks,
        )

        posterior_mu, posterior_logvar, posterior_z = self.encode_posterior(
            neighborhood,
            center,
            input_masks,
            deterministic=deterministic_posterior,
        )

        prior_output = self.predict_prior(
            history_latents=posterior_z,
            history_centers=center,
            history_masks=input_masks,
        )

        # First N output positions predict the N teeth.
        prior_mu = prior_output["prior_mu"][:, : self.num_teeth]
        prior_logvar = prior_output["prior_logvar"][:, : self.num_teeth]
        center_mu_normalized = prior_output[
            "center_mu_normalized"
        ][:, : self.num_teeth]
        mask_logits = prior_output["mask_logits"][:, : self.num_teeth]
        prior_hidden_states = prior_output[
            "hidden_states"
        ][:, : self.num_teeth]

        posterior_local_points = self.point_decoder(posterior_z)
        posterior_reconstruction = (
            posterior_local_points + center[:, :, None, :]
        )

        # Decode the prior mean for a stable geometry loss. At test time we
        # sample from the prior instead.
        prior_local_points = self.decode_prior_local_points(
            prior_mu,
            prior_hidden_states,
        )
        prior_centers = center_mu_normalized * self.center_scale
        prior_reconstruction = (
            prior_local_points + prior_centers[:, :, None, :]
        )

        return {
            "posterior_mu": posterior_mu,
            "posterior_logvar": posterior_logvar,
            "posterior_z": posterior_z,
            "posterior_reconstruction": posterior_reconstruction,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "prior_reconstruction": prior_reconstruction,
            "center_mu_normalized": center_mu_normalized,
            "mask_logits": mask_logits,
            "target_centers_normalized": center / self.center_scale,
            "masks": input_masks,
        }

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        latent_temperature: float = 1.0,
        center_temperature: float = 1.0,
        mask_temperature: float = 1.0,
        sample_masks: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        # Kept in the signature for configuration compatibility. Centers are
        # now predicted deterministically, so this value is intentionally unused.
        del center_temperature

        if device is None:
            device = self.sos_token.device
        if dtype is None:
            dtype = self.sos_token.dtype

        history_latents = torch.zeros(
            batch_size,
            self.num_teeth,
            self.latent_dim,
            device=device,
            dtype=dtype,
        )
        history_centers = torch.zeros(
            batch_size,
            self.num_teeth,
            3,
            device=device,
            dtype=dtype,
        )
        history_masks = torch.zeros(
            batch_size,
            self.num_teeth,
            device=device,
            dtype=torch.bool,
        )

        generated_points = torch.zeros(
            batch_size,
            self.num_teeth,
            self.group_size,
            3,
            device=device,
            dtype=dtype,
        )
        generated_mask_logits = torch.zeros(
            batch_size,
            self.num_teeth,
            2,
            device=device,
            dtype=dtype,
        )

        for tooth_index in range(self.num_teeth):
            prior_output = self.predict_prior(
                history_latents,
                history_centers,
                history_masks,
            )

            latent_mu = prior_output["prior_mu"][:, tooth_index]
            latent_logvar = prior_output["prior_logvar"][:, tooth_index]
            center_mu_normalized = prior_output[
                "center_mu_normalized"
            ][:, tooth_index]
            mask_logits = prior_output["mask_logits"][:, tooth_index]

            sampled_latent = self.reparameterize(
                latent_mu,
                latent_logvar,
                deterministic=False,
                temperature=latent_temperature,
                generator=generator,
            )
            # Use the predicted center mean directly. Randomness comes from
            # sampled_latent, whose distribution is explicitly learned.
            sampled_center = center_mu_normalized * self.center_scale

            if sample_masks:
                probabilities = torch.softmax(
                    mask_logits.float() / mask_temperature,
                    dim=-1,
                )
                sampled_mask = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1).bool()
            else:
                sampled_mask = mask_logits.argmax(dim=-1).bool()

            local_points = self.decode_prior_local_points(
                sampled_latent[:, None, :],
                prior_output["hidden_states"][:, tooth_index : tooth_index + 1],
            )[:, 0]
            points = local_points + sampled_center[:, None, :]

            # Raw geometry is always retained for debugging/saving decisions.
            generated_points[:, tooth_index] = points
            generated_mask_logits[:, tooth_index] = mask_logits

            # Only valid generated teeth become autoregressive history.
            history_latents[:, tooth_index] = torch.where(
                sampled_mask[:, None],
                sampled_latent,
                torch.zeros_like(sampled_latent),
            )
            history_centers[:, tooth_index] = torch.where(
                sampled_mask[:, None],
                sampled_center,
                torch.zeros_like(sampled_center),
            )
            history_masks[:, tooth_index] = sampled_mask

        return {
            "points": generated_points,
            "masks": history_masks,
            "mask_logits": generated_mask_logits,
            "latents": history_latents,
            "centers": history_centers,
        }
