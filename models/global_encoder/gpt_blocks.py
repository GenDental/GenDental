import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from vector_quantize_pytorch import ResidualVQ

from models.single_encoder.pointnet_plus_encoder import PointNetPlusEncoder


class MLP(nn.Module):
    """SwiGLU feed-forward network used in each Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # The first projection is split into gate/value halves by SwiGLU.
        projected_dim = int(hidden_size * mlp_ratio)
        if projected_dim % 2 != 0:
            projected_dim += 1
        swiglu_dim = projected_dim // 2

        self.gate_value_proj = nn.Linear(
            hidden_size,
            projected_dim,
            bias=False,
        )
        self.output_proj = nn.Linear(
            swiglu_dim,
            hidden_size,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_value_proj(hidden_states).chunk(2, dim=-1)
        hidden_states = F.silu(gate) * value
        hidden_states = self.output_proj(hidden_states)
        return self.dropout(hidden_states)


class Block(nn.Module):
    """Standard pre-norm causal Transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by "
                f"num_heads={num_heads}."
            )

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        # Positional information is already added once before the Transformer.
        # We intentionally do not apply the old pre-projection RoPE here,
        # because nn.MultiheadAttention performs its Q/K projections internally.
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_output_dropout = nn.Dropout(dropout)

        self.mlp = MLP(
            hidden_size=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.drop_path_1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.drop_path_2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Standard pre-norm residual connection:
        # x <- x + Attention(LN(x))
        residual = x
        normed_x = self.ln_1(x)

        attention_output, _ = self.attn(
            normed_x,
            normed_x,
            normed_x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        attention_output = self.attn_output_dropout(attention_output)
        x = residual + self.drop_path_1(attention_output)

        # x <- x + MLP(LN(x))
        x = x + self.drop_path_2(self.mlp(self.ln_2(x)))
        return x


class GPT_extractor(nn.Module):
    """Causal Transformer operating on SOS + tooth tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        trans_dim: int,
        group_size: int,
        num_groups: int,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        del pretrained

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.num_groups = num_groups

        drop_path_values = torch.linspace(
            0.0,
            drop_path_rate,
            num_layers,
        ).tolist()

        self.layers1 = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    drop_path=drop_path_values[layer_index],
                    mlp_ratio=mlp_ratio,
                )
                for layer_index in range(num_layers)
            ]
        )

        # LayerNorm is token-wise. Unlike GroupNorm over [B, C, L], it does
        # not mix statistics across future sequence positions.
        self.ln_f1 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor,
        classify: bool = False,
    ) -> torch.Tensor:
        del classify

        if h.ndim != 3:
            raise ValueError(
                f"Expected h with shape [B, L, C], got {tuple(h.shape)}."
            )

        for layer in self.layers1:
            h = layer(h, attn_mask)

        return self.ln_f1(h)


class CenterEmbedding(nn.Module):
    """
    Token-wise center-coordinate embedding.

    center_scale must reflect the coordinate scale of the dataset. It is a
    fixed scalar and therefore does not use statistics from future teeth.
    """

    def __init__(
        self,
        output_dim: int,
        center_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if center_scale <= 0:
            raise ValueError("center_scale must be positive.")

        self.center_scale = float(center_scale)
        hidden_dim = max(output_dim // 2, 64)

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        embeddings = self.net(centers / self.center_scale)
        return self.dropout(embeddings)


class PredictionHead(nn.Module):
    """Configurable MLP head used by point and mask prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        depth = max(int(depth), 1)
        hidden_dim = hidden_dim or input_dim

        layers = []
        current_dim = input_dim

        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPT_generator(nn.Module):
    """Decode each Transformer output token into points and a mask logit."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        trans_dim: int,
        group_size: int,
        dropout: float = 0.0,
        generation_noise_dim: int = 64,
        generation_noise_scale: float = 0.03,
    ) -> None:
        super().__init__()
        del embed_dim, num_heads

        self.trans_dim = trans_dim
        self.group_size = group_size
        self.generation_noise_dim = int(generation_noise_dim)
        self.generation_noise_scale = float(generation_noise_scale)
        if self.generation_noise_dim <= 0:
            raise ValueError("generation_noise_dim must be positive.")
        if self.generation_noise_scale < 0:
            raise ValueError("generation_noise_scale must be non-negative.")

        self.increase_dim = PredictionHead(
            input_dim=trans_dim,
            output_dim=3 * group_size,
            depth=max(num_layers, 1),
            hidden_dim=3 * group_size,
            dropout=dropout,
        )

        self.eos_predictor = PredictionHead(
            input_dim=trans_dim,
            output_dim=2,
            depth=max(num_layers, 1),
            hidden_dim=trans_dim,
            dropout=dropout,
        )
        self.noise_projection = nn.Sequential(
            nn.Linear(self.generation_noise_dim, trans_dim),
            nn.SiLU(),
            nn.Linear(trans_dim, trans_dim),
        )
        self.noise_residual = PredictionHead(
            input_dim=2 * trans_dim,
            output_dim=3 * group_size,
            depth=max(num_layers, 1),
            hidden_dim=3 * group_size,
            dropout=dropout,
        )

    def reset_stochastic_output(self) -> None:
        '''Start from the exact deterministic old-model prediction.'''
        last_layer = self.noise_residual.net[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("The stochastic residual must end with nn.Linear.")
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        h: torch.Tensor,
        generation_noise: Optional[torch.Tensor] = None,
        num_candidates: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = h.shape
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive.")

        base_points = self.increase_dim(h)
        if generation_noise is None:
            generation_noise = torch.randn(
                num_candidates,
                batch_size,
                self.generation_noise_dim,
                device=h.device,
                dtype=h.dtype,
            )
        elif generation_noise.ndim == 2:
            generation_noise = generation_noise.unsqueeze(0)
        elif generation_noise.ndim != 3:
            raise ValueError(
                "generation_noise must have shape [B, D] or [K, B, D]."
            )

        if generation_noise.shape[1:] != (
            batch_size,
            self.generation_noise_dim,
        ):
            raise ValueError(
                "generation_noise has incompatible shape: "
                f"{tuple(generation_noise.shape)}."
            )

        candidate_count = generation_noise.shape[0]
        noise_features = self.noise_projection(generation_noise)
        noise_features = noise_features[:, :, None, :].expand(
            -1, -1, sequence_length, -1
        )
        hidden_features = h[None].expand(
            candidate_count, -1, -1, -1
        )
        residual_inputs = torch.cat(
            [hidden_features, noise_features],
            dim=-1,
        ).reshape(
            candidate_count * batch_size,
            sequence_length,
            2 * self.trans_dim,
        )
        point_residual = self.noise_residual(residual_inputs).reshape(
            candidate_count,
            batch_size,
            sequence_length,
            3 * self.group_size,
        )
        candidate_points = (
            base_points[None]
            + self.generation_noise_scale * torch.tanh(point_residual)
        ).reshape(
            candidate_count,
            batch_size,
            sequence_length,
            self.group_size,
            3,
        )

        predicted_masks = self.eos_predictor(h)
        if candidate_count == 1:
            return candidate_points[0], predicted_masks
        return candidate_points, predicted_masks


class transformer(nn.Module):
    """
    GPT-style autoregressive tooth point-cloud model.

    Teacher-forcing alignment
    -------------------------
    Raw tooth sequence:
        [tooth_0, tooth_1, ..., tooth_{N-1}]

    Transformer input:
        [SOS, token(tooth_0), ..., token(tooth_{N-1})]

    Transformer output:
        output[0] predicts tooth_0
        output[1] predicts tooth_1
        ...
        output[N-1] predicts tooth_{N-1}
        output[N] is ignored by the outer training module

    Therefore the outer module should continue using:
        generated_points[:, :-1]  versus ground-truth teeth
        predicted_masks[:, :-1]   versus ground-truth masks
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
        encoder_config,
        style_dims: int,
        codebook_size: int,
        num_quantizers: int,
        codebook_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        share_encoder: bool = False,
        use_vq: bool = True,
        center_scale: float = 1.0,
        position_dropout: float = 0.0,
        generation_noise_dim: int = 64,
        generation_noise_scale: float = 0.03,
    ) -> None:
        super().__init__()
        del encoder_config, style_dims

        if num_groups < 2:
            raise ValueError(
                "num_groups must include SOS and at least one tooth token."
            )

        self.trans_dim = trans_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.num_quantizers = num_quantizers
        self.num_groups = num_groups
        self.num_teeth = num_groups - 1

        # Keep the original position-specific PointNet++ encoders.
        self.encoders = nn.ModuleList(
            [
                PointNetPlusEncoder(self.encoder_dims)
                for _ in range(self.num_teeth)
            ]
        )

        self.pre_vq_norm = nn.LayerNorm(self.encoder_dims)
        self.residual_vq = ResidualVQ(
            dim=self.encoder_dims,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            num_quantizers=self.num_quantizers,
            shared_codebook=True,
            kmeans_init=True,
            kmeans_iters=10,
        )

        self.encoder_to_transformer = (
            nn.Identity()
            if self.encoder_dims == self.trans_dim
            else nn.Linear(self.encoder_dims, self.trans_dim)
        )

        self.sos_token = nn.Parameter(
            torch.zeros(1, 1, self.trans_dim)
        )
        self.sos_center_embedding = nn.Parameter(
            torch.zeros(1, 1, self.trans_dim)
        )

        # Fixed anatomical tooth slots are better represented by a learned
        # absolute slot embedding than by the old pre-projection RoPE.
        self.sequence_position = nn.Parameter(
            torch.zeros(1, self.num_groups, self.trans_dim)
        )

        self.center_embedding = CenterEmbedding(
            output_dim=self.trans_dim,
            center_scale=center_scale,
            dropout=position_dropout,
        )

        # Normalize each feature source before fusion, then normalize the
        # fused residual-stream input once. Position features are not added
        # again inside every Transformer layer.
        self.token_norm = nn.LayerNorm(self.trans_dim)
        self.center_norm = nn.LayerNorm(self.trans_dim)
        self.slot_norm = nn.LayerNorm(self.trans_dim)
        self.input_norm = nn.LayerNorm(self.trans_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = GPT_extractor(
            embed_dim=self.trans_dim,
            num_heads=self.num_heads,
            num_layers=self.depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            num_groups=self.num_groups,
            pretrained=False,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            attn_dropout=attn_dropout,
            mlp_ratio=mlp_ratio,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.trans_dim,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            dropout=dropout,
            generation_noise_dim=generation_noise_dim,
            generation_noise_scale=generation_noise_scale,
        )

        # A persistent causal mask moves with the module and is not stored in
        # the checkpoint because it can always be reconstructed.
        causal_mask = torch.triu(
            torch.ones(
                self.num_groups,
                self.num_groups,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

        self.apply(self._init_weights)

        # Learned tokens are initialized exactly once with the same scale as
        # the other Transformer parameters.
        trunc_normal_(self.sos_token, std=0.02)
        trunc_normal_(self.sos_center_embedding, std=0.02)
        trunc_normal_(self.sequence_position, std=0.02)
        self.generator_blocks.reset_stochastic_output()

        self._rescale_residual_projections()

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

    def _rescale_residual_projections(self) -> None:
        """Reduce residual-branch initialization scale for deep networks."""
        if self.depth <= 0:
            return

        residual_std = 0.02 / math.sqrt(2.0 * self.depth)

        for block in self.blocks.layers1:
            trunc_normal_(block.attn.out_proj.weight, std=residual_std)
            trunc_normal_(block.mlp.output_proj.weight, std=residual_std)

    def _validate_inputs(
        self,
        neighborhood: torch.Tensor,
        center: torch.Tensor,
    ) -> None:
        if neighborhood.ndim != 4:
            raise ValueError(
                "neighborhood must have shape [B, N, P, 3], "
                f"got {tuple(neighborhood.shape)}."
            )

        batch_size, num_teeth, num_points, coordinate_dim = (
            neighborhood.shape
        )

        if num_teeth != self.num_teeth:
            raise ValueError(
                f"Expected {self.num_teeth} teeth, got {num_teeth}."
            )

        if num_points != self.group_size:
            raise ValueError(
                f"Expected {self.group_size} points per tooth, "
                f"got {num_points}."
            )

        if coordinate_dim != 3:
            raise ValueError(
                f"Expected point coordinate dimension 3, got {coordinate_dim}."
            )

        if center.shape != (batch_size, num_teeth, 3):
            raise ValueError(
                "center must have shape [B, N, 3], "
                f"got {tuple(center.shape)}."
            )

    def forward(
        self,
        neighborhood: torch.Tensor,
        center: torch.Tensor,
        classify: bool = False,
        generation_noise: Optional[torch.Tensor] = None,
        num_candidates: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(neighborhood, center)
        batch_size = neighborhood.shape[0]

        # Encode each tooth independently, preserving the original design.
        group_input_tokens = torch.stack(
            [
                self.encoders[tooth_index](
                    neighborhood[:, tooth_index, :, :]
                )
                for tooth_index in range(self.num_teeth)
            ],
            dim=1,
        )

        # Normalize PointNet++ features before vector quantization.
        group_input_tokens = self.pre_vq_norm(group_input_tokens)
        group_input_tokens, _, commit_loss = self.residual_vq(
            group_input_tokens
        )
        group_input_tokens = self.encoder_to_transformer(
            group_input_tokens
        )

        # Teacher-forcing input sequence:
        # [SOS, tooth_0, tooth_1, ..., tooth_{N-1}]
        sos_token = self.sos_token.expand(batch_size, -1, -1)
        sequence_tokens = torch.cat(
            [sos_token, group_input_tokens],
            dim=1,
        )

        # Center features are aligned with the input tokens, not the targets:
        # [SOS_center, center_0, ..., center_{N-1}].
        # Thus output i can only use centers up to i-1 under causal attention.
        tooth_center_embeddings = self.center_embedding(center)
        sos_center_embedding = self.sos_center_embedding.expand(
            batch_size,
            -1,
            -1,
        )
        center_embeddings = torch.cat(
            [sos_center_embedding, tooth_center_embeddings],
            dim=1,
        )

        slot_embeddings = self.sequence_position[
            :, : sequence_tokens.shape[1]
        ].expand(batch_size, -1, -1)

        # Normalize different feature sources before adding them. Add all
        # positional information exactly once before the Transformer.
        sequence_tokens = self.token_norm(sequence_tokens)
        center_embeddings = self.center_norm(center_embeddings)
        slot_embeddings = self.slot_norm(slot_embeddings)

        hidden_states = (
            sequence_tokens
            + center_embeddings
            + slot_embeddings
        )
        hidden_states = self.input_dropout(
            self.input_norm(hidden_states)
        )

        encoded_features = self.blocks(
            hidden_states,
            self.causal_mask,
            classify=classify,
        )

        generated_points, predicted_masks = self.generator_blocks(
            encoded_features,
            generation_noise=generation_noise,
            num_candidates=num_candidates,
        )

        return generated_points, predicted_masks, commit_loss