"""Depth decoders applied to the student's masked-patch features.

All depth decoders share the same forward signature:
    forward(features, cls_token=None, masks=None) -> (B, N, P, P)
Decoders that do not consume CLS or masks simply ignore those args.

Three variants are provided:
  - DepthDecoderMLP    per-token 3-layer MLP (lightweight baseline)
  - DepthDecoderXAttn  MultiMAE-style cross-attention from mask queries
  - DepthDecoderMAE    canonical MAE-style self-attention decoder, with
                       an optional CLS-persistence flag
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .ddp import get_2d_sincos_pos_embed
from .vit import Block


# ════════════════════════════════════════════════════════════════════════
# Variant 1: per-token MLP (smallest, the default)
# ════════════════════════════════════════════════════════════════════════
class DepthDecoderMLP(nn.Module):
    """Per-token MLP mapping each student patch feature to a P×P depth patch.

    Minimal baseline: 3-layer MLP, no attention. Depth is never in the
    encoder input, so the only gradient path for "predict depth" is through
    RGB features — that is the mechanism being tested.
    """

    def __init__(self, encoder_dim: int, hidden_dim: int = 512, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size * patch_size),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        cls_token: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        features:  (B, N, encoder_dim)  student's post-encoder patch tokens
        cls_token: ignored (kept for interface consistency)
        masks:     ignored (per-token prediction is mask-independent)
        returns:   (B, N, P, P)
        """
        out = self.net(features)
        B, N, _ = out.shape
        return out.reshape(B, N, self.patch_size, self.patch_size)


# ════════════════════════════════════════════════════════════════════════
# Variant 2: cross-attention decoder (mask queries cross-attend to encoder
#            tokens). Inspired by MultiMAE.
# ════════════════════════════════════════════════════════════════════════
class _XAttnBlock(nn.Module):
    """Pre-norm cross-attention + MLP block for the depth decoder."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, D = q.shape
        Nkv = kv.shape[1]

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        Q = self.q_proj(qn).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kvn).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kvn).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, Nq, D)
        q = q + self.out_proj(attn_out)

        q = q + self.mlp(self.norm_mlp(q))
        return q


class DepthDecoderXAttn(nn.Module):
    """Cross-attention decoder: mask queries attend to all encoder patch tokens.

    At each masked position, a learnable depth-query token cross-attends to
    every encoder patch token, aggregating global context to predict the
    missing patch's depth. More expressive than the per-token MLP because
    the query can pool cues from anywhere in the image.

    A single learnable [MASK] query is tiled to every patch position and
    added to positional embeddings so positions remain distinguishable. At
    unmasked positions the decoder still produces a (discarded) prediction.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        num_patches: int = 196,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_dim = decoder_dim
        self.num_patches = num_patches

        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_query = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        self.blocks = nn.ModuleList(
            [_XAttnBlock(decoder_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size)

        trunc_normal_(self.mask_query, std=0.02)
        trunc_normal_(self.dec_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        cls_token: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        features:  (B, N, encoder_dim)  student's post-encoder patch tokens
        cls_token: ignored (kept for interface consistency)
        masks:     (B, N) bool — unused; loss masks down to masked positions.
        returns:   (B, N, P, P)
        """
        B, N, _ = features.shape
        assert N == self.num_patches, (
            f"DepthDecoderXAttn expected N={self.num_patches} patches, got N={N}. "
            f"Rebuild decoder with num_patches={N} or change global_crop_size."
        )

        kv = self.encoder_proj(features)                            # (B, N, D_dec)
        q = self.mask_query.expand(B, N, -1) + self.dec_pos_embed   # (B, N, D_dec)

        for blk in self.blocks:
            q = blk(q, kv)

        q = self.norm(q)
        out = self.pred(q)                                          # (B, N, P*P)
        return out.reshape(B, N, self.patch_size, self.patch_size)


# ════════════════════════════════════════════════════════════════════════
# Variant 3: MAE-style self-attention decoder
# ════════════════════════════════════════════════════════════════════════
class DepthDecoderMAE(nn.Module):
    """MAE-style depth decoder with optional CLS-persistence.

    Pipeline (use_cls=False, default):
        x = Linear(encoder_patches, decoder_dim)
        x = x + fixed_2d_sincos_pos_embed
        for blk in decoder_blocks: x = SelfAttn+MLP(x)
        x = LayerNorm(x)
        depth_pred = Linear(x, P*P)

    Pipeline (use_cls=True, mirrors canonical MAE):
        cls = Linear(encoder_cls, decoder_dim) + learned_cls_pos
        x   = Linear(encoder_patches, decoder_dim) + fixed_2d_sincos_pos
        seq = concat([cls, x], dim=1)                   # (B, 1+N, D_dec)
        for blk in decoder_blocks: seq = SelfAttn+MLP(seq)
        seq = LayerNorm(seq)
        patches = seq[:, 1:]                            # drop CLS
        depth_pred = Linear(patches, P*P)

    No decoder-side learnable mask token: the encoder already substitutes a
    learnable mask token at masked positions before its own self-attention,
    so encoder outputs are mask-aware at every position.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 512,
        num_heads: int = 16,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        num_patches: int = 196,
        use_cls: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_dim = decoder_dim
        self.num_patches = num_patches
        self.use_cls = use_cls

        grid_size = int(math.isqrt(num_patches))
        assert grid_size * grid_size == num_patches, (
            f"DepthDecoderMAE assumes a square patch grid; got num_patches={num_patches}"
        )
        self.grid_size = grid_size

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim), requires_grad=False
        )

        if self.use_cls:
            self.decoder_cls_embed = nn.Linear(encoder_dim, decoder_dim)
            self.decoder_cls_pos = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        else:
            self.decoder_cls_embed = None
            self.decoder_cls_pos = None

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    init_values=None,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size, bias=True)

        self._init_pos_embed()
        self.apply(self._init_weights)

        if self.use_cls:
            trunc_normal_(self.decoder_cls_pos, std=0.02)

    def _init_pos_embed(self):
        pos = get_2d_sincos_pos_embed(self.decoder_dim, self.grid_size)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        cls_token: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        features:  (B, N, encoder_dim)  student's post-encoder patch tokens
        cls_token: (B, encoder_dim) | None
                   Required when self.use_cls=True; ignored otherwise.
        masks:     unused; encoder already handled masked positions.
        returns:   (B, N, P, P)
        """
        B, N, _ = features.shape
        assert N == self.num_patches, (
            f"DepthDecoderMAE expected N={self.num_patches} patches, got N={N}. "
            f"Rebuild decoder with num_patches={N} or change global_crop_size."
        )

        x = self.decoder_embed(features)      # (B, N, D_dec)
        x = x + self.decoder_pos_embed

        if self.use_cls:
            assert cls_token is not None, (
                "DepthDecoderMAE was constructed with use_cls=True but received "
                "cls_token=None. Pass the encoder's CLS token to the decoder."
            )
            cls = self.decoder_cls_embed(cls_token).unsqueeze(1)  # (B, 1, D_dec)
            cls = cls + self.decoder_cls_pos                       # (B, 1, D_dec)
            x = torch.cat([cls, x], dim=1)                         # (B, 1+N, D_dec)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        if self.use_cls:
            x = x[:, 1:]                       # drop CLS before per-patch projection

        out = self.decoder_pred(x)             # (B, N, P*P)
        return out.reshape(B, N, self.patch_size, self.patch_size)


# ════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════
def build_depth_decoder(
    kind: str,
    encoder_dim: int,
    patch_size: int,
    num_patches: int,
    args,
) -> nn.Module:
    """Dispatch on the --depth_decoder choice."""
    if kind == "mlp":
        return DepthDecoderMLP(
            encoder_dim=encoder_dim,
            hidden_dim=args.depth_decoder_hidden_dim,
            patch_size=patch_size,
        )
    elif kind == "xattn":
        return DepthDecoderXAttn(
            encoder_dim=encoder_dim,
            decoder_dim=args.depth_decoder_xattn_dim,
            num_heads=args.depth_decoder_xattn_heads,
            num_layers=args.depth_decoder_xattn_layers,
            mlp_ratio=4.0,
            patch_size=patch_size,
            num_patches=num_patches,
        )
    elif kind == "mae":
        return DepthDecoderMAE(
            encoder_dim=encoder_dim,
            decoder_dim=args.depth_decoder_mae_dim,
            num_heads=args.depth_decoder_mae_heads,
            num_layers=args.depth_decoder_mae_layers,
            mlp_ratio=4.0,
            patch_size=patch_size,
            num_patches=num_patches,
            use_cls=args.depth_decoder_mae_use_cls,
        )
    else:
        raise ValueError(
            f"Unknown --depth_decoder: {kind!r} (choices: 'mlp', 'xattn', 'mae')"
        )
