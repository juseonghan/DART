"""Vision Transformer backbone used as student and teacher.

RGB-only (in_chans=3) — depth never enters the encoder input. The encoder
substitutes a learnable mask token at masked positions before its own
self-attention, so encoder outputs are mask-aware at every position.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop_path=0.0,
        init_values=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), bias=ffn_bias)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
        self.ls2 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None

    def forward(self, x):
        y = self.attn(self.norm1(x))
        if self.ls1 is not None:
            y = self.ls1 * y
        x = x + self.drop_path1(y)
        y = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            y = self.ls2 * y
        x = x + self.drop_path2(y)
        return x


class ViT(nn.Module):
    """Vision Transformer for DINOv2 pretraining. RGB only (in_chans=3)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop_path_rate=0.1,
        drop_path_uniform=True,
        init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    proj_bias,
                    ffn_bias,
                    drop_path=dpr[i],
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        M = int(math.sqrt(N))
        patch_pos = patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h0, w0), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pos, patch_pos], dim=1).to(x.dtype)

    def prepare_tokens(self, x, masks=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        return x

    def forward(self, x, masks=None, is_training=False):
        if isinstance(x, list):
            return self._forward_multi(x, masks, is_training)

        x = self.prepare_tokens(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if is_training:
            return {
                "x_norm_clstoken": x[:, 0],
                "x_norm_patchtokens": x[:, 1:],
            }
        return x[:, 0]

    def _forward_multi(self, x_list, masks_list, is_training):
        tokens_list, sizes = [], []
        for i, x in enumerate(x_list):
            m = masks_list[i] if masks_list is not None else None
            t = self.prepare_tokens(x, m)
            tokens_list.append(t)
            sizes.append(t.shape[0])

        all_tokens = torch.cat(tokens_list, dim=0)
        for blk in self.blocks:
            all_tokens = blk(all_tokens)
        all_tokens = self.norm(all_tokens)

        outputs, idx = [], 0
        for s in sizes:
            chunk = all_tokens[idx:idx + s]
            outputs.append(
                {
                    "x_norm_clstoken": chunk[:, 0],
                    "x_norm_patchtokens": chunk[:, 1:],
                }
            )
            idx += s

        if is_training:
            return outputs[0], outputs[1] if len(outputs) > 1 else outputs[0]
        return outputs

    def forward_features(self, x, layers=None):
        """Spatial features for downstream evaluation. RGB only."""
        B, C, H, W = x.shape
        assert C == self.in_chans, (
            f"forward_features got C={C}, model expects in_chans={self.in_chans}"
        )
        h, w = H // self.patch_size, W // self.patch_size
        x = self.prepare_tokens(x)

        if layers is None:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            patch_tokens = x[:, 1:]
            return patch_tokens.transpose(1, 2).reshape(B, -1, h, w)
        else:
            outputs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in layers:
                    x_norm = self.norm(x)
                    patch_tokens = x_norm[:, 1:]
                    outputs.append(
                        patch_tokens.transpose(1, 2).reshape(B, -1, h, w)
                    )
            return outputs
