"""Distributed-training and positional-embedding utilities."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    """Initialize torch.distributed if RANK is set, else fall back to single-GPU."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        torch.cuda.set_device(0)
        return 0, 1, 0


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


# ────────────────────────────────────────────────────────────────────────
# 2D sin-cos positional embeddings (standard MAE-style construction)
# ────────────────────────────────────────────────────────────────────────
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Standard MAE 2D sin-cos positional embeddings.

    Returns: (grid_size*grid_size, embed_dim) numpy array. No CLS row — the
    depth decoder operates on patch tokens only (CLS, when used, gets a
    separate learnable position marker).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w varies fastest
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return _2d_sincos_from_grid(embed_dim, grid)


def _2d_sincos_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    emb_h = _1d_sincos_from_grid(embed_dim // 2, grid[0])
    emb_w = _1d_sincos_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _1d_sincos_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
