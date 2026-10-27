"""Loss functions for DINOv2 self-distillation plus depth reconstruction.

Implements:
  - DINOLoss: CLS-level self-distillation between student and teacher (centering
    handled here; the actual cross-entropy is computed in the training step
    because of the multi-crop structure).
  - iBOTLoss: patch-level self-distillation at masked positions.
  - KoLeoLoss: nearest-neighbour repulsive regularizer on student CLS tokens.
  - depth_reconstruction_loss: L1 between predicted and pseudo-depth at masked
    patches.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Holds the running center for teacher-CLS softmax centering.

    The cross-entropy across multi-crop view pairs is computed inside the
    training step (where the per-crop indexing lives); this module exposes
    `student_temp`, the `center` buffer, and `update_center`.
    """

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )


class iBOTLoss(nn.Module):
    """Patch-level self-distillation at iBOT-masked positions."""

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        batch_center = teacher_patch_tokens.mean(dim=(0, 1), keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def forward(self, student_patch_tokens, teacher_patch_tokens, masks, teacher_temp):
        t = F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)
        s = F.log_softmax(student_patch_tokens / self.student_temp, dim=-1)
        loss = -torch.sum(t * s, dim=-1)
        loss = (loss * masks.float()).sum() / masks.float().sum().clamp(min=1.0)
        self.update_center(teacher_patch_tokens.detach())
        return loss


class KoLeoLoss(nn.Module):
    """Nearest-neighbour repulsive regularizer on (L2-normalized) student CLS tokens."""

    def forward(self, x, eps=1e-8):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.normalize(x.float(), dim=-1, p=2, eps=eps)
            dots = torch.mm(x, x.t())
            n = x.shape[0]
            dots.view(-1)[:: n + 1].fill_(-1)
            _, I = torch.max(dots, dim=1)
            distances = torch.sqrt(((x - x[I]) ** 2).sum(dim=-1) + eps)
            loss = -torch.log(distances + eps).mean()
        return loss


def depth_reconstruction_loss(
    pred: torch.Tensor,
    target_depth: torch.Tensor,
    masks: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """L1 reconstruction loss between predicted and pseudo-depth at masked patches.

    Args:
      pred:         (B, N, P, P)   predicted depth per patch
      target_depth: (B, 1, H, W)   normalized pseudo-depth
      masks:        (B, N) bool    True at masked positions (loss computed here only)
      patch_size:   P

    Returns:
      Scalar L1 loss averaged over masked patches.
    """
    B, N, P, _ = pred.shape
    H, W = target_depth.shape[-2:]
    assert (H // P) * (W // P) == N, f"patch grid {H//P}x{W//P} != N ({N})"

    target = target_depth.squeeze(1)                   # (B, H, W)
    target = target.unfold(1, P, P).unfold(2, P, P)    # (B, h, w, P, P)
    target = target.contiguous().view(B, N, P, P)

    loss = (pred - target).abs()                       # (B, N, P, P)
    loss = loss.flatten(2).mean(dim=-1)                # (B, N)
    mask_f = masks.float()
    loss = (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return loss
