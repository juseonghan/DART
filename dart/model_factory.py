"""Construction of student/teacher backbones, projection heads, and depth decoder,
plus parameter-group construction for layerwise LR decay and weight-decay
exclusion, plus the EMA update for the teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .depth_decoders import build_depth_decoder
from .heads import DINOHead
from .vit import ViT


VIT_CONFIGS = {
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
}


def build_student_teacher(args):
    """Build student + teacher ViTs (in_chans=3) and student-only DepthDecoder."""
    cfg = VIT_CONFIGS[args.arch]
    embed_dim = cfg["embed_dim"]

    common_kwargs = dict(
        img_size=args.global_crop_size,
        patch_size=args.patch_size,
        in_chans=3,
        drop_path_rate=args.drop_path,
        drop_path_uniform=True,
        init_values=1e-5,
        **cfg,
    )

    student_backbone = ViT(**common_kwargs)
    teacher_backbone = ViT(**{**common_kwargs, "drop_path_rate": 0.0})

    head_kwargs = dict(
        in_dim=embed_dim,
        out_dim=args.head_n_prototypes,
        hidden_dim=args.head_hidden_dim,
        bottleneck_dim=args.head_bottleneck_dim,
        nlayers=3,
    )
    student_head = DINOHead(**head_kwargs)
    teacher_head = DINOHead(**head_kwargs)

    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    depth_decoder = build_depth_decoder(
        kind=args.depth_decoder,
        encoder_dim=embed_dim,
        patch_size=args.patch_size,
        num_patches=(args.global_crop_size // args.patch_size) ** 2,
        args=args,
    )

    return (
        student_backbone,
        teacher_backbone,
        student_head,
        teacher_head,
        depth_decoder,
    )


# ────────────────────────────────────────────────────────────────────────
# Parameter groups: layerwise LR decay, patch_embed LR multiplier,
# weight-decay exclusion for biases / norm scales / positional tokens.
# ────────────────────────────────────────────────────────────────────────
def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    if any(k in name for k in ("pos_embed", "cls_token", "mask_token")):
        return True
    return False


def build_param_groups(
    backbone: ViT,
    head: DINOHead,
    depth_decoder: nn.Module,
    patch_embed_lr_mult: float,
    layerwise_decay: float,
):
    num_layers = len(backbone.blocks)
    groups = []

    def _add(params_named, lr_mult, is_last_layer=False):
        decay = [
            p
            for n, p in params_named
            if p.requires_grad and not _is_no_decay(n, p)
        ]
        no_decay = [
            p for n, p in params_named if p.requires_grad and _is_no_decay(n, p)
        ]
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr_mult": lr_mult,
                    "wd_mult": 1.0,
                    "is_last_layer": is_last_layer,
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr_mult": lr_mult,
                    "wd_mult": 0.0,
                    "is_last_layer": is_last_layer,
                }
            )

    embed_named = [
        (n, p)
        for n, p in backbone.named_parameters()
        if any(
            k in n
            for k in ("patch_embed", "cls_token", "pos_embed", "mask_token")
        )
    ]
    _add(embed_named, lr_mult=patch_embed_lr_mult)

    for i, block in enumerate(backbone.blocks):
        lr_mult = layerwise_decay ** (num_layers - 1 - i)
        _add(list(block.named_parameters()), lr_mult=lr_mult)

    _add(list(backbone.norm.named_parameters()), lr_mult=1.0)

    head_non_last = [
        (n, p) for n, p in head.named_parameters() if "last_layer" not in n
    ]
    head_last = [(n, p) for n, p in head.named_parameters() if "last_layer" in n]
    _add(head_non_last, lr_mult=1.0, is_last_layer=False)
    _add(head_last, lr_mult=1.0, is_last_layer=True)

    # Depth decoder: full LR, standard WD exclusion
    _add(list(depth_decoder.named_parameters()), lr_mult=1.0, is_last_layer=False)

    return groups


@torch.no_grad()
def update_teacher(
    student_backbone,
    teacher_backbone,
    student_head,
    teacher_head,
    momentum,
):
    """EMA update of teacher backbone + head. The depth decoder is student-only."""
    for sp, tp in zip(
        student_backbone.parameters(), teacher_backbone.parameters()
    ):
        tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)
    for sp, tp in zip(student_head.parameters(), teacher_head.parameters()):
        tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)
