"""One training step: forward both teacher and student over global+local crops,
compute the four losses (DINO, iBOT, KoLeo, depth), and return the combined
loss plus a detached dict for logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import depth_reconstruction_loss


def train_one_step(
    batch,
    student_backbone,
    teacher_backbone,
    student_head,
    teacher_head,
    depth_decoder,
    dino_loss_fn,
    ibot_loss_fn,
    koleo_loss_fn,
    teacher_temp,
    args,
    device,
):
    """One training step. Returns (scalar loss, detached loss dict)."""

    global_crops = batch["global_crops"].to(device, non_blocking=True)            # (2B, 3, Hg, Wg)
    local_crops = batch["local_crops"].to(device, non_blocking=True)              # (nL*B, 3, Hl, Wl)
    global_depth_targets = batch["global_depth_targets"].to(
        device, non_blocking=True
    )                                                                              # (2B, 1, Hg, Wg)
    masks = batch["masks"].to(device, non_blocking=True)                          # (2B, N)

    n_global_total = global_crops.shape[0]
    B = n_global_total // 2

    # ── Teacher forward (no grad) ────────────────────────────────────
    with torch.no_grad():
        teacher_out = teacher_backbone(global_crops, is_training=True)
        teacher_cls = teacher_out["x_norm_clstoken"]
        teacher_patch = teacher_out["x_norm_patchtokens"]
        teacher_cls_after_head = teacher_head(teacher_cls)
        B2, N, D = teacher_patch.shape
        teacher_patch_after_head = teacher_head(
            teacher_patch.reshape(-1, D)
        ).reshape(B2, N, -1)

    # ── Student forward ──────────────────────────────────────────────
    student_global_out = student_backbone(
        global_crops, masks=masks, is_training=True
    )
    student_global_cls = student_global_out["x_norm_clstoken"]
    student_global_patch = student_global_out["x_norm_patchtokens"]   # (2B, N, D)

    student_local_out = student_backbone(local_crops, is_training=True)
    student_local_cls = student_local_out["x_norm_clstoken"]

    student_global_cls_head = student_head(student_global_cls)
    student_local_cls_head = student_head(student_local_cls)
    B2, N, D = student_global_patch.shape
    student_global_patch_head = student_head(
        student_global_patch.reshape(-1, D)
    ).reshape(B2, N, -1)

    # ── DINO loss: CLS self-distillation across crops ────────────────
    n_local = local_crops.shape[0] // B
    student_local_chunks = student_local_cls_head.chunk(n_local)

    dino_loss = torch.tensor(0.0, device=device)
    n_dino_terms = 0

    for s_crop_idx in range(2):
        s_start = s_crop_idx * B
        s = student_global_cls_head[s_start:s_start + B]
        s_log = F.log_softmax(s / dino_loss_fn.student_temp, dim=-1)
        for t_crop_idx in range(2):
            if s_crop_idx == t_crop_idx:
                continue
            t = teacher_cls_after_head[t_crop_idx * B:(t_crop_idx + 1) * B]
            t_soft = F.softmax(
                (t - dino_loss_fn.center) / teacher_temp, dim=-1
            )
            dino_loss = dino_loss - (t_soft * s_log).sum(dim=-1).mean()
            n_dino_terms += 1

    for local_chunk in student_local_chunks:
        s_log = F.log_softmax(local_chunk / dino_loss_fn.student_temp, dim=-1)
        for t_crop_idx in range(2):
            t = teacher_cls_after_head[t_crop_idx * B:(t_crop_idx + 1) * B]
            t_soft = F.softmax(
                (t - dino_loss_fn.center) / teacher_temp, dim=-1
            )
            dino_loss = dino_loss - (t_soft * s_log).sum(dim=-1).mean()
            n_dino_terms += 1

    dino_loss = dino_loss / n_dino_terms

    with torch.no_grad():
        dino_loss_fn.update_center(teacher_cls_after_head)

    # ── iBOT loss: masked patch self-distillation ────────────────────
    ibot_loss = ibot_loss_fn(
        student_global_patch_head, teacher_patch_after_head, masks, teacher_temp
    )

    # ── KoLeo loss on student global CLS (each crop separately) ─────
    koleo_loss = sum(
        koleo_loss_fn(student_global_cls[i * B:(i + 1) * B]) for i in range(2)
    )

    # ── Depth reconstruction loss ────────────────────────────────────
    # Pre-DINO-head patch features are the only path for depth gradient to
    # reach the encoder. Loss is at masked positions only (same mask as
    # iBOT). The CLS token is consumed only by decoders that opt into it
    # (currently DepthDecoderMAE with use_cls=True); other decoders ignore
    # it.
    depth_pred = depth_decoder(
        student_global_patch,
        cls_token=student_global_cls,
        masks=masks,
    )                                                                # (2B, N, P, P)
    depth_loss = depth_reconstruction_loss(
        depth_pred, global_depth_targets, masks, patch_size=args.patch_size
    )

    # ── Total ────────────────────────────────────────────────────────
    total_loss = (
        args.dino_loss_weight * dino_loss
        + args.ibot_loss_weight * ibot_loss
        + args.koleo_loss_weight * koleo_loss
        + args.depth_loss_weight * depth_loss
    )

    loss_dict = {
        "total": total_loss.detach(),
        "dino": dino_loss.detach(),
        "ibot": ibot_loss.detach(),
        "koleo": koleo_loss.detach(),
        "depth": depth_loss.detach(),
    }

    return total_loss, loss_dict
