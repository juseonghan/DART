"""DART pretraining entrypoint.

Builds the student/teacher ViTs, the depth decoder, and runs DINOv2-style
self-distillation augmented with an L1 depth-reconstruction objective at
iBOT-masked patch positions. Depth is used only as a supervision target;
the encoder never receives depth as input.

Total loss (per step):
    total = w_dino * DINO  +  w_ibot * iBOT  +  w_koleo * KoLeo  +  w_depth * L1_depth

Example:
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) train.py \
        --data_path /path/to/shards \
        --save_dir /save/path \
        --arch vit_base --patch_size 16 --num_workers 18 --base_lr 1.5e-3 \
        --epochs 200 --batch_size 128 --warmup_epochs 20 \
        --warmup_teacher_temp_epochs 60 --momentum_teacher 0.996 \
        --depth_loss_weight 0.1 \
        --save_freq 5 \
        --log_freq 50

To use the MAE-style decoder with CLS persistence (mirrors canonical MAE):
 --depth_decoder mae --depth_decoder_mae_use_cls
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dart.data import MaskGenerator, RGBMultiCropAugmentation, make_loader
from dart.ddp import is_main, setup_ddp
from dart.losses import DINOLoss, KoLeoLoss, iBOTLoss
from dart.model_factory import (
    VIT_CONFIGS,
    build_param_groups,
    build_student_teacher,
    update_teacher,
)
from dart.scheduler import CosineScheduler
from dart.train_step import train_one_step


def parse_args():
    parser = argparse.ArgumentParser("DART pretraining")

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/shards",
        help="Directory containing the (rgb.png, depth.npy) WebDataset shards.",
    )
    parser.add_argument("--shard_pattern", type=str, default="rgbd-*.tar")
    parser.add_argument("--total_samples", type=int, default=3_202_579)
    parser.add_argument("--save_dir", type=str, default="./ckpt_dart")
    parser.add_argument("--num_workers", type=int, default=8)

    # Architecture
    parser.add_argument(
        "--arch", type=str, default="vit_base", choices=list(VIT_CONFIGS.keys())
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--head_n_prototypes", type=int, default=65536)
    parser.add_argument("--head_hidden_dim", type=int, default=2048)
    parser.add_argument("--head_bottleneck_dim", type=int, default=256)

    # Depth decoder
    parser.add_argument(
        "--depth_decoder",
        type=str,
        default="mlp",
        choices=["mlp", "xattn", "mae"],
        help=(
            "Depth decoder architecture. "
            "'mlp' = per-token 3-layer MLP. "
            "'xattn' = cross-attention decoder (mask queries cross-attend to "
            "encoder tokens). "
            "'mae' = MAE-style self-attention decoder with fixed 2D sin-cos "
            "positional embeddings."
        ),
    )
    parser.add_argument(
        "--depth_decoder_hidden_dim",
        type=int,
        default=512,
        help="Hidden dim for --depth_decoder=mlp",
    )
    parser.add_argument("--depth_decoder_xattn_dim", type=int, default=384)
    parser.add_argument("--depth_decoder_xattn_heads", type=int, default=6)
    parser.add_argument("--depth_decoder_xattn_layers", type=int, default=2)
    parser.add_argument("--depth_decoder_mae_dim", type=int, default=512)
    parser.add_argument("--depth_decoder_mae_heads", type=int, default=16)
    parser.add_argument("--depth_decoder_mae_layers", type=int, default=8)
    parser.add_argument(
        "--depth_decoder_mae_use_cls",
        action="store_true",
        help=(
            "If set, the encoder's CLS token is prepended to the MAE decoder "
            "input, participates in all self-attention blocks, then is sliced "
            "off before the per-patch pixel projection. Only applies to "
            "--depth_decoder=mae."
        ),
    )

    # Crops
    parser.add_argument("--global_crop_size", type=int, default=224)
    parser.add_argument("--local_crop_size", type=int, default=96)
    parser.add_argument("--n_local_crops", type=int, default=8)
    parser.add_argument(
        "--global_crop_scale", type=float, nargs=2, default=[0.32, 1.0]
    )
    parser.add_argument(
        "--local_crop_scale", type=float, nargs=2, default=[0.05, 0.32]
    )

    # iBOT masking
    parser.add_argument(
        "--mask_ratio_range", type=float, nargs=2, default=[0.1, 0.5]
    )
    parser.add_argument("--mask_sample_prob", type=float, default=0.5)

    # Loss weights
    parser.add_argument("--dino_loss_weight", type=float, default=1.0)
    parser.add_argument("--ibot_loss_weight", type=float, default=1.0)
    parser.add_argument("--koleo_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--depth_loss_weight",
        type=float,
        default=0.25,
        help="Weight of the L1 depth reconstruction loss.",
    )

    # Optimiser
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_lr", type=float, default=0.004)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--weight_decay_end", type=float, default=0.4)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--freeze_last_layer_epochs", type=int, default=1)
    parser.add_argument("--patch_embed_lr_mult", type=float, default=0.2)
    parser.add_argument("--layerwise_decay", type=float, default=0.9)

    # Teacher
    parser.add_argument("--momentum_teacher", type=float, default=0.992)
    parser.add_argument("--final_momentum_teacher", type=float, default=1.0)
    parser.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    parser.add_argument("--teacher_temp", type=float, default=0.07)
    parser.add_argument("--warmup_teacher_temp_epochs", type=int, default=3)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.save_dir, exist_ok=True)

    eff_batch = args.batch_size * world_size
    args.lr = args.base_lr * math.sqrt(eff_batch / 1024)

    if is_main():
        print(
            f"[config] arch={args.arch}, patch_size={args.patch_size}, in_chans=3 (RGB)"
        )
        print(
            f"[config] batch_size={args.batch_size} x {world_size} GPUs = {eff_batch}"
        )
        print(
            f"[config] lr={args.lr:.6f} (base={args.base_lr}, sqrt scaling)"
        )
        print(
            f"[config] patch_embed_lr_mult={args.patch_embed_lr_mult}, "
            f"layerwise_decay={args.layerwise_decay}"
        )
        decoder_str = args.depth_decoder + (
            " (use_cls=True)"
            if (args.depth_decoder == "mae" and args.depth_decoder_mae_use_cls)
            else ""
        )
        print(
            f"[config] depth_loss_weight={args.depth_loss_weight}, "
            f"decoder={decoder_str}"
        )
        print(
            f"[config] total_samples={args.total_samples}, epochs={args.epochs}"
        )
        writer = SummaryWriter(os.path.join(args.save_dir, "tb_logs"))
    else:
        writer = None

    # ── Build model ──────────────────────────────────────────────────
    (
        student_backbone,
        teacher_backbone,
        student_head,
        teacher_head,
        depth_decoder,
    ) = build_student_teacher(args)

    student_backbone = student_backbone.to(device)
    teacher_backbone = teacher_backbone.to(device)
    student_head = student_head.to(device)
    teacher_head = teacher_head.to(device)
    depth_decoder = depth_decoder.to(device)

    if is_main():
        n_bb = sum(p.numel() for p in student_backbone.parameters()) / 1e6
        n_hd = sum(p.numel() for p in student_head.parameters()) / 1e6
        n_dd = sum(p.numel() for p in depth_decoder.parameters()) / 1e6
        print(
            f"[model] backbone: {n_bb:.1f}M | head: {n_hd:.1f}M | depth_decoder: {n_dd:.1f}M"
        )

    use_ddp = dist.is_initialized()
    student_backbone_raw = student_backbone
    student_head_raw = student_head
    depth_decoder_raw = depth_decoder

    param_groups = build_param_groups(
        student_backbone_raw,
        student_head_raw,
        depth_decoder_raw,
        patch_embed_lr_mult=args.patch_embed_lr_mult,
        layerwise_decay=args.layerwise_decay,
    )

    if use_ddp:
        student_backbone = DDP(student_backbone, device_ids=[local_rank])
        student_head = DDP(student_head, device_ids=[local_rank])
        depth_decoder = DDP(depth_decoder, device_ids=[local_rank])

    # ── Losses ───────────────────────────────────────────────────────
    dino_loss_fn = DINOLoss(args.head_n_prototypes).to(device)
    ibot_loss_fn = iBOTLoss(args.head_n_prototypes).to(device)
    koleo_loss_fn = KoLeoLoss()

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # ── Schedules ────────────────────────────────────────────────────
    mask_gen = MaskGenerator(
        input_size=args.global_crop_size // args.patch_size,
        mask_ratio_range=tuple(args.mask_ratio_range),
    )
    steps_per_epoch = args.total_samples // (args.batch_size * world_size)
    total_steps = args.epochs * steps_per_epoch

    if is_main():
        print(
            f"[schedule] ~{steps_per_epoch} steps/epoch, {total_steps} total steps"
        )

    lr_schedule = CosineScheduler(
        args.lr,
        args.min_lr,
        total_steps,
        warmup_iters=args.warmup_epochs * steps_per_epoch,
    )
    wd_schedule = CosineScheduler(
        args.weight_decay, args.weight_decay_end, total_steps
    )
    mom_schedule = CosineScheduler(
        args.momentum_teacher, args.final_momentum_teacher, total_steps
    )
    teacher_temp_schedule = CosineScheduler(
        args.teacher_temp,
        args.teacher_temp,
        total_steps,
        warmup_iters=args.warmup_teacher_temp_epochs * steps_per_epoch,
        start_warmup_value=args.warmup_teacher_temp,
    )
    last_layer_lr_schedule = CosineScheduler(
        args.lr,
        args.min_lr,
        total_steps,
        warmup_iters=args.warmup_epochs * steps_per_epoch,
    )
    freeze_steps = args.freeze_last_layer_epochs * steps_per_epoch
    last_layer_lr_schedule.schedule[:freeze_steps] = 0

    # ── Data loader ──────────────────────────────────────────────────
    transform = RGBMultiCropAugmentation(
        global_size=args.global_crop_size,
        local_size=args.local_crop_size,
        n_local=args.n_local_crops,
        global_scale=tuple(args.global_crop_scale),
        local_scale=tuple(args.local_crop_scale),
    )

    loader = make_loader(
        args.data_path,
        args.shard_pattern,
        transform,
        args.batch_size,
        args.num_workers,
        mask_gen,
        args.mask_sample_prob,
        steps_per_epoch=steps_per_epoch,
    )

    # ── Resume checkpoint ────────────────────────────────────────────
    start_epoch = 0
    ckpt_path = os.path.join(args.save_dir, "latest.pth")
    if os.path.exists(ckpt_path):
        if is_main():
            print(f"[resume] Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student_backbone_raw.load_state_dict(ckpt["student_backbone"])
        teacher_backbone.load_state_dict(ckpt["teacher_backbone"])
        student_head_raw.load_state_dict(ckpt["student_head"])
        teacher_head.load_state_dict(ckpt["teacher_head"])
        depth_decoder_raw.load_state_dict(ckpt["depth_decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        dino_loss_fn.load_state_dict(ckpt["dino_loss"])
        ibot_loss_fn.load_state_dict(ckpt["ibot_loss"])
        start_epoch = ckpt["epoch"] + 1
        if is_main():
            print(f"[resume] Resuming from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ────────────────────────────────────────────────
    if is_main():
        print(
            f"\n{'=' * 60}\nStarting training from epoch {start_epoch}\n{'=' * 60}\n"
        )

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_losses = {k: 0.0 for k in ["total", "dino", "ibot", "koleo", "depth"]}
        n_steps = 0

        for step, batch in enumerate(loader):
            lr = lr_schedule[global_step]
            wd = wd_schedule[global_step]
            mom = mom_schedule[global_step]
            teacher_temp = teacher_temp_schedule[global_step]
            last_layer_lr = last_layer_lr_schedule[global_step]

            for pg in optimizer.param_groups:
                if pg["is_last_layer"]:
                    pg["lr"] = last_layer_lr * pg["lr_mult"]
                else:
                    pg["lr"] = lr * pg["lr_mult"]
                pg["weight_decay"] = wd * pg["wd_mult"]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = train_one_step(
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
                )

            scaler.scale(loss).backward()

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(student_backbone.parameters())
                    + list(student_head.parameters())
                    + list(depth_decoder.parameters()),
                    args.clip_grad,
                )

            scaler.step(optimizer)
            scaler.update()

            update_teacher(
                student_backbone_raw,
                teacher_backbone,
                student_head_raw,
                teacher_head,
                mom,
            )

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            n_steps += 1

            if is_main() and (global_step % args.log_freq == 0):
                for k in ("total", "dino", "ibot", "koleo", "depth"):
                    writer.add_scalar(
                        f"step/{k}", loss_dict[k].item(), global_step
                    )
                writer.add_scalar("step/lr", lr, global_step)
                writer.add_scalar("step/wd", wd, global_step)
                writer.add_scalar("step/momentum", mom, global_step)
                writer.add_scalar("step/teacher_temp", teacher_temp, global_step)

            global_step += 1

        elapsed = time.time() - epoch_start
        if is_main() and n_steps > 0:
            avg = {k: v / n_steps for k, v in epoch_losses.items()}
            print(
                f"Epoch {epoch:3d} | loss {avg['total']:.4f} | "
                f"dino {avg['dino']:.4f} | ibot {avg['ibot']:.4f} | "
                f"koleo {avg['koleo']:.4f} | depth {avg['depth']:.4f} | "
                f"lr {lr:.6f} | mom {mom:.4f} | "
                f"t_temp {teacher_temp:.4f} | {elapsed / 60:.1f}min"
            )
            for k, v in avg.items():
                writer.add_scalar(f"epoch/{k}", v, epoch)
            writer.add_scalar("epoch/time_min", elapsed / 60, epoch)
            writer.flush()

        if is_main() and (
            (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1
        ):
            ckpt = {
                "student_backbone": student_backbone_raw.state_dict(),
                "teacher_backbone": teacher_backbone.state_dict(),
                "student_head": student_head_raw.state_dict(),
                "teacher_head": teacher_head.state_dict(),
                "depth_decoder": depth_decoder_raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "dino_loss": dino_loss_fn.state_dict(),
                "ibot_loss": ibot_loss_fn.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            path = os.path.join(args.save_dir, f"checkpoint_{epoch:04d}.pth")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
            torch.save(
                {"teacher_backbone": teacher_backbone.state_dict()},
                os.path.join(args.save_dir, f"teacher_{epoch:04d}.pth"),
            )
            print(f"  Saved checkpoint to {path}")

    if is_main():
        writer.close()
        print("\nTraining complete.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
