#!/usr/bin/env python3
"""
Extract RGB-D pairs from a directory of videos using Depth Anything V2,
then save as WebDataset shards for efficient pre-training data loading.

Designed for SLURM jobs with time limits:
  - Checkpoints after each video so the job can be resumed.
  - Catches SIGTERM (sent by SLURM before killing) for graceful shutdown.
  - Partial shards from interrupted runs are cleaned up on resume.

For each video:
  1. Detect FPS, extract 1 frame per second at original resolution.
  2. Run Depth Anything V2 at original resolution for best quality.
  3. Resize both RGB and depth to half the video's native resolution
     and write to WebDataset shards.
  4. Save 5 side-by-side RGB + depth sample visualizations.

Shard format (per sample):
  - __key__: "{video_name}_f{frame_index:06d}"
  - rgb.png:  uint8 PNG-encoded (H/2 x W/2 x 3 BGR)
  - depth.npy: float32 numpy array (H/2 x W/2), normalized to [0, 1]

Usage:
    python video_to_rgbd_shards.py \\
        --video_dir /path/to/videos \\
        --output_path /path/to/output \\
        --depth_anything_repo /path/to/Depth-Anything-V2 \\
        --model_checkpoint checkpoints/depth_anything_v2_vitl.pth \\
        --encoder vitl

Resume after interruption (same command – checkpoint is auto-detected):
    python video_to_rgbd_shards.py ... (same args)

Training dataloader example at the bottom of this file.
"""

import os
import sys
import glob
import json
import signal
import cv2
import numpy as np
import torch
import webdataset as wds
import random
import argparse
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Graceful SLURM shutdown
# ---------------------------------------------------------------------------
# SLURM sends SIGTERM before SIGKILL.  We catch it and finish the current
# video, checkpoint, then exit cleanly.

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    print(f"\n⚠  {sig_name} received – will checkpoint after current video "
          f"and exit gracefully.")


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
# Checkpoint JSON tracks:
#   completed_videos : list[str]  – video paths fully written to shards
#   total_written    : int        – cumulative sample count
#   next_shard       : int        – the shard index the *next* write will use
#
# On resume we delete any shard files with index >= next_shard (they may be
# partial / corrupt from an interrupted run) and skip completed videos.

def _ckpt_path(output_path: str) -> str:
    return os.path.join(output_path, "checkpoint.json")


def load_checkpoint(output_path: str) -> dict:
    p = _ckpt_path(output_path)
    if os.path.exists(p):
        with open(p) as f:
            ckpt = json.load(f)
        print(f"✓ Loaded checkpoint: {ckpt['total_written']} samples done, "
              f"{len(ckpt['completed_videos'])} videos complete, "
              f"next shard = {ckpt['next_shard']}")
        return ckpt
    return {"completed_videos": [], "total_written": 0, "next_shard": 0}


def save_checkpoint(output_path: str, completed: list, total_written: int,
                    next_shard: int) -> None:
    """Atomically write checkpoint via tmp + rename."""
    p = _ckpt_path(output_path)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "completed_videos": completed,
            "total_written": total_written,
            "next_shard": next_shard,
        }, f, indent=2)
    os.replace(tmp, p)  # atomic on POSIX


def cleanup_partial_shards(shard_dir: str, shard_prefix: str,
                           next_shard: int) -> None:
    """Remove shard files with index >= next_shard (potentially corrupt)."""
    for f in sorted(glob.glob(os.path.join(shard_dir, f"{shard_prefix}-*.tar"))):
        try:
            num = int(Path(f).stem.split("-")[-1])
        except ValueError:
            continue
        if num >= next_shard:
            os.remove(f)
            print(f"  Cleaned up partial shard: {Path(f).name}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Video → RGB-D WebDataset shards via Depth Anything V2 "
                    "(with SLURM-friendly checkpointing)",
    )
    p.add_argument("--video_dir", type=str, required=True,
                   help="Root directory of videos (searched recursively)")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output root – shards go to output_path/shards, "
                        "samples to output_path/samples")
    p.add_argument("--depth_anything_repo", type=str, required=True,
                   help="Path to the Depth-Anything-V2 repo (contains "
                        "depth_anything_v2/ package)")
    p.add_argument("--model_checkpoint", type=str, required=True,
                   help="Path to model checkpoint, e.g. "
                        "checkpoints/depth_anything_v2_vitl.pth")
    p.add_argument("--encoder", type=str, default="vitl",
                   choices=["vits", "vitb", "vitl", "vitg"],
                   help="Encoder variant (default: vitl)")
    p.add_argument("--shard_size", type=int, default=5000,
                   help="Max samples per tar shard (default: 5000)")
    p.add_argument("--subsample", type=int, default=1,
                   help="Keep every N-th video")
    p.add_argument("--num_viz_samples", type=int, default=5,
                   help="Side-by-side visualizations to save per video")
    p.add_argument("--video_exts", type=str, nargs="+",
                   default=["mp4", "avi", "mov", "mkv"],
                   help="Video file extensions to search for")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_depth_model(repo_path: str, ckpt_path: str, encoder: str,
                     device: str):
    """Load and return the Depth Anything V2 model."""
    sys.path.insert(0, repo_path)
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64,
                 "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128,
                 "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256,
                 "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384,
                 "out_channels": [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**configs[encoder])
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------

def discover_videos(root: str, exts: list[str]) -> list[str]:
    """Recursively find video files under *root*."""
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, f"**/*.{ext}"),
                               recursive=True))
        paths.extend(glob.glob(os.path.join(root, f"**/*.{ext.upper()}"),
                               recursive=True))
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalize a depth map to [0, 1] float32."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        return ((depth - d_min) / (d_max - d_min)).astype(np.float32)
    return np.zeros_like(depth, dtype=np.float32)


def save_side_by_side(rgb: np.ndarray, depth_norm: np.ndarray,
                      path: str) -> None:
    """Write an [RGB | Depth-colormap] visualization to *path*."""
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    combined = np.concatenate([rgb, depth_color], axis=1)
    cv2.imwrite(path, combined)


# ---------------------------------------------------------------------------
# Per-video processing (streaming, constant memory)
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    video_name: str,
    model,
    device: str,
    writer: wds.ShardWriter,
    sample_dir: str,
    num_viz: int,
) -> int:
    """
    Stream through *video_path*, extract 1 frame/sec, run depth estimation,
    resize to half the video's native resolution, and write each sample to
    the open *writer*.  Returns count of written samples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Could not open video")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"  ✗ Invalid FPS ({fps})")
        cap.release()
        return 0

    # Detect native resolution and compute half-res
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w = native_w // 2
    half_h = native_h // 2

    if half_w < 1 or half_h < 1:
        print(f"  ✗ Video too small ({native_w}x{native_h})")
        cap.release()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(round(fps))  # 1 frame per second
    estimated_samples = max(1, total_frames // frame_interval)

    print(f"  FPS={fps:.1f}  total_frames={total_frames}  "
          f"native={native_w}x{native_h}  half={half_w}x{half_h}  "
          f"≈{estimated_samples} samples")

    # Pre-select which sample indices to save as visualizations
    num_viz_actual = min(num_viz, estimated_samples)
    viz_indices = set(random.sample(range(estimated_samples), num_viz_actual))

    vid_sample_dir = os.path.join(sample_dir, video_name)
    os.makedirs(vid_sample_dir, exist_ok=True)

    sample_idx = 0
    frame_idx = 0
    written = 0

    pbar = tqdm(total=estimated_samples, desc="  Depth estimation",
                leave=False, unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            with torch.no_grad():
                depth_raw = model.infer_image(frame)  # HxW float numpy

            rgb_small = cv2.resize(frame, (half_w, half_h),
                                   interpolation=cv2.INTER_LINEAR)
            depth_small = cv2.resize(depth_raw, (half_w, half_h),
                                     interpolation=cv2.INTER_LINEAR)
            depth_norm = normalize_depth(depth_small)

            ok, rgb_enc = cv2.imencode(".png", rgb_small)
            if ok:
                key = f"{video_name}_f{sample_idx:06d}"
                writer.write({
                    "__key__": key,
                    "rgb.png": rgb_enc.tobytes(),
                    "depth.npy": depth_norm,
                })
                written += 1

            if sample_idx in viz_indices:
                viz_path = os.path.join(vid_sample_dir,
                                        f"sample_{sample_idx:05d}.png")
                save_side_by_side(rgb_small, depth_norm, viz_path)

            sample_idx += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Output directories
    shard_dir = os.path.join(args.output_path, "shards")
    sample_dir = os.path.join(args.output_path, "samples")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint: load previous state (if any)
    # ------------------------------------------------------------------
    ckpt = load_checkpoint(args.output_path)
    completed_set = set(ckpt["completed_videos"])
    total_written = ckpt["total_written"]
    next_shard = ckpt["next_shard"]

    SHARD_PREFIX = "rgbd"
    cleanup_partial_shards(shard_dir, SHARD_PREFIX, next_shard)

    # ------------------------------------------------------------------
    # Load depth model
    # ------------------------------------------------------------------
    print("Loading Depth Anything V2 model...")
    model = load_depth_model(args.depth_anything_repo, args.model_checkpoint,
                             args.encoder, device)
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # Discover & subsample videos  (deterministic order for resumability)
    # ------------------------------------------------------------------
    all_videos = discover_videos(args.video_dir, args.video_exts)
    # Subsample: keep every N-th video
    video_files = all_videos[1::args.subsample]
    video_files += [
        '/nfs/data/public/LEMON/hN__Tayw4NI.mp4',
        '/nfs/data/public/LEMON/ojpJOmBOeYk.mp4',
        '/nfs/data/public/LEMON/wB_k6T619g0.mp4',
        '/nfs/data/public/LEMON/PVuBZxqjh-o.mp4',
        '/nfs/data/public/LEMON/-ziJl5iTe8U.mp4',
        '/nfs/data/public/LEMON/uRCpXy-Yvmk.mp4',
        '/nfs/data/public/LEMON/D_Abmcuwqbc.mp4',
        '/nfs/data/public/LEMON/KQgy_MfvSTY.mp4',
        '/nfs/data/public/LEMON/PvlLX12-t7Q.mp4',
        '/nfs/data/public/LEMON/eaYhsQh5Xbg.mp4',
        '/nfs/data/public/LEMON/7xJaXrgnkOg.mp4',
    ]
    print(f"Found {len(all_videos)} video(s), subsampled to "
          f"{len(video_files)} (every {args.subsample}).\n")

    if not video_files:
        return

    # Deterministic shuffle (same seed → same order on every run, so
    # checkpoint resume picks up right where it left off).
    random.shuffle(video_files)

    # Filter out already-completed videos
    remaining = [(i, v) for i, v in enumerate(video_files)
                 if v not in completed_set]
    print(f"Already completed: {len(video_files) - len(remaining)} / "
          f"{len(video_files)} videos.  Remaining: {len(remaining)}.\n")

    if not remaining:
        print("Nothing to do – all videos already processed.")
        return

    # ------------------------------------------------------------------
    # Stream-write to shards  (constant memory, resumable)
    # ------------------------------------------------------------------
    shard_pattern = os.path.join(shard_dir, f"{SHARD_PREFIX}-%06d.tar")
    writer = wds.ShardWriter(shard_pattern, maxcount=args.shard_size)
    # Advance the internal shard counter so we don't overwrite existing
    # shards from previous runs.  ShardWriter lazily opens the first tar
    # on first write, so setting .shard beforehand is safe.
    writer.shard = next_shard

    completed_list = list(ckpt["completed_videos"])  # mutable copy

    for pos, (orig_idx, vpath) in enumerate(remaining):
        vname = Path(vpath).stem
        print(f"[{orig_idx + 1}/{len(video_files)}] {vpath}")

        n = process_video(
            video_path=vpath,
            video_name=vname,
            model=model,
            device=device,
            writer=writer,
            sample_dir=sample_dir,
            num_viz=args.num_viz_samples,
        )
        total_written += n
        completed_list.append(vpath)

        # writer.shard is the *next* shard that will be opened on write.
        current_next_shard = writer.shard

        # Checkpoint after every video (atomic write)
        save_checkpoint(args.output_path, completed_list, total_written,
                        current_next_shard)

        print(f"  → {n} samples  (cumulative: {total_written})  "
              f"[checkpointed, shard {current_next_shard}]\n")

        # Honour SIGTERM / SIGINT for graceful SLURM preemption
        if _shutdown_requested:
            print("⚠  Shutdown requested – closing writer and exiting.")
            writer.close()
            save_checkpoint(args.output_path, completed_list, total_written,
                            writer.shard)
            print(f"Checkpoint saved.  Resume with the same command.\n")
            sys.exit(0)

    writer.close()

    # Final checkpoint (marks everything done + final shard count)
    save_checkpoint(args.output_path, completed_list, total_written,
                    writer.shard)

    # Summary
    num_shards = len(glob.glob(os.path.join(shard_dir, f"{SHARD_PREFIX}-*.tar")))
    print("=" * 60)
    print(f"Done!  {total_written} samples across {num_shards} shard(s)")
    print(f"  Shards:  {shard_dir}")
    print(f"  Samples: {sample_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ===================================================================
# Example: training dataloader
# ===================================================================
#
#   import webdataset as wds
#   import numpy as np
#   import torch
#   from torchvision import transforms
#   from torch.utils.data import DataLoader
#   import io
#
#   SHARD_DIR = "/path/to/output/shards"
#   shard_urls = f"{SHARD_DIR}/rgbd-{{000000..000099}}.tar"
#
#   def decode_depth(key, data):
#       if not key.endswith(".npy"):
#           return None
#       return np.load(io.BytesIO(data))
#
#   # NOTE: samples are now at half the native video resolution (not
#   # a fixed square size), so you may want to resize/crop in the
#   # transform pipeline depending on your model's requirements.
#   rgb_transform = transforms.Compose([
#       transforms.ToTensor(),
#       transforms.Resize((224, 224)),  # or your model's input size
#       transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                            std=[0.229, 0.224, 0.225]),
#   ])
#
#   def preprocess(sample):
#       import cv2
#       rgb = cv2.imdecode(
#           np.frombuffer(sample["rgb.png"], np.uint8), cv2.IMREAD_COLOR)
#       rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
#       rgb = rgb_transform(rgb)
#
#       depth = sample["depth.npy"]
#       if isinstance(depth, bytes):
#           depth = np.load(io.BytesIO(depth))
#       depth = torch.from_numpy(depth).unsqueeze(0)  # 1xHxW
#       depth = torch.nn.functional.interpolate(
#           depth.unsqueeze(0), size=(224, 224), mode="bilinear",
#           align_corners=False,
#       ).squeeze(0)  # 1xHxW at target size
#
#       return rgb, depth
#
#   dataset = (
#       wds.WebDataset(shard_urls, shardshuffle=True)
#       .shuffle(2000)
#       .decode(decode_depth)
#       .map(preprocess)
#   )
#
#   loader = DataLoader(
#       dataset, batch_size=64, num_workers=4, pin_memory=True,
#   )
#
#   for rgb_batch, depth_batch in loader:
#       # rgb_batch:   (B, 3, 224, 224) float32
#       # depth_batch: (B, 1, 224, 224) float32 in [0,1]
#       ...