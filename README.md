# DART: Depth-as-Target Pretraining

Reference implementation accompanying the paper submission: DART: Depth-as-Target Pretraining for Surgical Vision Foundation Models

This codebase pretrains a Vision Transformer with DINOv2-style
self-distillation augmented by an L1 depth-reconstruction objective at
iBOT-masked patch positions. Depth is used only as a supervision target;
the encoder never receives depth as input, so fine-tuning and inference
remain RGB-only.

## Layout

```
dart/
  constants.py           Normalization statistics for RGB and pseudo-depth
  ddp.py                 DDP helpers and 2D sin-cos positional embeddings
  vit.py                 Vision Transformer (PatchEmbed, Block, ViT)
  heads.py               DINO projection head
  losses.py              DINO / iBOT / KoLeo / depth reconstruction losses
  depth_decoders.py      Three depth decoder variants (MLP / xattn / MAE)
  data.py                iBOT mask generator, multi-crop aug, WebDataset I/O
  scheduler.py           Cosine scheduler with warmup
  model_factory.py       Build student/teacher, param groups, EMA update
  train_step.py          Single training step (computes all four losses)
train.py                 Entrypoint: argparse + training loop
appendices.pdf           Appendix to accompany the main paper 
video_to_rgbd_shards.py  RGB-D shard generation code 
```

## Data format

Training data is read as `webdataset` shards containing paired
`(rgb.png, depth.npy)` samples. Pseudo-depth maps are pre-computed once
with a frozen monocular depth estimator (Depth Anything V2) and cached.

## Preprocessing 

To generate the RGB-D shards from the LEMON videos, run:

```bash 
python video_to_rgbd_shards.py \
    --video_dir /path/to/videos \
    --output_path /path/to/output \
    --depth_anything_repo /path/to/Depth-Anything-V2 \
    --model_checkpoint checkpoints/depth_anything_v2_vitl.pth \
    --encoder vitl
```
## Run

Single-node, two GPUs:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) train.py \
    --data_path /path/to/shards \
    --save_dir /save/path \
    --arch vit_base --patch_size 16 --num_workers 18 --base_lr 1.5e-3 \
    --epochs 200 --batch_size 128 --warmup_epochs 20 \
    --warmup_teacher_temp_epochs 60 --momentum_teacher 0.996 \
    --depth_loss_weight 0.1 \
    --save_freq 5 \
    --log_freq 50
```

The default depth decoder is a 3-layer MLP. To use the MAE-style
decoder with the encoder CLS persisted through the decoder's self-attention
blocks:

```bash
... --depth_decoder mae --depth_decoder_mae_use_cls
```
