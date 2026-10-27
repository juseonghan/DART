"""DART: Depth-as-Target pretraining for surgical vision foundation models.

Reference implementation accompanying the paper submission.
"""

from .constants import IMG_MEAN, IMG_STD, DEPTH_MEAN, DEPTH_STD
from .vit import ViT
from .heads import DINOHead
from .losses import DINOLoss, iBOTLoss, KoLeoLoss, depth_reconstruction_loss
from .depth_decoders import (
    DepthDecoderMLP,
    DepthDecoderXAttn,
    DepthDecoderMAE,
    build_depth_decoder,
)
from .data import (
    MaskGenerator,
    RGBMultiCropAugmentation,
    make_loader,
    collate_dinov2,
)
from .scheduler import CosineScheduler
from .model_factory import (
    VIT_CONFIGS,
    build_student_teacher,
    build_param_groups,
    update_teacher,
)
from .train_step import train_one_step
