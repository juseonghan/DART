"""Normalization constants for the pretraining corpus.

The mean/std values below are computed across the surgical RGB pretraining
corpus and the corresponding pseudo-depth maps (Depth Anything V2). They
are used to z-normalize inputs and depth targets during pretraining.
"""

# RGB statistics (per-channel mean / std, in [0, 1])
IMG_MEAN = [0.44476654, 0.35751768, 0.34613244]
IMG_STD = [0.28885465, 0.25989645, 0.26113421]

# Depth target statistics (scalar mean / std over standardized pseudo-depth)
DEPTH_MEAN = 0.3384192098550095
DEPTH_STD = 0.23974234373402492
