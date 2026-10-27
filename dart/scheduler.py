"""Cosine scheduler with linear warmup and optional freeze prefix.

Identical semantics to the DINOv2 reference scheduler: returns the
scheduled value for a given iteration index, clipping to `final_value`
beyond `total_iters`.
"""

from __future__ import annotations

import numpy as np


class CosineScheduler:
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        freeze = np.zeros(freeze_iters)
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        if len(iters) > 0:
            schedule = final_value + 0.5 * (base_value - final_value) * (
                1 + np.cos(np.pi * iters / len(iters))
            )
        else:
            schedule = np.array([base_value])
        self.schedule = np.concatenate([freeze, warmup, schedule])
        self.total_iters = total_iters
        self.final_value = final_value

    def __getitem__(self, it):
        if it >= len(self.schedule):
            return self.final_value
        return self.schedule[it]
