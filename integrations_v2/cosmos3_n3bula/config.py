# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the server chooses per process; the checkpoint's knobs come from Cosmos3 defaults."""

from __future__ import annotations

from dataclasses import dataclass

from cosmos3_nano_sim_auto.config import (
    DEFAULT_ATTENTION_SCOPE,
    DEFAULT_DENOISE_STEPS,
    DEFAULT_HISTORY_FRAMES,
    DEFAULT_TEMPORAL_WINDOW_SECONDS,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import AttentionScope

ATTENTION_BACKENDS = ("default", "flex", "cudnn")
"""``default`` leaves ``ChunkRollout`` its own choice, as Interactive Drive does."""
COMPILE_DENOISE_CHOICES = ("off", "default", "max-autotune", "max-autotune-no-cudagraphs")

DEFAULT_PROMPT = (
    "A multi-camera video of a car driving on a road, realistic photographic "
    "scenery, natural daylight, other vehicles on the road."
)
CHUNK_LATENT_FRAMES = 3
"""Latent frames a chunk generates: twelve pixel frames, as the checkpoint was trained."""


@dataclass(frozen=True)
class ServerSettings:
    checkpoint: str | None = None
    device: str = "cuda:0"
    fps: float = 30.0
    history_frames: int = DEFAULT_HISTORY_FRAMES
    denoise_steps: int | None = DEFAULT_DENOISE_STEPS
    attention_scope: AttentionScope = DEFAULT_ATTENTION_SCOPE
    temporal_window_seconds: float = DEFAULT_TEMPORAL_WINDOW_SECONDS
    return_control_images: bool = False
    """Also send each frame's drawn map as ``RenderedControlFrame.image`` (PNG)."""
    jpeg_quality: int = 90
    attention_backend: str = "default"
    """``default``, ``flex`` (block-sparse FlexAttention) or ``cudnn`` (dense SDPA)."""
    # The fastest measured single-GPU profile (RTX PRO 6000 Max-Q, 1 view): 15.8 FPS against
    # 14.4 eager. Applied to single-camera sessions only; a multi-camera rig runs eager. Interactive Drive leaves these off for these weights; the compiled rollout
    # matches eager for about 30 frames, then follows its own plausible trajectory.
    accelerated_rms_norm: bool = True
    """FlashDreams Accelerated's fused RMSNorm; applied to single-camera sessions only."""
    compile_denoise: str | None = "default"
    """A ``torch.compile`` mode for the noisy pass; ``None`` keeps it eager."""
    compile_decoder: bool = False
    """Measured no gain (about 11 ms per chunk either way) and slows the first chunk."""
    compile_control_encode: bool = True
    """Compiled control encoder for single-camera sessions; a rig shares the eager one."""
    stage_timing: bool = False
    """Log CUDA-synchronized per-chunk stage times; the syncs cost a little throughput."""
