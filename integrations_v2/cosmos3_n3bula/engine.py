# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint and the VAE weights, loaded once per server process.

Each scene definition opens a new rollout. Its encoder and decoder carry temporal
caches, so a session needs fresh caches, but not freshly loaded weights; the
upstream ``make_control_encode_streams``/``make_multiview_chunk_decoder`` factories
reload the weights on every call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from cosmos3_n3bula.config import ServerSettings


def _attention_impl(backend: str):
    """``None`` leaves ``ChunkRollout`` its own choice, as Interactive Drive does."""
    if backend == "default":
        return None
    from cosmos3_nano_sim_auto.impl.transformer.attention import (
        attention_implementation,
    )

    return attention_implementation(flex=backend == "flex")


class Engine:
    """Shared by every stream; streams are serialised, so nothing here is locked."""

    def __init__(self, settings: ServerSettings) -> None:
        from cosmos_interactive._taehv_super import SuperTAEHV
        from cosmos_interactive._vae import (
            TAEW2_2_SUPER_CHECKPOINT,
            wan22_vae_encoder_config,
        )
        from cosmos_interactive.checkpoint_load import resolve_checkpoint_source

        from cosmos3_nano_sim_auto.checkpoint import load_exported_checkpoint
        from cosmos3_nano_sim_auto.config import DEFAULT_CHECKPOINT

        self.settings = settings
        self.device = torch.device(settings.device)
        source = resolve_checkpoint_source(settings.checkpoint, default=DEFAULT_CHECKPOINT)
        self.network: Any = load_exported_checkpoint(source, device=str(self.device)).eval()
        self.dtype = self.network.config.dtype
        def load_encoder(compile: bool):
            return (
                wan22_vae_encoder_config(dtype=self.dtype, checkpoint_path=None, use_compile=compile)
                .setup()
                .to(self.device)
            )

        self._encoder = load_encoder(False)
        # One compiled encoder shared by several per-camera causal caches corrupts every
        # view after the first chunk (measured on a 3-camera rig), so only a single-camera
        # session uses the compiled copy.
        self._compiled_encoder = load_encoder(True) if settings.compile_control_encode else None
        self._encoder_device = next(self._encoder.parameters()).device
        self._taehv = (
            SuperTAEHV(
                checkpoint_path=TAEW2_2_SUPER_CHECKPOINT,
                use_cuda_graph=False,
                use_compile=settings.compile_decoder,
            )
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self._taehv_device = next(self._taehv.parameters()).device
        self.attention_impl = _attention_impl(settings.attention_backend)

    def _encode(self, clip: Tensor, cache=None, encoder=None) -> Tensor:
        if clip.ndim != 4 or clip.shape[1] != 3:
            raise ValueError(f"expected a [T, 3, H, W] clip, got {tuple(clip.shape)}.")
        encoder = encoder or self._encoder
        video = clip.to(device=self._encoder_device, dtype=self.dtype)[None]
        with torch.no_grad():
            latent = encoder(video) if cache is None else encoder(video, cache=cache)
        return latent[0].transpose(0, 1).contiguous()  # [C, T_lat, H_lat, W_lat]

    def encode(self, clip: Tensor) -> Tensor:
        """Encode a whole ``[T, 3, H, W]`` clip in ``[-1, 1]`` from an empty temporal state."""
        return self._encode(clip)

    def control_encode_streams(self, count: int) -> list[Callable[[Tensor], Tensor]]:
        """One stateful encode per camera, sharing the resident encoder weights."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}.")

        encoder = self._compiled_encoder if count == 1 and self._compiled_encoder is not None else self._encoder

        def make_stream() -> Callable[[Tensor], Tensor]:
            cache = encoder.initialize_autoregressive_cache()
            return lambda clip: self._encode(clip, cache, encoder)

        return [make_stream() for _ in range(count)]

    def chunk_decoder(self, num_views: int) -> Callable[[Tensor], Tensor]:
        """A streaming multiview decoder with a fresh cache over the resident TAEHV weights."""
        from cosmos3_nano_sim_auto.impl.decode import multiview_chunk_decoder

        cache = self._taehv.prepare_cache()

        def decode_step(z: Tensor) -> Tensor:
            with torch.no_grad():
                # TAEHV emits [0, 1]; the rollout speaks [-1, 1].
                return (
                    self._taehv.decode(z.to(device=self._taehv_device, dtype=self.dtype), cache=cache)
                    .mul(2)
                    .sub(1)
                    .float()
                )

        return multiview_chunk_decoder(decode_step, num_views=num_views)
