# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One scene definition's multi-camera rollout, a chunk at a time.

Follows ``MultiViewRenderBackend._start_rollout``: the map is drawn and encoded
a chunk at a time by ``RenderedControls``, so a run has no end. The rollout
prefills a window of controls ahead of what it generates, so the client must
stream frames ahead of the responses it waits for.
"""

from __future__ import annotations

import time

from loguru import logger

from cosmos3_n3bula.config import CHUNK_LATENT_FRAMES, DEFAULT_PROMPT
from cosmos3_n3bula.engine import Engine
from cosmos3_n3bula.scene.renderer import RigRenderer
from cosmos3_n3bula.stream.feed import FrameFeed
from cosmos3_n3bula.stream.output import (
    chunk_responses,
    encode_controls,
    encode_generated,
)
from cosmos3_n3bula.stream.rig import rig_cameras
from cosmos3_n3bula.stream.stage_clock import _StageClock
from cosmos3_nano_sim_auto.config import denoise_schedule
from cosmos3_nano_sim_auto.impl.conditioning.control_provider import RenderedControls
from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import FED
from cosmos3_nano_sim_auto.impl.decode import decoded_views
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout
from cosmos3_nano_sim_auto.inputs import (
    WAN_TEMPORAL_COMPRESSION,
    MultiviewSample,
    camera_view_ids,
    patch_grid,
    prompt_token_ids,
)


class CameraSession:
    """Generates the rig's frames for one definition from a :class:`FrameFeed`."""

    def __init__(self, engine: Engine, definition, feed: FrameFeed) -> None:
        config = definition.generation_config
        if config.enable_lidar:
            raise NotImplementedError("joint RGB+LiDAR generation is not served yet")
        settings = engine.settings
        self._settings = settings
        self._feed = feed
        self._clock = _StageClock(engine.device, settings.stage_timing)
        cameras = rig_cameras(definition)
        self.mount_ids = [one.message.mount_id for one in cameras]
        names = [one.canonical for one in cameras]
        spec = engine.network.config.spec
        patch = spec.vfm.latent_patch_size
        patch_h, patch_w = patch_grid(FED[1], FED[0], latent_patch_size=patch)
        geometry = ClipGeometry(
            num_views=len(cameras),
            frames_per_view=None,
            patch_h=patch_h,
            patch_w=patch_w,
            frames_per_chunk=CHUNK_LATENT_FRAMES,
            condition_frames=0,
            seconds_per_frame=WAN_TEMPORAL_COMPRESSION / settings.fps,
            rig_view_ids=camera_view_ids(names),
        )
        self._renderer = RigRenderer(
            definition,
            cameras,
            self._clock.wrap("feed_wait", feed.frame),
            device=engine.device,
            keep_controls=settings.return_control_images,
        )
        controls = RenderedControls(
            self._clock.wrap("render", self._renderer.render),
            [self._clock.wrap("control_encode", one) for one in engine.control_encode_streams(len(cameras))],
            geometry=geometry,
            latent_patch_size=patch,
        )
        sample = MultiviewSample(
            name=definition.run_id or "n3bula",
            model_mode="text2video",
            prompt=config.prompt or DEFAULT_PROMPT,
            resolution="480",
            cameras=tuple(names),
            control_paths=("",) * len(names),
            image_paths=(),
        )
        # Only per-camera captions read the duration; a single prompt ignores it.
        prompt = prompt_token_ids(sample, pixel_frames=1, fps=settings.fps)
        self._rollout = ChunkRollout(
            engine.network,
            geometry=geometry,
            control_tokens=controls,
            condition_tokens=None,
            text_ids=prompt.text_ids,
            view_text_tokens=prompt.view_text_tokens,
            fps=settings.fps,
            seed=int(config.random_seed),
            history_frames=settings.history_frames,
            token_frames=settings.history_frames,
            t_list=denoise_schedule(spec.vfm.fixed_step_t_list, settings.denoise_steps),
            attention_scope=settings.attention_scope,
            maskless_attention=True,
            fused_attention_merge=True,
            decomposed_temporal_window_seconds=settings.temporal_window_seconds,
            attention_impl=engine.attention_impl,
            # The fast profile is single-camera only: compiled denoise collapses a 6-camera rig
            # after its first chunk (eager stays coherent), the shared compiled encoder breaks
            # any rig, and NVIDIA's multi-view RMSNorm variant failed its quality gate.
            accelerated_rms_norm=settings.accelerated_rms_norm and len(cameras) == 1,
            compile_denoise=settings.compile_denoise if len(cameras) == 1 else None,
        )
        self._decode = engine.chunk_decoder(len(cameras))
        self._clock.take()  # the prefill's control window is not a chunk's cost
        self._decoded_to = 0
        self.emitted = 0
        # Answer time and frame count after each step; the first two steps (1 then 12
        # frames) carry torch.compile warm-up, so steady FPS starts after them.
        self.chunk_marks: list[tuple[float, int]] = []

    def step(self) -> list:
        """Generate the next chunk and answer its frames for every camera.

        Raises:
            StreamEnded: the client stopped before the controls this chunk needs.
        """
        clock = self._clock
        step = clock.wrap("step", self._rollout.step)
        trace, _ = step()
        latent = self._rollout.latent_for(self._decoded_to, trace.end)
        self._decoded_to = trace.end
        pixels = clock.wrap("decode", decoded_views)(latent, self._decode, parallel=None)
        first, last = self.emitted, self.emitted + int(pixels.shape[1])
        frames = [self._feed.frame(index) for index in range(first, last)]
        controls = (
            encode_controls(self._renderer.take(first, last))
            if self._settings.return_control_images
            else None
        )
        encoded = clock.wrap("jpeg", encode_generated)(pixels, self._settings.jpeg_quality)
        responses = chunk_responses(frames, self.mount_ids, encoded, controls)
        if clock.enabled:
            seconds = clock.take()
            wait = seconds.get("feed_wait", 0.0)
            seconds["render"] = seconds.get("render", 0.0) - wait
            inside = seconds["render"] + wait + seconds.get("control_encode", 0.0)
            seconds["denoise"] = seconds.get("step", 0.0) - inside
            logger.info(
                "N3BULA chunk frames={}..{} {}",
                first,
                last - 1,
                " ".join(f"{name}_ms={value * 1e3:.1f}" for name, value in sorted(seconds.items())),
            )
        self.emitted = last
        self._feed.release(last)
        self.chunk_marks.append((time.perf_counter(), last))
        return responses

    def steady_fps(self, warmup_steps: int = 2) -> float | None:
        """Frames per second after the warm-up steps, or ``None`` before two more steps."""
        if len(self.chunk_marks) < warmup_steps + 2:
            return None
        (start, before), (end, after) = self.chunk_marks[warmup_steps], self.chunk_marks[-1]
        return (after - before) / (end - start) if end > start else None
