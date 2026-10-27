# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Draw a pushed N3BULA scene for every camera of the rig with Ludus."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import ludus_renderer
import numpy as np
import torch
from ludus_renderer import TimestampedScene
from ludus_renderer._ops.primitives import CAMERA_TYPE_REGULAR
from ludus_renderer.torch import LudusCudaTimestampedContext
from torch import Tensor

from cosmos3_n3bula.scene.geometry import camera_max_ray_angle, camera_to_world
from cosmos3_n3bula.scene.pools import (
    WSM4_PALETTE,
    make_frame_pools,
    make_scene,
    make_static_pools,
    origin_array,
    unsupported_surface_bands,
)
from cosmos3_n3bula.stream.rig import RigCamera
from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import FED, _shrunk
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.camera import smoothable

_LOGGER = logging.getLogger(__name__)


def ludus_camera(camera, device) -> tuple[ludus_renderer.FThetaCamera, tuple[int, int]]:
    """An F-theta camera scaled to the largest size Ludus still smooths, and that size."""
    spec, params = camera.spec, camera.spec.ftheta_param
    native = (int(spec.resolution_w), int(spec.resolution_h))
    drawn = smoothable(native)
    sx, sy = drawn[0] / native[0], drawn[1] / native[1]
    poly = list(params.angle_to_pixeldist_poly)
    cx, cy = params.principal_point_x, params.principal_point_y
    return (
        ludus_renderer.FThetaCamera(
            principal_point=torch.tensor([cx * sx, cy * sy], dtype=torch.float32, device=device),
            image_size=torch.tensor(drawn, dtype=torch.float32, device=device),
            fw_poly=torch.tensor(poly, dtype=torch.float32, device=device) * ((sx + sy) / 2),
            max_ray_angle=camera_max_ray_angle(*native, cx, cy, poly),
            depth_max=max(camera.far_m, 1.0),
        ),
        drawn,
    )


class RigRenderer:
    """The rig's map controls, drawn a frame span at a time.

    :meth:`render` is a ``RigRenderFn``: ``[V*T, 480, 832, 3]`` uint8, view-outer.
    """

    def __init__(
        self,
        definition,
        cameras: Sequence[RigCamera],
        frames: Callable[[int], object],
        *,
        device: torch.device | str = "cuda",
        keep_controls: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._context = LudusCudaTimestampedContext(device=self._device)
        self._context.set_depth_scaling(True)
        self._context.set_msaa_samples(4)
        self._context.set_max_tessellation_levels(cube=0)
        self._context.upload_color_palette(WSM4_PALETTE)
        self._scene_id = self._context.upload_scene(TimestampedScene([], [], []))
        self._definition = definition
        self._origin = origin_array(definition)
        self._static = make_static_pools(definition, self._device)
        self._cameras = [(one.message, *ludus_camera(one.message, self._device)) for one in cameras]
        # Every rig camera is uploaded once and selected by index per render call.
        self._context.upload_cameras([ludus for _, ludus, _ in self._cameras])
        index = lambda value, dtype: torch.tensor([value], dtype=dtype, device=self._device)
        self._scene_index = index(self._scene_id, torch.int32)
        self._camera_indices = [index(view, torch.int32) for view in range(len(self._cameras))]
        self._camera_type = index(CAMERA_TYPE_REGULAR, torch.int32)
        # The resident frame's pole lines and cube-pool topology, for in-place updates.
        self._resident_poles = None
        self._resident_cubes: list[tuple[int, int]] | None = None
        self._frames = frames
        self._keep = keep_controls
        self._kept: dict[int, Tensor] = {}
        self._reported_omissions: set[int] = set()
        # Scanned once: the static bands do not change within a definition.
        self._static_omissions = unsupported_surface_bands(definition.static_surface_bands)

    def render(self, first: int, last: int, views: Sequence[int] | None = None) -> Tensor:
        if first < 0 or first >= last:
            raise ValueError(f"[{first}, {last}) is not a frame span.")
        drawn = list(range(len(self._cameras)) if views is None else views)
        per_view: list[list[Tensor]] = [[] for _ in drawn]
        # ponytail: one replace_scene and one render call per frame per camera;
        # a multi-timestamp scene would batch a whole span per camera.
        for index in range(first, last):
            frame = self._frames(index)
            self._load(frame)
            images = [self._draw(frame, view) for view in drawn]
            if self._keep:
                self._kept[index] = torch.cat(images)
            for column, image in zip(per_view, images):
                column.append(image)
        torch.cuda.synchronize(self._device)
        return torch.cat([image for column in per_view for image in column])

    def take(self, first: int, last: int) -> list[Tensor]:
        """Kept ``[V, 480, 832, 3]`` controls for frames ``[first, last)``, released."""
        return [self._kept.pop(index) for index in range(first, last)]

    def _load(self, frame) -> None:
        omitted = dict(self._static_omissions)
        for semantic, count in unsupported_surface_bands(frame.surface_bands).items():
            omitted[semantic] = omitted.get(semantic, 0) + count
        for semantic, count in omitted.items():
            if semantic not in self._reported_omissions:
                _LOGGER.warning(
                    "WSM4 omitted %s unsupported surface band(s) with semantic id %s",
                    count,
                    semantic,
                )
                self._reported_omissions.add(semantic)
        if self._static is None or frame.strips or frame.surface_bands:
            self._resident_poles = self._resident_cubes = None
            self._context.replace_scene(self._scene_id, make_scene(self._definition, frame, self._device))
            return
        static_lines, static_polygons = self._static
        frame_lines, cubes = make_frame_pools(frame, self._origin, self._device)
        poles = _pole_key(frame)
        topology = [(int(pool.prim_type_id), int(pool.scales.shape[0])) for pool in cubes]
        if poles == self._resident_poles and topology == self._resident_cubes and all(
            self._context.update_cube_pool_at_index(self._scene_id, index, pool)
            for index, pool in enumerate(cubes)
        ):
            return
        self._context.replace_scene(
            self._scene_id,
            TimestampedScene(
                polyline_pools=[*static_lines, *frame_lines],
                polygon_pools=static_polygons,
                cube_pools=cubes,
            ),
        )
        self._resident_poles, self._resident_cubes = poles, topology

    def _draw(self, frame, view: int) -> Tensor:
        camera, _, (width, height) = self._cameras[view]
        pose = np.linalg.inv(camera_to_world(frame, camera, self._origin)).astype(np.float32)
        image = self._context.render(
            self._scene_index,
            self._camera_indices[view],
            torch.tensor([int(frame.timestamp_us)], dtype=torch.int64, device=self._device),
            self._camera_type,
            torch.from_numpy(pose[None]).to(self._device),
            (height, width),
        )
        if self._context.needs_vflip:
            image = image.flip(1)
        return _shrunk(image, FED)


def _pole_key(frame) -> tuple:
    """What the frame's mesh pole lines derive from; equal keys draw identical poles."""
    return tuple(
        (mesh.semantic, mesh.origin_world_m.x, mesh.origin_world_m.y, mesh.origin_world_m.z)
        for mesh in frame.meshes
    )
