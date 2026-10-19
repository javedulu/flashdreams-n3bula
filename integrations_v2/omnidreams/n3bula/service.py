# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""N3BULA typed-scene adapter for the FlashDreams Ludus renderer."""

from __future__ import annotations

import os
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO

import grpc
import ludus_renderer
import numpy as np
import torch
from loguru import logger
from ludus_renderer import (
    PRIM_INTERSECTION,
    PRIM_LANE_BOUNDARY,
    PRIM_ROAD_ISLAND,
    TimestampedPolygonPool,
)
from ludus_renderer._ops.primitives import (
    CAMERA_TYPE_REGULAR,
    CUBE_FLAG_WIREFRAME,
    PRIM_CROSSWALK,
    PRIM_DOT_WHITE,
    PRIM_DOT_YELLOW,
    PRIM_LANE_LINE_WHITE_DASHED,
    PRIM_LANE_LINE_WHITE_SOLID,
    PRIM_LANE_LINE_YELLOW_DASHED,
    PRIM_LANE_LINE_YELLOW_SOLID,
    PRIM_OBSTACLE,
    PRIM_POLE,
    PRIM_ROAD_BOUNDARY,
    PRIM_ROAD_MARKING,
    PRIM_STATIC_OBSTACLE,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_TYPE_COUNT,
    PRIM_WAIT_LINE,
    CubePool,
    TimestampedPolylinePool,
    TimestampedScene,
)
from ludus_renderer.torch import LudusCudaTimestampedContext
from flashdreams.infra.postprocess import VideoChunk, to_bvtchw
from omnidreams.n3bula import n3bula_pb2, n3bula_pb2_grpc
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision.io import encode_jpeg

DEFAULT_PROMPT = (
    "A forward-facing dashcam video of a car driving on a road, realistic "
    "photographic scenery, natural daylight, other vehicles on the road."
)


def _process_rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


def _gpu_memory_mib() -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return -1.0, -1.0, -1.0
    try:
        device = torch.cuda.current_device()
        mib = 1024**2
        return (
            torch.cuda.memory_allocated(device) / mib,
            torch.cuda.memory_reserved(device) / mib,
            torch.cuda.max_memory_allocated(device) / mib,
        )
    except RuntimeError:
        return -1.0, -1.0, -1.0


SURFACE_BAND_PRIMITIVES = {
    n3bula_pb2.SEMANTIC_ID_ROAD_SURFACE: PRIM_INTERSECTION,
    n3bula_pb2.SEMANTIC_ID_SHOULDER: PRIM_ROAD_ISLAND,
    n3bula_pb2.SEMANTIC_ID_SIDEWALK: PRIM_TYPE_COUNT,
    n3bula_pb2.SEMANTIC_ID_CURB: PRIM_TYPE_COUNT + 1,
    n3bula_pb2.SEMANTIC_ID_PARKING: PRIM_TYPE_COUNT + 2,
    n3bula_pb2.SEMANTIC_ID_BIKE_LANE: PRIM_TYPE_COUNT + 3,
    n3bula_pb2.SEMANTIC_ID_TERRAIN: PRIM_TYPE_COUNT + 4,
}
SURFACE_BAND_PALETTE = {
    PRIM_INTERSECTION: (46, 46, 46, 255),
    PRIM_ROAD_ISLAND: (70, 70, 70, 255),
    PRIM_TYPE_COUNT: (128, 128, 128, 255),
    PRIM_TYPE_COUNT + 1: (112, 112, 112, 255),
    PRIM_TYPE_COUNT + 2: (76, 76, 76, 255),
    PRIM_TYPE_COUNT + 3: (56, 116, 104, 255),
    PRIM_TYPE_COUNT + 4: (87, 79, 69, 255),
}
ACTOR_COLORS = {
    n3bula_pb2.ACTOR_CATEGORY_CAR: ((0, 46, 136), (126, 206, 255)),
    n3bula_pb2.ACTOR_CATEGORY_TRUCK: ((204, 55, 0), (255, 192, 64)),
    n3bula_pb2.ACTOR_CATEGORY_PEDESTRIAN: ((148, 0, 62), (255, 124, 171)),
    n3bula_pb2.ACTOR_CATEGORY_CYCLIST: ((0, 80, 66), (102, 208, 198)),
    n3bula_pb2.ACTOR_CATEGORY_OTHER: ((53, 26, 20), (166, 136, 125)),
    n3bula_pb2.ACTOR_CATEGORY_EGO: ((230, 77, 51), (153, 26, 26)),
}
BOX_PRIMITIVES = {
    n3bula_pb2.SEMANTIC_ID_TRAFFIC_SIGN: PRIM_TRAFFIC_SIGN,
    n3bula_pb2.SEMANTIC_ID_SPEED_SIGN: PRIM_TRAFFIC_SIGN,
    n3bula_pb2.SEMANTIC_ID_TRAFFIC_LIGHT: PRIM_TRAFFIC_LIGHT,
    n3bula_pb2.SEMANTIC_ID_CROSSWALK: PRIM_CROSSWALK,
}
BOX_COLORS = {
    n3bula_pb2.SEMANTIC_ID_TRAFFIC_SIGN: (8, 2, 255),
    n3bula_pb2.SEMANTIC_ID_SPEED_SIGN: (8, 2, 255),
    n3bula_pb2.SEMANTIC_ID_TRAFFIC_LIGHT: (100, 100, 100),
    n3bula_pb2.SEMANTIC_ID_CROSSWALK: (139, 93, 255),
}
# Mirrors __pys__/render_ludus_msgpack_frame.py and cosmos builder::expandRoadMark; Ludus has no dash geometry.
DASH_LENGTH_M = 3.0
DASH_PERIOD_M = 12.0
DOT_LENGTH_M = 0.5
DOT_PERIOD_M = 1.5
DUAL_LINE_SEPARATION_M = 0.15
LUDUS_MAX_POLYLINE_POINTS = 256
STATIC_TIMESTAMP_US = 0  # Static road geometry is valid for every frame.
# Generation reads GPU pixels when set; PNG bytes then only serve the response.
GPU_CONTROL_INPUT = os.environ.get("N3BULA_GPU_CONTROL_INPUT", "0") == "1"
GPU_JPEG_ENCODING = os.environ.get("N3BULA_GPU_JPEG_ENCODING", "0") == "1"
RETURN_CONTROL_IMAGES = os.environ.get("N3BULA_RETURN_CONTROL_IMAGES", "0") == "1"
_PNG_ENCODER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="n3bula-control-png")
MARK_COMPONENTS = {
    n3bula_pb2.MARK_TYPE_ID_SOLID: ("solid",),
    n3bula_pb2.MARK_TYPE_ID_BROKEN: ("broken",),
    n3bula_pb2.MARK_TYPE_ID_SOLID_SOLID: ("solid", "solid"),
    n3bula_pb2.MARK_TYPE_ID_SOLID_BROKEN: ("solid", "broken"),
    n3bula_pb2.MARK_TYPE_ID_BROKEN_SOLID: ("broken", "solid"),
    n3bula_pb2.MARK_TYPE_ID_BROKEN_BROKEN: ("broken", "broken"),
    n3bula_pb2.MARK_TYPE_ID_BOTTS_DOTS: ("dots",),
}


def _rotation(euler) -> np.ndarray:
    return Rotation.from_euler(
        "ZYX", [euler.yaw_rad, euler.pitch_rad, euler.roll_rad]
    ).as_matrix()


def _camera_max_ray_angle(width: int, height: int, cx: float, cy: float,
                          coefficients: list[float]) -> float:
    radius_limit = np.hypot(max(cx, width - cx), max(cy, height - cy))
    polynomial = np.asarray(coefficients, dtype=np.float64)
    derivative = np.polynomial.polynomial.polyder(polynomial)
    angle = 0.0
    for index in range(1, 2001):
        candidate = index * np.pi / 2000
        radius = np.polynomial.polynomial.polyval(candidate, polynomial)
        slope = np.polynomial.polynomial.polyval(candidate, derivative)
        if slope <= 0 or radius > radius_limit:
            break
        angle = candidate
    if angle <= 0:
        raise ValueError("camera F-theta polynomial has no valid forward domain")
    return float(angle * 1.05)


def _line_primitive(strip) -> int:
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_ROAD_BOUNDARY:
        return PRIM_ROAD_BOUNDARY
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_CROSSWALK:
        return PRIM_CROSSWALK
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_STOP_LINE:
        return PRIM_WAIT_LINE
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_TRAFFIC_SIGN:
        return PRIM_TRAFFIC_SIGN
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_SPEED_SIGN:
        return PRIM_TRAFFIC_SIGN
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_TRAFFIC_LIGHT:
        return PRIM_TRAFFIC_LIGHT
    if strip.semantic == n3bula_pb2.SEMANTIC_ID_POLE:
        return PRIM_POLE
    if strip.semantic != n3bula_pb2.SEMANTIC_ID_LANE_MARKING:
        return PRIM_ROAD_MARKING
    return _lane_primitive(strip.colour, MARK_COMPONENTS.get(strip.pattern, ("solid",))[0])


def _lane_primitive(colour, component: str) -> int:
    if colour == n3bula_pb2.MARK_COLOR_ID_YELLOW:
        ids = {"solid": PRIM_LANE_LINE_YELLOW_SOLID, "broken": PRIM_LANE_LINE_YELLOW_DASHED,
               "dots": PRIM_DOT_YELLOW}
    elif colour == n3bula_pb2.MARK_COLOR_ID_WHITE:
        ids = {"solid": PRIM_LANE_LINE_WHITE_SOLID, "broken": PRIM_LANE_LINE_WHITE_DASHED,
               "dots": PRIM_DOT_WHITE}
    else:
        return PRIM_ROAD_MARKING
    return ids[component]


def _offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    shifted = points.copy()
    for index in range(len(points)):
        before = points[max(index - 1, 0)]
        after = points[min(index + 1, len(points) - 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = np.hypot(dx, dy)
        if length > 0.0:
            shifted[index, 0] = points[index, 0] - dy / length * offset
            shifted[index, 1] = points[index, 1] + dx / length * offset
    return shifted


def _dash_polyline(points: np.ndarray) -> list[np.ndarray]:
    """Paint DASH_LENGTH_M of every DASH_PERIOD_M, phase anchored at the chain start."""
    dashes, current, travelled = [], [], 0.0
    for a, b in zip(points, points[1:]):
        length = float(np.linalg.norm(b - a))
        if length <= 0.0:
            continue
        offset = 0.0
        while offset < length:
            phase = (travelled + offset) % DASH_PERIOD_M
            painted = phase < DASH_LENGTH_M
            next_edge = DASH_LENGTH_M if painted else DASH_PERIOD_M
            step = min(length - offset, next_edge - phase)
            if step < 1e-6:
                offset += 1e-6
                continue
            if painted:
                if not current:
                    current.append(a + (b - a) * (offset / length))
                current.append(a + (b - a) * ((offset + step) / length))
            if (not painted or phase + step >= DASH_LENGTH_M - 1e-9) and current:
                dashes.append(np.asarray(current))
                current = []
            offset += step
        travelled += length
    if current:
        dashes.append(np.asarray(current))
    return [dash for dash in dashes if len(dash) >= 2]


def _dot_polyline(points: np.ndarray) -> np.ndarray:
    """One vertex per DOT_PERIOD_M of arc length; Ludus draws each vertex as a dot."""
    dots, travelled, next_dot = [], 0.0, DOT_LENGTH_M / 2
    for a, b in zip(points, points[1:]):
        length = float(np.linalg.norm(b - a))
        while length > 0.0 and next_dot <= travelled + length:
            dots.append(a + (b - a) * ((next_dot - travelled) / length))
            next_dot += DOT_PERIOD_M
        travelled += length
    return np.asarray(dots).reshape(-1, 3)


def _strip_lines(strip, points: np.ndarray, already_expanded: bool = False) -> list[tuple[int, list[np.ndarray]]]:
    """Return (Ludus primitive, polylines) with the mark pattern expanded into geometry."""
    if strip.semantic != n3bula_pb2.SEMANTIC_ID_LANE_MARKING:
        return [(_line_primitive(strip), [points])]
    if len(points) == 1 and (already_expanded
            or strip.pattern == n3bula_pb2.MARK_TYPE_ID_BOTTS_DOTS):
        return [(_lane_primitive(strip.colour, "dots"), [points])]
    if len(points) < 2:
        return []
    components = MARK_COMPONENTS.get(strip.pattern, ("solid",))
    offsets = (DUAL_LINE_SEPARATION_M / 2, -DUAL_LINE_SEPARATION_M / 2) \
        if len(components) == 2 else (0.0,)
    layers = []
    for component, offset in zip(components, offsets):
        track = _offset_polyline(points, offset) if offset else points
        if component == "broken":
            chunks = _dash_polyline(track)
        elif component == "dots":
            chunks = [_dot_polyline(track)]
        else:
            chunks = [track]
        layers.append((_lane_primitive(strip.colour, component), chunks))
    return layers


def _split_polyline(points: np.ndarray):
    if len(points) <= LUDUS_MAX_POLYLINE_POINTS:
        yield points
        return
    for start in range(0, len(points) - 1, LUDUS_MAX_POLYLINE_POINTS - 1):
        yield points[start:start + LUDUS_MAX_POLYLINE_POINTS]


def _box_primitive(box) -> int:
    if box.semantic in (n3bula_pb2.SEMANTIC_ID_ACTOR, n3bula_pb2.SEMANTIC_ID_EGO):
        return PRIM_OBSTACLE
    return BOX_PRIMITIVES.get(box.semantic, PRIM_STATIC_OBSTACLE)


def _box_colors(box) -> list[float]:
    if box.semantic in (n3bula_pb2.SEMANTIC_ID_ACTOR, n3bula_pb2.SEMANTIC_ID_EGO):
        front, back = ACTOR_COLORS.get(
            box.category, ACTOR_COLORS[n3bula_pb2.ACTOR_CATEGORY_OTHER]
        )
    else:
        color = BOX_COLORS.get(box.semantic, (128, 128, 128))
        front = back = color
    return [channel / 255.0 for channel in (*front, *back)]


def _polyline_pools(timestamp_us: int, groups: dict[int, list[np.ndarray]], device) -> list:
    pools = []
    for primitive_id, lines in groups.items():
        minimum_points = 1 if primitive_id in (PRIM_DOT_WHITE, PRIM_DOT_YELLOW) else 2
        lines = [chunk for line in lines for chunk in _split_polyline(line)
                 if len(chunk) >= minimum_points]
        if not lines:
            continue
        lengths = np.asarray([len(line) for line in lines], dtype=np.int32)
        pools.append(
            TimestampedPolylinePool(
                timestamps_us=torch.tensor([timestamp_us], dtype=torch.int64, device=device),
                timestamped_varrays_prefix_sum=torch.tensor(
                    [len(lines)], dtype=torch.int32, device=device),
                varrays_prefix_sum=torch.tensor(
                    np.cumsum(lengths), dtype=torch.int32, device=device),
                vertices=torch.as_tensor(
                    np.concatenate(lines).astype(np.float32, copy=False), device=device),
                prim_type_id=primitive_id,
            )
        )
    return pools


def _point_signature(points) -> tuple:
    coordinates = tuple(
        tuple(round(float(value), 3) for value in (point.x, point.y, point.z))
        for point in points
    )
    return min(coordinates, coordinates[::-1])


def _surface_polygon_pools(timestamp_us: int, bands, origin, device) -> list:
    grouped = defaultdict(list)
    for band in bands:
        primitive = SURFACE_BAND_PRIMITIVES.get(band.semantic)
        left, right = band.left_world_m, band.right_world_m
        if primitive is None or len(left) < 2 or len(left) != len(right):
            continue

        vertices = []
        for left_point, right_point in zip(left, right):
            vertices.extend(
                [[point.x - origin[0], point.y - origin[1], point.z - origin[2]]
                 for point in (left_point, right_point)]
            )
        triangles = []
        for segment in range(len(left) - 1):
            first = segment * 2
            triangles.extend(((first, first + 1, first + 3),
                              (first, first + 3, first + 2)))
        grouped[primitive].append((
            torch.tensor(vertices, dtype=torch.float32, device=device),
            torch.tensor(triangles, dtype=torch.int32, device=device),
        ))

    pools = []
    for primitive, polygons in grouped.items():
        vertex_counts = [len(vertices) for vertices, _ in polygons]
        triangle_counts = [len(triangles) for _, triangles in polygons]
        pools.append(TimestampedPolygonPool(
            timestamps_us=torch.tensor([timestamp_us], dtype=torch.int64, device=device),
            timestamped_varrays_prefix_sum=torch.tensor(
                [len(polygons)], dtype=torch.int32, device=device),
            varrays_prefix_sum=torch.tensor(
                np.cumsum(vertex_counts), dtype=torch.int32, device=device),
            triangle_prefix_sum=torch.tensor(
                np.cumsum(triangle_counts), dtype=torch.int32, device=device),
            vertices=torch.cat([vertices for vertices, _ in polygons]),
            triangles=torch.cat([triangles for _, triangles in polygons]),
            prim_type_id=primitive,
        ))
    return pools


def _origin_array(definition) -> np.ndarray:
    return np.asarray(
        [definition.origin_world_m.x, definition.origin_world_m.y, definition.origin_world_m.z],
        dtype=np.float64,
    )


def _polyline_groups(definition, frame_strips, frame_bands, origin):
    lines: dict[int, list[np.ndarray]] = defaultdict(list)
    seen_signatures = set()
    for strip in definition.static_strips:
        points = np.asarray([[p.x, p.y, p.z] for p in strip.centre_world_m], dtype=np.float64)
        if len(points) == 0:
            continue
        for primitive, chunks in _strip_lines(strip, points):
            lines[primitive].extend(
                (chunk - origin).astype(np.float32) for chunk in chunks if len(chunk))
    for strip in frame_strips:
        points = np.asarray([[p.x, p.y, p.z] for p in strip.centre_world_m], dtype=np.float64)
        if len(points) == 0:
            continue
        for primitive, chunks in _strip_lines(strip, points, already_expanded=True):
            lines[primitive].extend(
                (chunk - origin).astype(np.float32) for chunk in chunks if len(chunk))
    for strip in definition.static_strips:
        if len(strip.centre_world_m) >= 2:
            seen_signatures.add(_point_signature(strip.centre_world_m))

    bands = [*definition.static_surface_bands, *frame_bands]
    for band in bands:
        if band.semantic != n3bula_pb2.SEMANTIC_ID_ROAD_SURFACE:
            continue
        for edge in (band.left_world_m, band.right_world_m):
            if len(edge) < 2:
                continue
            signature = _point_signature(edge)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            points = np.asarray([[p.x, p.y, p.z] for p in edge], dtype=np.float64)
            lines[PRIM_LANE_BOUNDARY].append((points - origin).astype(np.float32))
    return lines, bands


def _cube_pools(frame, origin, timestamp: int, device, lines) -> list:
    cube_groups = defaultdict(list)
    for box in frame.boxes:
        primitive = _box_primitive(box)
        render_flags = CUBE_FLAG_WIREFRAME if box.wireframe_overlay else 0
        rotation = Rotation.from_matrix(_rotation(box.rotation_world)).as_quat()
        translation = np.asarray(
            [box.centre_world_m.x, box.centre_world_m.y, box.centre_world_m.z],
            dtype=np.float64,
        ) - origin
        cube_groups[(primitive, render_flags)].append((
            translation.astype(np.float32),
            rotation.astype(np.float32),
            np.asarray([box.length_m, box.width_m, box.height_m], dtype=np.float32),
            np.asarray(_box_colors(box), dtype=np.float32),
        ))

    for mesh in frame.meshes:
        is_light = mesh.semantic == n3bula_pb2.SEMANTIC_ID_TRAFFIC_LIGHT
        is_speed_sign = mesh.semantic == n3bula_pb2.SEMANTIC_ID_SPEED_SIGN
        dimensions = np.asarray(
            (0.3, 0.3, 0.5) if is_light else ((0.16, 1.04, 1.04) if is_speed_sign
                                               else (0.01, 0.5, 0.8)),
            dtype=np.float32,
        )
        mount_height = 3.0 if is_light else (2.57 if is_speed_sign else 2.1)
        head = np.asarray(
            [mesh.origin_world_m.x, mesh.origin_world_m.y, mesh.origin_world_m.z],
            dtype=np.float64,
        )
        if head[2] < 0.5:
            head[2] += mount_height
        semantic = (n3bula_pb2.SEMANTIC_ID_TRAFFIC_LIGHT if is_light else
                    n3bula_pb2.SEMANTIC_ID_SPEED_SIGN if is_speed_sign else
                    n3bula_pb2.SEMANTIC_ID_TRAFFIC_SIGN)
        color = BOX_COLORS[semantic]
        mesh_rotation = Rotation.from_euler("z", mesh.yaw_rad).as_quat().astype(np.float32)
        primitive = BOX_PRIMITIVES[semantic]
        cube_groups[(primitive, 0)].append((
            (head - origin).astype(np.float32),
            mesh_rotation,
            dimensions,
            np.asarray([channel / 255.0 for channel in (*color, *color)], dtype=np.float32),
        ))
        head_bottom = head[2] - float(dimensions[2]) * 0.5
        if head_bottom > 0.05:
            pole = np.asarray(
                [[head[0], head[1], 0.0], [head[0], head[1], head_bottom]],
                dtype=np.float64,
            ) - origin
            lines[PRIM_POLE].append(pole.astype(np.float32))

    cube_pools = []
    for (primitive, render_flags), cubes in cube_groups.items():
        cube_count = len(cubes)
        cube_pools.append(CubePool(
            timestamps_us=torch.tensor([timestamp], dtype=torch.int64, device=device),
            cube_ts_prefix_sum=torch.arange(
                1, cube_count + 1, dtype=torch.int32, device=device),
            track_timestamps_us=torch.full(
                (cube_count,), timestamp, dtype=torch.int64, device=device),
            translations=torch.as_tensor(np.stack([cube[0] for cube in cubes]), device=device),
            quaternions=torch.as_tensor(np.stack([cube[1] for cube in cubes]), device=device),
            scales=torch.as_tensor(np.stack([cube[2] for cube in cubes]), device=device),
            colors=torch.as_tensor(np.stack([cube[3] for cube in cubes]), device=device),
            prim_type_id=primitive,
            render_flags=render_flags,
        ))

    return cube_pools


def _make_scene(definition, frame, device=torch.device("cpu")) -> TimestampedScene:
    origin = _origin_array(definition)
    timestamp = int(frame.timestamp_us)
    lines, bands = _polyline_groups(definition, frame.strips, frame.surface_bands, origin)
    cube_pools = _cube_pools(frame, origin, timestamp, device, lines)
    return TimestampedScene(
        polyline_pools=_polyline_pools(timestamp, lines, device),
        polygon_pools=_surface_polygon_pools(timestamp, bands, origin, device),
        cube_pools=cube_pools,
    )


def _make_static_pools(definition, device=torch.device("cpu")):
    """Build road geometry once per definition; None if it would share PRIM_POLE with per-frame poles."""
    origin = _origin_array(definition)
    lines, bands = _polyline_groups(definition, (), (), origin)
    if PRIM_POLE in lines:
        return None
    return (
        _polyline_pools(STATIC_TIMESTAMP_US, lines, device),
        _surface_polygon_pools(STATIC_TIMESTAMP_US, bands, origin, device),
    )


def _make_frame_pools(frame, origin, device=torch.device("cpu")):
    timestamp = int(frame.timestamp_us)
    lines: dict[int, list[np.ndarray]] = defaultdict(list)
    cube_pools = _cube_pools(frame, origin, timestamp, device, lines)
    return _polyline_pools(timestamp, lines, device), cube_pools


def _static_timestamp_pools(pools):
    return [
        TimestampedPolylinePool(
            timestamps_us=torch.full_like(pool.timestamps_us, STATIC_TIMESTAMP_US),
            timestamped_varrays_prefix_sum=pool.timestamped_varrays_prefix_sum,
            varrays_prefix_sum=pool.varrays_prefix_sum,
            vertices=pool.vertices,
            prim_type_id=pool.prim_type_id,
        )
        for pool in pools
    ]


def _same_polyline_pools(first, second) -> bool:
    if first is None or len(first) != len(second):
        return False
    tensor_fields = (
        "timestamps_us",
        "timestamped_varrays_prefix_sum",
        "varrays_prefix_sum",
        "vertices",
    )
    return all(
        left.prim_type_id == right.prim_type_id
        and all(torch.equal(getattr(left, name), getattr(right, name))
                for name in tensor_fields)
        for left, right in zip(first, second)
    )


def _encode_control_png(pixels: np.ndarray) -> tuple[bytes, float]:
    started = time.perf_counter()
    encoded = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(encoded, format="PNG")
    return encoded.getvalue(), time.perf_counter() - started


class _RenderedControl:
    def __init__(self, frame, camera_mount_id, pixels, png):
        self.frame = frame
        self.camera_mount_id = camera_mount_id
        self.pixels = pixels
        self.png = png  # PNG bytes, a Future of (bytes, encode seconds), or None

    def response(self, png=None):
        response = n3bula_pb2.RenderedControlFrame(
            camera_mount_id=self.camera_mount_id,
            source_frame_id=self.frame.source_frame_id,
            timestamp_us=self.frame.timestamp_us,
            ego_actor_id=self.frame.ego_actor_id,
        )
        if png is not None:
            response.image.CopyFrom(
                n3bula_pb2.Image(data=png, format=n3bula_pb2.PNG)
            )
        return response


class _StreamRenderer:
    def __init__(self):
        self.device = torch.device("cuda")
        self.context = LudusCudaTimestampedContext(device=self.device)
        self.context.set_depth_scaling(True)
        self.context.set_msaa_samples(4)
        self.context.set_max_tessellation_levels(cube=0)
        self.context.upload_color_palette(SURFACE_BAND_PALETTE)
        self.definition = None
        self.origin = None
        self.static_pools = None
        self.camera = None
        self.scene_id = None
        self.camera_mount_id = ""
        self.image_size = None
        self._scene_has_frame = False
        self._dynamic_polyline_pools = None
        self._cube_pool_primitives = None

    def set_definition(self, definition) -> None:
        if len(definition.cameras) != 1:
            raise ValueError("N3BULA stream definition must contain exactly one selected camera")
        camera = definition.cameras[0]
        if camera.spec.WhichOneof("camera_param") != "ftheta_param":
            raise ValueError(f"Ludus N3BULA stream requires an F-theta camera: {camera.mount_id}")
        params = camera.spec.ftheta_param
        if params.reference_poly != params.ANGLE_TO_PIXELDIST:
            raise ValueError(
                f"Ludus requires an angle-to-pixel-distance polynomial for {camera.mount_id}")
        poly = list(params.angle_to_pixeldist_poly)
        if len(poly) != 6:
            raise ValueError(f"F-theta camera {camera.mount_id} must provide six polynomial coefficients")
        width = camera.spec.resolution_w
        height = camera.spec.resolution_h
        self.image_size = (width, height)
        cx = params.principal_point_x
        cy = params.principal_point_y
        self.camera = ludus_renderer.FThetaCamera(
            principal_point=torch.tensor([cx, cy], dtype=torch.float32, device=self.device),
            image_size=torch.tensor(
                [width, height], dtype=torch.float32, device=self.device
            ),
            fw_poly=torch.tensor(poly, dtype=torch.float32, device=self.device),
            max_ray_angle=_camera_max_ray_angle(width, height, cx, cy, poly),
            depth_max=max(camera.far_m, 1.0),
        )
        self.context.upload_cameras([self.camera])
        if self.scene_id is None:
            self.scene_id = self.context.upload_scene(TimestampedScene([], [], []))
        self.camera_mount_id = camera.mount_id
        self.definition = definition
        self.origin = _origin_array(definition)
        self.static_pools = _make_static_pools(definition, self.device)
        self._scene_has_frame = False
        self._dynamic_polyline_pools = None
        self._cube_pool_primitives = None

    def _update_or_replace_scene(self, scene, frame_polyline_pools, cube_pools,
                                 allow_in_place: bool) -> bool:
        cube_primitives = sorted(int(pool.prim_type_id) for pool in cube_pools)
        can_update = (
            allow_in_place
            and self._scene_has_frame
            and _same_polyline_pools(
                self._dynamic_polyline_pools, frame_polyline_pools
            )
            and self._cube_pool_primitives == cube_primitives
        )
        if can_update:
            for pool in cube_pools:
                if not self.context.update_cube_pool(
                    self.scene_id, pool.prim_type_id, pool
                ):
                    can_update = False
                    break
        if can_update:
            return True

        self.context.replace_scene(self.scene_id, scene)
        self._scene_has_frame = True
        self._dynamic_polyline_pools = (
            frame_polyline_pools if allow_in_place else None
        )
        self._cube_pool_primitives = cube_primitives if allow_in_place else None
        return False

    def render(self, frame, timings=None, defer_png=False, include_png=True):
        if self.definition is None or self.camera is None or self.scene_id is None:
            raise ValueError("N3BULA stream frame arrived before its scene definition")
        if frame.camera_mount_id != self.camera_mount_id:
            raise ValueError("scene frame camera mount does not match the selected definition camera")
        scene_prepare_started = time.perf_counter()
        frame_polyline_pools = None
        cube_pools = None
        allow_in_place = self.static_pools is not None and not frame.strips and not frame.surface_bands
        if self.static_pools is None or frame.strips or frame.surface_bands:
            scene = _make_scene(self.definition, frame, self.device)
        else:
            static_polyline_pools, static_polygon_pools = self.static_pools
            frame_polyline_pools, cube_pools = _make_frame_pools(frame, self.origin, self.device)
            frame_polyline_pools = _static_timestamp_pools(frame_polyline_pools)
            scene = TimestampedScene(
                polyline_pools=[*static_polyline_pools, *frame_polyline_pools],
                polygon_pools=static_polygon_pools,
                cube_pools=cube_pools,
            )
        make_scene_done = time.perf_counter()
        update_started = time.perf_counter()
        updated_in_place = self._update_or_replace_scene(
            scene, frame_polyline_pools, cube_pools or [], allow_in_place
        )
        torch.cuda.synchronize(self.device)  # Keep replace_scene's GPU upload inside its timer.
        scene_done = time.perf_counter()
        if timings is not None:
            timings["make_scene_seconds"] = make_scene_done - scene_prepare_started
            scene_update_seconds = scene_done - update_started
            timings["replace_scene_seconds"] = (
                0.0 if updated_in_place else scene_update_seconds
            )
            timings["update_scene_seconds"] = (
                scene_update_seconds if updated_in_place else 0.0
            )
            timings["scene_prepare_seconds"] = scene_done - scene_prepare_started
        camera_setup_started = time.perf_counter()
        ego_rotation = _rotation(frame.ego_rotation_world)
        ego_position = np.asarray(
            [frame.ego_position_world_m.x, frame.ego_position_world_m.y, frame.ego_position_world_m.z],
            dtype=np.float64,
        ) - np.asarray(
            [self.definition.origin_world_m.x, self.definition.origin_world_m.y,
             self.definition.origin_world_m.z], dtype=np.float64
        )
        camera = self.definition.cameras[0]
        extrinsic_rotation = _rotation(camera.rotation_bs)
        camera_to_world = np.eye(4, dtype=np.float32)
        camera_to_world[:3, :3] = (ego_rotation @ extrinsic_rotation).astype(np.float32)
        camera_to_world[:3, 3] = (
            ego_position + ego_rotation @ np.asarray(
                [camera.position_bs_m.x, camera.position_bs_m.y, camera.position_bs_m.z],
                dtype=np.float64,
            )
        ).astype(np.float32)
        pose = torch.from_numpy(np.linalg.inv(camera_to_world)[None]).to(self.device)
        scene_ids = torch.tensor([self.scene_id], dtype=torch.int32, device=self.device)
        camera_ids = torch.zeros(1, dtype=torch.int32, device=self.device)
        timestamps = torch.tensor([frame.timestamp_us], dtype=torch.int64, device=self.device)
        camera_types = torch.tensor([CAMERA_TYPE_REGULAR], dtype=torch.int32, device=self.device)
        height = int(self.camera.image_size[1].item())
        width = int(self.camera.image_size[0].item())
        if timings is not None:
            timings["camera_setup_seconds"] = time.perf_counter() - camera_setup_started
        render_started = time.perf_counter()
        image = self.context.render(
            scene_ids, camera_ids, timestamps, camera_types, pose, (height, width)
        )[0, :, :, :3]
        if self.context.needs_vflip:
            image = image.flip(0)
        torch.cuda.synchronize(self.device)
        if timings is not None:
            timings["render_gpu_seconds"] = time.perf_counter() - render_started
        if not include_png:
            if timings is not None:
                timings["render_readback_seconds"] = 0.0
                timings["control_png_seconds"] = 0.0
            return _RenderedControl(frame, self.camera_mount_id, image, None)
        readback_started = time.perf_counter()
        pixels = image.detach().cpu().numpy()
        if pixels.dtype != np.uint8:
            if np.issubdtype(pixels.dtype, np.floating) and pixels.max(initial=0) <= 1.0:
                pixels = pixels * 255.0
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        if timings is not None:
            timings["render_readback_seconds"] = time.perf_counter() - readback_started
        if defer_png:
            if timings is not None:
                timings["control_png_seconds"] = 0.0
            png = _PNG_ENCODER.submit(_encode_control_png, pixels)
            return _RenderedControl(frame, self.camera_mount_id, image, png)
        png, encode_seconds = _encode_control_png(pixels)
        if timings is not None:
            timings["control_png_seconds"] = encode_seconds
        return _RenderedControl(frame, self.camera_mount_id, image, png)


class _OmniDreamsGenerationStream:
    def __init__(self, engine, generation_config, control_size, run_id=""):
        self.engine = engine
        self.wrapper = engine.conditioning_wrapper
        self.pipeline = self.wrapper.pipeline
        self.device = engine.device
        self.width, self.height = self.wrapper.video_resolution_wh
        self.gpu_control = GPU_CONTROL_INPUT and (
            tuple(control_size) == (int(self.width), int(self.height)))
        if GPU_CONTROL_INPUT and not self.gpu_control:
            logger.warning(
                "N3BULA GPU control input disabled: render {} differs from video resolution {}x{}",
                control_size, self.width, self.height)
        self._normalize_lut = None
        self.prompt = generation_config.prompt or DEFAULT_PROMPT
        self.random_seed = int(generation_config.random_seed)
        self.chunk_index = 0
        self.chunk_size = int(self.pipeline.get_num_output_frames(self.chunk_index))
        if self.chunk_size < 1:
            raise ValueError("OmniDreams pipeline returned a nonpositive chunk size")
        self.cache = None
        self.seed_image = None
        self.pending = []
        self.generated_frames = 0
        self.generation_seconds = 0.0
        self.flashvsr_session = getattr(engine, "flashvsr_session", None)
        self.flashvsr_pending = deque()
        self.flashvsr_seconds = 0.0
        self.run_id = run_id
        if self.flashvsr_session is not None:
            self.flashvsr_session.reset()

        if self.random_seed:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed % (2**32))
            torch.manual_seed(self.random_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_seed)

        if generation_config.HasField("seed_image"):
            seed = generation_config.seed_image
            if seed.format not in (n3bula_pb2.PNG, n3bula_pb2.JPEG):
                raise ValueError("N3BULA seed image must be PNG or JPEG")
            if not seed.data:
                raise ValueError("N3BULA seed image is empty")
            self.seed_image = seed.data
            self._initialize_cache(self._frames_to_tensor([self.seed_image])[:, :, :1])

    def _to_rgb_array(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize(
                (self.width, self.height), Image.Resampling.LANCZOS
            )
            return np.asarray(image, dtype=np.uint8)

    def _frames_to_tensor(self, frames: list[bytes], timings=None) -> torch.Tensor:
        decode_started = time.perf_counter()
        images = np.stack([self._to_rgb_array(frame) for frame in frames])
        if timings is not None:
            timings["control_decode_resize_seconds"] = time.perf_counter() - decode_started
        pack_started = time.perf_counter()
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        tensor = tensor / 127.5 - 1.0
        if timings is not None:
            timings["control_tensor_pack_seconds"] = time.perf_counter() - pack_started
        transfer_started = time.perf_counter()
        tensor = tensor.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if timings is not None:
            timings["control_h2d_seconds"] = time.perf_counter() - transfer_started
        return tensor[None, None]

    def _pixels_to_tensor(self, pixels: list[torch.Tensor], timings=None) -> torch.Tensor:
        if pixels[0].dtype != torch.uint8:
            raise ValueError("GPU control input requires uint8 Ludus pixels")
        pack_started = time.perf_counter()
        if self._normalize_lut is None:
            # Table computed on CPU so the gather matches the PNG path bitwise;
            # on-device float division differs by 1 ulp for ~43% of values.
            self._normalize_lut = (
                torch.arange(256, dtype=torch.float32) / 127.5 - 1.0
            ).to(self.device)
        tensor = self._normalize_lut[torch.stack(pixels).long()].permute(0, 3, 1, 2)
        if timings is not None:
            timings["control_tensor_pack_seconds"] = time.perf_counter() - pack_started
        return tensor[None, None]

    def _control_condition(self, controls, timings=None) -> torch.Tensor:
        if self.gpu_control:
            return self._pixels_to_tensor([control.pixels for control in controls], timings)
        return self._frames_to_tensor([control.png for control in controls], timings)

    def _control_png(self, control, timings) -> bytes | None:
        if control.png is None:
            return None
        if not isinstance(control.png, Future):
            return control.png
        wait_started = time.perf_counter()
        png, encode_seconds = control.png.result()
        timings["control_png_wait_seconds"] += time.perf_counter() - wait_started
        timings["control_png_async_seconds"] += encode_seconds
        return png

    def _initialize_cache(self, seed_tensor: torch.Tensor) -> None:
        self.cache = self.pipeline.initialize_cache(
            text=[[self.prompt]], image=seed_tensor
        )

    def add(self, frame, control, render_timings=None):
        if self.cache is None:
            self._initialize_cache(self._control_condition([control])[:, :, :1])
        self.pending.append((frame, control, render_timings or {}))
        if len(self.pending) < self.chunk_size:
            return None
        if len(self.pending) > self.chunk_size:
            raise RuntimeError("N3BULA generation chunk exceeded its configured frame count")
        return self._generate_pending(self.chunk_size)

    def flush(self):
        responses = []
        if self.pending:
            pending_count = len(self.pending)
            padding = self.chunk_size - pending_count
            controls = [item[1] for item in self.pending]
            if padding:
                controls.extend([controls[-1]] * padding)
            responses.extend(self._generate_pending(pending_count, controls))
        if self.flashvsr_session is not None:
            started = time.perf_counter()
            output_chunks = self.flashvsr_session.flush()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.flashvsr_seconds += time.perf_counter() - started
            tail_responses = self._encode_output_groups(
                self._flashvsr_output_groups(output_chunks),
                {
                    "control_png_wait_seconds": 0.0,
                    "control_png_async_seconds": 0.0,
                    "output_copy_seconds": 0.0,
                    "jpeg_encode_seconds": 0.0,
                    "jpeg_copy_seconds": 0.0,
                },
            )
            self.generated_frames += len(tail_responses)
            responses.extend(tail_responses)
            if self.flashvsr_pending:
                raise RuntimeError(
                    f"FlashVSR left {len(self.flashvsr_pending)} source frames without output"
                )
        return responses or None

    def _flashvsr_output_groups(self, output_chunks):
        groups = []
        for output_chunk in output_chunks:
            output = to_bvtchw(output_chunk.tensor, layout=output_chunk.layout)[0, 0]
            source_frames = [
                source_frame
                for input_chunk in output_chunk.metadata.get("input_chunks", ())
                for source_frame in input_chunk.get("n3bula_source_frames", ())
            ]
            if output.shape[0] != len(source_frames):
                raise RuntimeError(
                    "FlashVSR output count does not match its source-frame metadata: "
                    f"{output.shape[0]} != {len(source_frames)}"
                )
            source_items = []
            for source_frame in source_frames:
                if not self.flashvsr_pending:
                    raise RuntimeError("FlashVSR produced more frames than N3BULA submitted")
                source_item, expected_frame = self.flashvsr_pending.popleft()
                if expected_frame != source_frame:
                    raise RuntimeError(
                        "FlashVSR changed source-frame order or temporal metadata: "
                        f"expected {expected_frame}, got {source_frame}"
                    )
                source_items.append(source_item)
            groups.append((output, source_items))
        return groups

    def _encode_output_groups(self, output_groups, timings):
        responses = []
        for generated, source_items in output_groups:
            output_copy_started = time.perf_counter()
            generated = ((generated.float().clamp(-1, 1) + 1) * 127.5).round().byte()
            gpu_jpeg = GPU_JPEG_ENCODING and generated.device.type == "cuda"
            if gpu_jpeg:
                generated = generated.contiguous()
            else:
                generated = generated.permute(0, 2, 3, 1).cpu().numpy()
            timings["output_copy_seconds"] += time.perf_counter() - output_copy_started

            pngs = [
                self._control_png(item[1], timings)
                for item in source_items
            ] if RETURN_CONTROL_IMAGES else [None] * len(source_items)
            jpeg_data = None
            if gpu_jpeg:
                jpeg_started = time.perf_counter()
                encoded_jpegs = encode_jpeg(list(generated.unbind(0)), quality=90)
                torch.cuda.synchronize(self.device)
                timings["jpeg_encode_seconds"] += time.perf_counter() - jpeg_started
                jpeg_copy_started = time.perf_counter()
                jpeg_data = [encoded.cpu().numpy().tobytes() for encoded in encoded_jpegs]
                timings["jpeg_copy_seconds"] += time.perf_counter() - jpeg_copy_started
            else:
                jpeg_started = time.perf_counter()
                for index in range(len(source_items)):
                    encoded = BytesIO()
                    Image.fromarray(generated[index], mode="RGB").save(
                        encoded, format="JPEG", quality=90
                    )
                    jpeg_data = jpeg_data or []
                    jpeg_data.append(encoded.getvalue())
                timings["jpeg_encode_seconds"] += time.perf_counter() - jpeg_started

            for index, ((_, control, _), png) in enumerate(zip(source_items, pngs)):
                rendered = control.response(png)
                rendered.generated_image.CopyFrom(
                    n3bula_pb2.Image(
                        data=jpeg_data[index], format=n3bula_pb2.JPEG
                    )
                )
                responses.append(rendered)
        return responses

    def _generate_pending(self, output_count: int, controls=None):
        pending = self.pending
        if controls is None:
            controls = [item[1] for item in pending]
        timings = {
            "control_decode_resize_seconds": 0.0,
            "control_tensor_pack_seconds": 0.0,
            "control_h2d_seconds": 0.0,
            "control_png_wait_seconds": 0.0,
            "control_png_async_seconds": 0.0,
            "flashvsr_seconds": 0.0,
            "output_copy_seconds": 0.0,
            "jpeg_encode_seconds": 0.0,
            "jpeg_copy_seconds": 0.0,
        }
        source_frame_ids = [int(item[0].source_frame_id) for item in pending]
        timings["render_gpu_seconds"] = sum(
            item[2].get("render_gpu_seconds", 0.0) for item in pending
        )
        timings["scene_prepare_seconds"] = sum(
            item[2].get("scene_prepare_seconds", 0.0) for item in pending
        )
        timings["make_scene_seconds"] = sum(
            item[2].get("make_scene_seconds", 0.0) for item in pending
        )
        timings["replace_scene_seconds"] = sum(
            item[2].get("replace_scene_seconds", 0.0) for item in pending
        )
        timings["update_scene_seconds"] = sum(
            item[2].get("update_scene_seconds", 0.0) for item in pending
        )
        timings["camera_setup_seconds"] = sum(
            item[2].get("camera_setup_seconds", 0.0) for item in pending
        )
        timings["render_readback_seconds"] = sum(
            item[2].get("render_readback_seconds", 0.0) for item in pending
        )
        timings["control_png_seconds"] = sum(
            item[2].get("control_png_seconds", 0.0) for item in pending
        )
        condition = self._control_condition(controls, timings)
        generate_started = time.perf_counter()
        generated = self.pipeline.generate(
            self.chunk_index, self.cache, input=condition
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings["generate_seconds"] = time.perf_counter() - generate_started
        if generated.shape[2] < output_count:
            raise RuntimeError(
                f"OmniDreams returned {generated.shape[2]} frames for {output_count} scene controls"
            )
        finalize_started = time.perf_counter()
        self.pipeline.finalize(self.chunk_index, self.cache)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings["finalize_seconds"] = time.perf_counter() - finalize_started
        generated = generated[0, 0, :output_count]
        if self.flashvsr_session is not None:
            source_frames = [
                {
                    "run_id": self.run_id,
                    "source_frame_id": int(frame.source_frame_id),
                    "timestamp_us": int(frame.timestamp_us),
                    "simulation_time_s": float(frame.simulation_time_s),
                    "stale": bool(frame.stale),
                    "ego_actor_id": frame.ego_actor_id,
                    "camera_mount_id": control.camera_mount_id,
                }
                for frame, control, _ in pending
            ]
            self.flashvsr_pending.extend(zip(pending, source_frames))
            flashvsr_started = time.perf_counter()
            output_chunks = self.flashvsr_session.process(
                VideoChunk(
                    tensor=generated,
                    layout="tchw",
                    metadata={"n3bula_source_frames": source_frames},
                )
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["flashvsr_seconds"] = time.perf_counter() - flashvsr_started
            self.flashvsr_seconds += timings["flashvsr_seconds"]
            output_groups = self._flashvsr_output_groups(output_chunks)
        else:
            output_groups = [(generated, pending)]
        responses = self._encode_output_groups(output_groups, timings)

        count = len(responses)
        self.generated_frames += count
        self.generation_seconds += timings["generate_seconds"]
        logger.info(
            "N3BULA chunk {} frames {}-{}: {} controls, scene_prepare={:.1f}ms "
            "make_scene={:.1f}ms replace_scene={:.1f}ms "
            "camera_setup={:.1f}ms render_gpu={:.1f}ms "
            "render_readback={:.1f}ms control_png={:.1f}ms control_png_async={:.1f}ms "
            "control_png_wait={:.1f}ms control_decode_resize={:.1f}ms "
            "control_pack={:.1f}ms control_h2d={:.1f}ms generate={:.1f}ms "
            "scene_update={:.1f}ms finalize={:.1f}ms flashvsr={:.1f}ms output_copy={:.1f}ms "
            "jpeg={:.1f}ms jpeg_copy={:.1f}ms "
            "generated_fps={:.2f} rss_mib={:.1f} cuda_allocated_mib={:.1f} "
            "cuda_reserved_mib={:.1f} cuda_peak_allocated_mib={:.1f}",
            self.chunk_index,
            min(source_frame_ids),
            max(source_frame_ids),
            count,
            timings["scene_prepare_seconds"] * 1000,
            timings["make_scene_seconds"] * 1000,
            timings["replace_scene_seconds"] * 1000,
            timings["camera_setup_seconds"] * 1000,
            timings["render_gpu_seconds"] * 1000,
            timings["render_readback_seconds"] * 1000,
            timings["control_png_seconds"] * 1000,
            timings["control_png_async_seconds"] * 1000,
            timings["control_png_wait_seconds"] * 1000,
            timings["control_decode_resize_seconds"] * 1000,
            timings["control_tensor_pack_seconds"] * 1000,
            timings["control_h2d_seconds"] * 1000,
            timings["generate_seconds"] * 1000,
            timings.get("update_scene_seconds", 0.0) * 1000,
            timings["finalize_seconds"] * 1000,
            timings["flashvsr_seconds"] * 1000,
            timings["output_copy_seconds"] * 1000,
            timings["jpeg_encode_seconds"] * 1000,
            timings["jpeg_copy_seconds"] * 1000,
            count / timings["generate_seconds"] if timings["generate_seconds"] else 0.0,
            _process_rss_mib(),
            *_gpu_memory_mib(),
        )
        self.chunk_index += 1
        self.chunk_size = int(self.pipeline.get_num_output_frames(self.chunk_index))
        self.pending = []
        return responses


class N3bulaLudusService(n3bula_pb2_grpc.N3bulaLudusServiceServicer):
    """Render streamed Cosmos scenes with Ludus and return generated video frames."""

    def __init__(self, engine):
        self.engine = engine
        self._stream_lock = threading.Lock()

    def stream_scenes(self, request_iterator, context):
        with self._stream_lock:
            yield from self._stream_scenes(request_iterator, context)

    def _stream_scenes(self, request_iterator, context):
        renderer = _StreamRenderer()
        generation = None
        stream_started = time.perf_counter()
        control_seconds = 0.0
        render_gpu_seconds = 0.0
        render_readback_seconds = 0.0
        control_png_seconds = 0.0
        received_frames = 0
        generated_frames = 0
        generation_seconds = 0.0
        request_wait_seconds = 0.0
        response_yield_seconds = 0.0
        yielded_frames = 0
        flashvsr_seconds_total = 0.0
        request_iterator = iter(request_iterator)

        def timed_requests():
            nonlocal request_wait_seconds
            while True:
                wait_started = time.perf_counter()
                try:
                    request = next(request_iterator)
                except StopIteration:
                    request_wait_seconds += time.perf_counter() - wait_started
                    return
                request_wait_seconds += time.perf_counter() - wait_started
                yield request

        def yield_responses(responses):
            nonlocal response_yield_seconds, yielded_frames
            for response in responses or ():
                yield_started = time.perf_counter()
                yield response
                response_yield_seconds += time.perf_counter() - yield_started
                yielded_frames += 1
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "N3BULA direct Ludus generation currently requires a single-process pipeline",
            )
        try:
            logger.info("N3BULA Ludus stream connected; awaiting scene definition")
            for request in timed_requests():
                if request.schema_version != 1:
                    raise ValueError(f"unsupported N3BULA schema version: {request.schema_version}")
                event = request.WhichOneof("event")
                if event == "definition":
                    if generation is not None:
                        yield from yield_responses(generation.flush())
                        generated_frames += generation.generated_frames
                        generation_seconds += generation.generation_seconds
                        flashvsr_seconds_total += generation.flashvsr_seconds
                    renderer.set_definition(request.definition)
                    config = request.definition.generation_config
                    generation = _OmniDreamsGenerationStream(
                        self.engine,
                        config,
                        renderer.image_size,
                        run_id=request.definition.run_id,
                    )
                    logger.info(
                        "N3BULA stream configured: camera={}, resolution={}x{}, random_seed={}, prompt={!r}, gpu_control={}, return_control_images={}",
                        renderer.camera_mount_id,
                        generation.width,
                        generation.height,
                        generation.random_seed,
                        generation.prompt,
                        generation.gpu_control,
                        RETURN_CONTROL_IMAGES,
                    )
                elif event == "frame":
                    if generation is None:
                        raise ValueError("N3BULA scene definition must precede frames")
                    started = time.perf_counter()
                    render_timings = {}
                    control = renderer.render(
                        request.frame,
                        render_timings,
                        defer_png=RETURN_CONTROL_IMAGES and generation.gpu_control,
                        include_png=RETURN_CONTROL_IMAGES or not generation.gpu_control,
                    )
                    control_seconds += time.perf_counter() - started
                    render_gpu_seconds += render_timings["render_gpu_seconds"]
                    render_readback_seconds += render_timings["render_readback_seconds"]
                    control_png_seconds += render_timings["control_png_seconds"]
                    received_frames += 1
                    completed = generation.add(request.frame, control, render_timings)
                    if completed is not None:
                        yield from yield_responses(completed)
                else:
                    raise ValueError("N3BULA stream request has no event")

            if generation is not None:
                yield from yield_responses(generation.flush())
                generated_frames += generation.generated_frames
                generation_seconds += generation.generation_seconds
                flashvsr_seconds_total += generation.flashvsr_seconds
            elapsed = time.perf_counter() - stream_started
            logger.info(
                "N3BULA stream closed: controls={}, generated={}, render_total={:.3f}s "
                "render_gpu={:.3f}s render_readback={:.3f}s control_png={:.3f}s "
                "request_wait={:.3f}s response_yield_blocked={:.3f}s responses_yielded={}, "
                "generate={:.3f}s ({:.2f} generate-only fps), flashvsr={:.3f}s, "
                "end-to-end={:.2f} returned fps",
                received_frames,
                generated_frames,
                control_seconds,
                render_gpu_seconds,
                render_readback_seconds,
                control_png_seconds,
                request_wait_seconds,
                response_yield_seconds,
                yielded_frames,
                generation_seconds,
                0.0 if not generation_seconds
                else generated_frames / generation_seconds,
                flashvsr_seconds_total,
                0.0 if not elapsed else generated_frames / elapsed,
            )
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            logger.exception("N3BULA Ludus stream failed")
            context.abort(grpc.StatusCode.INTERNAL, str(error))

    def render_controls(self, request, context):
        if request.schema_version != 1 or not request.HasField("definition"):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "render_controls requires schema version 1 and a scene definition",
            )
        renderer = _StreamRenderer()
        try:
            renderer.set_definition(request.definition)
            frames = []
            for frame in request.frames:
                control = renderer.render(frame)
                frames.append(control.response(control.png))
            return n3bula_pb2.RenderedControls(frames=frames)
        except Exception as error:
            logger.exception("N3BULA Ludus render failed")
            context.abort(grpc.StatusCode.INTERNAL, str(error))