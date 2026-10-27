# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""N3BULA scene messages -> Ludus pools using the Cosmos3 WSM 4.0.1 palette."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from ludus_renderer import (
    TimestampedPolygonPool,
)
from ludus_renderer._ops.primitives import (
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
    PRIM_ROAD_MARKING,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    PRIM_TYPE_COUNT,
    PRIM_WAIT_LINE,
    CubePool,
    TimestampedPolylinePool,
    TimestampedScene,
)
from scipy.spatial.transform import Rotation

from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.scene.geometry import (
    DUAL_LINE_SEPARATION_M,
    dash_polyline,
    dot_polyline,
    offset_polyline,
    points_array,
    rotation_matrix,
    split_polyline,
    vec3,
)

STATIC_TIMESTAMP_US = 0  # Static road geometry is valid for every frame.

WSM4_LANE_UNKNOWN = PRIM_TYPE_COUNT
WSM4_BOUNDARY_PRIMITIVES = {
    pb.WSM4_BOUNDARY_BARRIER: PRIM_TYPE_COUNT + 1,
    pb.WSM4_BOUNDARY_FENCE: PRIM_TYPE_COUNT + 2,
    pb.WSM4_BOUNDARY_CURB: PRIM_TYPE_COUNT + 3,
    pb.WSM4_BOUNDARY_LOW_CURB: PRIM_TYPE_COUNT + 4,
    pb.WSM4_BOUNDARY_TALL_CURB: PRIM_TYPE_COUNT + 5,
    pb.WSM4_BOUNDARY_ROAD_BOUNDARY: PRIM_TYPE_COUNT + 6,
    pb.WSM4_BOUNDARY_OBSTACLE_BARS: PRIM_TYPE_COUNT + 7,
    pb.WSM4_BOUNDARY_WALL: PRIM_TYPE_COUNT + 8,
    pb.WSM4_BOUNDARY_OTHER: PRIM_TYPE_COUNT + 9,
}
WSM4_MARKING_PRIMITIVES = {
    pb.WSM4_ROAD_MARKING_STOP: PRIM_TYPE_COUNT + 10,
    pb.WSM4_ROAD_MARKING_KEEP_CLEAR: PRIM_TYPE_COUNT + 11,
    pb.WSM4_ROAD_MARKING_TEXT_PAINT: PRIM_TYPE_COUNT + 12,
    pb.WSM4_ROAD_MARKING_OTHER: PRIM_TYPE_COUNT + 13,
}
WSM4_BOUNDARY_COLORS = {
    pb.WSM4_BOUNDARY_BARRIER: (253, 1, 232),
    pb.WSM4_BOUNDARY_FENCE: (253, 1, 232),
    pb.WSM4_BOUNDARY_CURB: (207, 255, 191),
    pb.WSM4_BOUNDARY_LOW_CURB: (207, 255, 191),
    pb.WSM4_BOUNDARY_TALL_CURB: (207, 255, 191),
    pb.WSM4_BOUNDARY_ROAD_BOUNDARY: (0, 159, 96),
    pb.WSM4_BOUNDARY_OBSTACLE_BARS: (255, 207, 159),
    pb.WSM4_BOUNDARY_WALL: (0, 128, 159),
    pb.WSM4_BOUNDARY_OTHER: (239, 191, 255),
}
WSM4_MARKING_COLORS = {
    pb.WSM4_ROAD_MARKING_STOP: (255, 0, 0),
    pb.WSM4_ROAD_MARKING_KEEP_CLEAR: (0, 255, 32),
    pb.WSM4_ROAD_MARKING_TEXT_PAINT: (20, 254, 185),
    pb.WSM4_ROAD_MARKING_OTHER: (20, 254, 185),
}
WSM4_PALETTE = {
    PRIM_CROSSWALK: (139, 93, 255, 255),
    PRIM_WAIT_LINE: (108, 179, 59, 255),
    PRIM_POLE: (183, 69, 177, 255),
    PRIM_ROAD_MARKING: (20, 254, 185, 255),
    PRIM_LANE_LINE_WHITE_SOLID: (255, 255, 255, 255),
    PRIM_LANE_LINE_WHITE_DASHED: (255, 255, 255, 255),
    PRIM_DOT_WHITE: (255, 255, 255, 255),
    PRIM_LANE_LINE_YELLOW_SOLID: (255, 255, 0, 255),
    PRIM_LANE_LINE_YELLOW_DASHED: (255, 255, 0, 255),
    PRIM_DOT_YELLOW: (255, 255, 0, 255),
    WSM4_LANE_UNKNOWN: (181, 164, 71, 255),
    **{
        primitive: (*WSM4_BOUNDARY_COLORS[kind], 255)
        for kind, primitive in WSM4_BOUNDARY_PRIMITIVES.items()
    },
    **{
        primitive: (*WSM4_MARKING_COLORS[kind], 255)
        for kind, primitive in WSM4_MARKING_PRIMITIVES.items()
    },
}
ACTOR_COLORS = {
    pb.ACTOR_CATEGORY_CAR: ((0, 46, 136), (126, 206, 255)),
    pb.ACTOR_CATEGORY_TRUCK: ((204, 55, 0), (255, 192, 64)),
    pb.ACTOR_CATEGORY_PEDESTRIAN: ((148, 0, 62), (255, 124, 171)),
    pb.ACTOR_CATEGORY_CYCLIST: ((0, 80, 66), (102, 208, 198)),
    pb.ACTOR_CATEGORY_OTHER: ((53, 26, 20), (166, 136, 125)),
}
WSM4_SIGN_COLORS = {
    pb.WSM4_SIGN_CLASS_STOP: (220, 20, 20),
    pb.WSM4_SIGN_CLASS_YIELD: (255, 140, 0),
    pb.WSM4_SIGN_CLASS_DO_NOT_ENTER: (200, 0, 100),
    pb.WSM4_SIGN_CLASS_ONE_WAY: (40, 110, 220),
    pb.WSM4_SIGN_CLASS_ONE_WAY_LEFT: (0, 180, 230),
    pb.WSM4_SIGN_CLASS_ONE_WAY_RIGHT: (30, 60, 200),
    pb.WSM4_SIGN_CLASS_PEDESTRIAN_CROSSING: (255, 220, 0),
    pb.WSM4_SIGN_CLASS_SPEED_LIMIT: (8, 2, 255),
}
WSM4_SIGNAL_COLORS = {
    pb.WSM4_SIGNAL_STATE_UNKNOWN: (150, 150, 150),
    pb.WSM4_SIGNAL_STATE_STATELESS: (20, 20, 20),
    pb.WSM4_SIGNAL_STATE_RED: (240, 50, 50),
    pb.WSM4_SIGNAL_STATE_YELLOW: (255, 200, 0),
    pb.WSM4_SIGNAL_STATE_GREEN: (130, 240, 132),
    pb.WSM4_SIGNAL_STATE_RED_YELLOW: (208, 136, 72),
    pb.WSM4_SIGNAL_STATE_WHITE_VRU: (255, 255, 255),
}
BOX_PRIMITIVES = {
    pb.SEMANTIC_ID_TRAFFIC_SIGN: PRIM_TRAFFIC_SIGN,
    pb.SEMANTIC_ID_SPEED_SIGN: PRIM_TRAFFIC_SIGN,
    pb.SEMANTIC_ID_TRAFFIC_LIGHT: PRIM_TRAFFIC_LIGHT,
}
MARK_COMPONENTS = {
    pb.MARK_TYPE_ID_SOLID: ("solid",),
    pb.MARK_TYPE_ID_BROKEN: ("broken",),
    pb.MARK_TYPE_ID_SOLID_SOLID: ("solid", "solid"),
    pb.MARK_TYPE_ID_SOLID_BROKEN: ("solid", "broken"),
    pb.MARK_TYPE_ID_BROKEN_SOLID: ("broken", "solid"),
    pb.MARK_TYPE_ID_BROKEN_BROKEN: ("broken", "broken"),
    pb.MARK_TYPE_ID_BOTTS_DOTS: ("dots",),
}
_LINE_PRIMITIVES = {
    pb.SEMANTIC_ID_STOP_LINE: PRIM_WAIT_LINE,
    pb.SEMANTIC_ID_POLE: PRIM_POLE,
}


def lane_primitive(colour: int, component: str) -> int:
    if colour == pb.MARK_COLOR_ID_YELLOW:
        ids = {
            "solid": PRIM_LANE_LINE_YELLOW_SOLID,
            "broken": PRIM_LANE_LINE_YELLOW_DASHED,
            "dots": PRIM_DOT_YELLOW,
        }
    elif colour == pb.MARK_COLOR_ID_WHITE:
        ids = {
            "solid": PRIM_LANE_LINE_WHITE_SOLID,
            "broken": PRIM_LANE_LINE_WHITE_DASHED,
            "dots": PRIM_DOT_WHITE,
        }
    else:
        if component not in ("solid", "broken", "dots"):
            raise ValueError(f"unsupported WSM4 lane component {component!r}")
        return WSM4_LANE_UNKNOWN
    if component not in ids:
        raise ValueError(f"unsupported WSM4 lane component {component!r}")
    return ids[component]


def line_primitive(strip) -> int:
    if strip.semantic == pb.SEMANTIC_ID_LANE_MARKING:
        components = MARK_COMPONENTS.get(strip.pattern)
        if components is None:
            raise ValueError(f"lane marking has no WSM4 style: {strip.pattern}")
        return lane_primitive(strip.colour, components[0])
    if strip.semantic == pb.SEMANTIC_ID_ROAD_BOUNDARY:
        boundary = strip.wsm4_boundary_type
        if boundary == pb.WSM4_BOUNDARY_UNSPECIFIED:
            boundary = pb.WSM4_BOUNDARY_ROAD_BOUNDARY
        try:
            return WSM4_BOUNDARY_PRIMITIVES[boundary]
        except KeyError as error:
            raise ValueError(f"unknown WSM4 road-boundary type {boundary}") from error
    if strip.semantic in (pb.SEMANTIC_ID_STOP_LINE, pb.SEMANTIC_ID_POLE):
        return _LINE_PRIMITIVES[strip.semantic]
    if strip.semantic == pb.SEMANTIC_ID_CROSSWALK:
        raise ValueError("WSM4 crosswalks require a filled surface band, not a strip")
    raise ValueError(f"no WSM4 line mapping for semantic {strip.semantic}")


def strip_lines(
    strip, points: np.ndarray, already_expanded: bool = False
) -> list[tuple[int, list[np.ndarray]]]:
    """(Ludus primitive, polylines) with the mark pattern expanded into geometry."""
    if strip.semantic != pb.SEMANTIC_ID_LANE_MARKING:
        return [(line_primitive(strip), [points])]
    if len(points) == 1 and (
        already_expanded or strip.pattern == pb.MARK_TYPE_ID_BOTTS_DOTS
    ):
        return [(lane_primitive(strip.colour, "dots"), [points])]
    if len(points) < 2:
        return []
    components = MARK_COMPONENTS.get(strip.pattern)
    if components is None:
        raise ValueError(f"lane marking has no WSM4 style: {strip.pattern}")
    offsets = (
        (DUAL_LINE_SEPARATION_M / 2, -DUAL_LINE_SEPARATION_M / 2)
        if len(components) == 2
        else (0.0,)
    )
    layers = []
    for component, offset in zip(components, offsets):
        track = offset_polyline(points, offset) if offset else points
        if component == "broken":
            chunks = dash_polyline(track)
        elif component == "dots":
            chunks = [dot_polyline(track)]
        else:
            chunks = [track]
        layers.append((lane_primitive(strip.colour, component), chunks))
    return layers


def box_primitive(box) -> int:
    if box.semantic == pb.SEMANTIC_ID_ACTOR:
        return PRIM_OBSTACLE
    if box.semantic == pb.SEMANTIC_ID_EGO:
        raise ValueError("WSM4 has no ego actor category")
    if box.semantic == pb.SEMANTIC_ID_CROSSWALK:
        raise ValueError("WSM4 crosswalks require polygon geometry, not a proxy box")
    try:
        return BOX_PRIMITIVES[box.semantic]
    except KeyError as error:
        raise ValueError(f"no WSM4 box mapping for semantic {box.semantic}") from error


def box_colors(box) -> list[float]:
    if box.semantic == pb.SEMANTIC_ID_ACTOR:
        if box.category == pb.ACTOR_CATEGORY_EGO:
            raise ValueError("WSM4 has no ego actor category")
        try:
            front, back = ACTOR_COLORS[box.category]
        except KeyError as error:
            raise ValueError(f"unknown WSM4 actor category {box.category}") from error
    elif box.semantic in (pb.SEMANTIC_ID_TRAFFIC_SIGN, pb.SEMANTIC_ID_SPEED_SIGN):
        front, back = _sign_colors(
            box.wsm4_sign_class,
            speed_sign=box.semantic == pb.SEMANTIC_ID_SPEED_SIGN,
        )
    elif box.semantic == pb.SEMANTIC_ID_TRAFFIC_LIGHT:
        _validate_signal_glyph(box)
        state = box.wsm4_signal_state
        if state == pb.WSM4_SIGNAL_STATE_UNSPECIFIED:
            state = pb.WSM4_SIGNAL_STATE_UNKNOWN
        try:
            front = WSM4_SIGNAL_COLORS[state]
        except KeyError as error:
            raise ValueError(f"unknown WSM4 traffic-signal state {state}") from error
        back = (60, 60, 60)
    else:
        raise ValueError(f"no WSM4 box color mapping for semantic {box.semantic}")
    return [channel / 255.0 for channel in (*front, *back)]


def _validate_signal_glyph(signal) -> None:
    if signal.wsm4_signal_direction != pb.WSM4_SIGNAL_DIRECTION_NONE:
        raise NotImplementedError("WSM4 directional traffic-signal glyphs are not rendered yet")
    if signal.wsm4_signal_user != pb.WSM4_SIGNAL_USER_VEHICLE:
        raise NotImplementedError("WSM4 pedestrian/cyclist signal glyphs are not rendered yet")


def _polyline_pools(timestamp_us: int, groups: dict[int, list[np.ndarray]], device) -> list:
    pools = []
    for primitive_id, lines in groups.items():
        minimum = 1 if primitive_id in (
            PRIM_DOT_WHITE,
            PRIM_DOT_YELLOW,
            WSM4_LANE_UNKNOWN,
        ) else 2
        lines = [
            chunk
            for line in lines
            for chunk in split_polyline(line)
            if len(chunk) >= minimum
        ]
        if not lines:
            continue
        lengths = np.asarray([len(line) for line in lines], dtype=np.int32)
        pools.append(
            TimestampedPolylinePool(
                timestamps_us=torch.tensor([timestamp_us], dtype=torch.int64, device=device),
                timestamped_varrays_prefix_sum=torch.tensor(
                    [len(lines)], dtype=torch.int32, device=device
                ),
                varrays_prefix_sum=torch.tensor(
                    np.cumsum(lengths), dtype=torch.int32, device=device
                ),
                vertices=torch.as_tensor(
                    np.concatenate(lines).astype(np.float32, copy=False), device=device
                ),
                prim_type_id=primitive_id,
            )
        )
    return pools


def unsupported_surface_bands(bands) -> dict[int, int]:
    """Count SUMO-only filled surfaces that have no WSM4 control primitive."""
    counts: dict[int, int] = defaultdict(int)
    for band in bands:
        if (
            band.semantic != pb.SEMANTIC_ID_CROSSWALK
            and band.wsm4_marking_type == pb.WSM4_ROAD_MARKING_UNSPECIFIED
        ):
            counts[int(band.semantic)] += 1
    return dict(counts)


def _sign_colors(sign_class: int, *, speed_sign: bool = False) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if sign_class == pb.WSM4_SIGN_CLASS_UNSPECIFIED and speed_sign:
        sign_class = pb.WSM4_SIGN_CLASS_SPEED_LIMIT
    if sign_class == pb.WSM4_SIGN_CLASS_UNSPECIFIED:
        return (8, 2, 255), (8, 2, 255)
    try:
        return WSM4_SIGN_COLORS[sign_class], (40, 40, 40)
    except KeyError as error:
        raise ValueError(f"unknown WSM4 traffic-sign class {sign_class}") from error


def _surface_polygon_pools(timestamp_us: int, bands, origin, device) -> list:
    grouped = defaultdict(list)
    for band in bands:
        primitive = surface_band_primitive(band)
        left, right = band.left_world_m, band.right_world_m
        if primitive is None:
            continue
        if len(left) < 2 or len(left) != len(right):
            raise ValueError(
                "WSM4 surface band requires matching left/right edge points, at least two each"
            )
        # Interleaved left/right vertices, two triangles per segment.
        vertices = np.stack(
            (points_array(left), points_array(right)), axis=1
        ).reshape(-1, 3) - origin
        triangles = [
            tri
            for first in range(0, 2 * (len(left) - 1), 2)
            for tri in ((first, first + 1, first + 3), (first, first + 3, first + 2))
        ]
        grouped[primitive].append(
            (
                torch.tensor(vertices, dtype=torch.float32, device=device),
                torch.tensor(triangles, dtype=torch.int32, device=device),
            )
        )
    pools = []
    for primitive, polygons in grouped.items():
        pools.append(
            TimestampedPolygonPool(
                timestamps_us=torch.tensor([timestamp_us], dtype=torch.int64, device=device),
                timestamped_varrays_prefix_sum=torch.tensor(
                    [len(polygons)], dtype=torch.int32, device=device
                ),
                varrays_prefix_sum=torch.tensor(
                    np.cumsum([len(v) for v, _ in polygons]), dtype=torch.int32, device=device
                ),
                triangle_prefix_sum=torch.tensor(
                    np.cumsum([len(t) for _, t in polygons]), dtype=torch.int32, device=device
                ),
                vertices=torch.cat([v for v, _ in polygons]),
                triangles=torch.cat([t for _, t in polygons]),
                prim_type_id=primitive,
            )
        )
    return pools


def surface_band_primitive(band) -> int | None:
    marking = band.wsm4_marking_type
    if marking in (
        pb.WSM4_ROAD_MARKING_FORWARD,
        pb.WSM4_ROAD_MARKING_LEFT,
        pb.WSM4_ROAD_MARKING_RIGHT,
        pb.WSM4_ROAD_MARKING_LEFT_FORWARD,
        pb.WSM4_ROAD_MARKING_RIGHT_FORWARD,
        pb.WSM4_ROAD_MARKING_MERGE_LEFT,
        pb.WSM4_ROAD_MARKING_MERGE_RIGHT,
    ):
        raise NotImplementedError(
            "WSM4 arrow glyph projection is not implemented for Ludus controls"
        )
    if marking in WSM4_MARKING_PRIMITIVES:
        return WSM4_MARKING_PRIMITIVES[marking]
    if marking != pb.WSM4_ROAD_MARKING_UNSPECIFIED:
        raise ValueError(f"unknown WSM4 road-marking type {marking}")
    if band.semantic == pb.SEMANTIC_ID_CROSSWALK:
        return PRIM_CROSSWALK
    return None


def origin_array(definition) -> np.ndarray:
    return vec3(definition.origin_world_m)


def _polyline_groups(definition, frame_strips, frame_bands, origin):
    lines: dict[int, list[np.ndarray]] = defaultdict(list)
    for strips, expanded in ((definition.static_strips, False), (frame_strips, True)):
        for strip in strips:
            points = points_array(strip.centre_world_m)
            if len(points) == 0:
                continue
            for primitive, chunks in strip_lines(strip, points, expanded):
                lines[primitive].extend(
                    (chunk - origin).astype(np.float32) for chunk in chunks if len(chunk)
                )
    bands = [*definition.static_surface_bands, *frame_bands]
    return lines, bands


def _cube_pools(frame, origin, timestamp: int, device, lines) -> list:
    """Boxes and mesh stand-ins as cube pools; mesh poles are appended to ``lines``."""
    groups = defaultdict(list)
    for box in frame.boxes:
        flags = CUBE_FLAG_WIREFRAME if box.wireframe_overlay else 0
        groups[(box_primitive(box), flags)].append(
            (
                (vec3(box.centre_world_m) - origin).astype(np.float32),
                Rotation.from_matrix(rotation_matrix(box.rotation_world))
                .as_quat()
                .astype(np.float32),
                np.asarray([box.length_m, box.width_m, box.height_m], dtype=np.float32),
                np.asarray(box_colors(box), dtype=np.float32),
            )
        )
    for mesh in frame.meshes:
        is_light = mesh.semantic == pb.SEMANTIC_ID_TRAFFIC_LIGHT
        is_speed = mesh.semantic == pb.SEMANTIC_ID_SPEED_SIGN
        if mesh.semantic not in (
            pb.SEMANTIC_ID_TRAFFIC_LIGHT,
            pb.SEMANTIC_ID_TRAFFIC_SIGN,
            pb.SEMANTIC_ID_SPEED_SIGN,
        ):
            raise ValueError(f"no WSM4 mesh mapping for semantic {mesh.semantic}")
        dimensions = np.asarray(
            (0.3, 0.3, 0.5) if is_light else (0.16, 1.04, 1.04) if is_speed else (0.01, 0.5, 0.8),
            dtype=np.float32,
        )
        head = vec3(mesh.origin_world_m)
        if head[2] < 0.5:
            head[2] += 3.0 if is_light else 2.57 if is_speed else 2.1
        if is_light:
            _validate_signal_glyph(mesh)
            state = mesh.wsm4_signal_state
            if state == pb.WSM4_SIGNAL_STATE_UNSPECIFIED:
                state = pb.WSM4_SIGNAL_STATE_UNKNOWN
            try:
                front = WSM4_SIGNAL_COLORS[state]
            except KeyError as error:
                raise ValueError(f"unknown WSM4 traffic-signal state {state}") from error
            back = (60, 60, 60)
            primitive = PRIM_TRAFFIC_LIGHT
        else:
            front, back = _sign_colors(mesh.wsm4_sign_class, speed_sign=is_speed)
            primitive = PRIM_TRAFFIC_SIGN
        groups[(primitive, 0)].append(
            (
                (head - origin).astype(np.float32),
                Rotation.from_euler("z", mesh.yaw_rad).as_quat().astype(np.float32),
                dimensions,
                np.asarray([c / 255.0 for c in (*front, *back)], dtype=np.float32),
            )
        )
        bottom = head[2] - float(dimensions[2]) * 0.5
        if bottom > 0.05:
            pole = np.asarray([[head[0], head[1], 0.0], [head[0], head[1], bottom]]) - origin
            lines[PRIM_POLE].append(pole.astype(np.float32))

    pools = []
    for (primitive, flags), cubes in groups.items():
        count = len(cubes)
        pools.append(
            CubePool(
                timestamps_us=torch.tensor([timestamp], dtype=torch.int64, device=device),
                cube_ts_prefix_sum=torch.arange(1, count + 1, dtype=torch.int32, device=device),
                track_timestamps_us=torch.full((count,), timestamp, dtype=torch.int64, device=device),
                translations=torch.as_tensor(np.stack([c[0] for c in cubes]), device=device),
                quaternions=torch.as_tensor(np.stack([c[1] for c in cubes]), device=device),
                scales=torch.as_tensor(np.stack([c[2] for c in cubes]), device=device),
                colors=torch.as_tensor(np.stack([c[3] for c in cubes]), device=device),
                prim_type_id=primitive,
                render_flags=flags,
            )
        )
    return pools


def make_scene(definition, frame, device) -> TimestampedScene:
    """The whole scene for one frame, rebuilt from scratch."""
    origin = origin_array(definition)
    timestamp = int(frame.timestamp_us)
    lines, bands = _polyline_groups(definition, frame.strips, frame.surface_bands, origin)
    cubes = _cube_pools(frame, origin, timestamp, device, lines)
    return TimestampedScene(
        polyline_pools=_polyline_pools(timestamp, lines, device),
        polygon_pools=_surface_polygon_pools(timestamp, bands, origin, device),
        cube_pools=cubes,
    )


def make_static_pools(definition, device):
    """Road geometry built once per definition; None if it would share PRIM_POLE with mesh poles."""
    origin = origin_array(definition)
    lines, bands = _polyline_groups(definition, (), (), origin)
    if PRIM_POLE in lines:
        return None
    return (
        _polyline_pools(STATIC_TIMESTAMP_US, lines, device),
        _surface_polygon_pools(STATIC_TIMESTAMP_US, bands, origin, device),
    )


def make_frame_pools(frame, origin, device):
    """Per-frame cubes plus mesh poles, the poles restamped onto the static clock."""
    lines: dict[int, list[np.ndarray]] = defaultdict(list)
    cubes = _cube_pools(frame, origin, int(frame.timestamp_us), device, lines)
    return _polyline_pools(STATIC_TIMESTAMP_US, lines, device), cubes
