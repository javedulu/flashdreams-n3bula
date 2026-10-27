# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy geometry for N3BULA scenes and rigs: no torch, Ludus or protobuf imports."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation

# Mirrors the OmniDreams N3BULA service; Ludus has no dash geometry of its own.
DASH_LENGTH_M = 3.0
DASH_PERIOD_M = 12.0
DOT_LENGTH_M = 0.5
DOT_PERIOD_M = 1.5
DUAL_LINE_SEPARATION_M = 0.15
LUDUS_MAX_POLYLINE_POINTS = 256


def rotation_matrix(euler) -> np.ndarray:
    """Rz(yaw) * Ry(pitch) * Rx(roll) from an ``Euler3d`` message."""
    return Rotation.from_euler(
        "ZYX", [euler.yaw_rad, euler.pitch_rad, euler.roll_rad]
    ).as_matrix()


def vec3(point) -> np.ndarray:
    """A ``Vec3d`` message as a float64 array."""
    return np.asarray([point.x, point.y, point.z], dtype=np.float64)


def points_array(points) -> np.ndarray:
    """A repeated ``Vec3d`` field as ``[K, 3]`` float64."""
    return np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float64).reshape(
        -1, 3
    )


def camera_to_world(frame, camera, origin: np.ndarray) -> np.ndarray:
    """``[4, 4]`` camera-to-world for one mount at one frame, origin-relative."""
    ego_rotation = rotation_matrix(frame.ego_rotation_world)
    pose = np.eye(4)
    pose[:3, :3] = ego_rotation @ rotation_matrix(camera.rotation_bs)
    pose[:3, 3] = vec3(frame.ego_position_world_m) - origin + ego_rotation @ vec3(
        camera.position_bs_m
    )
    return pose


def camera_max_ray_angle(
    width: int, height: int, cx: float, cy: float, coefficients: list[float]
) -> float:
    """Largest ray angle (radians, padded 5%) the F-theta polynomial maps into the image."""
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


def offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    """Shift a polyline sideways (left positive) in the XY plane."""
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


def dash_polyline(points: np.ndarray) -> list[np.ndarray]:
    """Paint DASH_LENGTH_M of every DASH_PERIOD_M, phase anchored at the chain start."""
    dashes, current, travelled = [], [], 0.0
    for a, b in pairwise(points):
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


def dot_polyline(points: np.ndarray) -> np.ndarray:
    """One vertex per DOT_PERIOD_M of arc length; Ludus draws each vertex as a dot."""
    dots, travelled, next_dot = [], 0.0, DOT_LENGTH_M / 2
    for a, b in pairwise(points):
        length = float(np.linalg.norm(b - a))
        while length > 0.0 and next_dot <= travelled + length:
            dots.append(a + (b - a) * ((next_dot - travelled) / length))
            next_dot += DOT_PERIOD_M
        travelled += length
    return np.asarray(dots).reshape(-1, 3)


def split_polyline(points: np.ndarray):
    """Yield chunks of at most LUDUS_MAX_POLYLINE_POINTS sharing their end vertex."""
    if len(points) <= LUDUS_MAX_POLYLINE_POINTS:
        yield points
        return
    for start in range(0, len(points) - 1, LUDUS_MAX_POLYLINE_POINTS - 1):
        yield points[start : start + LUDUS_MAX_POLYLINE_POINTS]
