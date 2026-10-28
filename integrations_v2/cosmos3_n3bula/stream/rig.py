# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Bind N3BULA camera mounts to the model's canonical camera seats."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, cos, degrees, pi, sin

from cosmos3_nano_sim_auto.impl.conditioning.camera_rig import CAMERAS, view_ids

# Trained seats in body-frame yaw and horizontal FOV degrees.
_CAMERA_SEAT_GEOMETRY = {
    "camera_front_wide_120fov": (0.0, 120.0),
    "camera_cross_right_120fov": (-pi / 2, 120.0),
    "camera_rear_right_70fov": (-3 * pi / 4, 70.0),
    "camera_rear_tele_30fov": (pi, 30.0),
    "camera_rear_left_70fov": (3 * pi / 4, 70.0),
    "camera_cross_left_120fov": (pi / 2, 120.0),
    "camera_front_tele_30fov": (0.0, 30.0),
    "camera_front_fisheye_200fov": (0.0, 200.0),
    "camera_left_fisheye_200fov": (pi / 2, 200.0),
    "camera_right_fisheye_200fov": (-pi / 2, 200.0),
    "camera_rear_fisheye_200fov": (pi, 200.0),
}


@dataclass(frozen=True)
class RigCamera:
    canonical: str
    """The model seat, one of :data:`CAMERAS`; decides the learned view embedding."""
    message: object
    """The definition's ``Camera``; its ``mount_id`` is what responses are labelled with."""


def canonical_name(camera) -> str:
    """Resolve an explicit model seat or infer the closest seat from pose and FOV."""
    for name in (camera.mount_id, camera.spec.logical_id):
        if name in CAMERAS:
            return name

    if camera.spec.WhichOneof("camera_param") == "ftheta_param":
        half_fov = camera.spec.ftheta_param.max_angle
        if 0.0 < half_fov < pi:
            yaw = camera.rotation_bs.yaw_rad
            fov = 2.0 * degrees(half_fov)
            matches = []
            for name in CAMERAS:
                seat_yaw, seat_fov = _CAMERA_SEAT_GEOMETRY[name]
                yaw_error = abs(degrees(atan2(sin(yaw - seat_yaw), cos(yaw - seat_yaw))))
                fov_error = abs(fov - seat_fov)
                if yaw_error <= 45.0 and fov_error <= 45.0:
                    matches.append((yaw_error + fov_error * 0.5, name))
            if matches:
                return min(matches)[1]

    raise ValueError(
        f"camera {camera.mount_id!r} names no model seat matching its identity or pose/FOV; "
        f"set mount_id or spec.logical_id to one of {', '.join(CAMERAS)}"
    )


def rig_cameras(definition) -> tuple[RigCamera, ...]:
    """The definition's cameras in canonical rig order, validated for the model."""
    if not definition.cameras:
        raise ValueError("N3BULA scene definition has no cameras")
    cameras = sorted(
        (RigCamera(canonical_name(camera), camera) for camera in definition.cameras),
        key=lambda one: CAMERAS.index(one.canonical),
    )
    mounts = [one.message.mount_id for one in cameras]
    if len(set(mounts)) != len(mounts):
        raise ValueError(f"camera mount ids repeat: {', '.join(mounts)}")
    view_ids([one.canonical for one in cameras])  # Refuses two mounts on one seat.
    for one in cameras:
        spec = one.message.spec
        if spec.WhichOneof("camera_param") != "ftheta_param":
            raise ValueError(f"Ludus N3BULA stream requires an F-theta camera: {one.message.mount_id}")
        params = spec.ftheta_param
        if params.reference_poly != params.ANGLE_TO_PIXELDIST:
            raise ValueError(
                f"Ludus requires an angle-to-pixel-distance polynomial for {one.message.mount_id}"
            )
        if len(params.angle_to_pixeldist_poly) != 6:
            raise ValueError(
                f"F-theta camera {one.message.mount_id} must provide six polynomial coefficients"
            )
        if not spec.resolution_w or not spec.resolution_h:
            raise ValueError(f"camera {one.message.mount_id} has no resolution")
    return tuple(cameras)
