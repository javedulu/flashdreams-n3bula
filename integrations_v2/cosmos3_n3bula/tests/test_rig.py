# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

from math import pi

import pytest

from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.stream.rig import canonical_name, rig_cameras

pytestmark = pytest.mark.ci_cpu


def camera(mount_id: str, logical_id: str = "", rotation_bs=None, **spec) -> pb.Camera:
    ftheta = pb.FthetaCameraParam(
        principal_point_x=960,
        principal_point_y=540,
        reference_poly=pb.FthetaCameraParam.ANGLE_TO_PIXELDIST,
        angle_to_pixeldist_poly=[0, 800, 0, 0, 0, 0],
    )
    fields = {
        "resolution_w": 1920,
        "resolution_h": 1080,
        "ftheta_param": ftheta,
        "logical_id": logical_id,
    }
    fields.update(spec)
    return pb.Camera(
        mount_id=mount_id,
        spec=pb.CameraSpec(**fields),
        rotation_bs=rotation_bs or pb.Euler3d(),
    )


def definition(*cameras: pb.Camera) -> pb.SceneDefinition:
    return pb.SceneDefinition(cameras=cameras)


def test_cameras_sorted_into_rig_order():
    rig = rig_cameras(
        definition(
            camera("camera_front_tele_30fov"),
            camera("rear", logical_id="camera_rear_left_70fov"),
            camera("camera_front_wide_120fov"),
        )
    )
    assert [one.canonical for one in rig] == [
        "camera_front_wide_120fov",
        "camera_rear_left_70fov",
        "camera_front_tele_30fov",
    ]
    assert rig[1].message.mount_id == "rear"


@pytest.mark.parametrize(
    "cameras, message",
    [
        ((), "no cameras"),
        ((camera("dashcam"),), "names no model seat"),
        (
            (camera("a", "camera_front_wide_120fov"), camera("b", "camera_front_wide_120fov")),
            "named twice",
        ),
        ((camera("camera_front_wide_120fov", resolution_w=0),), "no resolution"),
        (
            (
                camera(
                    "camera_front_wide_120fov",
                    ftheta_param=pb.FthetaCameraParam(
                        reference_poly=pb.FthetaCameraParam.PIXELDIST_TO_ANGLE,
                        angle_to_pixeldist_poly=[0] * 6,
                    ),
                ),
            ),
            "angle-to-pixel-distance",
        ),
    ],
)
def test_rig_refusals(cameras, message):
    with pytest.raises(ValueError, match=message):
        rig_cameras(definition(*cameras))


def test_canonical_name_prefers_mount_id():
    assert canonical_name(camera("camera_rear_tele_30fov", "camera_front_wide_120fov")) == (
        "camera_rear_tele_30fov"
    )


@pytest.mark.parametrize(
    "mount_id, yaw, expected",
    [
        ("FWC_C", 0.0, "camera_front_wide_120fov"),
        ("external-right-camera", -pi / 2, "camera_cross_right_120fov"),
    ],
)
def test_unknown_mount_ids_match_trained_seat_from_pose_and_fov(mount_id, yaw, expected):
    unknown = camera(
        mount_id,
        ftheta_param=pb.FthetaCameraParam(
            principal_point_x=960,
            principal_point_y=540,
            reference_poly=pb.FthetaCameraParam.ANGLE_TO_PIXELDIST,
            angle_to_pixeldist_poly=[0, 800, 0, 0, 0, 0],
            max_angle=pi / 3,
        ),
        rotation_bs=pb.Euler3d(yaw_rad=yaw),
    )

    rig = rig_cameras(definition(unknown))

    assert rig[0].canonical == expected
    assert rig[0].message.mount_id == mount_id


def test_generation_config_has_no_image_conditioning_fields():
    fields = pb.GenerationConfig.DESCRIPTOR.fields_by_name
    assert set(fields) == {"prompt", "random_seed", "enable_lidar", "lidar_output"}
