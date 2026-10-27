# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.scene.geometry import (
    DASH_LENGTH_M,
    DASH_PERIOD_M,
    LUDUS_MAX_POLYLINE_POINTS,
    camera_to_world,
    dash_polyline,
    dot_polyline,
    offset_polyline,
    split_polyline,
)
from cosmos3_n3bula.stream.output import chunk_responses

pytestmark = pytest.mark.ci_cpu


def line(length: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])


def test_dashes_paint_the_dash_length_every_period():
    dashes = dash_polyline(line(30.0))
    assert [round(dash[0, 0], 6) for dash in dashes] == [0.0, DASH_PERIOD_M, 2 * DASH_PERIOD_M]
    assert all(np.isclose(dash[-1, 0] - dash[0, 0], DASH_LENGTH_M) for dash in dashes)


def test_dots_and_offsets():
    assert len(dot_polyline(line(3.0))) == 2
    assert np.allclose(offset_polyline(line(5.0), 1.0)[:, 1], 1.0)  # Left is +y.


def test_split_shares_end_vertices():
    points = np.zeros((LUDUS_MAX_POLYLINE_POINTS + 10, 3))
    points[:, 0] = np.arange(len(points))
    chunks = list(split_polyline(points))
    assert len(chunks) == 2 and chunks[0][-1, 0] == chunks[1][0, 0]


def test_camera_follows_ego_yaw_and_mount():
    frame = pb.SceneFrame(
        ego_position_world_m=pb.Vec3d(x=101.0, y=50.0),
        ego_rotation_world=pb.Euler3d(yaw_rad=np.pi / 2),
    )
    mount = pb.Camera(position_bs_m=pb.Vec3d(x=2.0, z=1.5))
    pose = camera_to_world(frame, mount, origin=np.asarray([100.0, 50.0, 0.0]))
    # A mount 2 m ahead of an ego facing +y sits 2 m along +y, origin-relative.
    assert np.allclose(pose[:3, 3], [1.0, 2.0, 1.5])
    assert np.allclose(pose[:3, 0], [0.0, 1.0, 0.0])


def test_responses_are_frame_outer_and_labelled():
    frames = [pb.SceneFrame(source_frame_id=i, timestamp_us=100 * i, ego_actor_id="ego") for i in (7, 8)]
    generated = [[b"f0", b"r0"], [b"f1", b"r1"]]
    responses = chunk_responses(frames, ["front", "rear"], generated, controls=generated)
    assert [(r.source_frame_id, r.camera_mount_id, r.generated_image.data) for r in responses] == [
        (7, "front", b"f0"),
        (7, "rear", b"r0"),
        (8, "front", b"f1"),
        (8, "rear", b"r1"),
    ]
    assert responses[0].generated_image.format == pb.JPEG
    assert responses[0].image.format == pb.PNG and responses[0].ego_actor_id == "ego"
    assert not chunk_responses(frames, ["front", "rear"], generated)[0].HasField("image")


def test_responses_refuse_misaligned_chunks():
    frames = [pb.SceneFrame(), pb.SceneFrame()]
    with pytest.raises(ValueError, match="come apart"):
        chunk_responses(frames, ["front"], [[b"x"]])
    with pytest.raises(ValueError, match="carries 2 views"):
        chunk_responses(frames[:1], ["front"], [[b"x", b"y"]])
