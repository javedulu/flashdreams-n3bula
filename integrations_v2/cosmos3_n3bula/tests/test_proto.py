# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cosmos3_n3bula.proto import n3bula_pb2 as pb

pytestmark = pytest.mark.ci_cpu


def test_point_cloud_round_trip():
    xyz = np.arange(12, dtype="<f4").reshape(4, 3)
    sweep = pb.LidarFrame(
        lidar_mount_id="top",
        source=pb.LIDAR_SOURCE_CONTROL,
        point_cloud=pb.LidarPointCloud(
            point_count=4,
            frame=pb.LIDAR_COORDINATE_FRAME_WORLD,
            positions_m=xyz.tobytes(),
            semantic=np.full(4, pb.SEMANTIC_ID_POLE, "u1").tobytes(),
        ),
    )
    wire = pb.RenderedControlFrame(camera_mount_id="front", lidar=[sweep])
    cloud = pb.RenderedControlFrame.FromString(wire.SerializeToString()).lidar[0].point_cloud
    assert np.array_equal(np.frombuffer(cloud.positions_m, "<f4").reshape(-1, 3), xyz)
    assert set(np.frombuffer(cloud.semantic, "u1")) == {pb.SEMANTIC_ID_POLE}


def test_existing_wire_numbers_unchanged():
    fields = pb.RenderedControlFrame.DESCRIPTOR.fields_by_name
    assert {name: fields[name].number for name in ("image", "generated_image")} == {
        "image": 4,
        "generated_image": 6,
    }
    assert pb.GenerationConfig.DESCRIPTOR.fields_by_name["random_seed"].number == 3
    assert pb.DESCRIPTOR.package == "omnidreams.n3bula"
