# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import grpc
import pytest
import torch
from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.service import N3bulaCosmosService
from cosmos3_n3bula.stream.feed import StreamEnded

pytestmark = pytest.mark.ci_cpu


class RpcAbort(Exception):
    def __init__(self, code, details):
        self.code = code
        self.details = details


class FakeContext:
    def abort(self, code, details):
        raise RpcAbort(code, details)


class FakeEngine:
    device = torch.device("cpu")


def camera(mount_id="camera_front_wide_120fov"):
    return pb.Camera(
        mount_id=mount_id,
        spec=pb.CameraSpec(
            resolution_w=1920,
            resolution_h=1080,
            logical_id=mount_id,
            ftheta_param=pb.FthetaCameraParam(
                reference_poly=pb.FthetaCameraParam.ANGLE_TO_PIXELDIST,
                angle_to_pixeldist_poly=[0, 800, 0, 0, 0, 0],
            ),
        ),
    )


def test_render_controls_rejects_invalid_schema_before_rendering():
    service = N3bulaCosmosService(FakeEngine())

    with pytest.raises(RpcAbort) as raised:
        service.render_controls(pb.SceneBatch(schema_version=2), FakeContext())

    assert raised.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "schema version 1" in raised.value.details


def test_render_controls_encodes_and_labels_every_frame(monkeypatch):
    class FakeRenderer:
        def __init__(self, definition, cameras, frames, *, device):
            self.cameras = cameras

        def render(self, first, last):
            return torch.arange(len(self.cameras) * (last - first) * 3).reshape(
                len(self.cameras) * (last - first), 1, 1, 3
            )

    renderer_module = ModuleType("cosmos3_n3bula.scene.renderer")
    renderer_module.RigRenderer = FakeRenderer
    monkeypatch.setitem(sys.modules, renderer_module.__name__, renderer_module)
    monkeypatch.setattr(
        "cosmos3_n3bula.service.encode_controls",
        lambda frames: [[f"png-{frame}-{view}".encode() for view in range(frames.shape[1])] for frame in range(frames.shape[0])],
    )
    first = pb.SceneFrame(source_frame_id=4, timestamp_us=100, ego_actor_id="ego-4")
    second = pb.SceneFrame(source_frame_id=5, timestamp_us=200, ego_actor_id="ego-5")
    request = pb.SceneBatch(
        schema_version=1,
        definition=pb.SceneDefinition(cameras=[camera()]),
        frames=[first, second],
    )

    response = N3bulaCosmosService(FakeEngine()).render_controls(request, FakeContext())

    assert [(frame.camera_mount_id, frame.source_frame_id) for frame in response.frames] == [
        ("camera_front_wide_120fov", 4),
        ("camera_front_wide_120fov", 5),
    ]
    assert [frame.image.data for frame in response.frames] == [b"png-0-0", b"png-1-0"]
    assert [frame.ego_actor_id for frame in response.frames] == ["ego-4", "ego-5"]
    assert all(frame.image.format == pb.PNG for frame in response.frames)


def test_stream_closure_exports_cache_and_returns_emitted_frames(monkeypatch):
    class FakeSession:
        def __init__(self, engine, definition, feed):
            self.mount_ids = ["front"]
            self.emitted = 0

        def step(self):
            if self.emitted:
                raise StreamEnded("client closed")
            self.emitted += 1
            return [pb.RenderedControlFrame(camera_mount_id="front", source_frame_id=9)]

        def steady_fps(self):
            return None

    session_module = ModuleType("cosmos3_n3bula.stream.session")
    session_module.CameraSession = FakeSession
    monkeypatch.setitem(sys.modules, session_module.__name__, session_module)

    class Cache:
        saves = 0

        def save(self):
            self.saves += 1

    cache = Cache()
    definition = pb.SceneDefinition(run_id="run-1")
    request = pb.SceneStreamRequest(schema_version=1, definition=definition)
    responses = list(
        N3bulaCosmosService(FakeEngine(), cache).stream_scenes(iter([request]), FakeContext())
    )

    assert [response.source_frame_id for response in responses] == [9]
    assert cache.saves == 1