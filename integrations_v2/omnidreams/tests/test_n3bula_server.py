# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the N3BULA gRPC server runner."""

from types import SimpleNamespace

import pytest
import torch
from omnidreams.n3bula import server as n3bula_server

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    """Minimal pipeline double for engine setup tests."""

    def __init__(self) -> None:
        self.device = None
        self.evaluating = False

    def to(self, *, device):
        """Record the requested device and return this pipeline."""
        self.device = device
        return self

    def eval(self):
        """Record evaluation mode and return this pipeline."""
        self.evaluating = True
        return self


def test_build_engine_sets_up_single_view_pipeline(monkeypatch) -> None:
    """Build engine state after moving the single-view pipeline to its device."""
    pipeline = _FakePipeline()

    class FakePipelineConfig:
        """Provide a minimal single-view pipeline configuration."""

        diffusion_model = SimpleNamespace(transformer=SimpleNamespace(num_views=1))
        """Single-view transformer metadata consumed by ``build_engine``."""

        def setup(self):
            """Return the test pipeline instance."""
            return pipeline

    monkeypatch.setattr(n3bula_server, "OmnidreamsPipeline", _FakePipeline)
    monkeypatch.setitem(
        n3bula_server.OMNIDREAMS_CONFIGS, "test-config", FakePipelineConfig()
    )

    engine = n3bula_server.build_engine("test-config", "640p", device="cpu")

    assert engine.device == torch.device("cpu")
    assert engine.conditioning_wrapper.pipeline is pipeline
    assert engine.conditioning_wrapper.video_resolution_wh == (1168, 640)
    assert engine.flashvsr_session is None
    assert engine.flashvsr_resolution_wh is None
    assert pipeline.device == torch.device("cpu")
    assert pipeline.evaluating


def test_parse_args_keeps_flashvsr_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("N3BULA_ENABLE_FLASHVSR", raising=False)

    args = n3bula_server.parse_args([])

    assert args.enable_flashvsr is False
    assert args.flashvsr_scale == 2
    assert args.flashvsr_sparse_ratio == 1.5


def test_build_engine_rejects_multiview_pipeline(monkeypatch) -> None:
    """Reject pipeline configurations that request multiple views."""

    class FakePipelineConfig:
        """Provide a multi-view pipeline configuration."""

        diffusion_model = SimpleNamespace(transformer=SimpleNamespace(num_views=2))
        """Multi-view transformer metadata consumed by ``build_engine``."""

    monkeypatch.setitem(
        n3bula_server.OMNIDREAMS_CONFIGS, "multi-view", FakePipelineConfig()
    )

    with pytest.raises(ValueError, match="single-view"):
        n3bula_server.build_engine("multi-view", "640p", device="cpu")


def test_create_server_registers_n3bula_service(monkeypatch) -> None:
    """Bind and start a server with the N3BULA servicer registered."""
    registered = []
    engine = SimpleNamespace()

    def register_service(servicer, server):
        """Capture the servicer and server passed to the registration helper."""
        registered.append((servicer, server))

    monkeypatch.setattr(
        n3bula_server.n3bula_pb2_grpc,
        "add_N3bulaLudusServiceServicer_to_server",
        register_service,
    )
    server, bound_port = n3bula_server.create_server(engine, "127.0.0.1", 0, 1)
    try:
        assert bound_port > 0
        server.start()
        assert len(registered) == 1
        assert isinstance(registered[0][0], n3bula_server.N3bulaLudusService)
        assert registered[0][1] is server
    finally:
        server.stop(0).wait()
