# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA gRPC server for the N3BULA scene-stream service."""

from __future__ import annotations

import argparse
import gc
import signal
import threading
from concurrent import futures
from dataclasses import dataclass

import grpc
import torch
from loguru import logger
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.impl.pipeline import OmnidreamsPipeline
from omnidreams.n3bula import n3bula_pb2_grpc
from omnidreams.n3bula.service import N3bulaLudusService

RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "480p": (832, 480),
    "640p": (1168, 640),
    "704p": (1280, 704),
    "720p": (1280, 720),
}
"""Pixel width and height supported by the N3BULA image profiles."""


@dataclass(slots=True)
class _PipelineRuntime:
    """Pipeline state and output shape consumed by the scene service."""

    pipeline: OmnidreamsPipeline
    """Configured OmniDreams pipeline used for generation."""

    video_resolution_wh: tuple[int, int]
    """Output width and height selected by the server resolution option."""


@dataclass(slots=True)
class N3bulaEngine:
    """Model device and conditioning pipeline used by N3BULA generation."""

    device: torch.device
    """CUDA device that owns the model and generated tensors."""

    conditioning_wrapper: _PipelineRuntime
    """Pipeline and output shape consumed by the N3BULA scene service."""


def build_engine(
    pipeline_config_name: str,
    resolution: str,
    device: torch.device | str = "cuda:0",
) -> N3bulaEngine:
    """Set up a supported single-view OmniDreams pipeline for N3BULA.

    Args:
        pipeline_config_name: Registered OmniDreams pipeline configuration.
        resolution: Supported output resolution slug.
        device: Device that owns the model; defaults to ``cuda:0``.

    Returns:
        Engine state consumed by ``N3bulaLudusService``.

    Raises:
        ValueError: The config, resolution, or view count is unsupported.
        TypeError: The config does not instantiate an ``OmnidreamsPipeline``.
    """
    if resolution not in RESOLUTION_MAP:
        raise ValueError(f"unsupported video resolution: {resolution}")
    try:
        pipeline_config = OMNIDREAMS_CONFIGS[pipeline_config_name]
    except KeyError as error:
        raise ValueError(
            f"unknown OmniDreams pipeline config: {pipeline_config_name}"
        ) from error

    transformer_config = pipeline_config.diffusion_model.transformer
    if transformer_config.num_views != 1:
        raise ValueError("N3BULA direct generation requires a single-view pipeline")

    device = torch.device(device)
    pipeline = pipeline_config.setup().to(device=device)
    if not isinstance(pipeline, OmnidreamsPipeline):
        raise TypeError(f"expected OmnidreamsPipeline, got {type(pipeline).__name__}")
    pipeline.eval()
    return N3bulaEngine(
        device=device,
        conditioning_wrapper=_PipelineRuntime(
            pipeline=pipeline,
            video_resolution_wh=RESOLUTION_MAP[resolution],
        ),
    )


def create_server(
    engine: N3bulaEngine,
    host: str,
    port: int,
    max_workers: int,
) -> tuple[grpc.Server, int]:
    """Register the N3BULA service and bind a gRPC server.

    Args:
        engine: Initialized model state for scene generation.
        host: Interface address to bind.
        port: TCP port, or zero to request an ephemeral port.
        max_workers: Maximum concurrent gRPC handlers.

    Returns:
        The unstarted server and its bound TCP port.

    Raises:
        RuntimeError: The server cannot bind the requested address.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    n3bula_pb2_grpc.add_N3bulaLudusServiceServicer_to_server(
        N3bulaLudusService(engine), server
    )
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"could not bind N3BULA gRPC server to {host}:{port}")
    return server, bound_port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse server options from ``argv`` or the process command line."""
    parser = argparse.ArgumentParser(
        description="Serve OmniDreams generation over the N3BULA scene-stream API."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument(
        "--pipeline_config_name",
        default="omnidreams-perf",
        choices=sorted(OMNIDREAMS_CONFIGS),
    )
    parser.add_argument(
        "--resolution",
        default="640p",
        choices=sorted(RESOLUTION_MAP),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Start the CUDA-backed N3BULA gRPC server and handle shutdown signals."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the N3BULA gRPC server")
    torch.cuda.set_device(0)
    engine = build_engine(args.pipeline_config_name, args.resolution, "cuda:0")
    server, bound_port = create_server(engine, args.host, args.port, args.max_workers)
    address = f"{args.host}:{bound_port}"
    shutdown_requested = threading.Event()

    def request_shutdown(_signum, _frame) -> None:
        shutdown_requested.set()

    previous_handlers = {
        sig: signal.signal(sig, request_shutdown)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    server.start()
    logger.info(
        "N3BULA gRPC server listening on {} with config={} resolution={}",
        address,
        args.pipeline_config_name,
        args.resolution,
    )
    try:
        shutdown_requested.wait()
    finally:
        server.stop(grace=5).wait()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        del server, engine
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
