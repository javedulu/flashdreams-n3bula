# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA gRPC server for the N3BULA scene-stream service on Cosmos3 Nano Sim Auto."""

from __future__ import annotations

import argparse
import signal
import threading
from concurrent import futures
from pathlib import Path

import grpc
from loguru import logger

from cosmos3_n3bula.config import (
    ATTENTION_BACKENDS,
    COMPILE_DENOISE_CHOICES,
    ServerSettings,
)
from cosmos3_n3bula.proto import n3bula_pb2_grpc

MAX_MESSAGE_BYTES = 100 * 1024 * 1024


def create_server(service, host: str, port: int, max_workers: int) -> tuple[grpc.Server, int]:
    """Register ``service`` on an unstarted server and return it with its bound port."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ],
    )
    n3bula_pb2_grpc.add_N3bulaLudusServiceServicer_to_server(service, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise RuntimeError(f"could not bind N3BULA gRPC server to {host}:{port}")
    return server, bound


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Cosmos3 Nano Sim Auto generation over the N3BULA scene-stream API."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--checkpoint", default=None, help="Local path or URL; default is the published one.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--return-control-images", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default="default")
    defaults = ServerSettings()
    parser.add_argument(
        "--accelerated-rms-norm", action=argparse.BooleanOptionalAction, default=defaults.accelerated_rms_norm
    )
    parser.add_argument(
        "--compile-denoise",
        choices=COMPILE_DENOISE_CHOICES,
        default=defaults.compile_denoise or "off",
        help="off reproduces the eager rollout exactly; the default compiles for throughput",
    )
    parser.add_argument("--compile-decoder", action=argparse.BooleanOptionalAction, default=defaults.compile_decoder)
    parser.add_argument(
        "--compile-control-encode", action=argparse.BooleanOptionalAction, default=defaults.compile_control_encode
    )
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument(
        "--compile-cache-dir",
        default=None,
        help="torch.compile artifact cache shared across processes and containers "
        "(default ~/.cache/n3bula/cosmos3-compile-cache; 'none' disables)",
    )
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> ServerSettings:
    return ServerSettings(
        checkpoint=args.checkpoint,
        device=args.device,
        fps=args.fps,
        return_control_images=args.return_control_images,
        jpeg_quality=args.jpeg_quality,
        attention_backend=args.attention_backend,
        accelerated_rms_norm=args.accelerated_rms_norm,
        compile_denoise=None if args.compile_denoise == "off" else args.compile_denoise,
        compile_decoder=args.compile_decoder,
        compile_control_encode=args.compile_control_encode,
        stage_timing=args.stage_timing,
    )


def main() -> None:
    import torch

    from cosmos3_n3bula.engine import Engine
    from cosmos3_n3bula.service import N3bulaCosmosService

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the N3BULA Cosmos3 server")
    settings = settings_from_args(args)
    torch.cuda.set_device(torch.device(settings.device))
    from cosmos3_n3bula.compile_cache import DEFAULT_ROOT, CompileCache

    cache_root = None if args.compile_cache_dir == "none" else Path(args.compile_cache_dir or DEFAULT_ROOT)
    compile_cache = CompileCache(cache_root, torch.device(settings.device))
    compile_cache.load()
    server, port = create_server(
        N3bulaCosmosService(Engine(settings), compile_cache), args.host, args.port, args.max_workers
    )
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    server.start()
    logger.info("N3BULA Cosmos3 gRPC server listening on {}:{}", args.host, port)
    try:
        stop.wait()
    finally:
        server.stop(grace=5).wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
