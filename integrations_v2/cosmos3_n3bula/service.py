# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""The N3BULA gRPC service backed by the Cosmos3 multi-camera rollout."""

from __future__ import annotations

import threading
import time

import grpc
import torch
from loguru import logger

from cosmos3_n3bula.engine import Engine
from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.proto import n3bula_pb2_grpc
from cosmos3_n3bula.stream.feed import FrameFeed, StreamEnded, first_definition
from cosmos3_n3bula.stream.output import encode_controls
from cosmos3_n3bula.stream.rig import rig_cameras


class N3bulaCosmosService(n3bula_pb2_grpc.N3bulaLudusServiceServicer):
    """Render streamed N3BULA scenes with Ludus and return Cosmos3 generated frames."""

    def __init__(self, engine: Engine, compile_cache=None) -> None:
        self.engine = engine
        self.compile_cache = compile_cache
        # ponytail: one stream at a time on one GPU; queue a second client.
        self._lock = threading.Lock()

    def stream_scenes(self, request_iterator, context):
        with self._lock:
            yield from self._stream(iter(request_iterator), context)

    def _stream(self, requests, context):
        from cosmos3_n3bula.stream.session import CameraSession

        try:
            definition = first_definition(requests)
            while definition is not None:
                feed = FrameFeed(requests)
                started, session = time.perf_counter(), None
                closed = False
                next_definition = None
                try:
                    session = CameraSession(self.engine, definition, feed)
                    logger.info(
                        "N3BULA stream configured: run={} cameras={} mode=text2video seeded=False",
                        definition.run_id,
                        session.mount_ids,
                    )
                    while True:
                        yield from session.step()
                except StreamEnded as ended:
                    ended_message = str(ended)
                    # StreamEnded is normal flow control; its traceback keeps the closed session alive.
                    ended.__traceback__ = None
                    emitted = 0 if session is None else session.emitted
                    elapsed = time.perf_counter() - started
                    rate = None if session is None else session.steady_fps()
                    steady = "n/a" if rate is None else f"{rate:.2f}"
                    logger.info(
                        "N3BULA scene closed ({}): received={} generated={} in {:.1f}s "
                        "fps={:.2f} steady_fps={} (excluding the two warm-up steps)",
                        ended_message,
                        feed.received,
                        emitted,
                        elapsed,
                        emitted / elapsed if elapsed > 0 else 0.0,
                        steady,
                    )
                    next_definition = feed.next_definition
                    closed = True
                    if self.compile_cache is not None:
                        try:
                            self.compile_cache.save()
                        except Exception:  # noqa: BLE001 - caching must not fail a stream.
                            logger.exception("compile cache export failed")
                finally:
                    session = None
                    feed = None
                    if closed and self.engine.device.type == "cuda":
                        try:
                            torch.cuda.synchronize(self.engine.device)
                            logger.info(
                                "N3BULA CUDA memory after scene close: "
                                "allocated_mib={:.1f} reserved_mib={:.1f}",
                                torch.cuda.memory_allocated(self.engine.device) / 2**20,
                                torch.cuda.memory_reserved(self.engine.device) / 2**20,
                            )
                        except RuntimeError:
                            logger.exception("N3BULA CUDA memory snapshot failed after scene close")
                definition = next_definition
        except NotImplementedError as error:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:  # noqa: BLE001 - every failure must reach the client as a status.
            logger.exception("N3BULA Cosmos3 stream failed")
            context.abort(grpc.StatusCode.INTERNAL, str(error))

    def render_controls(self, request, context):
        if request.schema_version != 1 or not request.HasField("definition"):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "render_controls requires schema version 1 and a scene definition",
            )
        from cosmos3_n3bula.scene.renderer import RigRenderer

        try:
            cameras = rig_cameras(request.definition)
            frames = list(request.frames)
            if not frames:
                return pb.RenderedControls()
            renderer = RigRenderer(
                request.definition, cameras, frames.__getitem__, device=self.engine.device
            )
            drawn = renderer.render(0, len(frames))
            per_frame = drawn.view(len(cameras), len(frames), *drawn.shape[1:]).transpose(0, 1)
            images = encode_controls(per_frame)
            return pb.RenderedControls(
                frames=[
                    pb.RenderedControlFrame(
                        camera_mount_id=one.message.mount_id,
                        source_frame_id=frame.source_frame_id,
                        timestamp_us=frame.timestamp_us,
                        ego_actor_id=frame.ego_actor_id,
                        image=pb.Image(data=images[index][view], format=pb.PNG),
                    )
                    for index, frame in enumerate(frames)
                    for view, one in enumerate(cameras)
                ]
            )
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:  # noqa: BLE001 - every failure must reach the client as a status.
            logger.exception("N3BULA Ludus render failed")
            context.abort(grpc.StatusCode.INTERNAL, str(error))
