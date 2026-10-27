# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Generated rig frames -> ``RenderedControlFrame`` responses."""

from __future__ import annotations

from collections.abc import Sequence

from cosmos3_n3bula.proto import n3bula_pb2 as pb


def chunk_responses(
    frames: Sequence,
    mount_ids: Sequence[str],
    generated: Sequence[Sequence[bytes]],
    controls: Sequence[Sequence[bytes]] | None = None,
) -> list[pb.RenderedControlFrame]:
    """One response per frame per camera, frame-outer and in rig order.

    Args:
        frames: the ``SceneFrame`` each generated frame answers.
        mount_ids: the rig's mount ids, in view order.
        generated: ``[T][V]`` JPEG bytes.
        controls: ``[T][V]`` PNG bytes of the drawn map, or ``None``.
    """
    if len(generated) != len(frames) or (controls is not None and len(controls) != len(frames)):
        raise ValueError(
            f"{len(generated)} generated frame(s) for {len(frames)} scene frame(s); "
            "the control feed and the rollout have come apart"
        )
    responses = []
    for index, frame in enumerate(frames):
        if len(generated[index]) != len(mount_ids):
            raise ValueError(f"frame {index} carries {len(generated[index])} views, not {len(mount_ids)}")
        for view, mount_id in enumerate(mount_ids):
            response = pb.RenderedControlFrame(
                camera_mount_id=mount_id,
                source_frame_id=frame.source_frame_id,
                timestamp_us=frame.timestamp_us,
                ego_actor_id=frame.ego_actor_id,
                generated_image=pb.Image(data=generated[index][view], format=pb.JPEG),
            )
            if controls is not None:
                response.image.CopyFrom(pb.Image(data=controls[index][view], format=pb.PNG))
            responses.append(response)
    return responses


def encode_generated(pixels, quality: int) -> list[list[bytes]]:
    """``[V, T, 3, H, W]`` in ``[-1, 1]`` -> ``[T][V]`` JPEG bytes.

    CUDA pixels are encoded on the device (NVJPEG), so only the compressed bytes
    cross to the host; the CPU path costs about 90 ms per 12-frame chunk.
    """
    import torch
    from torchvision.io import encode_jpeg

    images = pixels.add(1).mul(127.5).round_().clamp_(0, 255).to(dtype=torch.uint8)
    views, frames = int(images.shape[0]), int(images.shape[1])
    ordered = [images[v, t].contiguous() for t in range(frames) for v in range(views)]
    try:
        encoded = encode_jpeg(ordered, quality=quality)
    except RuntimeError:
        if not images.is_cuda:
            raise
        encoded = encode_jpeg([image.cpu() for image in ordered], quality=quality)
    return [[bytes(encoded[t * views + v].cpu().numpy()) for v in range(views)] for t in range(frames)]


def encode_controls(drawn) -> list[list[bytes]]:
    """``[T]`` of ``[V, H, W, 3]`` uint8 -> ``[T][V]`` PNG bytes."""
    from torchvision.io import encode_png

    return [
        [bytes(encode_png(view.permute(2, 0, 1).cpu()).numpy()) for view in frame]
        for frame in drawn
    ]
