---
title: 'N3BULA'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

# N3BULA gRPC Service

N3BULA is a gRPC scene-stream interface backed here by OmniDreams generation.
The service runs in a CUDA-enabled container and listens on port `50051`.

<img
  src="../_static/model_clips/n3bula/n3bula-post.avif"
  alt="Animated view of the N3BULA scene-stream service."
  loading="lazy"
  style="max-width: 100%; height: auto;"
/>

## Requirements

- Linux host with a supported NVIDIA GPU and compatible NVIDIA driver.
- Docker Buildx to build the development and release images.
- NVIDIA Container Toolkit to expose the GPU to the running container.
- Hugging Face access for model weights; set `HF_TOKEN` if the repository
  requires authentication.

The checked-in native extensions target Blackwell (`12.0a`). See the
[Docker guide](../../../docker/README.md#n3bula-grpc-service-image) for image
and GPU profile details.

## Build

From the repository root, build both the development and release images:

```bash
bash docker/build_n3bula_with_docker.sh omnidreams-n3bula:latest
```

The build validates the selected pipeline config and resolution. To bake a
different registered profile into the release image, set the build-time
environment values:

```bash
FLASHDREAMS_PIPELINE_CONFIG=omnidreams-optimized-rtx-pro-6000 \
FLASHDREAMS_RESOLUTION=704p \
bash docker/build_n3bula_with_docker.sh omnidreams-n3bula:rtx-pro-6000
```

## Run

Publish the gRPC port on localhost and persist downloaded model files in a
named volume:

```bash
docker run --rm --gpus all \
  --publish 127.0.0.1:50051:50051 \
  --mount type=volume,src=omnidreams-hf,dst=/home/flashdreams/.cache/huggingface \
  --env HF_TOKEN \
  omnidreams-n3bula:latest
```

The image defaults to `omnidreams-perf` at `640p`. Set
`FLASHDREAMS_PIPELINE_CONFIG` or `FLASHDREAMS_RESOLUTION` on `docker run` to
select a different registered config or resolution. Supported resolutions are
`480p`, `640p`, `704p`, and `720p`. Model weights are downloaded on first use.

## Optional FlashVSR upscaling

The service can post-process generated frames with FlashVSR for 2x or 4x output. It is off by
default because it is slow on a single GPU. See [N3BULA + FlashVSR](n3bula_flashvsr.md) for the
design, side-by-side comparisons and integration learnings.

## Network security

The service currently uses gRPC without TLS or client authentication. The run
example binds the host port to localhost. Do not expose the service to
untrusted networks; use an authenticated, encrypted gateway if remote access is
required.

For container configuration and troubleshooting, see the
[N3BULA Docker instructions](../../../docker/README.md#n3bula-grpc-service-image).