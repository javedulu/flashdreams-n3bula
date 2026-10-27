# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0
#
# Build from the repository root: ./integrations_v2/cosmos3_n3bula/build_with_docker.sh

ARG FLASHDREAMS_DEVEL_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel
ARG FLASHDREAMS_RUNTIME_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

FROM ${FLASHDREAMS_DEVEL_IMAGE} AS build

ARG FLASHDREAMS_UID=1000
ARG CUDA_ARCH_LIST=12.0a

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/flashdreams \
    FLASHDREAMS_CACHE_DIR=/home/flashdreams/.cache/flashdreams \
    HF_HOME=/home/flashdreams/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/opt/flashdreams/.venv \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    LUDUS_RENDERER_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    LUDUS_RENDERER_NATIVE_BUILD_ROOT=/opt/flashdreams/artifacts

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates \
        ffmpeg libnccl-dev libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN useradd --non-unique --uid "${FLASHDREAMS_UID}" --create-home --shell /bin/bash flashdreams

WORKDIR /opt/flashdreams
COPY --chown=${FLASHDREAMS_UID}:1000 . /opt/flashdreams

# torch and the CUDA wheels come from the base image, not from the lockfile.
RUN --mount=type=cache,target=/root/.cache/uv,id=cosmos3-n3bula-uv,sharing=locked \
    uv venv --python /usr/bin/python3 --system-site-packages "$UV_PROJECT_ENVIRONMENT" \
    && uv sync --package flashdreams-cosmos3-n3bula --inexact --no-dev \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package triton \
        --no-install-package cuda-toolkit \
        --no-install-package nvidia-cuda-cupti \
        --no-install-package nvidia-cuda-nvrtc \
        --no-install-package nvidia-cuda-runtime \
        --no-install-package nvidia-cudnn-cu13 \
        --no-install-package nvidia-cufft \
        --no-install-package nvidia-cufile \
        --no-install-package nvidia-curand \
        --no-install-package nvidia-cusolver \
        --no-install-package nvidia-cusparse \
        --no-install-package nvidia-cusparselt-cu13 \
        --no-install-package nvidia-nccl-cu13 \
        --no-install-package nvidia-nvjitlink \
        --no-install-package nvidia-nvtx \
        --no-install-package nvidia-nvshmem-cu13 \
    && mkdir -p /opt/flashdreams/artifacts "$FLASHDREAMS_CACHE_DIR" "$HF_HOME" \
    && chown -R "${FLASHDREAMS_UID}:1000" /opt/flashdreams/artifacts /home/flashdreams

USER flashdreams
WORKDIR /opt/flashdreams/integrations_v2

# No GPU at build time: link against the CUDA driver stub to compile Ludus.
RUN mkdir -p /tmp/cuda-driver-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-driver-stub/libcuda.so.1 \
    && LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
       LD_LIBRARY_PATH=/tmp/cuda-driver-stub:${LD_LIBRARY_PATH} \
       TORCH_EXTENSIONS_DIR=/opt/flashdreams/artifacts/torch_extensions \
       MAX_JOBS=8 \
       /opt/flashdreams/.venv/bin/python cosmos3_n3bula/docker/prebuild_extensions.py \
    && rm -rf /tmp/cuda-driver-stub

# Fails the build if the service or the checked-in protobuf code cannot import.
RUN /opt/flashdreams/.venv/bin/python -c \
    "from cosmos3_n3bula.server import create_server, parse_args; from cosmos3_n3bula.proto import n3bula_pb2_grpc; print('cosmos3-n3bula import check OK')"

CMD ["bash"]

FROM ${FLASHDREAMS_RUNTIME_IMAGE} AS release

ARG FLASHDREAMS_UID=1000
ARG CUDA_ARCH_LIST=12.0a

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/flashdreams \
    FLASHDREAMS_CACHE_DIR=/home/flashdreams/.cache/flashdreams \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    LUDUS_RENDERER_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    LUDUS_RENDERER_NATIVE_BUILD_ROOT=/opt/flashdreams/artifacts \
    TORCH_EXTENSIONS_DIR=/opt/flashdreams/artifacts/torch_extensions \
    HF_HOME=/home/flashdreams/.cache/huggingface

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates ffmpeg git libnccl2 libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --non-unique --uid "${FLASHDREAMS_UID}" --create-home --shell /bin/bash flashdreams \
    && mkdir -p /home/flashdreams/.cache/flashdreams /home/flashdreams/.cache/huggingface \
    && chown -R "${FLASHDREAMS_UID}:1000" /home/flashdreams

COPY --from=build --chown=${FLASHDREAMS_UID}:1000 /opt/flashdreams /opt/flashdreams
COPY --chmod=0755 --from=build /opt/flashdreams/integrations_v2/cosmos3_n3bula/docker/entrypoint.sh /usr/local/bin/cosmos3-n3bula-entrypoint

USER flashdreams
WORKDIR /opt/flashdreams/integrations_v2
EXPOSE 50051
ENTRYPOINT ["/usr/local/bin/cosmos3-n3bula-entrypoint"]
