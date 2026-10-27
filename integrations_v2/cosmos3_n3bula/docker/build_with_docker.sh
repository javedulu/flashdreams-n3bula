#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0
#
# Usage: build_with_docker.sh [release-image-tag]
# CUDA_ARCH_LIST must match the target GPU (e.g. 12.0a Blackwell, 9.0a Hopper, 8.0 A100).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
IMAGE="${1:-cosmos3-n3bula:latest}"
DEV_IMAGE="${FLASHDREAMS_COSMOS3_N3BULA_DEV_IMAGE:-cosmos3-n3bula:development}"
HF_CACHE="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
FLASHDREAMS_CACHE="${FLASHDREAMS_CACHE_HOST_DIR:-$HOME/.cache/flashdreams-cosmos3-n3bula}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [release-image-tag]" >&2
    exit 2
fi

BUILD_ARGS=(
    --file "$SCRIPT_DIR/Dockerfile"
    --build-arg "FLASHDREAMS_UID=${FLASHDREAMS_UID:-1000}"
    --build-arg "CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-12.0a}"
    --provenance=false
    --load
)

docker buildx build "${BUILD_ARGS[@]}" \
    --target build \
    --tag "$DEV_IMAGE" \
    "$ROOT"

docker buildx build "${BUILD_ARGS[@]}" \
    --target release \
    --tag "$IMAGE" \
    "$ROOT"

mkdir -p "$HF_CACHE" "$FLASHDREAMS_CACHE"
echo "Built development image $DEV_IMAGE and release image $IMAGE"
printf 'Run: docker run --rm --gpus all -p 50051:50051 \\\n  -v "%s:/home/flashdreams/.cache/huggingface" \\\n  -v "%s:/home/flashdreams/.cache/flashdreams" %s\n' \
    "$HF_CACHE" "$FLASHDREAMS_CACHE" "$IMAGE"
