#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLASHDREAMS_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGE="${1:-omnidreams-n3bula:latest}"
DEV_IMAGE="${FLASHDREAMS_N3BULA_DEV_IMAGE:-omnidreams-n3bula:development}"
FLASHDREAMS_UID="${FLASHDREAMS_UID:-1000}"
FLASHDREAMS_PIPELINE_CONFIG="${FLASHDREAMS_PIPELINE_CONFIG:-omnidreams-perf}"
FLASHDREAMS_RESOLUTION="${FLASHDREAMS_RESOLUTION:-640p}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [release-image-tag]" >&2
    exit 2
fi

BUILD_ARGS=(
    --file "$SCRIPT_DIR/Dockerfile.n3bula"
    --build-arg "FLASHDREAMS_UID=$FLASHDREAMS_UID" \
    --build-arg "FLASHDREAMS_PIPELINE_CONFIG=$FLASHDREAMS_PIPELINE_CONFIG" \
    --build-arg "FLASHDREAMS_RESOLUTION=$FLASHDREAMS_RESOLUTION" \
    --provenance=false \
    --load \
)

docker buildx build "${BUILD_ARGS[@]}" \
    --target development \
    --tag "$DEV_IMAGE" \
    "$FLASHDREAMS_ROOT"

docker buildx build "${BUILD_ARGS[@]}" \
    --target release \
    --tag "$IMAGE" \
    "$FLASHDREAMS_ROOT"

echo "Built development image $DEV_IMAGE and release image $IMAGE"