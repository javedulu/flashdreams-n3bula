#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ARGS=(--host "${FLASHDREAMS_HOST:-0.0.0.0}" --port "${FLASHDREAMS_PORT:-50051}" --device "${FLASHDREAMS_DEVICE:-cuda:0}")
[[ -n "${FLASHDREAMS_CHECKPOINT:-}" ]] && ARGS+=(--checkpoint "$FLASHDREAMS_CHECKPOINT")
[[ "${N3BULA_RETURN_CONTROL_IMAGES:-0}" == "1" ]] && ARGS+=(--return-control-images)
# Extra server flags, e.g. "--stage-timing", or the eager profile
# "--compile-denoise off --no-accelerated-rms-norm --no-compile-control-encode".
if [[ -n "${COSMOS3_N3BULA_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "$COSMOS3_N3BULA_SERVER_ARGS"
    ARGS+=("${EXTRA_ARGS[@]}")
fi
# Keep torch.compile/Triton artifacts on the mounted cache so they survive the container.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/flashdreams/.cache/torchinductor-cosmos3-n3bula}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/flashdreams/.cache/triton-cosmos3-n3bula}"

exec /opt/flashdreams/.venv/bin/python -m cosmos3_n3bula.server "${ARGS[@]}" "$@"
