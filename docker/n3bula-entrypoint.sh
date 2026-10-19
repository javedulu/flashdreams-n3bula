#!/usr/bin/env bash
set -euo pipefail

exec /opt/flashdreams/.venv/bin/python -m omnidreams.n3bula.server \
    --host "${FLASHDREAMS_HOST:-0.0.0.0}" \
    --port "${FLASHDREAMS_PORT:-50051}" \
    --pipeline_config_name "${FLASHDREAMS_PIPELINE_CONFIG:-omnidreams-perf}" \
    --resolution "${FLASHDREAMS_RESOLUTION:-640p}" \
    "$@"