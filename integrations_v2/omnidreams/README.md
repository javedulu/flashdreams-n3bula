<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams

OmniDreams is an HD-map-conditioned streaming driving world-model integration.
Its public model variants are `OmnidreamsPipelineConfig` literals in
`omnidreams.config`; reusable applications own interactive I/O.

## Pipeline configurations

- `OMNIDREAMS_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_RTX_5090_PIPELINE_CONFIG`
- `OMNIDREAMS_RTX_5090_FAST_PIPELINE_CONFIG`
- `OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG`

## Install

```bash
uv sync --package flashdreams-omnidreams --inexact
```

The default install contains the inference and gRPC service dependencies, not
the Interactive Drive or Crazy Robotaxi applications. Add
`--extra applications` to install those application packages, and add
`--extra interactive-drive` when launching the Interactive Drive UI.
Session replay tools require `--extra replay`; regenerate protobuf bindings
with `--extra codegen`.

Checkpoints and example scenes download from Hugging Face on first use. Export
`HF_TOKEN` when the selected repository requires authentication.

Native third-party sources download on first use into
`artifacts/omnidreams/thirdparty`. The N3BULA development image synchronizes
these sources and prebuilds the Sage3 native extension; its release image
applies the server patch and copies the compiled extension into a CUDA runtime
base. The artifact is stored in both images at
`/opt/flashdreams/artifacts/omnidreams/native-build`, so release rebuilds for
server-only changes reuse the development build. Outside those images, first
use may still perform the build. Set
`FLASHDREAMS_OMNIDREAMS_TRY_THIRDPARTY_RESYNC=1` to attempt a clean redownload
on each load. Do not use resync while another OmniDreams process is using the
sources. The default value is `0` (only downloads missing sources).

## Applications

- [Interactive Drive](apps/interactive_drive/README.md)
- [Crazy Robotaxi](apps/crazy_robotaxi/README.md)

Launch Interactive Drive in a browser with:

```bash
uv sync --package flashdreams-omnidreams --extra applications --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
```

## Programmatic pipeline access

```python
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG

pipeline = OMNIDREAMS_PIPELINE_CONFIG.setup().to("cuda").eval()
```

## Tests

```bash
uv sync --package flashdreams-omnidreams --extra dev --group test --inexact
uv run --no-sync pytest integrations_v2/omnidreams/tests -m ci_cpu
```
