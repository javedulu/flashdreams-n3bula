<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/horizontal-dark.svg">
    <img alt="FlashDreams" src="assets/logo/horizontal-light.svg" width="600">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://nvidia.github.io/flashdreams/main/index.html"><img alt="Documentation" src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
</p>

**FlashDreams** is a high-performance inference and serving library for
interactive autoregressive video and world models. It began as the optimized
runtime behind the [NVIDIA OmniDreams closed-loop demo for GTC 2026][omnidreams-blog]
and has grown into a general platform for real-time world-model applications
across gaming, autonomous vehicles, robotics, simulated or virtual
environments, and more.


## Quickstart

The complete setup is in our
[quick-start guide](https://nvidia.github.io/flashdreams/main/quickstart/index.html#install).

## Checkout Our Demos! 🎉

Quick-start to our interactive driving demo is [here](https://nvidia.github.io/flashdreams/main/quickstart/index.html#run-your-first-model).

![](docs/source/_static/demo_clips/interactive-drive.avif)

Checkout our game-like crazy-robotaxi demo [here](https://nvidia.github.io/flashdreams/main/demos/crazy_robotaxi/).

![](docs/source/_static/demo_clips/crazy-robotaxi.avif)

CUDA 12.8 is available as an opt-in side profile (Linux only) via the
`cuda12` dependency group:

```bash
uv sync --group cuda12 --extra runners
```

The two groups are declared mutually exclusive in `pyproject.toml`, so uv
deactivates the default `cuda13` group automatically when `--group cuda12`
is passed.

Then launch your first model by following [the Get Started guide](https://nvidia.github.io/flashdreams/main/quickstart/index.html#run-your-first-model).
For example, the Self-Forcing T2V v2 application is:

```bash
uv run --project integrations_v2/self_forcing \
    flashdreams-run-v2 t2v-self-forcing-wan2.1-t2v-1.3b \
    --output-path artifacts/t2v-self-forcing-wan2.1-t2v-1.3b.mp4 --timeout unbound -- \
    --prompt "A cat surfing." --total-blocks 7
```

You can also install FlashDreams as a library from PyPI:

```bash
pip install flashdreams
```

### Try the interactive driving demo

Drive a world model in real time with the unified OmniDreams `local-window` or
`webrtc` launch mode. See the
**[interactive demo guide](https://nvidia.github.io/flashdreams/main/models/omnidreams.html#launch-the-interactive-demo)**.

OmniDreams also includes a GPU-backed N3BULA gRPC service. For container build
and run instructions, see the [N3BULA service guide](docs/source/models/n3bula.md)
and [Docker instructions](docker/README.md#n3bula-grpc-service-image). The
current service uses unauthenticated, unencrypted gRPC; keep it on a trusted
network or put it behind an appropriately secured gateway.

## Supported models

FlashDreams ships first-party integrations under
[`integrations_v2/`](integrations_v2/). Each model has a dedicated docs page with
runner slugs, multi-GPU commands, and (where available) profiling benchmarks.

| Model | Family |
| --- | --- |
| [Self-Forcing](https://nvidia.github.io/flashdreams/main/models/self_forcing.html) | Streaming Wan2.1 T2V |
| [OmniDreams](https://nvidia.github.io/flashdreams/main/models/omnidreams.html) | HDMap-conditioned driving world model |
| [LingBot-World](https://nvidia.github.io/flashdreams/main/models/lingbot_world.html) | Camera-controllable I2V world model |
| [Waypoint 1.5](https://nvidia.github.io/flashdreams/main/models/waypoint.html) | Interactive image-established keyboard/mouse-controlled world model |
| [Wan2.1](https://nvidia.github.io/flashdreams/main/models/wan21.html) | Bidirectional T2V / I2V |
| [Causal-Forcing](https://nvidia.github.io/flashdreams/main/models/causal_forcing.html) | Streaming Wan2.1 T2V / I2V |
| [Causal Wan2.2](https://nvidia.github.io/flashdreams/main/models/causal_wan22.html) | FastVideo Causal Wan 2.2 14B MoE T2V |
| [FlashVSR](https://nvidia.github.io/flashdreams/main/models/flashvsr.html) | Streaming video super-resolution |
| [Cosmos-Predict2.5](https://nvidia.github.io/flashdreams/main/models/cosmos_predict2.html) | Bidirectional T2V / I2V |

### Serving integrations

| Integration | Interface |
| --- | --- |
| [N3BULA](docs/source/models/n3bula.md) | OmniDreams scene generation over gRPC |

See [the model gallery](https://nvidia.github.io/flashdreams/main/models/index.html) and
[the new method guide](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
to add your own.

Try [Cosmos3 Nano Sim Auto](integrations_v2/cosmos3_nano_sim_auto/README.md)
for multiview driving with prepared HD maps and optional LiDAR.

## Developer guides

### For Demo Developers

Make demos using world-models with our [Demo API](https://nvidia.github.io/flashdreams/main/documentation/demo_api/).

### For Model Developers

Implement world-models with our [Inferencing API](https://nvidia.github.io/flashdreams/main/documentation/inferencing_api/).

## Contributing

For how to contribute, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
New integrations, bug reports, feature requests, performance tuning, and
documentation edits are all welcome.

Use [GitHub Issues](https://github.com/NVIDIA/flashdreams/issues) to report defects or request improvements.

Join us on the [NVIDIA Omniverse Discord](https://discord.com/invite/nvidiaomniverse)
to share your results and take part in technical discussion! Channel: [`#flashdreams`](https://discord.gg/cMt2mHm4aN)

## Security

To report a potential security vulnerability, follow the coordinated
disclosure process in [`SECURITY.md`](SECURITY.md).

## License

FlashDreams is released under the [Apache License 2.0](LICENSE). Third-party
components and their licenses are listed in
[`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES) and [`NOTICE`](NOTICE). The
repository is REUSE-compliant; see [`REUSE.toml`](REUSE.toml) and
[`LICENSES/`](LICENSES/). The independently packaged
[Ludus renderer](integrations_v2/omnidreams/impl/ludus-renderer/LICENSE) retains its own Apache license file;
[upstream cudaraster terms](integrations_v2/omnidreams/impl/ludus-renderer/ludus_renderer/_cpp/cudaraster/LICENSE)
and per-file exceptions are described in the third-party notices.

## External materials

**NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.  Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.**

## Citation

If FlashDreams is useful in your research or product, please cite the project:

```bibtex
@misc{flashdreams2026,
  title        = {FlashDreams: High-performance inference and serving for
                  interactive autoregressive video and world models},
  author       = {{FlashDreams Contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/NVIDIA/flashdreams}},
}

@misc{nvidia2026omnidreams,
  title         = {{NVIDIA} {OmniDreams}: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation},
  author        = {Basant, Aarti and Kar, Amlan and Paschalidou, Despoina and Wei, Fangyin and Ferroni, Francesco and Garcia Cobo, Guillermo and Turki, Haithem and Ling, Huan and Seo, Jaewoo and Lucas, James and Wu, Jay Zhangjie and Wang, Jialiang and Lorraine, Jonathan and Gao, Jun and He, Kai and Tothova, Katarina and Xie, Kevin and Tyszkiewicz, Micha{\l} and Wu, Qi and de Lutio, Riccardo and Li, Ruilong and Fidler, Sanja and Kim, Seung Wook and Shen, Tianchang and Cao, Tianshi and Pfaff, Tobias and Lew, William and Wu, Xindi and Ren, Xuanchi and Lu, Yifan and Zhang, Yuxuan and Gojcic, Zan and Wang, Zian},
  year          = {2026},
  eprint        = {2606.03159},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.03159},
  url           = {https://arxiv.org/abs/2606.03159},
}
```

[omnidreams-blog]: https://research.nvidia.com/labs/sil/projects/omnidreams-blog/
