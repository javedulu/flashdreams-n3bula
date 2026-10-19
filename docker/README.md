# `docker/` -- flashdreams container image

This folder contains the Dockerfile and build tooling for a flashdreams-ready
container image. Build it locally or push to your own registry.

The N3BULA service image uses the matching PyTorch 2.12.1 / CUDA 13.0 devel
base, exposes its system PyTorch/CUDA packages through the project venv, and
installs only the OmniDreams inference/gRPC dependencies by default. WebRTC,
replay, and proto-generation dependencies are excluded from the runtime image.

---

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | Image integration. Based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. |
| `Dockerfile.n3bula` | Self-contained OmniDreams gRPC service image with the N3BULA/Ludus RPC enabled. |
| `build_with_docker.sh` | Build + push a multi-arch (`linux/arm64` + `linux/amd64`) image to a registry you specify. |
| `build_n3bula_with_docker.sh` | Build the N3BULA gRPC image from the FlashDreams checkout root. |
| `n3bula-entrypoint.sh` | Start the service with the pipeline and resolution baked into the image. |

---

## Building locally

For a quick single-arch local image (no registry push):

```bash
docker build -t flashdreams:local -f docker/Dockerfile .
```

Then use it with docker:

```bash
docker run --rm --gpus all -it flashdreams:local bash
```

---

## Building + pushing (multi-arch)

Use `build_with_docker.sh` to produce a multi-arch manifest (linux/arm64 +
linux/amd64) and push it to your registry:

```bash
# Log in to your target registry first
docker login <your-registry>

# Build and push -- at least one fully-qualified tag is required
bash docker/build_with_docker.sh <your-registry>/flashdreams:<your-tag>

# Multiple tags are supported
bash docker/build_with_docker.sh reg1/flashdreams:v1.0 reg2/flashdreams:latest
```

---

## Multi-arch builds

`build_with_docker.sh` expects a Buildx builder that can build both
`linux/arm64` and `linux/amd64`. A default local builder can use QEMU
emulation for the non-native architecture, or you can configure your own
remote Buildx nodes outside this repository.

---

## Troubleshooting

**`ERROR: failed to solve: ... network ...` during build.**
`build_with_docker.sh` passes `--allow network.host --network host` so apt
and PyPI traffic can use the host's configured network path.

**Buildx can't find an arm64 node.**
Run `docker buildx ls` and confirm your selected builder supports
`linux/arm64`. If it does not, configure a multi-platform builder or use
QEMU emulation for the non-native architecture.

**`docker buildx build ... --load` complains about multi-platform.**
`--load` imports a single image into the local Docker daemon and is
incompatible with multi-arch output. Drop one of the `--platform` values
if you need a local-only build for testing.

## N3BULA gRPC service image

The N3BULA Dockerfile has two targets. `development` starts from the CUDA devel
image, installs build tools, and compiles the OmniDreams Sage3 and Ludus renderer
extensions for Blackwell (`12.0a`).
`release` generates the Python protobuf bindings, validates the N3BULA service,
and packages its compiled extensions into the matching CUDA runtime image. The
release image does not contain `nvcc` or compiler packages. Both images are
built directly from this fork checkout, where the N3BULA service is already
present.
BuildKit reuses the development image's extension layer for release setup.

From the root of the `flashdreams-n3bula` checkout, build both tags with:

```bash
bash docker/build_n3bula_with_docker.sh
```

The helper defaults to `omnidreams-n3bula:development` and
`omnidreams-n3bula:latest` for the development and release images. To override
the release tag:

```bash
bash docker/build_n3bula_with_docker.sh omnidreams-n3bula:latest
```

The image defaults to the repo's `omnidreams-perf` native-FP8 DiT recipe at
1168x640 (`640p`), the settings measured at 24.54 generated frames/s on an
RTX PRO 6000 Blackwell. That number measures synchronized `pipeline.generate`
only; it excludes setup, control transfer, finalization, CPU copies, encoding,
and N3BULA/Ludus application overhead, so it is not an end-to-end delivery
rate. Both images include the FlashDreams config and native acceleration code;
CUDA execution still requires an NVIDIA container runtime and the host driver.
Only the development target needs `nvcc`; the release target uses the runtime
base and retains both prebuilt extensions under
`/opt/flashdreams/artifacts/omnidreams/native-build`.
The runtime loaders import matching Sage3 and Ludus artifacts before attempting
JIT compilation, so server startup and the first scene stream do not require
`CUDA_HOME`, `nvcc`, or `ninja`. Keep Git in
the release image because runtime third-party provenance checks use it. BuildKit
cache mounts retain UV downloads and fingerprinted Torch extension outputs
across release-only changes; the serialized extension cache also removes stale
PyTorch lock files before reuse.
The release stage also uninstalls `cmake`, `ninja`, and `uv` inherited from the
PyTorch runtime base.
Model checkpoints download on first use. A cache bind mount is optional but
recommended when replacing containers.

Choose another checked-in GPU config when building, for example:

```bash
FLASHDREAMS_PIPELINE_CONFIG=omnidreams-optimized-rtx-pro-6000 \
FLASHDREAMS_RESOLUTION=704p \
bash docker/build_n3bula_with_docker.sh omnidreams-n3bula:rtx-pro-6000
```

Other registered profiles include `omnidreams-optimized-gb300`,
`omnidreams-rtx-5090`, `omnidreams-perf-responsive`,
`omnidreams-fast-perf-responsive`, and
`omnidreams-optimized-rtx-pro-6000-responsive`. Responsive configs retain
recent visual history for long-rollout experiments and can be selected per run
with `--flashdreams-pipeline-config`. The image build validates the selected config and
resolution against the checked-in server registry. The measured 24.54 frames/s
result is specific to `omnidreams-perf` at `640p`; other GPU profiles must be
measured separately. Set `FLASHDREAMS_UID` to match the host cache owner when
it is not UID 1000. N3BULA can select a profile per run with
`--flashdreams-pipeline-config` and `--flashdreams-resolution`.

The build uses the source in the current checkout directly. Set
`FLASHDREAMS_N3BULA_DEV_IMAGE`,
`FLASHDREAMS_PIPELINE_CONFIG`, and `FLASHDREAMS_RESOLUTION` to customize the
development tag or the profile baked into the release image.
