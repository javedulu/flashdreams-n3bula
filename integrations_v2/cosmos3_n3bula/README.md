# Cosmos3 N3BULA

Serves the N3BULA scene-stream gRPC API (`omnidreams.n3bula.N3bulaLudusService`)
with the multi-camera Cosmos3 Nano Sim Auto model.

```bash
uv sync --package flashdreams-cosmos3-n3bula --extra dev
uv run cosmos3-n3bula-server --port 50051
uv run pytest -m ci_cpu integrations_v2/cosmos3_n3bula/tests
./integrations_v2/cosmos3_n3bula/compile_protos.sh   # after editing proto/n3bula.proto
```

Docker builds development and release images (build context is the repo root; set `CUDA_ARCH_LIST` to the target GPU, default `12.0a`):

```bash
CUDA_ARCH_LIST=12.0a ./integrations_v2/cosmos3_n3bula/docker/build_with_docker.sh cosmos3-n3bula:latest
docker run --rm --gpus all -p 50051:50051 \
  -v $HOME/.cache/huggingface:/home/flashdreams/.cache/huggingface \
  -v $HOME/.cache/flashdreams-cosmos3-n3bula:/home/flashdreams/.cache/flashdreams \
  cosmos3-n3bula:latest
```

The Hugging Face volume stores downloaded shards; the FlashDreams volume stores merged
checkpoints. Keep both mounted to avoid downloading or merging the model again. The
entrypoint reads `FLASHDREAMS_CHECKPOINT`, `FLASHDREAMS_PORT`, `FLASHDREAMS_DEVICE`
and `N3BULA_RETURN_CONTROL_IMAGES=1`; extra arguments go to the server.

- Canonical `mount_id` or `spec.logical_id` values select their matching model
  seats. Unknown IDs infer the nearest trained seat from body-frame yaw and F-theta
  `max_angle`; the external `mount_id` remains unchanged in responses. Arbitrary
  camera geometry is not supported. Responses are ordered frame-outer, rig-order inner.
- The native model runs either unseeded `text2video` (`condition_frames=0`) or
  seeded `image2video` (`condition_frames=1`). Use `camera_seed_images` for all
  cameras or none; `seed_image` is accepted only for a one-camera rig. Seeds must
  be real opening RGB frames from the same scene state. A Ludus control render is
  not an RGB seed.
- The rollout draws a window of controls ahead of what it returns, so clients
  must stream frames ahead of the responses they wait for.
- One stream at a time, one GPU, one process. `enable_lidar` returns `UNIMPLEMENTED`.
- The proto shares the `omnidreams.n3bula` package name, so this package's
  `n3bula_pb2` and OmniDreams' cannot be imported into one process.

## Performance defaults

The server defaults to its fastest measured profile for both text2video and
image2video: compiled denoising (`--compile-denoise default`), fused RMSNorm and
a compiled control encoder. Generated frames are JPEG-encoded on the GPU, Ludus
uploads each rig camera once and updates actor cubes in place, and the VAE and
TAEHV weights load once per process.

On one RTX PRO 6000 Blackwell Max-Q (300 W), one front-wide view, 4 denoising
steps, 12 frames per chunk:

| profile | chunk p50 | FPS |
|---|---:|---:|
| eager (`--compile-denoise off --no-accelerated-rms-norm --no-compile-control-encode`) | 831 ms | 14.4 |
| default (compiled) | 756 ms | 15.8 |
| default, image2video | 784 ms | 15.2 |

These settings apply to single-camera sessions only. A multi-camera rig runs
eager: compiled denoising collapsed a 6-camera rig after its first chunk, and a
compiled control encoder shared by per-camera causal caches broke any rig. Eager
multi-camera throughput is 2.01 FPS per camera for 6 views.

The compiled rollout matches eager for about 30 frames and then follows its own
trajectory; it fails a 28 dB worst-frame PSNR check against eager. Use the
eager flags when output must reproduce the eager rollout. `--stage-timing` logs
per-chunk render, control-encode, denoise, decode and JPEG times. The first
chunk after a cold start compiles for several minutes; keep
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` on persistent storage.

Compiled kernels are also exported as portable `torch.compile` artifacts to
`~/.cache/n3bula/cosmos3-compile-cache/torch-<version>-<GPU>/` after any stream
that compiled something new, and every file there is loaded at startup, so a
new process or container on the same PyTorch build and GPU starts warm. Image
builds have no GPU to autotune on, so the artifacts live on the mounted cache
rather than in the image. `--compile-cache-dir none` disables this. Each
"scene closed" log line reports overall `fps` and `steady_fps`, which excludes
the two warm-up steps (1 then 12 frames).

## WSM 4.0.1 status

The pool mapper uses the WSM4 palette for supported lane, boundary, sign, signal,
crosswalk, waitline, pole and actor classes. The protobuf adds typed WSM4 sign,
signal, road-marking and boundary annotations without changing existing field
numbers. This is not yet full WSM4 conformance: the SUMO sender does not populate
the new labels; arrow and directional/pedestrian/cyclist signal glyphs are
rejected; crosswalks must be filled polygons rather than proxy boxes; and
SUMO-only surface bands are omitted with a warning and semantic counts. Static
sign/light proxies still use approximate extents. Do not waive these gaps as
generic WSM4 output. The migration contract tracks them under M2/T07/T09.

The model's seed mode and application feedback mode are separate choices.
NVIDIA Interactive Drive consumes user input, samples a trajectory chunk and
updates its own vehicle state. This stream consumes externally submitted SUMO
states and does not feed generated images or model actions back into SUMO. Its
chunk-feedback option only bounds transport backpressure; it is not closed-loop
driving.

Run the prototype's protobuf regeneration, CPU-marked Python suite and the independent
`gCosmosViz` palette unit test from the SUMO checkout with
`bash __pys__/test_cosmos3_n3bula_remote.sh`. On 2026-10-26, this passed 35 Python tests and 9
palette tests on cy9h3r. These are unit-level checks, not a WSM4 end-to-end conformance result.
