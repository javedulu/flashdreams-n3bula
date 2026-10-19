# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO, StringIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flashdreams.infra.postprocess import VideoChunk
from omnidreams.n3bula import n3bula_pb2
from omnidreams.n3bula import service as n3bula_service
from omnidreams.n3bula.service import (
    SURFACE_BAND_PALETTE,
    SURFACE_BAND_PRIMITIVES,
    N3bulaLudusService,
    _make_scene,
    _OmniDreamsGenerationStream,
    _StreamRenderer,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    def __init__(self) -> None:
        self.initialized_text = None
        self.initialized_image = None
        self.inputs = []
        self.generated_indices = []
        self.finalized_indices = []
        self.output_device = torch.device("cpu")

    def get_num_output_frames(self, index: int) -> int:
        return 2

    def initialize_cache(self, *, text, image):
        self.initialized_text = text
        self.initialized_image = image
        return object()

    def generate(self, index, cache, *, input):
        self.inputs.append(input)
        self.generated_indices.append(index)
        return torch.zeros((1, 1, 2, 3, 3, 4), device=self.output_device)

    def finalize(self, index, cache) -> None:
        self.finalized_indices.append(index)


class _FakeBufferedFlashVSRSession:
    def __init__(self) -> None:
        self.chunks = []
        self.submitted_metadata = []
        self.reset_count = 0

    def reset(self) -> None:
        self.chunks.clear()
        self.reset_count += 1

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        self.chunks.append(chunk)
        self.submitted_metadata.append(chunk.metadata)
        return []

    def flush(self) -> list[VideoChunk]:
        if not self.chunks:
            return []
        tensor = torch.cat([chunk.tensor for chunk in self.chunks])
        output = F.interpolate(
            tensor, scale_factor=2, mode="bilinear", align_corners=False
        )
        metadata = {"input_chunks": tuple(chunk.metadata for chunk in self.chunks)}
        self.chunks.clear()
        return [VideoChunk(tensor=output, layout="tchw", metadata=metadata)]


def _png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (4, 3), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _rendered_control(
    color: tuple[int, int, int],
    frame=None,
    device: torch.device = torch.device("cpu"),
):
    pixels = torch.from_numpy(
        np.broadcast_to(np.asarray(color, dtype=np.uint8), (3, 4, 3)).copy()
    ).to(device)
    return n3bula_service._RenderedControl(
        frame or n3bula_pb2.SceneFrame(), "FWC_C", pixels, _png(color)
    )


def _engine(pipeline: _FakePipeline, device: torch.device = torch.device("cpu")):
    return SimpleNamespace(
        conditioning_wrapper=SimpleNamespace(
            pipeline=pipeline,
            video_resolution_wh=(4, 3),
        ),
        device=device,
    )


def test_n3bula_camera_and_image_schemas_preserve_precision() -> None:
    definition = n3bula_pb2.SceneDefinition(
        cameras=[
            n3bula_pb2.Camera(
                spec=n3bula_pb2.CameraSpec(
                    opencv_pinhole_param=n3bula_pb2.OpenCVPinholeCameraParam(
                        principal_point_x=1920.123456789,
                        focal_length_y=1080.987654321,
                        radial_coeffs=[1.234567890123e-7],
                    ),
                    logical_id="front",
                    shutter_type=n3bula_pb2.GLOBAL,
                ),
                position_bs_m=n3bula_pb2.Vec3d(x=12_345_678.123456789),
                rotation_bs=n3bula_pb2.Euler3d(yaw_rad=1.234567890123e-7),
            )
        ],
        generation_config=n3bula_pb2.GenerationConfig(
            seed_image=n3bula_pb2.Image(data=b"seed", format=n3bula_pb2.PNG)
        ),
    )

    restored = n3bula_pb2.SceneDefinition.FromString(definition.SerializeToString())
    camera = restored.cameras[0]

    assert camera.spec.opencv_pinhole_param.principal_point_x == 1920.123456789
    assert camera.spec.opencv_pinhole_param.radial_coeffs[0] == 1.234567890123e-7
    assert camera.position_bs_m.x == 12_345_678.123456789
    assert camera.rotation_bs.yaw_rad == 1.234567890123e-7
    assert restored.generation_config.seed_image.format == n3bula_pb2.PNG
    assert restored.generation_config.seed_image.data == b"seed"


def test_generation_uses_first_ludus_control_as_seed_without_returning_control_images() -> None:
    pipeline = _FakePipeline()
    generation = _OmniDreamsGenerationStream(
        _engine(pipeline),
        n3bula_pb2.GenerationConfig(prompt="drive through daylight"),
        (4, 3),
    )
    first = _rendered_control((255, 0, 0))
    second = _rendered_control((0, 255, 0))

    assert generation.add(n3bula_pb2.SceneFrame(), first) is None
    assert pipeline.initialized_image is not None
    assert torch.allclose(
        pipeline.initialized_image[0, 0, 0, :, 0, 0],
        torch.tensor([1.0, -1.0, -1.0]),
    )
    responses = generation.add(n3bula_pb2.SceneFrame(), second)

    assert responses is not None and len(responses) == 2
    assert all(not response.HasField("image") for response in responses)
    assert all(
        response.generated_image.format == n3bula_pb2.JPEG for response in responses
    )
    assert all(response.generated_image.data for response in responses)
    assert pipeline.initialized_text == [["drive through daylight"]]
    assert pipeline.generated_indices == [0]
    assert pipeline.finalized_indices == [0]
    assert pipeline.inputs[0].shape == (1, 1, 2, 3, 3, 4)


def test_flashvsr_flush_preserves_temporal_metadata_and_response_pairing() -> None:
    pipeline = _FakePipeline()
    session = _FakeBufferedFlashVSRSession()
    engine = _engine(pipeline)
    engine.flashvsr_session = session
    frames = [
        n3bula_pb2.SceneFrame(
            source_frame_id=41,
            timestamp_us=1_230_000,
            simulation_time_s=12.3,
            stale=True,
            ego_actor_id="ego-1",
            camera_mount_id="FWC_C",
        ),
        n3bula_pb2.SceneFrame(
            source_frame_id=42,
            timestamp_us=1_263_333,
            simulation_time_s=12.333333,
            ego_actor_id="ego-1",
            camera_mount_id="FWC_C",
        ),
    ]
    generation = _OmniDreamsGenerationStream(
        engine,
        n3bula_pb2.GenerationConfig(prompt="drive through daylight"),
        (4, 3),
        run_id="run-17",
    )

    generation.add(frames[0], _rendered_control((255, 0, 0), frames[0]))
    assert generation.add(
        frames[1], _rendered_control((0, 255, 0), frames[1])
    ) == []
    responses = generation.flush()

    assert responses is not None
    assert [response.source_frame_id for response in responses] == [41, 42]
    assert [response.timestamp_us for response in responses] == [1_230_000, 1_263_333]
    assert generation.generated_frames == 2
    assert session.reset_count == 1
    assert session.chunks == []
    assert session.submitted_metadata == [
        {
            "n3bula_source_frames": [
                {
                    "run_id": "run-17",
                    "source_frame_id": 41,
                    "timestamp_us": 1_230_000,
                    "simulation_time_s": 12.3,
                    "stale": True,
                    "ego_actor_id": "ego-1",
                    "camera_mount_id": "FWC_C",
                },
                {
                    "run_id": "run-17",
                    "source_frame_id": 42,
                    "timestamp_us": 1_263_333,
                    "simulation_time_s": 12.333333,
                    "stale": False,
                    "ego_actor_id": "ego-1",
                    "camera_mount_id": "FWC_C",
                },
            ]
        }
    ]
    with Image.open(BytesIO(responses[0].generated_image.data)) as image:
        assert image.size == (8, 6)


def test_generation_returns_control_images_when_debug_enabled(monkeypatch) -> None:
    monkeypatch.setattr(n3bula_service, "RETURN_CONTROL_IMAGES", True)
    pipeline = _FakePipeline()
    generation = _OmniDreamsGenerationStream(
        _engine(pipeline),
        n3bula_pb2.GenerationConfig(prompt="drive through daylight"),
        (4, 3),
    )
    first = _rendered_control((255, 0, 0))
    second = _rendered_control((0, 255, 0))

    generation.add(n3bula_pb2.SceneFrame(), first)
    responses = generation.add(n3bula_pb2.SceneFrame(), second)

    assert responses is not None
    assert [response.image.data for response in responses] == [first.png, second.png]


def test_gpu_control_path_skips_control_png_encoding_by_default(monkeypatch) -> None:
    monkeypatch.setattr(n3bula_service, "GPU_CONTROL_INPUT", True)
    pipeline = _FakePipeline()
    generation = _OmniDreamsGenerationStream(
        _engine(pipeline),
        n3bula_pb2.GenerationConfig(prompt="drive through daylight"),
        (4, 3),
    )
    first = n3bula_service._RenderedControl(
        n3bula_pb2.SceneFrame(), "FWC_C", torch.zeros((3, 4, 3), dtype=torch.uint8), None
    )
    second = n3bula_service._RenderedControl(
        n3bula_pb2.SceneFrame(), "FWC_C", torch.ones((3, 4, 3), dtype=torch.uint8), None
    )

    generation.add(n3bula_pb2.SceneFrame(), first)
    responses = generation.add(n3bula_pb2.SceneFrame(), second)

    assert responses is not None
    assert all(not response.HasField("image") for response in responses)


def test_generation_uses_explicit_seed_image_and_prompt() -> None:
    pipeline = _FakePipeline()
    config = n3bula_pb2.GenerationConfig(
        prompt="night highway",
        seed_image=n3bula_pb2.Image(data=_png((0, 0, 255)), format=n3bula_pb2.PNG),
        random_seed=17,
    )

    _OmniDreamsGenerationStream(_engine(pipeline), config, (4, 3))

    assert pipeline.initialized_text == [["night highway"]]
    assert np.allclose(
        pipeline.initialized_image[0, 0, 0, :, 0, 0].numpy(),
        np.array([-1.0, -1.0, 1.0]),
    )


def test_scene_updates_cubes_in_place_and_replaces_on_topology_change() -> None:
    class _SceneContext:
        def __init__(self):
            self.replace_calls = 0
            self.update_calls = 0
            self.cube_counts = {}

        def replace_scene(self, scene_id, scene):
            self.replace_calls += 1
            self.cube_counts = {
                pool.prim_type_id: pool.scales.shape[0]
                for pool in scene.cube_pools
            }

        def update_cube_pool(self, scene_id, primitive, pool):
            self.update_calls += 1
            if self.cube_counts.get(primitive) != pool.scales.shape[0]:
                return False
            return True

    renderer = object.__new__(_StreamRenderer)
    renderer.context = _SceneContext()
    renderer.scene_id = 1
    renderer._scene_has_frame = False
    renderer._dynamic_polyline_pools = None
    renderer._cube_pool_primitives = None

    def line_pool(vertices):
        return SimpleNamespace(
            prim_type_id=8,
            timestamps_us=torch.tensor([0]),
            timestamped_varrays_prefix_sum=torch.tensor([1]),
            varrays_prefix_sum=torch.tensor([len(vertices)]),
            vertices=torch.tensor(vertices, dtype=torch.float32),
        )

    def cube_pool(count):
        return SimpleNamespace(
            prim_type_id=5,
            scales=torch.zeros((count, 3)),
        )

    first_line = line_pool([[0, 0, 0], [1, 0, 0]])
    first_cube = cube_pool(2)
    first_scene = SimpleNamespace(cube_pools=[first_cube])
    assert not renderer._update_or_replace_scene(
        first_scene, [first_line], [first_cube], allow_in_place=True
    )

    next_cube = cube_pool(2)
    assert renderer._update_or_replace_scene(
        SimpleNamespace(cube_pools=[next_cube]),
        [line_pool([[0, 0, 0], [1, 0, 0]])],
        [next_cube],
        allow_in_place=True,
    )
    assert renderer.context.replace_calls == 1
    assert renderer.context.update_calls == 1

    larger_cube = cube_pool(3)
    assert not renderer._update_or_replace_scene(
        SimpleNamespace(cube_pools=[larger_cube]),
        [line_pool([[0, 0, 0], [1, 0, 0]])],
        [larger_cube],
        allow_in_place=True,
    )
    assert renderer.context.replace_calls == 2

    moved_line_cube = cube_pool(3)
    assert not renderer._update_or_replace_scene(
        SimpleNamespace(cube_pools=[moved_line_cube]),
        [line_pool([[0, 1, 0], [1, 1, 0]])],
        [moved_line_cube],
        allow_in_place=True,
    )
    assert renderer.context.replace_calls == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NVJPEG")
def test_generation_encodes_cuda_outputs_with_nvjpeg(monkeypatch) -> None:
    monkeypatch.setattr(n3bula_service, "GPU_JPEG_ENCODING", True)
    pipeline = _FakePipeline()
    pipeline.output_device = torch.device("cuda")
    generation = _OmniDreamsGenerationStream(
        _engine(pipeline, device=torch.device("cuda")),
        n3bula_pb2.GenerationConfig(prompt="drive through daylight"),
        (4, 3),
    )

    generation.add(
        n3bula_pb2.SceneFrame(),
        _rendered_control((255, 0, 0), device=torch.device("cuda")),
    )
    responses = generation.add(
        n3bula_pb2.SceneFrame(),
        _rendered_control((0, 255, 0), device=torch.device("cuda")),
    )

    assert responses is not None and len(responses) == 2
    for response in responses:
        assert response.generated_image.format == n3bula_pb2.JPEG
        with Image.open(BytesIO(response.generated_image.data)) as image:
            assert image.size == (4, 3)
            assert image.mode == "RGB"


class _FakeStreamRenderer:
    camera_mount_id = "FWC_C"
    image_size = (4, 3)

    def set_definition(self, definition) -> None:
        pass

    def render(self, frame, timings=None, defer_png=False, include_png=True):
        if timings is not None:
            timings.update(
                scene_prepare_seconds=0.0,
                camera_setup_seconds=0.0,
                render_gpu_seconds=0.0,
                render_readback_seconds=0.0,
                control_png_seconds=0.0,
            )
        return _rendered_control(
            (255, 0, 0), frame=frame
        )


class _FakeContext:
    def abort(self, code, message) -> None:
        raise RuntimeError(message)


def test_ludus_scene_fills_surface_bands_with_local_palette_primitives() -> None:
    definition = n3bula_pb2.SceneDefinition()
    definition.origin_world_m.x = 100.0
    definition.origin_world_m.y = 200.0
    definition.origin_world_m.z = 3.0
    road = definition.static_surface_bands.add(
        semantic=n3bula_pb2.SEMANTIC_ID_ROAD_SURFACE
    )
    road.left_world_m.add(x=100.0, y=200.0, z=3.0)
    road.left_world_m.add(x=100.0, y=210.0, z=3.0)
    road.right_world_m.add(x=104.0, y=200.0, z=3.0)
    road.right_world_m.add(x=104.0, y=210.0, z=3.0)
    shoulder = definition.static_surface_bands.add(
        semantic=n3bula_pb2.SEMANTIC_ID_SHOULDER
    )
    shoulder.left_world_m.add(x=104.0, y=200.0, z=3.0)
    shoulder.left_world_m.add(x=104.0, y=210.0, z=3.0)
    shoulder.right_world_m.add(x=106.0, y=200.0, z=3.0)
    shoulder.right_world_m.add(x=106.0, y=210.0, z=3.0)
    for semantic in (
        n3bula_pb2.SEMANTIC_ID_SIDEWALK,
        n3bula_pb2.SEMANTIC_ID_CURB,
        n3bula_pb2.SEMANTIC_ID_PARKING,
        n3bula_pb2.SEMANTIC_ID_BIKE_LANE,
        n3bula_pb2.SEMANTIC_ID_TERRAIN,
    ):
        band = definition.static_surface_bands.add(semantic=semantic)
        band.left_world_m.add(x=106.0, y=200.0, z=3.0)
        band.left_world_m.add(x=106.0, y=210.0, z=3.0)
        band.right_world_m.add(x=107.0, y=200.0, z=3.0)
        band.right_world_m.add(x=107.0, y=210.0, z=3.0)
    boundary = definition.static_strips.add(
        semantic=n3bula_pb2.SEMANTIC_ID_ROAD_BOUNDARY
    )
    boundary.centre_world_m.add(x=100.0, y=200.0, z=3.0)
    boundary.centre_world_m.add(x=100.0, y=210.0, z=3.0)
    frame = n3bula_pb2.SceneFrame(timestamp_us=42)

    scene = _make_scene(definition, frame)

    polygon_pools = {pool.prim_type_id: pool for pool in scene.polygon_pools}
    assert set(polygon_pools) == set(SURFACE_BAND_PALETTE)
    road_pool = polygon_pools[SURFACE_BAND_PRIMITIVES[road.semantic]]
    assert road_pool.prim_type_id == SURFACE_BAND_PRIMITIVES[road.semantic]
    assert road_pool.timestamps_us.tolist() == [42]
    assert road_pool.vertices.tolist() == [
        [0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        [0.0, 10.0, 0.0], [4.0, 10.0, 0.0],
    ]
    assert road_pool.triangles.tolist() == [[0, 1, 3], [0, 3, 2]]
    assert road_pool.triangle_prefix_sum.tolist() == [2]
    assert SURFACE_BAND_PALETTE[road_pool.prim_type_id] == (46, 46, 46, 255)
    shoulder_pool = polygon_pools[SURFACE_BAND_PRIMITIVES[shoulder.semantic]]
    assert shoulder_pool.triangles.tolist() == [[0, 1, 3], [0, 3, 2]]
    assert SURFACE_BAND_PALETTE[shoulder_pool.prim_type_id] == (70, 70, 70, 255)
    expected_extension_colors = {
        n3bula_pb2.SEMANTIC_ID_SIDEWALK: (128, 128, 128, 255),
        n3bula_pb2.SEMANTIC_ID_CURB: (112, 112, 112, 255),
        n3bula_pb2.SEMANTIC_ID_PARKING: (76, 76, 76, 255),
        n3bula_pb2.SEMANTIC_ID_BIKE_LANE: (56, 116, 104, 255),
        n3bula_pb2.SEMANTIC_ID_TERRAIN: (87, 79, 69, 255),
    }
    for semantic, color in expected_extension_colors.items():
        primitive = SURFACE_BAND_PRIMITIVES[semantic]
        assert SURFACE_BAND_PALETTE[primitive] == color
        assert polygon_pools[primitive].triangle_prefix_sum.tolist() == [2]
    assert all(pool.vertices.device.type == "cpu" for pool in polygon_pools.values())
    lane_boundary = next(
        pool for pool in scene.polyline_pools
        if pool.prim_type_id == n3bula_service.PRIM_LANE_BOUNDARY
    )
    assert lane_boundary.timestamped_varrays_prefix_sum.tolist() == [1]


def test_ludus_scene_renders_mesh_proxies_actor_colors_and_single_point_dots() -> None:
    definition = n3bula_pb2.SceneDefinition()
    frame = n3bula_pb2.SceneFrame(timestamp_us=99)
    actor = frame.boxes.add(
        semantic=n3bula_pb2.SEMANTIC_ID_ACTOR,
        category=n3bula_pb2.ACTOR_CATEGORY_TRUCK,
        length_m=4.0,
        width_m=2.0,
        height_m=1.5,
        wireframe_overlay=True,
    )
    actor.centre_world_m.x = 10.0
    actor.centre_world_m.y = 20.0
    mesh = frame.meshes.add(
        asset_path="sign.obj",
        semantic=n3bula_pb2.SEMANTIC_ID_TRAFFIC_SIGN,
    )
    mesh.origin_world_m.x = 4.0
    mesh.origin_world_m.y = 5.0
    speed_sign = frame.meshes.add(
        asset_path="speed-sign.obj",
        semantic=n3bula_pb2.SEMANTIC_ID_SPEED_SIGN,
    )
    speed_sign.origin_world_m.x = 6.0
    speed_sign.origin_world_m.y = 5.0
    light = frame.meshes.add(
        asset_path="traffic-light.obj",
        semantic=n3bula_pb2.SEMANTIC_ID_TRAFFIC_LIGHT,
    )
    light.origin_world_m.x = 8.0
    light.origin_world_m.y = 5.0
    crosswalk = frame.boxes.add(
        semantic=n3bula_pb2.SEMANTIC_ID_CROSSWALK,
        length_m=4.0,
        width_m=7.0,
        height_m=0.06,
    )
    crosswalk.centre_world_m.x = 12.0
    dot = frame.strips.add(
        semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
        colour=n3bula_pb2.MARK_COLOR_ID_WHITE,
        pattern=n3bula_pb2.MARK_TYPE_ID_SOLID,
    )
    dot.centre_world_m.add(x=1.0, y=2.0, z=0.0)

    scene = _make_scene(definition, frame)

    actor_pool = next(
        pool for pool in scene.cube_pools
        if pool.prim_type_id == n3bula_service.PRIM_OBSTACLE
    )
    assert np.allclose(
        actor_pool.colors[0].numpy(),
        np.asarray([204, 55, 0, 255, 192, 64], dtype=np.float32) / 255.0,
    )
    assert actor_pool.render_flags & n3bula_service.CUBE_FLAG_WIREFRAME
    expected_mesh_proxies = (
        (n3bula_service.PRIM_TRAFFIC_SIGN, [4.0, 5.0, 2.1], [0.01, 0.5, 0.8], [8, 2, 255]),
        (n3bula_service.PRIM_TRAFFIC_SIGN, [6.0, 5.0, 2.57], [0.16, 1.04, 1.04], [8, 2, 255]),
        (n3bula_service.PRIM_TRAFFIC_LIGHT, [8.0, 5.0, 3.0], [0.3, 0.3, 0.5], [100, 100, 100]),
    )
    for primitive, position, dimensions, color in expected_mesh_proxies:
        mesh_pool, mesh_index = next(
            (pool, index)
            for pool in scene.cube_pools if pool.prim_type_id == primitive
            for index in range(len(pool.translations))
            if np.allclose(pool.translations[index].numpy(), position)
        )
        assert np.allclose(mesh_pool.scales[mesh_index].numpy(), dimensions)
        assert np.allclose(
            mesh_pool.colors[mesh_index].numpy(),
            np.asarray([*color, *color], dtype=np.float32) / 255.0,
        )
    crosswalk_pool = next(
        pool for pool in scene.cube_pools
        if pool.prim_type_id == n3bula_service.PRIM_CROSSWALK
    )
    assert np.allclose(
        crosswalk_pool.colors[0].numpy(),
        np.asarray([139, 93, 255, 139, 93, 255], dtype=np.float32) / 255.0,
    )
    pole_pool = next(
        pool for pool in scene.polyline_pools
        if pool.prim_type_id == n3bula_service.PRIM_POLE
    )
    assert pole_pool.timestamped_varrays_prefix_sum.tolist() == [3]
    dot_pool = next(
        pool for pool in scene.polyline_pools
        if pool.prim_type_id == n3bula_service.PRIM_DOT_WHITE
    )
    assert dot_pool.vertices.tolist() == [[1.0, 2.0, 0.0]]
    assert n3bula_service._line_primitive(SimpleNamespace(
        semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
        colour=n3bula_pb2.MARK_COLOR_ID_ORANGE,
        pattern=n3bula_pb2.MARK_TYPE_ID_SOLID,
    )) == n3bula_service.PRIM_ROAD_MARKING


def test_ludus_scene_expands_broken_double_and_dot_lane_marks() -> None:
    definition = n3bula_pb2.SceneDefinition()
    frame = n3bula_pb2.SceneFrame(timestamp_us=1)
    broken = definition.static_strips.add(
        semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
        colour=n3bula_pb2.MARK_COLOR_ID_WHITE,
        pattern=n3bula_pb2.MARK_TYPE_ID_BROKEN,
    )
    broken.centre_world_m.extend(
        [n3bula_pb2.Vec3d(x=float(x), y=0.0, z=0.0) for x in (0, 10, 20, 30)]
    )
    double = definition.static_strips.add(
        semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
        colour=n3bula_pb2.MARK_COLOR_ID_YELLOW,
        pattern=n3bula_pb2.MARK_TYPE_ID_SOLID_SOLID,
    )
    double.centre_world_m.extend(
        [n3bula_pb2.Vec3d(x=0.0, y=5.0, z=0.0), n3bula_pb2.Vec3d(x=10.0, y=5.0, z=0.0)]
    )
    dots = definition.static_strips.add(
        semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
        colour=n3bula_pb2.MARK_COLOR_ID_WHITE,
        pattern=n3bula_pb2.MARK_TYPE_ID_BOTTS_DOTS,
    )
    dots.centre_world_m.extend(
        [n3bula_pb2.Vec3d(x=0.0, y=10.0, z=0.0), n3bula_pb2.Vec3d(x=4.5, y=10.0, z=0.0)]
    )
    for start in (0.0, 12.0, 24.0):
        segment = frame.strips.add(
            semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
            colour=n3bula_pb2.MARK_COLOR_ID_WHITE,
            pattern=n3bula_pb2.MARK_TYPE_ID_SOLID,
        )
        segment.centre_world_m.extend([
            n3bula_pb2.Vec3d(x=start, y=20.0, z=0.0),
            n3bula_pb2.Vec3d(x=start + 3.0, y=20.0, z=0.0),
        ])

    scene = _make_scene(definition, frame)
    pools = {pool.prim_type_id: pool for pool in scene.polyline_pools}
    broken_pool = pools[n3bula_service.PRIM_LANE_LINE_WHITE_DASHED]
    assert broken_pool.timestamped_varrays_prefix_sum.tolist() == [3]
    assert broken_pool.varrays_prefix_sum.tolist() == [2, 4, 6]
    assert np.allclose(broken_pool.vertices[:, 0], [0.0, 3.0, 12.0, 15.0, 24.0, 27.0])

    frame_dash_pool = pools[n3bula_service.PRIM_LANE_LINE_WHITE_SOLID]
    assert frame_dash_pool.timestamped_varrays_prefix_sum.tolist() == [3]
    assert np.allclose(frame_dash_pool.vertices[:, 0], [0.0, 3.0, 12.0, 15.0, 24.0, 27.0])

    double_pool = pools[n3bula_service.PRIM_LANE_LINE_YELLOW_SOLID]
    assert double_pool.timestamped_varrays_prefix_sum.tolist() == [2]
    assert np.allclose(double_pool.vertices[[0, 2], 1], [5.075, 4.925])
    dot_pool = pools[n3bula_service.PRIM_DOT_WHITE]
    assert dot_pool.timestamped_varrays_prefix_sum.tolist() == [1]
    assert np.allclose(dot_pool.vertices[:, 0], [0.25, 1.75, 3.25])


def test_ludus_scene_splits_long_polylines_at_renderer_limit() -> None:
    definition = n3bula_pb2.SceneDefinition()
    frame = n3bula_pb2.SceneFrame(timestamp_us=1)
    strip = frame.strips.add(
        semantic=n3bula_pb2.SEMANTIC_ID_LANE_MARKING,
        colour=n3bula_pb2.MARK_COLOR_ID_WHITE,
        pattern=n3bula_pb2.MARK_TYPE_ID_SOLID,
    )
    strip.centre_world_m.extend(
        [n3bula_pb2.Vec3d(x=float(x), y=0.0, z=0.0) for x in range(300)]
    )

    scene = _make_scene(definition, frame)
    pool = next(pool for pool in scene.polyline_pools
                if pool.prim_type_id == n3bula_service.PRIM_LANE_LINE_WHITE_SOLID)
    assert pool.timestamped_varrays_prefix_sum.tolist() == [2]
    assert pool.varrays_prefix_sum.tolist() == [256, 301]


def test_ludus_service_stream_flushes_pending_frames_on_half_close_and_reopens(monkeypatch) -> None:
    monkeypatch.setattr(n3bula_service, "RETURN_CONTROL_IMAGES", True)
    pipeline = _FakePipeline()
    service = N3bulaLudusService(_engine(pipeline))
    original_renderer = n3bula_service._StreamRenderer
    n3bula_service._StreamRenderer = _FakeStreamRenderer
    definition = n3bula_pb2.SceneDefinition(
        generation_config=n3bula_pb2.GenerationConfig(prompt="stream test")
    )
    log_output = StringIO()
    log_handler = n3bula_service.logger.add(log_output, format="{message}")
    requests = [
        n3bula_pb2.SceneStreamRequest(schema_version=1, definition=definition),
        n3bula_pb2.SceneStreamRequest(
            schema_version=1,
            frame=n3bula_pb2.SceneFrame(source_frame_id=10, timestamp_us=10),
        ),
        n3bula_pb2.SceneStreamRequest(schema_version=1, definition=definition),
        n3bula_pb2.SceneStreamRequest(
            schema_version=1,
            frame=n3bula_pb2.SceneFrame(source_frame_id=11, timestamp_us=20),
        ),
        n3bula_pb2.SceneStreamRequest(
            schema_version=1,
            frame=n3bula_pb2.SceneFrame(source_frame_id=12, timestamp_us=30),
        ),
    ]
    try:
        responses = list(service.stream_scenes(iter(requests), _FakeContext()))
        next_stream = list(service.stream_scenes(iter([
            n3bula_pb2.SceneStreamRequest(schema_version=1, definition=definition),
            n3bula_pb2.SceneStreamRequest(
                schema_version=1,
                frame=n3bula_pb2.SceneFrame(source_frame_id=13, timestamp_us=40),
            ),
        ]), _FakeContext()))
    finally:
        n3bula_service.logger.remove(log_handler)
        n3bula_service._StreamRenderer = original_renderer

    assert [response.source_frame_id for response in responses] == [10, 11, 12]
    assert [response.source_frame_id for response in next_stream] == [13]
    assert all(response.image.format == n3bula_pb2.PNG for response in responses)
    assert all(response.generated_image.format == n3bula_pb2.JPEG for response in responses)
    assert "stream closed: controls=3, generated=3" in log_output.getvalue()