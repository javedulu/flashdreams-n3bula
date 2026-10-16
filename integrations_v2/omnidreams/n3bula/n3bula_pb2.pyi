from omnidreams.impl.grpc.protos import camera_pb2 as _camera_pb2
from omnidreams.impl.grpc.protos import video_model_pb2 as _video_model_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SemanticId(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEMANTIC_ID_VOID: _ClassVar[SemanticId]
    SEMANTIC_ID_LANE_MARKING: _ClassVar[SemanticId]
    SEMANTIC_ID_ROAD_BOUNDARY: _ClassVar[SemanticId]
    SEMANTIC_ID_CROSSWALK: _ClassVar[SemanticId]
    SEMANTIC_ID_STOP_LINE: _ClassVar[SemanticId]
    SEMANTIC_ID_TRAFFIC_SIGN: _ClassVar[SemanticId]
    SEMANTIC_ID_TRAFFIC_LIGHT: _ClassVar[SemanticId]
    SEMANTIC_ID_POLE: _ClassVar[SemanticId]
    SEMANTIC_ID_ACTOR: _ClassVar[SemanticId]
    SEMANTIC_ID_EGO: _ClassVar[SemanticId]
    SEMANTIC_ID_ROAD_SURFACE: _ClassVar[SemanticId]
    SEMANTIC_ID_SHOULDER: _ClassVar[SemanticId]
    SEMANTIC_ID_SIDEWALK: _ClassVar[SemanticId]
    SEMANTIC_ID_CURB: _ClassVar[SemanticId]
    SEMANTIC_ID_PARKING: _ClassVar[SemanticId]
    SEMANTIC_ID_BIKE_LANE: _ClassVar[SemanticId]
    SEMANTIC_ID_TERRAIN: _ClassVar[SemanticId]
    SEMANTIC_ID_SPEED_SIGN: _ClassVar[SemanticId]

class MarkColorId(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MARK_COLOR_ID_WHITE: _ClassVar[MarkColorId]
    MARK_COLOR_ID_YELLOW: _ClassVar[MarkColorId]
    MARK_COLOR_ID_ORANGE: _ClassVar[MarkColorId]
    MARK_COLOR_ID_BLUE: _ClassVar[MarkColorId]
    MARK_COLOR_ID_GREEN: _ClassVar[MarkColorId]
    MARK_COLOR_ID_RED: _ClassVar[MarkColorId]
    MARK_COLOR_ID_VIOLET: _ClassVar[MarkColorId]
    MARK_COLOR_ID_BLACK: _ClassVar[MarkColorId]
    MARK_COLOR_ID_UNRESOLVED: _ClassVar[MarkColorId]

class MarkTypeId(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MARK_TYPE_ID_UNDECIDED: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_SOLID: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_BROKEN: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_SOLID_SOLID: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_SOLID_BROKEN: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_BROKEN_SOLID: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_BROKEN_BROKEN: _ClassVar[MarkTypeId]
    MARK_TYPE_ID_BOTTS_DOTS: _ClassVar[MarkTypeId]

class ActorCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTOR_CATEGORY_CAR: _ClassVar[ActorCategory]
    ACTOR_CATEGORY_TRUCK: _ClassVar[ActorCategory]
    ACTOR_CATEGORY_PEDESTRIAN: _ClassVar[ActorCategory]
    ACTOR_CATEGORY_CYCLIST: _ClassVar[ActorCategory]
    ACTOR_CATEGORY_OTHER: _ClassVar[ActorCategory]
    ACTOR_CATEGORY_EGO: _ClassVar[ActorCategory]

class CameraRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAMERA_ROLE_GROUND_TRUTH: _ClassVar[CameraRole]
    CAMERA_ROLE_CONDITIONING: _ClassVar[CameraRole]

class LineWidthMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINE_WIDTH_MODE_WORLD_METRES: _ClassVar[LineWidthMode]
    LINE_WIDTH_MODE_SCREEN_PIXELS: _ClassVar[LineWidthMode]
SEMANTIC_ID_VOID: SemanticId
SEMANTIC_ID_LANE_MARKING: SemanticId
SEMANTIC_ID_ROAD_BOUNDARY: SemanticId
SEMANTIC_ID_CROSSWALK: SemanticId
SEMANTIC_ID_STOP_LINE: SemanticId
SEMANTIC_ID_TRAFFIC_SIGN: SemanticId
SEMANTIC_ID_TRAFFIC_LIGHT: SemanticId
SEMANTIC_ID_POLE: SemanticId
SEMANTIC_ID_ACTOR: SemanticId
SEMANTIC_ID_EGO: SemanticId
SEMANTIC_ID_ROAD_SURFACE: SemanticId
SEMANTIC_ID_SHOULDER: SemanticId
SEMANTIC_ID_SIDEWALK: SemanticId
SEMANTIC_ID_CURB: SemanticId
SEMANTIC_ID_PARKING: SemanticId
SEMANTIC_ID_BIKE_LANE: SemanticId
SEMANTIC_ID_TERRAIN: SemanticId
SEMANTIC_ID_SPEED_SIGN: SemanticId
MARK_COLOR_ID_WHITE: MarkColorId
MARK_COLOR_ID_YELLOW: MarkColorId
MARK_COLOR_ID_ORANGE: MarkColorId
MARK_COLOR_ID_BLUE: MarkColorId
MARK_COLOR_ID_GREEN: MarkColorId
MARK_COLOR_ID_RED: MarkColorId
MARK_COLOR_ID_VIOLET: MarkColorId
MARK_COLOR_ID_BLACK: MarkColorId
MARK_COLOR_ID_UNRESOLVED: MarkColorId
MARK_TYPE_ID_UNDECIDED: MarkTypeId
MARK_TYPE_ID_SOLID: MarkTypeId
MARK_TYPE_ID_BROKEN: MarkTypeId
MARK_TYPE_ID_SOLID_SOLID: MarkTypeId
MARK_TYPE_ID_SOLID_BROKEN: MarkTypeId
MARK_TYPE_ID_BROKEN_SOLID: MarkTypeId
MARK_TYPE_ID_BROKEN_BROKEN: MarkTypeId
MARK_TYPE_ID_BOTTS_DOTS: MarkTypeId
ACTOR_CATEGORY_CAR: ActorCategory
ACTOR_CATEGORY_TRUCK: ActorCategory
ACTOR_CATEGORY_PEDESTRIAN: ActorCategory
ACTOR_CATEGORY_CYCLIST: ActorCategory
ACTOR_CATEGORY_OTHER: ActorCategory
ACTOR_CATEGORY_EGO: ActorCategory
CAMERA_ROLE_GROUND_TRUTH: CameraRole
CAMERA_ROLE_CONDITIONING: CameraRole
LINE_WIDTH_MODE_WORLD_METRES: LineWidthMode
LINE_WIDTH_MODE_SCREEN_PIXELS: LineWidthMode

class Vec3d(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Euler3d(_message.Message):
    __slots__ = ("yaw_rad", "pitch_rad", "roll_rad")
    YAW_RAD_FIELD_NUMBER: _ClassVar[int]
    PITCH_RAD_FIELD_NUMBER: _ClassVar[int]
    ROLL_RAD_FIELD_NUMBER: _ClassVar[int]
    yaw_rad: float
    pitch_rad: float
    roll_rad: float
    def __init__(self, yaw_rad: _Optional[float] = ..., pitch_rad: _Optional[float] = ..., roll_rad: _Optional[float] = ...) -> None: ...

class Camera(_message.Message):
    __slots__ = ("mount_id", "spec", "position_bs_m", "rotation_bs", "role", "line_width_mode", "near_m", "far_m", "rate_hz", "distortion", "calibration_source", "calibration_hash")
    MOUNT_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    POSITION_BS_M_FIELD_NUMBER: _ClassVar[int]
    ROTATION_BS_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    LINE_WIDTH_MODE_FIELD_NUMBER: _ClassVar[int]
    NEAR_M_FIELD_NUMBER: _ClassVar[int]
    FAR_M_FIELD_NUMBER: _ClassVar[int]
    RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    DISTORTION_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_SOURCE_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_HASH_FIELD_NUMBER: _ClassVar[int]
    mount_id: str
    spec: _camera_pb2.CameraSpec
    position_bs_m: Vec3d
    rotation_bs: Euler3d
    role: CameraRole
    line_width_mode: LineWidthMode
    near_m: float
    far_m: float
    rate_hz: float
    distortion: _containers.RepeatedScalarFieldContainer[float]
    calibration_source: str
    calibration_hash: int
    def __init__(self, mount_id: _Optional[str] = ..., spec: _Optional[_Union[_camera_pb2.CameraSpec, _Mapping]] = ..., position_bs_m: _Optional[_Union[Vec3d, _Mapping]] = ..., rotation_bs: _Optional[_Union[Euler3d, _Mapping]] = ..., role: _Optional[_Union[CameraRole, str]] = ..., line_width_mode: _Optional[_Union[LineWidthMode, str]] = ..., near_m: _Optional[float] = ..., far_m: _Optional[float] = ..., rate_hz: _Optional[float] = ..., distortion: _Optional[_Iterable[float]] = ..., calibration_source: _Optional[str] = ..., calibration_hash: _Optional[int] = ...) -> None: ...

class Strip(_message.Message):
    __slots__ = ("centre_world_m", "width", "semantic", "colour", "pattern", "instance_id")
    CENTRE_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_FIELD_NUMBER: _ClassVar[int]
    COLOUR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    centre_world_m: _containers.RepeatedCompositeFieldContainer[Vec3d]
    width: float
    semantic: SemanticId
    colour: MarkColorId
    pattern: MarkTypeId
    instance_id: int
    def __init__(self, centre_world_m: _Optional[_Iterable[_Union[Vec3d, _Mapping]]] = ..., width: _Optional[float] = ..., semantic: _Optional[_Union[SemanticId, str]] = ..., colour: _Optional[_Union[MarkColorId, str]] = ..., pattern: _Optional[_Union[MarkTypeId, str]] = ..., instance_id: _Optional[int] = ...) -> None: ...

class SurfaceBand(_message.Message):
    __slots__ = ("left_world_m", "right_world_m", "semantic", "instance_id")
    LEFT_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    RIGHT_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    left_world_m: _containers.RepeatedCompositeFieldContainer[Vec3d]
    right_world_m: _containers.RepeatedCompositeFieldContainer[Vec3d]
    semantic: SemanticId
    instance_id: int
    def __init__(self, left_world_m: _Optional[_Iterable[_Union[Vec3d, _Mapping]]] = ..., right_world_m: _Optional[_Iterable[_Union[Vec3d, _Mapping]]] = ..., semantic: _Optional[_Union[SemanticId, str]] = ..., instance_id: _Optional[int] = ...) -> None: ...

class Box(_message.Message):
    __slots__ = ("centre_world_m", "rotation_world", "length_m", "width_m", "height_m", "semantic", "category", "instance_id", "gradient_along_local_x", "wireframe_overlay")
    CENTRE_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    ROTATION_WORLD_FIELD_NUMBER: _ClassVar[int]
    LENGTH_M_FIELD_NUMBER: _ClassVar[int]
    WIDTH_M_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_M_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_ALONG_LOCAL_X_FIELD_NUMBER: _ClassVar[int]
    WIREFRAME_OVERLAY_FIELD_NUMBER: _ClassVar[int]
    centre_world_m: Vec3d
    rotation_world: Euler3d
    length_m: float
    width_m: float
    height_m: float
    semantic: SemanticId
    category: ActorCategory
    instance_id: int
    gradient_along_local_x: bool
    wireframe_overlay: bool
    def __init__(self, centre_world_m: _Optional[_Union[Vec3d, _Mapping]] = ..., rotation_world: _Optional[_Union[Euler3d, _Mapping]] = ..., length_m: _Optional[float] = ..., width_m: _Optional[float] = ..., height_m: _Optional[float] = ..., semantic: _Optional[_Union[SemanticId, str]] = ..., category: _Optional[_Union[ActorCategory, str]] = ..., instance_id: _Optional[int] = ..., gradient_along_local_x: _Optional[bool] = ..., wireframe_overlay: _Optional[bool] = ...) -> None: ...

class MeshInstance(_message.Message):
    __slots__ = ("asset_path", "origin_world_m", "yaw_rad", "semantic", "instance_id")
    ASSET_PATH_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    YAW_RAD_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    asset_path: str
    origin_world_m: Vec3d
    yaw_rad: float
    semantic: SemanticId
    instance_id: int
    def __init__(self, asset_path: _Optional[str] = ..., origin_world_m: _Optional[_Union[Vec3d, _Mapping]] = ..., yaw_rad: _Optional[float] = ..., semantic: _Optional[_Union[SemanticId, str]] = ..., instance_id: _Optional[int] = ...) -> None: ...

class GenerationConfig(_message.Message):
    __slots__ = ("prompt", "seed_image", "random_seed")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    SEED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    RANDOM_SEED_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    seed_image: _video_model_pb2.Image
    random_seed: int
    def __init__(self, prompt: _Optional[str] = ..., seed_image: _Optional[_Union[_video_model_pb2.Image, _Mapping]] = ..., random_seed: _Optional[int] = ...) -> None: ...

class SceneDefinition(_message.Message):
    __slots__ = ("run_id", "configuration_revision", "cameras", "camera_errors", "static_strips", "static_surface_bands", "origin_world_m", "generation_config")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_REVISION_FIELD_NUMBER: _ClassVar[int]
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ERRORS_FIELD_NUMBER: _ClassVar[int]
    STATIC_STRIPS_FIELD_NUMBER: _ClassVar[int]
    STATIC_SURFACE_BANDS_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    GENERATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    configuration_revision: int
    cameras: _containers.RepeatedCompositeFieldContainer[Camera]
    camera_errors: _containers.RepeatedScalarFieldContainer[str]
    static_strips: _containers.RepeatedCompositeFieldContainer[Strip]
    static_surface_bands: _containers.RepeatedCompositeFieldContainer[SurfaceBand]
    origin_world_m: Vec3d
    generation_config: GenerationConfig
    def __init__(self, run_id: _Optional[str] = ..., configuration_revision: _Optional[int] = ..., cameras: _Optional[_Iterable[_Union[Camera, _Mapping]]] = ..., camera_errors: _Optional[_Iterable[str]] = ..., static_strips: _Optional[_Iterable[_Union[Strip, _Mapping]]] = ..., static_surface_bands: _Optional[_Iterable[_Union[SurfaceBand, _Mapping]]] = ..., origin_world_m: _Optional[_Union[Vec3d, _Mapping]] = ..., generation_config: _Optional[_Union[GenerationConfig, _Mapping]] = ...) -> None: ...

class SceneFrame(_message.Message):
    __slots__ = ("source_frame_id", "timestamp_us", "simulation_time_s", "stale", "ego_actor_id", "ego_position_world_m", "ego_rotation_world", "camera_mount_id", "surface_bands", "strips", "boxes", "meshes", "palette_hash", "palette_version")
    SOURCE_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_US_FIELD_NUMBER: _ClassVar[int]
    SIMULATION_TIME_S_FIELD_NUMBER: _ClassVar[int]
    STALE_FIELD_NUMBER: _ClassVar[int]
    EGO_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    EGO_POSITION_WORLD_M_FIELD_NUMBER: _ClassVar[int]
    EGO_ROTATION_WORLD_FIELD_NUMBER: _ClassVar[int]
    CAMERA_MOUNT_ID_FIELD_NUMBER: _ClassVar[int]
    SURFACE_BANDS_FIELD_NUMBER: _ClassVar[int]
    STRIPS_FIELD_NUMBER: _ClassVar[int]
    BOXES_FIELD_NUMBER: _ClassVar[int]
    MESHES_FIELD_NUMBER: _ClassVar[int]
    PALETTE_HASH_FIELD_NUMBER: _ClassVar[int]
    PALETTE_VERSION_FIELD_NUMBER: _ClassVar[int]
    source_frame_id: int
    timestamp_us: int
    simulation_time_s: float
    stale: bool
    ego_actor_id: str
    ego_position_world_m: Vec3d
    ego_rotation_world: Euler3d
    camera_mount_id: str
    surface_bands: _containers.RepeatedCompositeFieldContainer[SurfaceBand]
    strips: _containers.RepeatedCompositeFieldContainer[Strip]
    boxes: _containers.RepeatedCompositeFieldContainer[Box]
    meshes: _containers.RepeatedCompositeFieldContainer[MeshInstance]
    palette_hash: int
    palette_version: str
    def __init__(self, source_frame_id: _Optional[int] = ..., timestamp_us: _Optional[int] = ..., simulation_time_s: _Optional[float] = ..., stale: _Optional[bool] = ..., ego_actor_id: _Optional[str] = ..., ego_position_world_m: _Optional[_Union[Vec3d, _Mapping]] = ..., ego_rotation_world: _Optional[_Union[Euler3d, _Mapping]] = ..., camera_mount_id: _Optional[str] = ..., surface_bands: _Optional[_Iterable[_Union[SurfaceBand, _Mapping]]] = ..., strips: _Optional[_Iterable[_Union[Strip, _Mapping]]] = ..., boxes: _Optional[_Iterable[_Union[Box, _Mapping]]] = ..., meshes: _Optional[_Iterable[_Union[MeshInstance, _Mapping]]] = ..., palette_hash: _Optional[int] = ..., palette_version: _Optional[str] = ...) -> None: ...

class SceneBatch(_message.Message):
    __slots__ = ("schema_version", "definition", "frames")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    definition: SceneDefinition
    frames: _containers.RepeatedCompositeFieldContainer[SceneFrame]
    def __init__(self, schema_version: _Optional[int] = ..., definition: _Optional[_Union[SceneDefinition, _Mapping]] = ..., frames: _Optional[_Iterable[_Union[SceneFrame, _Mapping]]] = ...) -> None: ...

class SceneStreamRequest(_message.Message):
    __slots__ = ("schema_version", "definition", "frame")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    schema_version: int
    definition: SceneDefinition
    frame: SceneFrame
    def __init__(self, schema_version: _Optional[int] = ..., definition: _Optional[_Union[SceneDefinition, _Mapping]] = ..., frame: _Optional[_Union[SceneFrame, _Mapping]] = ...) -> None: ...

class RenderedControlFrame(_message.Message):
    __slots__ = ("camera_mount_id", "source_frame_id", "timestamp_us", "image", "ego_actor_id", "generated_image")
    CAMERA_MOUNT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_US_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    EGO_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    camera_mount_id: str
    source_frame_id: int
    timestamp_us: int
    image: _video_model_pb2.Image
    ego_actor_id: str
    generated_image: _video_model_pb2.Image
    def __init__(self, camera_mount_id: _Optional[str] = ..., source_frame_id: _Optional[int] = ..., timestamp_us: _Optional[int] = ..., image: _Optional[_Union[_video_model_pb2.Image, _Mapping]] = ..., ego_actor_id: _Optional[str] = ..., generated_image: _Optional[_Union[_video_model_pb2.Image, _Mapping]] = ...) -> None: ...

class RenderedControls(_message.Message):
    __slots__ = ("frames",)
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    frames: _containers.RepeatedCompositeFieldContainer[RenderedControlFrame]
    def __init__(self, frames: _Optional[_Iterable[_Union[RenderedControlFrame, _Mapping]]] = ...) -> None: ...
