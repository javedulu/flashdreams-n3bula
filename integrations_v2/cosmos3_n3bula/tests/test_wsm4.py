# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from ludus_renderer._ops.primitives import PRIM_TYPE_COUNT

from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.scene.pools import (
    WSM4_BOUNDARY_PRIMITIVES,
    WSM4_LANE_UNKNOWN,
    WSM4_PALETTE,
    _surface_polygon_pools,
    box_colors,
    box_primitive,
    lane_primitive,
    line_primitive,
    surface_band_primitive,
    unsupported_surface_bands,
)
pytestmark = pytest.mark.ci_cpu


def test_wsm4_sign_palette_and_speed_limit_back_face():
    stop = pb.Box(
        semantic=pb.SEMANTIC_ID_TRAFFIC_SIGN,
        wsm4_sign_class=pb.WSM4_SIGN_CLASS_STOP,
    )
    assert box_colors(stop) == [
        220 / 255,
        20 / 255,
        20 / 255,
        40 / 255,
        40 / 255,
        40 / 255,
    ]

    generic = pb.Box(semantic=pb.SEMANTIC_ID_TRAFFIC_SIGN)
    generic_colors = box_colors(generic)
    assert generic_colors[:3] == generic_colors[3:] == [8 / 255, 2 / 255, 1]

    speed = pb.Box(semantic=pb.SEMANTIC_ID_SPEED_SIGN)
    speed_colors = box_colors(speed)
    assert speed_colors[:3] == [8 / 255, 2 / 255, 1]
    assert speed_colors[3:] == [40 / 255] * 3


def test_wsm4_signal_states_and_unrendered_glyphs_are_explicit():
    signal = pb.Box(
        semantic=pb.SEMANTIC_ID_TRAFFIC_LIGHT,
        wsm4_signal_state=pb.WSM4_SIGNAL_STATE_GREEN,
    )
    assert box_colors(signal) == [
        130 / 255,
        240 / 255,
        132 / 255,
        60 / 255,
        60 / 255,
        60 / 255,
    ]
    assert box_colors(pb.Box(semantic=pb.SEMANTIC_ID_TRAFFIC_LIGHT))[:3] == [
        150 / 255,
        150 / 255,
        150 / 255,
    ]
    directional = pb.Box(
        semantic=pb.SEMANTIC_ID_TRAFFIC_LIGHT,
        wsm4_signal_direction=pb.WSM4_SIGNAL_DIRECTION_LEFT,
    )
    with pytest.raises(NotImplementedError, match="directional"):
        box_colors(directional)


def test_wsm4_lane_and_boundary_palette():
    assert lane_primitive(pb.MARK_COLOR_ID_WHITE, "solid") != WSM4_LANE_UNKNOWN
    assert lane_primitive(pb.MARK_COLOR_ID_YELLOW, "broken") != WSM4_LANE_UNKNOWN
    assert lane_primitive(pb.MARK_COLOR_ID_UNRESOLVED, "solid") == WSM4_LANE_UNKNOWN
    assert WSM4_PALETTE[WSM4_LANE_UNKNOWN] == (181, 164, 71, 255)
    assert line_primitive(
        pb.Strip(
            semantic=pb.SEMANTIC_ID_ROAD_BOUNDARY,
            wsm4_boundary_type=pb.WSM4_BOUNDARY_CURB,
        )
    ) == WSM4_BOUNDARY_PRIMITIVES[pb.WSM4_BOUNDARY_CURB]
    assert line_primitive(
        pb.Strip(semantic=pb.SEMANTIC_ID_ROAD_BOUNDARY)
    ) == WSM4_BOUNDARY_PRIMITIVES[pb.WSM4_BOUNDARY_ROAD_BOUNDARY]
    assert WSM4_PALETTE[WSM4_BOUNDARY_PRIMITIVES[pb.WSM4_BOUNDARY_CURB]] == (
        207,
        255,
        191,
        255,
    )


def test_wsm4_polygon_classes_and_missing_surface_report():
    crosswalk = pb.SurfaceBand(semantic=pb.SEMANTIC_ID_CROSSWALK)
    stop_area = pb.SurfaceBand(wsm4_marking_type=pb.WSM4_ROAD_MARKING_STOP)
    arrow = pb.SurfaceBand(wsm4_marking_type=pb.WSM4_ROAD_MARKING_LEFT)
    crosswalk_primitive = surface_band_primitive(crosswalk)
    assert crosswalk_primitive is not None
    assert WSM4_PALETTE[crosswalk_primitive] == (139, 93, 255, 255)
    assert surface_band_primitive(stop_area) == PRIM_TYPE_COUNT + 10
    with pytest.raises(NotImplementedError, match="arrow glyph"):
        surface_band_primitive(arrow)
    with pytest.raises(ValueError, match="unknown WSM4 road-marking type"):
        surface_band_primitive(pb.SurfaceBand(wsm4_marking_type=999))

    omitted = unsupported_surface_bands(
        [
            pb.SurfaceBand(semantic=pb.SEMANTIC_ID_ROAD_SURFACE),
            pb.SurfaceBand(semantic=pb.SEMANTIC_ID_ROAD_SURFACE),
            crosswalk,
            stop_area,
        ]
    )
    assert omitted == {pb.SEMANTIC_ID_ROAD_SURFACE: 2}


def test_wsm4_marking_polygon_rejects_malformed_edges():
    band = pb.SurfaceBand(wsm4_marking_type=pb.WSM4_ROAD_MARKING_STOP)
    band.left_world_m.add(x=0, y=0, z=0)
    with pytest.raises(ValueError, match="matching left/right edge points"):
        _surface_polygon_pools(0, [band], np.zeros(3), "cpu")


def test_wsm4_rejects_nonconforming_actor_and_crosswalk_proxies():
    with pytest.raises(ValueError, match="no ego actor category"):
        box_primitive(pb.Box(semantic=pb.SEMANTIC_ID_EGO))
    with pytest.raises(ValueError, match="polygon geometry"):
        box_primitive(pb.Box(semantic=pb.SEMANTIC_ID_CROSSWALK))


def test_wsm4_wire_additions_preserve_existing_fields():
    strip = pb.Strip.DESCRIPTOR.fields_by_name
    band = pb.SurfaceBand.DESCRIPTOR.fields_by_name
    box = pb.Box.DESCRIPTOR.fields_by_name
    mesh = pb.MeshInstance.DESCRIPTOR.fields_by_name
    assert strip["instance_id"].number == 6
    assert strip["wsm4_boundary_type"].number == 7
    assert band["instance_id"].number == 4
    assert band["wsm4_marking_type"].number == 5
    assert box["wireframe_overlay"].number == 10
    assert box["wsm4_sign_class"].number == 11
    assert mesh["instance_id"].number == 5
    assert mesh["wsm4_signal_user"].number == 9
