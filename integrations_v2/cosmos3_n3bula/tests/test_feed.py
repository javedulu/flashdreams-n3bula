# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import pytest

from cosmos3_n3bula.proto import n3bula_pb2 as pb
from cosmos3_n3bula.stream.feed import (
    FrameFeed,
    SceneChanged,
    StreamEnded,
    first_definition,
)

pytestmark = pytest.mark.ci_cpu


def frame(timestamp_us: int) -> pb.SceneStreamRequest:
    return pb.SceneStreamRequest(
        schema_version=1, frame=pb.SceneFrame(source_frame_id=timestamp_us, timestamp_us=timestamp_us)
    )


def define(run_id: str) -> pb.SceneStreamRequest:
    return pb.SceneStreamRequest(schema_version=1, definition=pb.SceneDefinition(run_id=run_id))


def test_frames_are_pulled_lazily_and_in_order():
    requests = iter([frame(10), frame(20), frame(30)])
    feed = FrameFeed(requests)
    assert feed.frame(1).timestamp_us == 20
    assert feed.received == 2
    assert feed.frame(0).timestamp_us == 10
    assert next(requests).frame.timestamp_us == 30


def test_close_ends_the_feed_every_time():
    feed = FrameFeed(iter([frame(10)]))
    with pytest.raises(StreamEnded):
        feed.frame(1)
    with pytest.raises(StreamEnded):
        feed.frame(1)
    assert feed.next_definition is None


def test_new_definition_ends_the_scene_and_is_kept():
    feed = FrameFeed(iter([frame(10), define("second"), frame(5)]))
    with pytest.raises(SceneChanged):
        feed.frame(3)
    assert feed.next_definition.run_id == "second"


def test_release_forgets_answered_frames():
    feed = FrameFeed(iter([frame(10), frame(20)]))
    feed.frame(1)
    feed.release(1)
    with pytest.raises(KeyError):
        feed.frame(0)
    assert feed.frame(1).timestamp_us == 20


@pytest.mark.parametrize(
    "requests, message",
    [
        ([frame(20), frame(20)], "must increase"),
        ([pb.SceneStreamRequest(schema_version=2, frame=pb.SceneFrame())], "schema version"),
        ([pb.SceneStreamRequest(schema_version=1)], "no event"),
    ],
)
def test_bad_frames_are_refused(requests, message):
    with pytest.raises(ValueError, match=message):
        FrameFeed(iter(requests)).frame(len(requests) - 1)


def test_first_definition():
    assert first_definition(iter([define("one")])).run_id == "one"
    assert first_definition(iter([])) is None
    with pytest.raises(ValueError, match="must precede"):
        first_definition(iter([frame(10)]))
