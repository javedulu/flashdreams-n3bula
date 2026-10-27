# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Turn the pushed N3BULA request stream into frames the rollout can pull.

The rollout asks for control frames by index, often a window ahead of what it
generates; the feed reads requests until that index has arrived. Frame ``i`` of
a definition is pixel frame ``i`` of its run.
"""

from __future__ import annotations

from collections.abc import Iterator

SCHEMA_VERSION = 1


class StreamEnded(Exception):
    """The client stopped sending frames for this definition."""


class SceneChanged(StreamEnded):
    """A new definition arrived; it is left on :attr:`FrameFeed.next_definition`."""


class FrameFeed:
    """Frames of one definition, read from the request stream on demand."""

    def __init__(self, requests: Iterator) -> None:
        self._requests = requests
        self._frames: dict[int, object] = {}
        self._received = 0
        self._last_timestamp: int | None = None
        self._ended: StreamEnded | None = None
        self.next_definition = None

    @property
    def received(self) -> int:
        return self._received

    def frame(self, index: int):
        """Frame ``index``, blocking on the client until it has been sent.

        Raises:
            StreamEnded: the stream closed, or a new definition began, first.
            KeyError: the frame was already released.
        """
        while index >= self._received:
            self._pull()
        return self._frames[index]

    def release(self, before: int) -> None:
        """Forget frames before ``before``; they have been answered."""
        for index in [i for i in self._frames if i < before]:
            del self._frames[index]

    def _pull(self) -> None:
        if self._ended is not None:
            raise self._ended
        request = next(self._requests, None)
        if request is None:
            self._ended = StreamEnded("the client closed the stream")
            raise self._ended
        if request.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported N3BULA schema version: {request.schema_version}")
        event = request.WhichOneof("event")
        if event == "definition":
            self.next_definition = request.definition
            self._ended = SceneChanged("a new scene definition began")
            raise self._ended
        if event != "frame":
            raise ValueError("N3BULA stream request has no event")
        frame = request.frame
        if self._last_timestamp is not None and frame.timestamp_us <= self._last_timestamp:
            raise ValueError(
                f"frame timestamps must increase: {frame.timestamp_us} after "
                f"{self._last_timestamp}"
            )
        self._last_timestamp = frame.timestamp_us
        self._frames[self._received] = frame
        self._received += 1


def first_definition(requests: Iterator):
    """The stream's opening definition; frames before it are an error."""
    request = next(requests, None)
    if request is None:
        return None
    if request.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported N3BULA schema version: {request.schema_version}")
    if request.WhichOneof("event") != "definition":
        raise ValueError("N3BULA scene definition must precede frames")
    return request.definition
