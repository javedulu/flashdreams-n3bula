# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cosmos3_n3bula.stream.stage_clock import _StageClock

pytestmark = pytest.mark.ci_cpu


def test_enabled_stage_clock_synchronizes_and_records(monkeypatch):
    synchronized_devices = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: synchronized_devices.append(device),
    )
    clock = _StageClock(torch.device("cpu"), enabled=True)

    assert clock.wrap("render", lambda: "complete")() == "complete"
    assert synchronized_devices == [torch.device("cpu"), torch.device("cpu")]
    assert clock.take()["render"] >= 0
    assert clock.take() == {}
