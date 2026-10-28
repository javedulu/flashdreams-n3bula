# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable

import torch


class _StageClock:
    """CUDA-synchronized wall time per named stage, summed over one chunk."""

    def __init__(self, device, enabled: bool) -> None:
        self._device = device
        self.enabled = enabled
        self.seconds: dict[str, float] = {}

    def wrap(self, name: str, function: Callable) -> Callable:
        if not self.enabled:
            return function

        def timed(*args, **kwargs):
            torch.cuda.synchronize(self._device)
            started = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize(self._device)
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
            return result

        return timed

    def take(self) -> dict[str, float]:
        seconds, self.seconds = self.seconds, {}
        return seconds
