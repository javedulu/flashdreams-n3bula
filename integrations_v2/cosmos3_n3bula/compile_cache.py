# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Portable torch.compile artifacts, so a new process or container starts warm.

Inductor and Triton autotune on the GPU, which an image build does not have, so
the compiled kernels cannot be baked into an image. Instead each process loads
every exported artifact file at startup and, after a stream that compiled new
kernels or autotuning results, exports what it holds. Files are content-addressed;
an export removes older files whose keys it fully covers and keeps the rest, so a
later partial export never drops an earlier one. They live under
``<root>/<torch version>-<GPU>``: another PyTorch build or GPU starts cold rather
than loading mismatched kernels.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch
from loguru import logger

DEFAULT_ROOT = Path.home() / ".cache" / "n3bula" / "cosmos3-compile-cache"
_PREFIX = "compile-artifacts-"
# The expensive kinds: generated kernels and autotuning results. An export whose only
# new keys are other kinds (an aot_autograd graph key differs in every process) is
# skipped, or each warm start would write another full-size file.
_WORTH_EXPORTING = frozenset(("inductor", "autotune"))


def _keys(info) -> set[tuple[str, str]]:
    artifacts = getattr(info, "artifacts", None) or {}
    return {(str(kind), str(key)) for kind, keys in artifacts.items() for key in keys}


class CompileCache:
    def __init__(self, root: Path | None, device: torch.device) -> None:
        # Artifact keys already on disk, overall and per file.
        self._known: set[tuple[str, str]] = set()
        self._files: dict[Path, set[tuple[str, str]]] = {}
        if root is None:
            self.directory = None
            return
        gpu = re.sub(r"[^A-Za-z0-9.]+", "-", torch.cuda.get_device_name(device)).strip("-")
        self.directory = Path(root) / f"torch-{torch.__version__}-{gpu}"

    def load(self) -> None:
        """Prime the compile caches from every exported file; call before compiling."""
        if self.directory is None or not self.directory.is_dir():
            return
        loaded = 0
        for path in sorted(self.directory.glob(f"{_PREFIX}*.bin")):
            try:
                info = torch.compiler.load_cache_artifacts(path.read_bytes())
            except Exception as error:  # noqa: BLE001 - a stale file must not stop the server.
                logger.warning("skipping unreadable compile artifacts {}: {}", path.name, error)
                continue
            if info is None:
                logger.warning("skipping unreadable compile artifacts {}", path.name)
                continue
            keys = _keys(info)
            self._files[path] = keys
            self._known |= keys
            loaded += 1
        logger.info(
            "compile cache: loaded {} file(s), {} artifact(s) from {}",
            loaded,
            len(self._known),
            self.directory,
        )

    def save(self) -> None:
        """Export this process's compiled artifacts if they are new."""
        if self.directory is None:
            return
        result = torch.compiler.save_cache_artifacts()
        if result is None:
            return
        data, info = result
        keys = _keys(info)
        fresh = keys - self._known
        if not data or not any(kind in _WORTH_EXPORTING for kind, _ in fresh):
            return
        digest = hashlib.sha256(data).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{_PREFIX}{digest[:16]}.bin"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        self._known |= keys
        # Drop older files the new export fully covers, so the directory does not grow
        # with every new graph shape.
        for older, older_keys in list(self._files.items()):
            if older != path and older_keys <= keys:
                older.unlink(missing_ok=True)
                del self._files[older]
        self._files[path] = keys
        logger.info(
            "compile cache: exported {:.1f} MB ({} new of {} artifacts) to {}",
            len(data) / 1e6,
            len(fresh),
            len(keys),
            path,
        )
