# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

import hashlib
from types import SimpleNamespace

import pytest
import torch
from cosmos3_n3bula.compile_cache import CompileCache

pytestmark = pytest.mark.ci_cpu


def test_cache_directory_is_scoped_to_torch_and_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA RTX PRO 6000")

    cache = CompileCache(tmp_path, torch.device("cuda:0"))

    assert cache.directory == tmp_path / f"torch-{torch.__version__}-NVIDIA-RTX-PRO-6000"


def test_load_collects_valid_keys_and_skips_stale_files(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test-gpu")
    cache = CompileCache(tmp_path, torch.device("cpu"))
    cache.directory.mkdir()
    (cache.directory / "compile-artifacts-a.bin").write_bytes(b"valid")
    (cache.directory / "compile-artifacts-b.bin").write_bytes(b"stale")
    (cache.directory / "compile-artifacts-c.bin").write_bytes(b"empty")
    (cache.directory / "compile-artifacts-d.tmp").write_bytes(b"ignored")

    def load_artifacts(data):
        if data == b"stale":
            raise ValueError("corrupt cache")
        if data == b"empty":
            return None
        return SimpleNamespace(artifacts={"inductor": ["kernel-a"], "autotune": ["tune-a"]})

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load_artifacts)

    cache.load()

    assert cache._known == {("inductor", "kernel-a"), ("autotune", "tune-a")}
    assert list(cache._files) == [cache.directory / "compile-artifacts-a.bin"]


def test_save_replaces_only_files_fully_covered_by_new_export(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test-gpu")
    cache = CompileCache(tmp_path, torch.device("cpu"))
    cache.directory.mkdir(parents=True)
    covered = cache.directory / "compile-artifacts-covered.bin"
    partial = cache.directory / "compile-artifacts-partial.bin"
    covered.write_bytes(b"covered")
    partial.write_bytes(b"partial")
    loaded = {
        b"covered": SimpleNamespace(artifacts={"inductor": ["kernel-a"]}),
        b"partial": SimpleNamespace(artifacts={"inductor": ["kernel-b"]}),
    }
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda data: loaded[data])
    cache.load()
    exported = b"new artifact payload"
    monkeypatch.setattr(
        torch.compiler,
        "save_cache_artifacts",
        lambda: (
            exported,
            SimpleNamespace(artifacts={"inductor": ["kernel-a", "kernel-c"]}),
        ),
    )

    cache.save()

    output = cache.directory / f"compile-artifacts-{hashlib.sha256(exported).hexdigest()[:16]}.bin"
    assert output.read_bytes() == exported
    assert not covered.exists()
    assert partial.exists()
    assert not list(cache.directory.glob("*.tmp"))


def test_save_skips_non_kernel_artifacts_and_known_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test-gpu")
    cache = CompileCache(tmp_path, torch.device("cpu"))
    cache.directory.mkdir(parents=True)
    cached = cache.directory / "compile-artifacts-known.bin"
    cached.write_bytes(b"known")
    known_info = SimpleNamespace(artifacts={"inductor": ["kernel-a"]})
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda _data: known_info)
    cache.load()

    monkeypatch.setattr(
        torch.compiler,
        "save_cache_artifacts",
        lambda: (b"trace", SimpleNamespace(artifacts={"aot_autograd": ["trace-a"]})),
    )
    cache.save()
    assert sorted(path.name for path in cache.directory.iterdir()) == [cached.name]

    monkeypatch.setattr(
        torch.compiler,
        "save_cache_artifacts",
        lambda: (b"duplicate", known_info),
    )
    cache.save()
    assert sorted(path.name for path in cache.directory.iterdir()) == [cached.name]