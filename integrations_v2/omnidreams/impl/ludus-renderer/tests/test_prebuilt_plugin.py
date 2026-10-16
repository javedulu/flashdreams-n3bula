import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch.utils.cpp_extension

from ludus_renderer._ops import _plugin


def test_get_plugin_imports_prebuilt_module_without_jit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_name = "ludus_renderer_plugin"
    extension_dir = tmp_path / "torch_extensions" / plugin_name
    extension_dir.mkdir(parents=True)
    (extension_dir / f"{plugin_name}.so").touch()

    plugin = ModuleType(plugin_name)
    loader = type("Loader", (), {"exec_module": lambda self, module: None})()
    spec = importlib.machinery.ModuleSpec(plugin_name, loader)

    monkeypatch.setattr(_plugin, "_cached_plugin", None)
    monkeypatch.setenv(_plugin._PREBUILT_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda *args: spec,
    )
    monkeypatch.setattr(
        importlib.util,
        "module_from_spec",
        lambda unused_spec: plugin,
    )
    monkeypatch.setitem(sys.modules, plugin_name, plugin)
    monkeypatch.setattr(
        torch.utils.cpp_extension,
        "load",
        lambda **kwargs: pytest.fail("JIT must not run when a prebuilt plugin exists"),
    )

    assert _plugin._get_plugin() is plugin