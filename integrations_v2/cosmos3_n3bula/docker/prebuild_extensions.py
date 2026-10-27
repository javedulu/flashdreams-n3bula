# SPDX-FileCopyrightText: Copyright (c) 2026 Javed Shaik
# SPDX-License-Identifier: Apache-2.0

"""Compile Ludus's CUDA plugin at image build time so the first request does not."""

import os

from torch.utils import cpp_extension


def main() -> None:
    arch = os.environ["LUDUS_RENDERER_CUDA_ARCH_LIST"]
    original = cpp_extension.load

    def load_with_arch(*args, **kwargs):
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        return original(*args, **kwargs)

    cpp_extension.load = load_with_arch
    try:
        from ludus_renderer._ops._plugin import _get_plugin

        _get_plugin()
    finally:
        cpp_extension.load = original


if __name__ == "__main__":
    main()
