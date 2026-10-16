import os

import torch.utils.cpp_extension as cpp_extension


def _prebuild_ludus() -> None:
    cuda_arch_list = os.environ["LUDUS_RENDERER_CUDA_ARCH_LIST"]
    original_load = cpp_extension.load

    def load_with_cuda_arch(*args, **kwargs):
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
        return original_load(*args, **kwargs)

    cpp_extension.load = load_with_cuda_arch
    try:
        from ludus_renderer._ops._plugin import _get_plugin

        _get_plugin()
    finally:
        cpp_extension.load = original_load


def main() -> None:
    from omnidreams.impl.native import omnidreams_singleview as native

    extension = native.load_extension(verbose=True)
    assert extension is not None, native.extension_load_error()
    _prebuild_ludus()


if __name__ == "__main__":
    main()