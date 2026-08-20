"""Build script for the ``fused_layernorm`` package and its CUDA extension.

Builds one extension module, ``fused_layernorm_cuda``, from ``csrc/bindings.cpp`` and
``csrc/layernorm_cuda_kernel.cu``, plus the pure-Python package ``fused_layernorm``.

Requires a CUDA toolkit (``nvcc``) and a CUDA-enabled PyTorch at build time. Static
project metadata lives in ``pyproject.toml``; the values repeated here mirror it so the
file also works with plain ``python setup.py build_ext --inplace``.

Compiler flags: ``-O3`` only. ``--use_fast_math`` is intentionally NOT passed: it changes
the numerics (flushes denormals to zero, uses approximate division/sqrt and fast intrinsic
transcendental functions). The ``setup.py`` committed together with the August-2025 data
(``git show 4ea14d2:setup.py``) passed no extra flags; the July-2025 build (``2924f53``) used
``--use_fast_math``; which binary produced each committed result file is not recorded. Set
``TORCH_CUDA_ARCH_LIST`` (e.g. ``"8.0"``) to choose target architectures; otherwise PyTorch
detects the local GPU.

Installing: prefer ``pip install --no-build-isolation .`` (or ``python setup.py build_ext
--inplace``). With default build isolation pip builds the extension against a fresh torch
downloaded into the isolated build env (``pyproject.toml`` lists ``torch`` under
``[build-system] requires``), which can differ from the torch you run and then fail at
import time with an ABI / CUDA-version mismatch.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_layernorm",
    version="0.3.0",
    packages=["fused_layernorm"],
    python_requires=">=3.9",
    ext_modules=[
        CUDAExtension(
            name="fused_layernorm_cuda",
            sources=["csrc/bindings.cpp", "csrc/layernorm_cuda_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
