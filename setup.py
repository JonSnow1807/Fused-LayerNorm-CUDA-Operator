from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_layernorm_cuda',
    ext_modules=[
        CUDAExtension('fused_layernorm_cuda', [
            'csrc/bindings.cpp',
            'csrc/layernorm_cuda_kernel.cu',  # Use the file on GitHub
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
