from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='adder_cuda',
    ext_modules=[
        CUDAExtension('adder_cuda', [
            'adder_cuda.cpp',
            'adder.cu',
        ],extra_compile_args=['-O3'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })