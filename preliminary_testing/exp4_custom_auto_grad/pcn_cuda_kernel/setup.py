import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='pcn_kernels',
    version='1.0',
    author='Charles Hetterich',
    author_email='hetterich.charles@gmail.com',
    description='pcn_kernels',
    long_description='pcn_kernels',
    ext_modules=[
        CUDAExtension(
            name='pcn_kernels',
            sources=sources,
            include_dirs=include_dirs,
            # extra_compile_args={'cxx': ['-02'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)