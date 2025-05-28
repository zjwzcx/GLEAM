# setup.py
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


pathfinding_2D_cuda = CUDAExtension(
    name='bfs_cuda_2D',
    sources=[
		'gleam/utils/bfs_cuda_2D.cpp',
		'gleam/utils/bfs_cuda_kernel_2D.cu',
	],
)

# Define the setup configuration
# Original author: Nikita Rudin (legged_gym)
setup(
    name='legged_gym',
    version='1.0.0',
    author='Xiao Chen',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='cx123@ie.cuhk.edu.hk',
    description='GLEAM based on Isaac Gym environments',
    install_requires=[
        'isaacgym',
        'gym',
        'matplotlib',
        "tensorboard",
        "cloudpickle",
        "pandas",
        "yapf~=0.30.0",
        "wandb",
        "opencv-python>=3.0.0"
    ],
    ext_modules=[
		pathfinding_2D_cuda
	],
    cmdclass={
        'build_ext': BuildExtension
    },
)
