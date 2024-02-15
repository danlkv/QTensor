from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

cuSZp_install = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuSZp')
cuSZp_include = os.path.join(cuSZp_install, 'include')
cuSZp_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuSZp', 'src')
# Retrieve list of source files
cuSZp_src_files = []
for root, dirs, files in os.walk(cuSZp_src):
    for file in files:
        if file.endswith('.cu'):
            cuSZp_src_files.append(os.path.join(root, file))
cuSZp_src_files.append('cuSZp_interface.cpp')

# define the extension module
cuSZp_extension = cpp_extension.CUDAExtension(
    name='cuszp',
    sources=cuSZp_src_files,
    include_dirs=[cuSZp_include],
)

# build the extension module
setup(
    name='cuszp',
    ext_modules=[cuSZp_extension],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
