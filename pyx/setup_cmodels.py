from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles = ['cmodels.pyx', '../cpp/gr4h.cc', '../cpp/delay.cc', '../cpp/model.cc']
ext_modules = [Extension('cmodels', sourcefiles, language = 'c++', include_dirs = ['../cpp/', numpy.get_include()], extra_compile_args=['-g', '-std=gnu++11'], extra_link_args=['-g'],)]

setup(
    name = 'cmodels',
    ext_modules = cythonize(ext_modules)
)
