from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob, os
includes = [numpy.get_include(), '/usr/include', 'etc', 'io', 'lib', 'math', 'src/2pt', 'tree', 'src']

pyfcfc = Extension("pyfcfc",
                  sources=['src/2pt/pyfcfc.pyx'],
                  include_dirs=includes,
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  language='c',
                  extra_compile_args=["-fopenmp", "-DOMP"],
                  extra_link_args=["-fopenmp"]
             )
load_conf = Extension("_load_conf",
                  sources=['src/2pt/_load_conf.pyx'],
                  include_dirs=includes,
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  language='c',
                  extra_compile_args=["-fopenmp", "-DOMP"],
                  extra_link_args=["-fopenmp"]
)
setup(name='pyfcfc',
      ext_modules=cythonize([pyfcfc, load_conf]),
      packages=['pyfcfc'])