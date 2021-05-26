from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob, os

from numpy.compat import py3k
try:
      os.remove("src/2pt/pyfcfc.c")
except: pass
includes = [numpy.get_include(), '/usr/include', 'etc', 'io', 'lib', 'math', 'src/2pt', 'tree', 'src']
sources = glob.glob(f"src/2pt/*.c") + glob.glob(f"io/*.c") + glob.glob(f"lib/*.c") + glob.glob(f"math/*.c") + glob.glob(f"tree/*.c")
pyfcfc_lc = Extension("pyfcfc.lightcones",
                  sources=['src/2pt/pyfcfc.pyx'] + sources,
                  include_dirs=includes,
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  language='c',
                  extra_compile_args=["-DOMP", "-fopenmp"],
                  extra_link_args=["-fopenmp"]
             )

try:
      os.remove("src/2pt_box/pyfcfc.c")
except: pass
includes = [numpy.get_include(), '/usr/include', 'etc', 'io', 'lib', 'math', 'src/2pt_box', 'tree', 'src']
sources = glob.glob(f"src/2pt_box/*.c") + glob.glob(f"io/*.c") + glob.glob(f"lib/*.c") + glob.glob(f"math/*.c") + glob.glob(f"tree/*.c")
pyfcfc_box = Extension("pyfcfc.boxes",
                  sources=['src/2pt_box/pyfcfc.pyx'] + sources,
                  include_dirs=includes,
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  language='c',
                  extra_compile_args=["-DOMP", "-fopenmp"],
                  extra_link_args=["-fopenmp"]
             )

setup(name='pyfcfc',
      ext_modules=cythonize([pyfcfc_lc, pyfcfc_box], gdb_debug=True),
      packages=['pyfcfc'])