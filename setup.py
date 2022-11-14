from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob, os

from numpy.compat import py3k


fcfc_prefix = "FCFC-main"
default_includes = [numpy.get_include(), '/usr/include']
fcfc_include_dirs = ["io", "lib", "math", "tree", "util"]
fcfc_include_dirs_2pt_box = ["fcfc/2pt_box"]
fcfc_include_dirs_2pt = ["fcfc/2pt"]
fcfc_pyx_box = f"{fcfc_prefix}/src/{fcfc_include_dirs_2pt_box[0]}/pyfcfc.pyx"
fcfc_pyx = f"{fcfc_prefix}/src/{fcfc_include_dirs_2pt[0]}/pyfcfc.pyx"
try:
      os.remove(f"{fcfc_pyx_box.replace('.pyx', '.c')}")
      os.remove(f"{fcfc_pyx.replace('.pyx', '.c')}")
except: pass


includes =  [f"{fcfc_prefix}/src/{d}" for d in fcfc_include_dirs_2pt_box] + \
            [f"{fcfc_prefix}/src/{d}" for d in fcfc_include_dirs]
sources = []
for d in includes:
      sources += glob.glob(f"{d}/*.c")
pyfcfc_box = Extension("pyfcfc.boxes",
                  sources=[fcfc_pyx_box] + sources,
                  include_dirs=includes + default_includes,
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  language='c',
                  extra_compile_args=["-DOMP", "-fopenmp", "-march=native", "-DWITH_SIMD"],
                  extra_link_args=["-fopenmp"]
             )

includes =  [f"{fcfc_prefix}/src/{d}" for d in fcfc_include_dirs_2pt] + \
            [f"{fcfc_prefix}/src/{d}" for d in fcfc_include_dirs]
sources = []
for d in includes:
      sources += glob.glob(f"{d}/*.c")
pyfcfc_sky = Extension("pyfcfc.sky",
                  sources=[fcfc_pyx] + sources,
                  include_dirs=includes + default_includes,
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  language='c',
                  extra_compile_args=["-DOMP", "-fopenmp", "-march=native", "-DWITH_SIMD"],
                  extra_link_args=["-fopenmp"]
             )

setup(name='pyfcfc',
      ext_modules=cythonize([pyfcfc_box, pyfcfc_sky], gdb_debug=True),
      packages=['pyfcfc'],
      annotate = True
      )