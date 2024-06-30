#from distutils.extension import Extension
from setuptools import setup, find_packages, Extension
#from Cython.Distutils.extension import Extension
from Cython.Distutils.build_ext import new_build_ext
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob, os
from Cython.Compiler.Main import default_options

from numpy.compat import py3k


fcfc_prefix = "FCFC-main"
default_includes = [numpy.get_include(), '/usr/include']
fcfc_include_dirs = ["io", "lib", "math", "tree", "util"]
fcfc_include_dirs_2pt_box = ["fcfc/2pt_box"]
fcfc_include_dirs_2pt = ["fcfc/2pt"]
extra_compile_args = ["-fopenmp", "-march=native", "-O3", "-flto"]
cython_compile_time_env = {}
define_macros = [("OMP", None)]

if os.getenv('PYFCFC_WITH_SIMD') is not None:
      define_macros += [("WITH_SIMD", None)]
      cython_compile_time_env["WITH_SIMD"] = 1
else:
      cython_compile_time_env["WITH_SIMD"] = 0
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
                  extra_compile_args=extra_compile_args,
                  extra_link_args=["-fopenmp"],
                  define_macros = define_macros,
                  cython_compile_time_env=cython_compile_time_env,
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
                  extra_compile_args=extra_compile_args,
                  extra_link_args=["-fopenmp"],
                  define_macros = define_macros,
                  cython_compile_time_env=cython_compile_time_env,
             )
print(vars(pyfcfc_box))
setup(name='pyfcfc',
      ext_modules=cythonize([pyfcfc_box, pyfcfc_sky], gdb_debug=False, compile_time_env=cython_compile_time_env),
      packages=['pyfcfc'],
      annotate = False,
      author = "Daniel Forero & Cheng Zhao",
      author_email = "dfforero10@gmail.com"
      )