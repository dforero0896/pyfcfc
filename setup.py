"""Build script for the pyfcfc Cython extensions.

Environment variables
---------------------
PYFCFC_WITH_SIMD : set to 1 to compile with SIMD (AVX/AVX2/AVX512,
    auto-detected from the build machine's instruction set) support.
    This implies `-march=native`, so the resulting binaries only run on
    machines with the same (or a compatible) instruction set.
PYFCFC_MARCH_NATIVE : set to 1 to add `-march=native` without SIMD.
PYFCFC_NO_LTO : set to 1 to disable link-time optimisation.
PYFCFC_EXTRA_CFLAGS : extra space-separated compiler flags appended to
    the compilation of the C sources, e.g. "-mno-avx512f" to restrict a
    SIMD build to AVX2, or "-O2" to reduce the memory footprint of the
    compilation.
"""

import glob
import os
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# Use paths relative to the project root (pip runs setup.py from there);
# absolute paths break editable installs.
FCFC_SRC = os.path.join("FCFC-main", "src")

# FCFC components: (python module name, source directory under FCFC-main/src)
COMPONENTS = [
    ("pyfcfc.boxes", "fcfc/2pt_box"),
    ("pyfcfc.sky", "fcfc/2pt"),
]
# Shared FCFC source directories.
COMMON_DIRS = ["io", "lib", "math", "tree", "util"]

# Catalogue file readers of the upstream C program: in library mode the
# catalogues always come from Python, so the FITS/HDF5 readers are dead
# code (their callers in build_tree.c are gone, and they compile to
# stubs without WITH_CFITSIO/WITH_HDF5).  read_ascii.c is kept because
# `read_ascii_table' backs the optional Z_CMVDST_FILE distance table of
# the survey-like component, and write_file.c backs the optional
# PAIR_COUNT_FILE / CF_OUTPUT_FILE / ... outputs.
UNUSED_SOURCES = {
    "read_fits.c", "read_hdf5.c",
}

WITH_SIMD = bool(os.getenv("PYFCFC_WITH_SIMD"))
MARCH_NATIVE = WITH_SIMD or bool(os.getenv("PYFCFC_MARCH_NATIVE"))
NO_LTO = bool(os.getenv("PYFCFC_NO_LTO"))


def get_flags():
    """Return (extra_compile_args, extra_link_args) for the C code."""
    if sys.platform == "darwin":
        omp_compile = ["-Xpreprocessor", "-fopenmp"]
        omp_link = ["-lomp"]
    else:
        omp_compile = ["-fopenmp"]
        omp_link = ["-fopenmp"]
    compile_args = ["-O3"] + omp_compile
    if MARCH_NATIVE:
        compile_args.append("-march=native")
    if not NO_LTO:
        compile_args.append("-flto")
        omp_link.append("-flto")
    extra = os.getenv("PYFCFC_EXTRA_CFLAGS")
    if extra:
        compile_args.extend(extra.split())
    return compile_args, omp_link


def make_extension(name, comp_dir):
    comp_path = os.path.join(FCFC_SRC, comp_dir)
    include_dirs = (
        [comp_path]
        + [os.path.join(FCFC_SRC, d) for d in COMMON_DIRS]
        + [numpy.get_include()]
    )
    # The Cython source; cythonize() turns it into a C source.
    sources = [os.path.join(comp_path, "pyfcfc.pyx")]
    for directory in [comp_path] + [os.path.join(FCFC_SRC, d)
                                    for d in COMMON_DIRS]:
        for cfile in sorted(glob.glob(os.path.join(directory, "*.c"))):
            base = os.path.basename(cfile)
            # Skip stale Cython-generated files; cythonize() regenerates
            # and appends the fresh one.
            if base == "pyfcfc.c":
                continue
            if base in UNUSED_SOURCES:
                continue
            sources.append(cfile)

    define_macros = [("OMP", None)]
    if WITH_SIMD:
        define_macros.append(("WITH_SIMD", None))

    compile_args, link_args = get_flags()
    return Extension(
        name,
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c",
    )


if __name__ == "__main__":
    extensions = [make_extension(name, comp) for name, comp in COMPONENTS]
    setup(
        ext_modules=cythonize(
            extensions,
            compiler_directives={"language_level": "3"},
            annotate=False,
            gdb_debug=False,
        ),
    )
