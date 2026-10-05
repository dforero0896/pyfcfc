"""
pyfcfc: Python bindings for the Fast Correlation Function Calculator (FCFC).

FCFC (https://github.com/cheng-zhao/FCFC, by Cheng Zhao) is a C toolkit for
computing cosmological two-point correlation functions, using k-d tree or
ball tree dual-tree traversal, with optional OpenMP parallelisation and SIMD
vectorisation.

This package wraps the two FCFC pair-counting components:

- :mod:`pyfcfc.boxes` (FCFC_2PT_BOX): two-point correlation functions for
  periodic simulation boxes, with the line of sight along z;
- :mod:`pyfcfc.sky` (FCFC_2PT): two-point correlation functions for
  survey-like data, optionally converting {RA, Dec, redshift} to comoving
  Cartesian coordinates internally.

Both modules expose a single function, ``py_compute_cf(data_cats, data_wts,
sedges, pedges=None, nmu=1, **kwargs)``, which takes in-memory (NumPy)
catalogues and returns a dictionary with the pair counts, correlation
functions, multipoles and/or projected correlation functions.

The :mod:`pyfcfc.utils` module provides helpers to combine pair counts from
random-catalog splits, integrate correlation functions, and convert results
to the state format of `pycorr <https://github.com/cosmodesi/pycorr>`_.
"""

__version__ = "0.2.0"

__all__ = ["boxes", "sky", "utils", "external", "__version__"]

_BUILD_HINT = (
    "pyfcfc.{name} is a compiled Cython extension that could not be "
    "imported. Please (re)build the package from the root of the pyfcfc "
    "repository with `pip install .` (optionally setting "
    "PYFCFC_WITH_SIMD=1 for SIMD acceleration). Original error: {exc}"
)


def __getattr__(name):
    # Lazy imports of the compiled extensions, so that `pyfcfc.utils`
    # remains usable (and error messages stay helpful) even when the
    # extensions have not been built yet.
    if name in ("boxes", "sky"):
        import importlib
        try:
            return importlib.import_module("." + name, __name__)
        except ImportError as exc:
            raise ImportError(
                _BUILD_HINT.format(name=name, exc=exc)) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from . import utils  # noqa: E402  (pure python, always importable)
from . import external  # noqa: E402  (optional converters; lsstypes is
                        #  imported lazily inside the functions)
