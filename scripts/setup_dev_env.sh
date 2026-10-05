#!/usr/bin/env bash
# Recreate a development environment for pyfcfc (Debian/Ubuntu-flavoured).
#
#   bash scripts/setup_dev_env.sh [VENV_DIR]
#
# Installs the system build dependencies (gcc, make, python headers),
# creates a virtual environment with the Python dependencies (including
# the optional pycorr/Corrfunc extras and Jupyter for the notebooks),
# and installs pyfcfc itself in editable mode.
set -euo pipefail

VENV="${1:-$HOME/.venvs/pyfcfc}"

echo ">> system packages (gcc, make, python3-dev, python3-venv)"
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq gcc make python3-dev python3-venv
else
    echo "   apt-get not found: please install a C compiler, make and the"
    echo "   Python development headers manually."
fi

echo ">> virtual environment at ${VENV}"
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip wheel setuptools

echo ">> python dependencies"
"${VENV}/bin/pip" install numpy scipy cython matplotlib pytest

echo ">> optional extras: pycorr + Corrfunc (comparison/interop utilities)"
"${VENV}/bin/pip" install "git+https://github.com/cosmodesi/pycorr.git" || \
    echo "   (pycorr install failed: interop tests/examples will be skipped)"
if [ -d "${VENV}/lib" ] && "${VENV}/bin/python" -c "import pycorr" 2>/dev/null; then
    "${VENV}/bin/pip" install "git+https://github.com/cosmodesi/Corrfunc@desi" || \
        echo "   (Corrfunc install failed: pycorr engine unavailable)"
fi

echo ">> optional extras: jupyter (to run/refresh the notebooks)"
"${VENV}/bin/pip" install nbconvert ipykernel || true

echo ">> pyfcfc (editable, without build isolation)"
"${VENV}/bin/pip" install -e . --no-build-isolation

cat <<EOF

Done. Activate with:

    source ${VENV}/bin/activate

Useful next steps:

    python -m pytest tests -v                  # test suite (61 tests)
    python examples/example_features_tour.py  # guided feature tour
    make examples                              # run all examples
    PYFCFC_WITH_SIMD=1 pip install -e . --no-build-isolation   # AVX kernels
EOF
