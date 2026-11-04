"""Precision regression tests for :mod:`pyfcfc.jax`.

JAX's x64 switch is process-global (and ``tests/test_soft_vjp_jax.py``
enables it at import time), so the no-x64 behaviour is exercised in a
subprocess with a clean JAX configuration.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, x64):
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_import_does_not_enable_x64_and_f32_stays_f32():
    """pyfcfc.jax must not force x64, and f32 in -> f32 counts/grads out."""
    proc = _run("""
        import numpy as np
        import jax

        assert jax.config.jax_enable_x64 is False
        import pyfcfc.jax as pjax
        assert jax.config.jax_enable_x64 is False

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 100, (40, 3)).astype(np.float32)
        w = np.ones(40, dtype=np.float32)
        kw = dict(module="boxes", s_edges=np.arange(0, 51, 10),
                  pairs=("AA",), soft_sigma=5.0, bin=0, box=120.0)

        counts = pjax.soft_pair_counts([x], [w], **kw)["AA"]
        assert counts.dtype == np.float32, counts.dtype

        nrm = pjax.pair_normalizations([w], ("AA",))["AA"]
        assert nrm.dtype == np.float32, nrm.dtype

        def loss(px):
            return pjax.soft_pair_counts([px], [w], **kw)["AA"].sum()

        g = jax.grad(loss)(x)
        assert g.dtype == np.float32, g.dtype
        assert np.all(np.isfinite(g))
        print("OK")
    """, x64=False)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_x64_inputs_keep_double_precision():
    """With x64 enabled, f64 catalogues still use the f64 path."""
    proc = _run("""
        import numpy as np
        import jax

        assert jax.config.jax_enable_x64 is True
        import pyfcfc.jax as pjax

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 100, (40, 3)).astype(np.float64)
        w = np.ones(40, dtype=np.float64)
        kw = dict(module="boxes", s_edges=np.arange(0, 51, 10),
                  pairs=("AA",), soft_sigma=5.0, bin=0, box=120.0)

        counts = pjax.soft_pair_counts([x], [w], **kw)["AA"]
        assert counts.dtype == np.float64, counts.dtype

        g = jax.grad(lambda px: pjax.soft_pair_counts(
            [px], [w], **kw)["AA"].sum())(x)
        assert g.dtype == np.float64, g.dtype
        print("OK")
    """, x64=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
