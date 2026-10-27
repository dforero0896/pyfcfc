"""Optional JAX integration test for the soft histogramming VJP.

Skipped when jax is not installed.  Checks that a `jax.custom_vjp`
primitive built on ``py_compute_cf`` (forward) and ``soft_hist_vjp``
(adjoint) produces gradients identical to finite differences, and that
the chain rule through a JAX-side parameterisation matches as well.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pyfcfc.boxes import py_compute_cf, soft_hist_vjp  # noqa: E402

BOX = 400.0
S_EDGES = np.arange(0, 191, 10, dtype=np.float64)
SIGMA = 4.0


def _soft_dd_np(x, w):
    res = py_compute_cf([np.asarray(x, np.float64)], [np.asarray(w, np.float64)],
                        S_EDGES, None, 1, label=['A'], bin=0, pair=['AA'],
                        soft_hist=True, soft_sigma=SIGMA, box=BOX)
    return res['pairs']['AA'] * res['normalization']['AA']


@jax.custom_vjp
def soft_dd(x, w):
    return jnp.asarray(_soft_dd_np(x, w))


def _fwd(x, w):
    return soft_dd(x, w), (x, w)


def _bwd(res, g):
    x, w = res
    out = soft_hist_vjp([np.asarray(x, np.float64)],
                        [np.asarray(w, np.float64)], S_EDGES, None, 1,
                        pair='AA', cotangent=np.asarray(g, np.float64),
                        soft_sigma=SIGMA, bin=0, box=BOX)
    return jnp.asarray(out['grads']['A']), jnp.zeros_like(w)


soft_dd.defvjp(_fwd, _bwd)


def test_jax_custom_vjp_grad():
    rng = np.random.default_rng(11)
    x = rng.uniform(150, 250, (50, 3))
    w = np.ones(len(x))
    g = jnp.asarray(rng.normal(size=len(S_EDGES) - 1))

    def loss(x):
        return jnp.dot(g, soft_dd(x, w))

    gx = np.asarray(jax.grad(loss)(jnp.asarray(x)))

    eps = 1e-6
    v = rng.normal(size=x.shape)
    fd = (float(loss(jnp.asarray(x + eps * v))) -
          float(loss(jnp.asarray(x - eps * v)))) / (2 * eps)
    an = float(np.sum(gx * v))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))


def test_jax_chain_rule_through_parameterisation():
    rng = np.random.default_rng(12)
    p = rng.uniform(150, 250, (50, 3))
    w = np.ones(len(p))
    c = np.array([200., 200., 200.])
    g = jnp.asarray(rng.normal(size=len(S_EDGES) - 1))

    def loss(theta):
        x = c + (p - c) * theta          # scalar stretch, JAX-side
        return jnp.dot(g, soft_dd(jnp.asarray(x), w))

    theta0 = jnp.float64(1.0)
    an = float(jax.grad(loss)(theta0))
    eps = 1e-6
    fd = (float(loss(theta0 + eps)) - float(loss(theta0 - eps))) / (2 * eps)
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))


# ----------------------------------------------------------------------
# the optional pyfcfc.jax module: cached custom_vjp primitive, gradients
# in positions and weights with the scale-free kernel (soft_sigma_log)
# ----------------------------------------------------------------------

from pyfcfc import jax as pjax  # noqa: E402

LOG_EDGES = np.geomspace(2.0, 120.0, 13)


def _log_kwargs(**over):
    kw = dict(module="boxes", s_edges=LOG_EDGES, pairs=("AB",),
              labels=["A", "B"], soft_sigma_log=True, bin=0, box=BOX)
    kw.update(over)
    return kw


def test_pjax_primitive_cache_reuse():
    rng = np.random.default_rng(21)
    x = rng.uniform(150, 250, (60, 3))
    y = rng.uniform(150, 250, (50, 3))
    w = np.ones(len(x))
    wr = np.ones(len(y))
    n0 = len(pjax._PRIM_CACHE)

    c1 = pjax.soft_pair_counts([x, y], [w, wr], **_log_kwargs())
    n1 = len(pjax._PRIM_CACHE)
    assert n1 == n0 + 1

    # repeated forward calls, and a backward pass through the same
    # static configuration, must reuse the cached primitive
    c2 = pjax.soft_pair_counts([x, y], [w, wr], **_log_kwargs())
    g = jnp.asarray(rng.normal(size=c1["AB"].shape))
    jax.grad(lambda p: jnp.sum(
        pjax.soft_pair_counts([p, jnp.asarray(y)],
                              [jnp.asarray(w), jnp.asarray(wr)],
                              **_log_kwargs())["AB"] * g))(jnp.asarray(x))
    assert len(pjax._PRIM_CACHE) == n1
    np.testing.assert_allclose(np.asarray(c1["AB"]), np.asarray(c2["AB"]),
                               rtol=1e-12)

    # a different static configuration adds exactly one new entry
    pjax.soft_pair_counts([x, y], [w, wr],
                          **_log_kwargs(soft_sigma_log=False,
                                        soft_sigma=3.0))
    assert len(pjax._PRIM_CACHE) == n1 + 1


def test_pjax_grad_fd_sigma_log():
    rng = np.random.default_rng(22)
    x = rng.uniform(150, 250, (60, 3))
    y = rng.uniform(150, 250, (50, 3))
    wx = rng.uniform(0.5, 1.5, len(x))
    g = jnp.asarray(rng.normal(size=len(LOG_EDGES) - 1))
    yj = jnp.asarray(y)
    wyj = jnp.ones(len(y))

    def loss_x(p):
        c = pjax.soft_pair_counts([p, yj], [jnp.asarray(wx), wyj],
                                  **_log_kwargs())
        return jnp.sum(c["AB"] * g)

    def loss_w(w):
        c = pjax.soft_pair_counts([jnp.asarray(x), yj], [w, wyj],
                                  **_log_kwargs())
        return jnp.sum(c["AB"] * g)

    eps = 1e-6
    v = rng.normal(size=x.shape)
    an = float(jnp.sum(jax.grad(loss_x)(jnp.asarray(x)) * v))
    fd = (float(loss_x(jnp.asarray(x + eps * v))) -
          float(loss_x(jnp.asarray(x - eps * v)))) / (2 * eps)
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))

    u = rng.uniform(0.5, 1.5, len(x))
    an = float(jnp.sum(jax.grad(loss_w)(jnp.asarray(wx)) * u))
    fd = (float(loss_w(jnp.asarray(wx + eps * u))) -
          float(loss_w(jnp.asarray(wx - eps * u)))) / (2 * eps)
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))
