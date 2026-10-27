"""Optional JAX integration test for the soft histogramming VJP.

Skipped when jax is not installed.  Checks that a `jax.custom_vjp`
primitive built on ``py_compute_cf`` (forward) and ``soft_hist_vjp``
(adjoint) produces gradients identical to finite differences, and that
the chain rule through a JAX-side parameterisation matches as well.
The ``pyfcfc.jax`` sections additionally cover ``jax.jit``: the cached
primitive is traceable (its host calls are registered with
``jax.pure_callback``), jitted forwards and gradients match the eager
results, and JAX's compilation caches are reused across repeated calls
(no recompilation for unchanged shapes / static configurations -
verified via cache introspection where available and via first-run vs
subsequent-run timings on any jax version; reproducible numbers:
``python examples/benchmark_jit_cache.py``).
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


# ----------------------------------------------------------------------
# jax.jit: the cached primitive is traceable (its C host calls are
# registered via jax.pure_callback), jitted forwards and gradients match
# the eager results, and JAX's compilation caches are reused
#
# Tolerances below allow last-bit (relative ~1e-15) run-to-run noise:
# with several OpenMP threads the dynamic work distribution changes the
# order of the parallel floating-point reductions inside the C kernels
# (a single thread is fully deterministic).
# ----------------------------------------------------------------------

S_EDGES_2D = np.arange(0, 191, 10, dtype=np.float64)
PI_EDGES = np.arange(0, 81, 10, dtype=np.float64)


def _cache_size(fn):
    """Number of compilations cached by a jax.jit wrapper, or None when
    the (private) introspection API is unavailable in this jax version."""
    sz = getattr(fn, "_cache_size", None)
    if not callable(sz):
        return None
    try:
        return sz()
    except Exception:      # e.g. AttributeError: no cache entries yet
        return None


def test_pjax_jit_forward_matches_eager():
    rng = np.random.default_rng(31)
    x = rng.uniform(150, 250, (60, 3))
    y = rng.uniform(150, 250, (50, 3))
    wx = rng.uniform(0.5, 1.5, len(x))
    wy = np.ones(len(y))
    kw = _log_kwargs(soft_sigma=0.25, pairs=("AB", "BA"))
    n0 = len(pjax._PRIM_CACHE)

    eager = pjax.soft_pair_counts([x, y], [wx, wy], **kw)
    assert len(pjax._PRIM_CACHE) == n0 + 1

    counts_fn = jax.jit(lambda p, q, u, v:
                        pjax.soft_pair_counts([p, q], [u, v], **kw))
    args = tuple(map(jnp.asarray, (x, y, wx, wy)))
    out = counts_fn(*args)
    assert set(out) == {"AB", "BA"}
    for pair in ("AB", "BA"):
        np.testing.assert_allclose(np.asarray(out[pair]),
                                   np.asarray(eager[pair]), rtol=1e-12)

    # repeated jitted calls: stable results (up to the parallel-reduction
    # last-bit noise), no recompilation, and the same cached primitive
    # (no new _PRIM_CACHE entries)
    out2 = counts_fn(*args)
    for pair in ("AB", "BA"):
        np.testing.assert_allclose(np.asarray(out2[pair]),
                                   np.asarray(out[pair]), rtol=1e-12)
    cs = _cache_size(counts_fn)
    assert cs is None or cs == 1
    assert len(pjax._PRIM_CACHE) == n0 + 1


def test_pjax_jit_grad_matches_eager():
    rng = np.random.default_rng(32)
    x = rng.uniform(150, 250, (60, 3))
    y = rng.uniform(150, 250, (50, 3))
    wx = rng.uniform(0.5, 1.5, len(x))
    wy = np.ones(len(y))
    kw = _log_kwargs(soft_sigma=0.26)
    g = jnp.asarray(rng.normal(size=len(LOG_EDGES) - 1))
    Y, WY = jnp.asarray(y), jnp.asarray(wy)

    def loss(p, u):
        c = pjax.soft_pair_counts([p, Y], [u, WY], **kw)["AB"]
        return jnp.sum(c * g)

    X, U = jnp.asarray(x), jnp.asarray(wx)
    eager = jax.grad(loss, argnums=(0, 1))(X, U)
    jit_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
    got = jit_grad(X, U)
    for e, a in zip(eager, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e),
                                   rtol=1e-10, atol=1e-12)

    # grad of a jitted loss agrees as well, and value_and_grad under jit
    # returns both pieces consistently
    got2 = jax.grad(jax.jit(loss), argnums=(0, 1))(X, U)
    for a, b in zip(got, got2):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a),
                                   rtol=1e-10, atol=1e-12)
    v, gv = jax.jit(jax.value_and_grad(loss))(X, U)
    assert np.isclose(float(v), float(loss(X, U)), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gv), np.asarray(got[0]),
                               rtol=1e-10, atol=1e-12)

    # repeated jitted gradient calls do not recompile
    jit_grad(X, U)
    jit_grad(X, U)
    cs = _cache_size(jit_grad)
    assert cs is None or cs == 1

    # finite differences through the *jitted* loss (the random cotangent
    # keeps the loss non-conserved: the plain sum of soft counts is
    # exactly preserved, so its coordinate gradient vanishes)
    eps = 1e-6
    vdir = rng.normal(size=x.shape)
    jl = jax.jit(loss)
    fd = (float(jl(jnp.asarray(x + eps * vdir), U)) -
          float(jl(jnp.asarray(x - eps * vdir), U))) / (2 * eps)
    an = float(jnp.sum(jit_grad(X, U)[0] * jnp.asarray(vdir)))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))


def test_pjax_jit_compilation_cache():
    rng = np.random.default_rng(33)
    x = rng.uniform(150, 250, (60, 3))
    y = rng.uniform(150, 250, (50, 3))
    wx = rng.uniform(0.5, 1.5, len(x))
    wy = np.ones(len(y))
    kw = _log_kwargs(soft_sigma=0.27)
    n0 = len(pjax._PRIM_CACHE)
    g = jnp.asarray(rng.normal(size=len(LOG_EDGES) - 1))

    def counts(p, q, u, v):
        return pjax.soft_pair_counts([p, q], [u, v], **kw)["AB"]

    X, Y, U, W = map(jnp.asarray, (x, y, wx, wy))
    jf = jax.jit(counts)
    jf(X, Y, U, W)
    cs1 = _cache_size(jf)
    if cs1 is None:
        pytest.skip("jit compilation-cache introspection unavailable "
                    "in this jax version")
    # repeated calls with the same signature add no cache entries
    # (relative assertions: the absolute count is a jax-internal detail)
    jf(X, Y, U, W)
    jf(X, Y, U, W)
    assert _cache_size(jf) == cs1

    # a new input shape compiles (at least) one additional executable ...
    y2 = rng.uniform(150, 250, (45, 3))
    Y2, W2 = jnp.asarray(y2), jnp.asarray(np.ones(len(y2)))
    jf(X, Y2, U, W2)
    cs2 = _cache_size(jf)
    assert cs2 > cs1, "a new input shape must trigger a new compilation"
    # ... which repeated calls with that shape then reuse
    jf(X, Y2, U, W2)
    assert _cache_size(jf) == cs2

    # a second jit wrapper around the same static configuration reuses
    # the cached primitive: no new _PRIM_CACHE entry, one compilation
    jf2 = jax.jit(lambda p, q, u, v: counts(p, q, u, v))
    jf2(X, Y, U, W)
    cs3 = _cache_size(jf2)
    assert cs3 is not None
    jf2(X, Y, U, W)
    assert _cache_size(jf2) == cs3
    assert len(pjax._PRIM_CACHE) == n0 + 1

    # the jitted gradient of the same primitive also caches a single
    # compilation across repeated calls
    jg = jax.jit(jax.grad(lambda p: jnp.sum(counts(p, Y, U, W) * g)))
    jg(X)
    cs4 = _cache_size(jg)
    assert cs4 is not None
    jg(X)
    jg(X)
    assert _cache_size(jg) == cs4
    assert len(pjax._PRIM_CACHE) == n0 + 1


def test_pjax_jit_first_vs_subsequent_timing():
    """First jitted call pays tracing + XLA compilation; subsequent calls
    with the same signature reuse the cached executable and must be much
    cheaper.  Timing-based, so it also works on jax versions without
    compilation-cache introspection.  (Reproducible, prettier numbers:
    ``python examples/benchmark_jit_cache.py``.)"""
    import time

    rng = np.random.default_rng(37)
    n = 400
    x = rng.uniform(150, 250, (n, 3))
    y = rng.uniform(150, 250, (n, 3))
    kw = _log_kwargs(soft_sigma=0.29)
    g = jnp.asarray(rng.normal(size=len(LOG_EDGES) - 1))
    X, Y = jnp.asarray(x), jnp.asarray(y)
    U, W = jnp.asarray(np.ones(n)), jnp.asarray(np.ones(n))

    def counts(p):
        return pjax.soft_pair_counts([p, Y], [U, W], **kw)["AB"]

    def time_call(fn, *args, reps=5):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        first = time.perf_counter() - t0
        warm = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            warm.append(time.perf_counter() - t0)
        return first, min(warm)

    jf = jax.jit(counts)
    f_first, f_warm = time_call(jf, X)
    jg = jax.jit(jax.grad(lambda p: jnp.sum(counts(p) * g)))
    g_first, g_warm = time_call(jg, X)

    for tag, first, warm in (("forward", f_first, f_warm),
                             ("grad", g_first, g_warm)):
        if first < 0.05:
            pytest.skip(f"cold jitted {tag} finished in {first * 1e3:.1f} ms"
                        " - too fast to distinguish from a cached run"
                        " (persistent compilation cache?)")
        assert warm < 0.5 * first, (
            f"jitted {tag}: first call {first * 1e3:.1f} ms, subsequent "
            f"calls {warm * 1e3:.1f} ms - expected the cached executable "
            "to be at least 2x faster (no recompilation)")

    # cross-check with the cache introspection when available: exactly
    # one compilation per jitted callable happened above
    csf, csg = _cache_size(jf), _cache_size(jg)
    assert csf is None or csf == 1
    assert csg is None or csg == 1


def _walk_primitives(jaxpr):
    """Primitive names in a jaxpr, descending into nested (closed) jaxprs.

    Depending on the jax version the custom_vjp / pure_callback equations
    may appear at the top level or nested inside wrapper equations, and
    the exact primitive spellings can vary - so collect everything and
    match on substrings.
    """
    names = []
    for eqn in jaxpr.eqns:
        names.append(str(eqn.primitive))
        for v in eqn.params.values():
            sub = getattr(v, "jaxpr", None)      # ClosedJaxpr
            if sub is None and hasattr(v, "eqns"):
                sub = v                          # Jaxpr
            if sub is not None:
                try:
                    names.extend(_walk_primitives(sub))
                except AttributeError:
                    pass
    return names


def test_pjax_jit_traceable():
    # make_jaxpr uses the same tracing machinery as jax.jit: the forward
    # is captured as one custom_vjp call, and differentiating exposes
    # both C host calls (forward + adjoint) as callback equations
    rng = np.random.default_rng(34)
    x = rng.uniform(150, 250, (60, 3))
    y = rng.uniform(150, 250, (50, 3))
    wx = rng.uniform(0.5, 1.5, len(x))
    kw = _log_kwargs(soft_sigma=0.28)
    g = jnp.asarray(rng.normal(size=len(LOG_EDGES) - 1))
    Y, U, W = jnp.asarray(y), jnp.asarray(wx), jnp.asarray(np.ones(len(y)))

    def f(p):
        return pjax.soft_pair_counts([p, Y], [U, W], **kw)["AB"]

    X = jnp.asarray(x)
    names = _walk_primitives(jax.make_jaxpr(f)(X).jaxpr)
    if not names:
        pytest.skip("jaxpr introspection produced no equations in this "
                    "jax version")
    assert any(("callback" in n) or ("custom_vjp" in n) for n in names), \
        f"no custom_vjp/pure_callback equation found in {names}"

    gnames = _walk_primitives(
        jax.make_jaxpr(jax.grad(lambda p: jnp.sum(f(p) * g)))(X).jaxpr)
    ncb = sum("callback" in n for n in gnames)
    assert ncb >= 2, ("expected forward + adjoint host callbacks in the "
                      f"gradient jaxpr, found {ncb} in {gnames}")


def test_pjax_jit_multipole_chain():
    # full (s, mu) pipeline under jit: counts -> Landy-Szalay ->
    # multipoles, gradients w.r.t. positions and weights
    rng = np.random.default_rng(35)
    d = rng.uniform(100, 300, (70, 3))
    r = rng.uniform(100, 300, (90, 3))
    wd = rng.uniform(0.5, 1.5, len(d))
    wr = np.ones(len(r))
    kw = dict(module="boxes", s_edges=S_EDGES_2D, pairs=("DD", "DR", "RR"),
              labels=["D", "R"], soft_sigma=4.1, soft_mu=True,
              soft_mu_sigma=0.08, nmu=4, bin=1, box=BOX)
    R, WR = jnp.asarray(r), jnp.asarray(wr)
    g0 = jnp.asarray(rng.normal(size=len(S_EDGES_2D) - 1))
    g2 = jnp.asarray(rng.normal(size=len(S_EDGES_2D) - 1))

    def loss(p, u):
        c = pjax.soft_pair_counts([p, R], [u, WR], **kw)
        n = pjax.pair_normalizations([u, WR], ("DD", "DR", "RR"),
                                     labels=["D", "R"])
        mp = pjax.multipoles(pjax.xi_landy_szalay(c, n), (0, 2), 4)
        return jnp.sum(mp[0] * g0 + mp[1] * g2)

    D, U = jnp.asarray(d), jnp.asarray(wd)
    eager = jax.grad(loss, argnums=(0, 1))(D, U)
    got = jax.jit(jax.grad(loss, argnums=(0, 1)))(D, U)
    for e, a in zip(eager, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e),
                                   rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(float(jax.jit(loss)(D, U)), float(loss(D, U)),
                               rtol=1e-12)

    # finite differences through the jitted chain
    eps = 1e-6
    v = rng.normal(size=d.shape)
    jl = jax.jit(loss)
    fd = (float(jl(jnp.asarray(d + eps * v), U)) -
          float(jl(jnp.asarray(d - eps * v), U))) / (2 * eps)
    an = float(jnp.sum(got[0] * jnp.asarray(v)))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))


def test_pjax_jit_wp_chain():
    # full (s_perp, pi) pipeline under jit: counts -> Landy-Szalay ->
    # projected w_p, gradient w.r.t. the data positions
    rng = np.random.default_rng(36)
    d = rng.uniform(100, 300, (70, 3))
    r = rng.uniform(100, 300, (90, 3))
    wd = rng.uniform(0.5, 1.5, len(d))
    wr = np.ones(len(r))
    kw = dict(module="boxes", s_edges=S_EDGES_2D, pedges=PI_EDGES,
              pairs=("DD", "DR", "RR"), labels=["D", "R"], soft_sigma=4.2,
              soft_pi=True, soft_pi_sigma=5.0, bin=2, box=BOX)
    R, WR = jnp.asarray(r), jnp.asarray(wr)
    gwp = jnp.asarray(rng.normal(size=len(S_EDGES_2D) - 1))

    def loss(p, u):
        c = pjax.soft_pair_counts([p, R], [u, WR], **kw)
        n = pjax.pair_normalizations([u, WR], ("DD", "DR", "RR"),
                                     labels=["D", "R"])
        wp = pjax.projected_wp(pjax.xi_landy_szalay(c, n), PI_EDGES)
        return jnp.sum(wp * gwp)

    D, U = jnp.asarray(d), jnp.asarray(wd)
    eager = jax.grad(loss, argnums=(0, 1))(D, U)
    got = jax.jit(jax.grad(loss, argnums=(0, 1)))(D, U)
    for e, a in zip(eager, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e),
                                   rtol=1e-10, atol=1e-12)

    eps = 1e-6
    v = rng.normal(size=d.shape)
    jl = jax.jit(loss)
    fd = (float(jl(jnp.asarray(d + eps * v), U)) -
          float(jl(jnp.asarray(d - eps * v), U))) / (2 * eps)
    an = float(jnp.sum(got[0] * jnp.asarray(v)))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * abs(fd))
