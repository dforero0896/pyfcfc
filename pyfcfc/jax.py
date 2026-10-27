"""Optional JAX integration: differentiable soft-histogram pair counts.

This module wraps the pyfcfc soft histogramming forward pass
(``py_compute_cf`` with ``soft_hist=True``) and its reverse-mode adjoint
(``soft_hist_vjp``) into a ``jax.custom_vjp`` primitive, and provides the
small post-processing helpers (estimator, multipoles, projected w_p) with
the exact FCFC conventions, so that everything after the pair counting is
differentiated natively by JAX.

Typical use::

    import jax, jax.numpy as jnp
    from pyfcfc.jax import (soft_pair_counts, pair_normalizations,
                            xi_landy_szalay, multipoles, projected_wp)

    def counts_fn(pos, wts):            # pos/wts: sequence of arrays
        return soft_pair_counts(pos, wts, module="boxes",
                                s_edges=s_edges, pedges=pedges,
                                pairs=("DD", "DR", "RR"), labels="DR"[:2],
                                soft_hist=True, soft_sigma=2.0,
                                soft_mu=True, soft_mu_sigma=0.1,
                                box=box)

    def xi(theta):                      # any JAX-differentiable model
        c = counts_fn(model(theta), wts)
        n = pair_normalizations(wts, ("DD", "DR", "RR"))
        return xi_landy_szalay(c, n)

    grad_fn = jax.grad(lambda th: jnp.sum(g * multipoles(xi(th), [0, 2], nmu)))

``soft_pair_counts`` is differentiable with respect to the catalogue
positions *and* the weights; all configuration (bins, kernel widths,
labels, box, ...) is static.  The backward pass evaluates one
``soft_hist_vjp`` call per requested pair count and accumulates the
cotangents of the involved catalogues.  Both passes run in the C library
via ``jax.pure_callback`` host calls, so the primitive can be evaluated
eagerly or inside ``jax.jit`` (forward *and* gradients); its
``custom_vjp`` object is created once per static configuration and
cached, keeping JAX's tracing/compilation caches effective.

Requires ``jax`` (CPU is fine); double precision is enabled on import
(``jax_enable_x64``), matching the double-precision C kernels.  The
``jax.pure_callback`` vmapping keyword is resolved against the installed
jax (``vmap_method="sequential"`` on jax >= 0.4.31, ``vectorized=False``
on older releases), so the module works across jax versions.

JIT behaviour: one ``custom_vjp`` primitive is created per static
configuration and cached in ``_PRIM_CACHE``; ``jax.jit`` then compiles
one executable per input signature (shape/dtype).  The first jitted call
with a given signature pays the tracing + XLA compilation cost, every
subsequent call reuses the cached executable (no recompilation).
``examples/benchmark_jit_cache.py`` measures first-run vs subsequent-run
timings; ``tests/test_soft_vjp_jax.py`` asserts the same (compilation
cache sizes and timings).
"""

import inspect

import numpy as np

import jax
import jax.numpy as jnp

from . import boxes as _boxes
from . import sky as _sky

jax.config.update("jax_enable_x64", True)

__all__ = ["soft_pair_counts", "pair_normalizations", "xi_landy_szalay",
           "multipoles", "projected_wp", "legpoly"]


# The vmapping control of jax.pure_callback was renamed in jax 0.4.31:
# `vectorized=False` (older) became `vmap_method="sequential"` (newer).
# Both request the same behaviour here - the C kernels are not vmapped,
# jax loops over a batched axis instead - so resolve the spelling once
# against the installed jax (and fall back to the other one on a
# TypeError, in case the signature ever disagrees with the behaviour).
try:
    _PURE_CB_KW = ({"vmap_method": "sequential"}
                   if "vmap_method" in
                   inspect.signature(jax.pure_callback).parameters
                   else {"vectorized": False})
except (TypeError, ValueError):        # not introspectable: assume newest
    _PURE_CB_KW = {"vmap_method": "sequential"}


def _pure_callback(callback, result_shape_dtypes, *args):
    """jax.pure_callback with the vmapping kwarg of the installed jax."""
    global _PURE_CB_KW
    try:
        return jax.pure_callback(callback, result_shape_dtypes, *args,
                                 **_PURE_CB_KW)
    except TypeError:
        _PURE_CB_KW = ({"vectorized": False}
                       if "vmap_method" in _PURE_CB_KW
                       else {"vmap_method": "sequential"})
        return jax.pure_callback(callback, result_shape_dtypes, *args,
                                 **_PURE_CB_KW)


def _mod(module):
    if module in ("boxes", "box", _boxes):
        return _boxes
    if module in ("sky", _sky):
        return _sky
    raise ValueError(f"unknown module {module!r} (use 'boxes' or 'sky')")


# custom_vjp objects are cached per configuration: creating a new one on
# every call would defeat JAX's tracing/compilation caches (which are
# keyed on the callable) for users who wrap their model in jax.jit.  With
# this cache, repeated calls with the same static configuration reuse the
# same primitive, exactly as a module-level @jax.custom_vjp definition
# would (the decorator is only syntactic sugar for jax.custom_vjp(fun)).
_PRIM_CACHE = {}


def _hashable(x):
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, np.ndarray):
        return tuple(x.ravel().tolist())
    return x


def soft_pair_counts(positions, weights, *, module="boxes", s_edges,
                     pedges=None, nmu=1, pairs=("DD",), labels=None,
                     soft_sigma=None, soft_mu=False, soft_mu_sigma=None,
                     soft_pi=False, soft_pi_sigma=None,
                     soft_sigma_log=False, **fcfc_kwargs):
    """Raw soft-histogram pair counts as a JAX-differentiable function.

    Parameters
    ----------
    positions, weights : sequence of array_like
        The catalogues, shapes (N_c, 3) and (N_c,).  These are the
        *differentiable* arguments.
    module : str
        ``'boxes'`` (periodic, FCFC_2PT_BOX) or ``'sky'`` (survey-like,
        FCFC_2PT).
    s_edges, pedges, nmu : array_like / int
        Binning, as in ``py_compute_cf``.
    pairs : sequence of str
        Pair counts to evaluate, e.g. ``('DD', 'DR', 'RR')`` with labels
        taken from ``labels`` (default ``'ABCD...'``).
    soft_* :
        Soft histogramming options, as in ``py_compute_cf``
        (``soft_hist=True`` is implied).
    **fcfc_kwargs :
        Further FCFC options (``box``, ``bin``, ``data_struct``, ...).

    Returns
    -------
    dict
        ``pair -> jnp.ndarray`` of raw (unnormalised) counts, shape
        (ns,), (ns, nmu) or (ns, np).  Gradients flow to `positions`
        and `weights` through the C adjoint.

    Notes
    -----
    The ``custom_vjp`` primitive is created once per static configuration
    (module, bins, kernel options, labels, pairs, extra FCFC options) and
    cached in ``_PRIM_CACHE``, so repeated calls - also inside
    ``jax.jit``-compiled outer functions - reuse the same callable and
    hence JAX's tracing/compilation caches.  The pair counting and its
    adjoint run in the C library on host arrays, registered with XLA via
    ``jax.pure_callback``; the primitive is therefore traceable and works
    eagerly as well as under ``jax.jit``, for both the forward pass and
    ``jax.grad``.  As with any OpenMP-parallel floating-point reduction,
    multi-threaded runs may differ in the last bits (relative ~1e-15)
    between calls, since the dynamic work distribution changes the
    summation order; single-threaded runs are fully deterministic.
    """
    mod = _mod(module)
    pos = [jnp.asarray(p, jnp.float64) for p in positions]
    wts = [jnp.asarray(w, jnp.float64) for w in weights]
    ncat = len(pos)
    if labels is None:
        labels = [chr(ord('A') + i) for i in range(ncat)]
    labels = list(labels)
    pairs = tuple(pairs)
    s_edges = np.asarray(s_edges, dtype=np.float64)
    pedges = None if pedges is None else np.asarray(pedges, np.float64)

    opts = dict(fcfc_kwargs)
    opts.update(label=labels, pair=list(pairs), soft_hist=True)
    if soft_sigma is not None:
        opts['soft_sigma'] = soft_sigma
    if soft_mu:
        opts['soft_mu'] = True
        if soft_mu_sigma is not None:
            opts['soft_mu_sigma'] = soft_mu_sigma
    if soft_pi:
        opts['soft_pi'] = True
        if soft_pi_sigma is not None:
            opts['soft_pi_sigma'] = soft_pi_sigma
    if soft_sigma_log:
        opts['soft_sigma_log'] = True

    def _raw(p_list, w_list):
        res = mod.py_compute_cf(p_list, w_list, s_edges, pedges, nmu, **opts)
        nrm = res['normalization']
        return tuple(np.asarray(res['pairs'][p] * nrm[p]) for p in pairs)

    # static output shape of every pair count, set by the binning scheme
    # (bin=0: (ns,); bin=1 (s, mu): (ns, nmu); bin=2 (s_perp, pi): (ns, npi))
    bin_scheme = int(fcfc_kwargs.get('bin', 0))
    if soft_mu and bin_scheme != 1:
        raise ValueError("soft_mu=True requires the (s, mu) binning scheme: "
                         "pass bin=1 and nmu")
    if soft_pi and bin_scheme != 2:
        raise ValueError("soft_pi=True requires the (s_perp, pi) binning "
                         "scheme: pass bin=2 and pedges")
    nb = len(s_edges) - 1
    if bin_scheme == 1:
        out_shape = (nb, int(nmu))
    elif bin_scheme == 2:
        if pedges is None:
            raise ValueError("the (s_perp, pi) binning scheme (bin=2) "
                             "requires pedges")
        out_shape = (nb, len(pedges) - 1)
    else:
        out_shape = (nb,)

    def base(*args):
        # The C library runs on host arrays.  jax.pure_callback exposes it
        # to XLA as a host call, so the primitive is traceable: it can be
        # evaluated eagerly *and* inside jax.jit (forward and gradient).
        def _call(*host_args):
            return _raw([np.asarray(a) for a in host_args[:ncat]],
                        [np.asarray(a) for a in host_args[ncat:]])
        shapes = (jax.ShapeDtypeStruct(out_shape, jnp.float64),) * len(pairs)
        return _pure_callback(_call, shapes, *args)

    vjp_opts = dict(fcfc_kwargs)
    vjp_opts['label'] = labels      # soft_hist_vjp takes the pair separately

    key = _hashable(dict(module=str(module), s_edges=s_edges,
                         pedges=pedges, nmu=int(nmu), pairs=pairs,
                         labels=labels, opts=opts, fcfc=fcfc_kwargs))
    prim = _PRIM_CACHE.get(key)
    if prim is None:
        prim = jax.custom_vjp(base)

        def fwd(*args):
            return base(*args), args

        def bwd(args, g):
            # The whole adjoint accumulation runs in one host callback, so
            # jax.grad of a jax.jit-compiled function works as well (the
            # residual catalogues and the output cotangents are passed as
            # callback arguments).
            shapes = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype)
                           for a in args)

            def _call(*host_args):
                p_list = [np.asarray(a) for a in host_args[:ncat]]
                w_list = [np.asarray(a) for a in host_args[ncat:2 * ncat]]
                g_list = [np.asarray(a) for a in host_args[2 * ncat:]]
                gx = [np.zeros_like(p) for p in p_list]
                gw = [np.zeros_like(w) for w in w_list]
                for pair, cot in zip(pairs, g_list):
                    out = mod.soft_hist_vjp(p_list, w_list, s_edges, pedges,
                                            nmu, pair=pair,
                                            cotangent=cot,
                                            soft_sigma=soft_sigma,
                                            soft_mu=soft_mu,
                                            soft_mu_sigma=soft_mu_sigma,
                                            soft_pi=soft_pi,
                                            soft_pi_sigma=soft_pi_sigma,
                                            soft_sigma_log=soft_sigma_log,
                                            grad_wts=True, **vjp_opts)
                    for idx, lab in enumerate(labels):
                        if lab in out['grads']:
                            gx[idx] = gx[idx] + np.asarray(out['grads'][lab])
                        if out['grad_wts'] and lab in out['grad_wts']:
                            gw[idx] = (gw[idx] +
                                       np.asarray(out['grad_wts'][lab]))
                return tuple(gx + gw)

            return _pure_callback(_call, shapes, *args, *g)

        prim.defvjp(fwd, bwd)
        _PRIM_CACHE[key] = prim
    out = prim(*pos, *wts)
    return {p: o for p, o in zip(pairs, out)}


def pair_normalizations(weights, pairs, labels=None):
    """FCFC pair-count normalizations (ordered-pair convention), in JAX."""
    wts = [jnp.asarray(w, jnp.float64) for w in weights]
    ncat = len(wts)
    if labels is None:
        labels = [chr(ord('A') + i) for i in range(ncat)]
    idx = {lab: i for i, lab in enumerate(labels)}
    out = {}
    for pair in pairs:
        a, b = idx[pair[0]], idx[pair[1]]
        if a == b:
            sw = wts[a].sum()
            out[pair] = sw * sw - jnp.sum(wts[a] ** 2)
        else:
            out[pair] = wts[a].sum() * wts[b].sum()
    return out


def xi_landy_szalay(counts, norms, keys=('DD', 'DR', 'RR')):
    """Landy-Szalay xi from raw counts and norms.

    ``keys`` gives the (auto-data, cross, auto-random) pair-count names,
    e.g. ('DD', 'DR', 'RR') or ('GG', 'GR', 'RR')."""
    k1, k2, k3 = keys
    dd = counts[k1] / norms[k1]
    dr = counts[k2] / norms[k2]
    rr = counts[k3] / norms[k3]
    return (dd - 2 * dr + rr) / rr


def legpoly(ell, x):
    """Legendre polynomials, same expressions as FCFC's legpoly()."""
    x2 = x * x
    if ell == 0:
        return 1.0 + 0.0 * x
    if ell == 1:
        return x
    if ell == 2:
        return 1.5 * x2 - 0.5
    if ell == 3:
        return 2.5 * x * (x2 - 0.6)
    if ell == 4:
        return 4.375 * x2 * x2 - 3.75 * x2 + 0.375
    raise ValueError(f"unsupported multipole order {ell}")


def multipoles(xi, ells, nmu):
    """xi_ell(s_k) = sum_j xi(s_k, mu_j) (2 ell + 1) / nmu P_ell(mu_j),
    with mu_j = (j + 1/2) / nmu (FCFC's integration convention).

    Parameters: xi of shape (ns, nmu); returns (nell, ns)."""
    xi = jnp.asarray(xi)
    mu_c = (jnp.arange(nmu) + 0.5) / nmu
    return jnp.stack([jnp.sum(xi * legpoly(ell, mu_c)[None, :] *
                              (2 * ell + 1) / nmu, axis=1) for ell in ells])


def projected_wp(xi, pi_edges):
    """w_p(s_perp) = 2 * sum_j xi(s_perp, pi_j) * dpi_j (FCFC convention).

    Parameters: xi of shape (ns, np); pi_edges of length np + 1 (input
    units)."""
    dpi = jnp.asarray(np.diff(np.asarray(pi_edges, dtype=np.float64)))
    return 2.0 * jnp.sum(jnp.asarray(xi) * dpi[None, :], axis=1)
