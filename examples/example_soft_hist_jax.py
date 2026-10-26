"""Differentiable pair counting: pyfcfc soft histogramming inside JAX.

The soft histogramming method (``soft_hist=True``) is smooth in the
catalogue coordinates, and pyfcfc ships its reverse-mode derivative
(``soft_hist_vjp``).  This example shows the intended way to plug it into
an autodiff engine: a ``jax.custom_vjp`` primitive whose forward pass
calls the C pair counter and whose backward pass calls the C adjoint.

    @jax.custom_vjp
    def soft_dd(x, w): ...          # raw soft-histogram DD counts

With that primitive registered, *any* JAX transformation works on top of
the C code: ``jax.grad``, ``jax.value_and_grad``, composition with JAX
side computations, etc.  Two demonstrations are run here:

1. **Adjoint validation**: the gradient of a random linear functional of
   the pair counts, obtained with ``jax.grad`` through the custom VJP,
   against central finite differences of the forward pass.

2. **Differentiable model fitting**: an Alcock-Paczyński-like test.  The
   catalogue is anisotropically stretched, ``x = c + (p - c) * (a, a, b)``,
   and the two stretch parameters ``(a, b)`` are recovered by gradient
   descent (Adam) on the mismatch between the soft DD histogram of the
   stretched catalogue and that of the true one, using ``jax.grad``
   through the custom VJP for the parameter gradients.

It saves ``examples/figures/soft_hist_jax.png``.  Requires ``jax``
(``pip install jax``); run from the root of the repository:

    python examples/example_soft_hist_jax.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from mocks import make_clustered_box  # noqa: E402
from pyfcfc.boxes import py_compute_cf, soft_hist_vjp  # noqa: E402

try:
    import jax
    import jax.numpy as jnp
except ImportError:                                   # pragma: no cover
    print("this example requires jax:  pip install jax")
    sys.exit(1)

# The C library and the adjoint are double precision: enable x64 in JAX.
jax.config.update("jax_enable_x64", True)

BOX = 400.0
S_EDGES = np.arange(0, 191, 10, dtype=np.float64)     # smax < box / 2
SIGMA = 4.0
NS = len(S_EDGES) - 1
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ============================================================================
#  The differentiable primitive: soft-histogram DD counts in JAX
# ============================================================================

def soft_dd_np(x, w, s_edges=S_EDGES, sigma=SIGMA, box=BOX):
    """Forward pass (NumPy / C): raw soft-histogram DD pair counts."""
    res = py_compute_cf([np.asarray(x, dtype=np.float64)],
                        [np.asarray(w, dtype=np.float64)],
                        s_edges, None, 1, label=['A'], bin=0, pair=['AA'],
                        soft_hist=True, soft_sigma=sigma, box=box)
    return res['pairs']['AA'] * res['normalization']['AA']


@jax.custom_vjp
def soft_dd(x, w):
    """Raw soft-histogram DD pair counts, differentiable in (x, w)."""
    return jnp.asarray(soft_dd_np(x, w))


def _soft_dd_fwd(x, w):
    return soft_dd(x, w), (x, w)


def _soft_dd_bwd(res, g):
    """Backward pass (C adjoint): cotangents on the bins -> (x, w)."""
    x, w = res
    out = soft_hist_vjp([np.asarray(x, dtype=np.float64)],
                        [np.asarray(w, dtype=np.float64)],
                        S_EDGES, None, 1, pair='AA',
                        cotangent=np.asarray(g, dtype=np.float64),
                        soft_sigma=SIGMA, bin=0, box=BOX,
                        grad_wts=True)
    return (jnp.asarray(out['grads']['A']),
            jnp.asarray(out['grad_wts']['A']))


soft_dd.defvjp(_soft_dd_fwd, _soft_dd_bwd)


# ============================================================================
#  Demo 1: the custom VJP against finite differences
# ============================================================================

def validate_adjoint(rng, n=60):
    x = rng.uniform(150, 250, (n, 3))        # all separations < smax:
    w = np.ones(n)                           # the forward is smooth here
    g = jnp.asarray(rng.normal(size=NS))

    def loss(x):
        return jnp.dot(g, soft_dd(x, w))

    gx = jax.grad(loss)(jnp.asarray(x))

    # central finite differences of the same loss, component by component
    fd = np.zeros_like(x)
    eps = 1e-6
    for i in range(n):
        for c in range(3):
            xp = x.copy()
            xp[i, c] += eps
            xm = x.copy()
            xm[i, c] -= eps
            fd[i, c] = (float(loss(jnp.asarray(xp))) -
                        float(loss(jnp.asarray(xm)))) / (2 * eps)
    return x, np.asarray(gx), fd


# ============================================================================
#  Demo 2: gradient descent on an anisotropic stretch (AP-like test)
# ============================================================================

CENTRE = np.array([200., 200., 200.])


def stretch(p, theta):
    """Anisotropic stretch about the cube centre: JAX-differentiable."""
    a, b = theta
    return CENTRE + (p - CENTRE) * jnp.array([a, a, b])


def fit_stretch(p, w, theta0, theta_true, n_iter=200, lr=6e-3):
    target = jnp.asarray(soft_dd_np(stretch(jnp.asarray(p),
                                            jnp.asarray(theta_true)), w))

    def loss(theta):
        cnt = soft_dd(stretch(jnp.asarray(p), theta), w)
        d = cnt - target
        return 0.5 * jnp.dot(d, d)

    grad_fn = jax.value_and_grad(loss)

    # Adam
    theta = jnp.asarray(theta0)
    m = v = jnp.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        val, gr = grad_fn(theta)
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr * gr
        mh = m / (1 - b1 ** step)
        vh = v / (1 - b2 ** step)
        theta = theta - lr * mh / (jnp.sqrt(vh) + eps)
        hist.append((float(val), float(theta[0]), float(theta[1])))
        if step % 15 == 0 or step == 1:
            print(f"  step {step:3d}: loss {float(val):12.4f}   "
                  f"a = {float(theta[0]):.5f}  b = {float(theta[1]):.5f}")
    return np.array(hist), np.asarray(theta), time.perf_counter() - t0


# ============================================================================
#  Main
# ============================================================================

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(7)

    # ---- demo 1 ------------------------------------------------------
    print("validating the custom VJP against finite differences ...")
    x_v, gx, fd = validate_adjoint(rng)
    rel = np.abs(gx - fd) / np.maximum(np.abs(fd), 1e-8 * np.abs(fd).max())
    print(f"  max |jax.grad - FD| / |FD| = {rel.max():.3e}")

    # ---- demo 2 ------------------------------------------------------
    # clustered catalogue, squeezed into the kink-free cube
    p, w = make_clustered_box(700, BOX, seed=42, n_centers=18)
    p = 150.0 + p / BOX * 100.0
    w = np.ones(len(p))
    theta_true = np.array([1.0, 1.0])
    theta0 = np.array([1.06, 0.94])
    print(f"recovering the stretch parameters from theta0 = {theta0} "
          f"(true {theta_true}) ...")
    hist, theta_fit, dt = fit_stretch(p, w, theta0, theta_true)
    print(f"  fit: a = {theta_fit[0]:.5f}, b = {theta_fit[1]:.5f} "
          f"({dt:.1f} s for {len(hist)} Adam steps)")

    cnt_true = soft_dd_np(stretch(jnp.asarray(p), jnp.asarray(theta_true)), w)
    cnt_0 = soft_dd_np(stretch(jnp.asarray(p), jnp.asarray(theta0)), w)
    cnt_f = soft_dd_np(stretch(jnp.asarray(p), theta_fit), w)

    # ---- figure ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    ax = axes[0, 0]
    lim = max(np.abs(gx).max(), np.abs(fd).max())
    ax.plot([-lim, lim], [-lim, lim], ":", color="0.4", label="y = x")
    ax.scatter(fd.ravel(), gx.ravel(), s=7, alpha=0.6,
               color=plt.get_cmap("viridis")(0.3))
    ax.set_xlabel("finite-difference gradient")
    ax.set_ylabel("jax.grad through the custom VJP")
    ax.set_title(f"Adjoint validation (max rel err {rel.max():.1e})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.semilogy(hist[:, 0], lw=1.5,
                color=plt.get_cmap("viridis")(0.3))
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss  $\\frac{1}{2}|DD(\\theta)-DD(\\theta_*)|^2$")
    ax.set_title("Differentiable histogram fitting")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(hist[:, 1], hist[:, 2], "-o", ms=3, lw=1.2,
            color=plt.get_cmap("viridis")(0.3), label="gradient descent")
    ax.plot(theta0[0], theta0[1], "x", ms=10, mew=2.5, color="crimson",
            label="start")
    ax.plot(theta_true[0], theta_true[1], "*", ms=16, color="k",
            label="truth")
    ax.set_xlabel("$a$ ($s_\\perp$ stretch)")
    ax.set_ylabel("$b$ (line-of-sight stretch)")
    ax.set_title("Parameter recovery trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    cen = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    ax.step(np.repeat(S_EDGES, 2)[1:-1], np.repeat(cnt_true, 2), where="mid",
            color="k", lw=1.5, label="target  $\\theta_*$")
    ax.plot(cen, cnt_0, "--", color="crimson", lw=1.3,
            label=f"start ({theta0[0]:.2f}, {theta0[1]:.2f})")
    ax.plot(cen, cnt_f, "-o", ms=3, lw=1.2,
            color=plt.get_cmap("viridis")(0.3), label="fitted")
    ax.set_xlabel("$s$")
    ax.set_ylabel("soft DD pair counts")
    ax.set_title("Soft histograms along the fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "soft_hist_jax.png")
    fig.savefig(out, dpi=140)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
