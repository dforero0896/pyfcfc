"""Differentiable projected correlation function w_p with pyfcfc + JAX.

With ``soft_hist=True`` *and* ``soft_pi=True`` the (s_perp, pi) pair counts
are smooth in the catalogue coordinates, so everything built on top of
them is differentiable: this example registers a ``jax.custom_vjp``
primitive for the raw 2D counts (GG, GR, RR) whose backward pass calls
``soft_hist_vjp`` per pair count, and then builds the Landy-Szalay
xi(s_perp, pi) and the projected correlation function

    w_p(s_perp) = 2 * sum_j xi(s_perp, pi_j) * dpi_j      (FCFC convention)

with plain JAX ops, so JAX differentiates those steps natively and the
C adjoint supplies the counts' cotangents.

Demonstrations:

  1. ``jax.grad`` of a linear functional of w_p against finite differences
     of the forward C pipeline.
  2. Recovery of a transverse (perpendicular to the line of sight)
     stretch of the galaxy catalogue from w_p alone - the classic
     projected-statistic observable - by Adam on the JAX-differentiable
     loss 1/2 |w_p(a) - w_p(a_*)|^2.

It saves ``examples/figures/soft_hist_wp.png``.  Requires ``jax``
(``pip install jax``).  Run from the root of the repository:

    python examples/example_soft_hist_wp.py
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

jax.config.update("jax_enable_x64", True)

BOX = 400.0
S_EDGES = np.arange(0, 151, 10, dtype=np.float64)
PI_EDGES = np.arange(0, 91, 10, dtype=np.float64)
NS, NPI = len(S_EDGES) - 1, len(PI_EDGES) - 1
SIGMA_S = 5.0
SIGMA_PI = 5.0
A_TRUE = 0.05                     # transverse stretch to recover
CENTRE = np.array([200., 200., 200.])
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OK_MASK = None          # (ns, np) bool, fixed support mask (set in main)

FWD_KW = dict(label=['G', 'R'], bin=2, pair=['GG', 'GR', 'RR'],
              soft_hist=True, soft_sigma=SIGMA_S, soft_pi=True,
              soft_pi_sigma=SIGMA_PI, box=BOX)


def counts_np(gpos, rpos, wg, wr):
    res = py_compute_cf([gpos, rpos], [wg, wr], S_EDGES, PI_EDGES, 1,
                        **FWD_KW)
    n = res['normalization']
    return tuple(res['pairs'][p] * n[p] for p in ('GG', 'GR', 'RR'))


@jax.custom_vjp
def soft_counts(gpos):
    """Raw soft (s_perp, pi) counts (GG, GR, RR); differentiable in gpos.

    The random catalogue and the weights are closed over (constant).
    """
    return tuple(jnp.asarray(c) for c in counts_np(gpos, RAND, WG, WR))


def _fwd(gpos):
    return soft_counts(gpos), gpos


def _bwd(gpos, g):
    gg = np.zeros_like(np.asarray(gpos))
    for pair, cot in zip(('GG', 'GR', 'RR'), g):
        out = soft_hist_vjp([np.asarray(gpos), RAND], [WG, WR], S_EDGES,
                            PI_EDGES, 1, pair=pair,
                            cotangent=np.asarray(cot), soft_sigma=SIGMA_S,
                            soft_pi=True, soft_pi_sigma=SIGMA_PI, bin=2,
                            box=BOX, label=['G', 'R'])
        if 'G' in out['grads']:
            gg = gg + out['grads']['G']   # R cotangents not needed here
    return (jnp.asarray(gg),)


soft_counts.defvjp(_fwd, _bwd)


def wp_of(gpos):
    """Landy-Szalay xi(s_perp, pi) and w_p(s_perp), in pure JAX."""
    ngg, ngr, nrr = soft_counts(gpos)
    nrm_g = WG.sum() ** 2 - (WG ** 2).sum()
    nrm_r = WR.sum() ** 2 - (WR ** 2).sum()
    nrm_x = WG.sum() * WR.sum()
    # FIXED mask (set in main) of cells without random-pair support
    nrr_n = nrr / nrm_r
    xi = (ngg / nrm_g - 2 * ngr / nrm_x + nrr / nrm_r) / \
        jnp.where(OK_MASK, nrr_n, 1.0) * OK_MASK
    dpi = jnp.asarray(np.diff(PI_EDGES))
    return 2.0 * jnp.sum(xi * dpi[None, :], axis=1)


def stretch(pos, a):
    """Transverse stretch about the cube centre (LOS untouched)."""
    return CENTRE + (pos - CENTRE) * jnp.array([1 + a, 1 + a, 1.0])


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    global RAND, WG, WR
    rng = np.random.default_rng(77)
    g, _ = make_clustered_box(900, BOX, seed=21, n_centers=10)
    g = 150.0 + g / BOX * 100.0
    RAND, _ = make_clustered_box(900, BOX, seed=22, n_centers=10)
    RAND = 150.0 + RAND / BOX * 100.0
    WG = rng.uniform(0.8, 1.2, len(g))
    WR = rng.uniform(0.8, 1.2, len(RAND))
    g_j = jnp.asarray(g)

    # fixed support mask from the trajectory endpoints
    global OK_MASK
    nrm_r = WR.sum() ** 2 - (WR ** 2).sum()
    ok = (counts_np(np.asarray(g_j), RAND, WG, WR)[2] / nrm_r > 1e-4)
    ok &= (counts_np(np.asarray(stretch(g_j, jnp.float64(A_TRUE))), RAND,
                     WG, WR)[2] / nrm_r > 1e-4)
    OK_MASK = jnp.asarray(ok)
    print(f"support mask: {int(ok.sum())}/{ok.size} (s_perp, pi) cells "
          f"retained")

    # ------------------------------------------------------------------
    # 1. gradient check of a w_p functional
    # ------------------------------------------------------------------
    print("checking jax.grad of a w_p functional vs FD ...")
    G = jnp.asarray(rng.normal(size=NS))

    def lin(a):
        return jnp.sum(G * wp_of(stretch(g_j, a)))

    a0 = jnp.float64(0.02)
    grad = float(jax.grad(lin)(a0))
    eps = 1e-6
    fd = (float(lin(a0 + eps)) - float(lin(a0 - eps))) / (2 * eps)
    print(f"  adjoint {grad:.10e}  FD {fd:.10e}  rel {abs(grad-fd)/abs(fd):.2e}")

    # ------------------------------------------------------------------
    # 2. recover the transverse stretch from w_p
    # ------------------------------------------------------------------
    print(f"recovering transverse stretch a (true {A_TRUE}) from w_p ...")

    def loss(a):
        d = wp_of(stretch(g_j, a)) - wp_of(stretch(g_j, jnp.float64(A_TRUE)))
        return 0.5 * jnp.sum(d * d)

    a = jnp.float64(0.0)
    m = v = 0.0
    n_iter = 90
    hist = []
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        val, gr = jax.value_and_grad(loss)(a)
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        mh = m / (1 - 0.9 ** step)
        vh = v / (1 - 0.999 ** step)
        lr = 3e-3 * (1 - step / n_iter) ** 2 + 1e-5
        a = a - lr * mh / (np.sqrt(vh) + 1e-8)
        hist.append((float(val), float(a)))
        if step % 40 == 0 or step == 1:
            print(f"  step {step:3d}: loss {float(val):12.5f}  a = {float(a):.5f}")
    dt = time.perf_counter() - t0
    hist = np.array(hist)
    print(f"  fit a = {hist[-1, 1]:.5f} (true {A_TRUE})  [{dt:.0f} s]")

    wp_t = np.asarray(wp_of(stretch(g_j, jnp.float64(A_TRUE))))
    wp_0 = np.asarray(wp_of(g_j))
    wp_f = np.asarray(wp_of(stretch(g_j, jnp.float64(hist[-1, 1]))))

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    cen = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    vir = plt.get_cmap("viridis")

    ax = axes[0, 0]
    ax.bar([0, 1], [fd, grad], 0.6, color=[vir(0.25), vir(0.75)])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["finite differences", "jax.grad + soft_hist_vjp"])
    ax.set_ylabel("$d\\,G\\cdot w_p / da$")
    ax.set_title(f"w_p gradient check (rel err {abs(grad-fd)/abs(fd):.1e})")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    ax.semilogy(hist[:, 0], lw=1.5, color=vir(0.25))
    ax.set_xlabel("Adam step")
    ax.set_ylabel("$\\frac{1}{2}|w_p(a)-w_p(a_*)|^2$")
    ax.set_title("Fitting the transverse stretch from $w_p$")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(hist[:, 1], lw=1.5, color=vir(0.25), label="gradient descent")
    ax.axhline(A_TRUE, ls=":", color="k", label="truth")
    ax.axhline(0.0, ls=":", color="crimson", label="start")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("$\\alpha_\\perp - 1$")
    ax.set_title("Recovered transverse stretch")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(cen, wp_t, "k-", lw=1.6, label="target $w_p$")
    ax.plot(cen, wp_0, "--", color="crimson", lw=1.2, label="start ($a=0$)")
    ax.plot(cen, wp_f, "-o", ms=4, lw=1.2, color=vir(0.75), label="fitted")
    ax.set_xlabel("$s_\\perp$ [Mpc/h]")
    ax.set_ylabel("$w_p(s_\\perp)$")
    ax.set_title("Projected correlation function along the fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "soft_hist_wp.png")
    fig.savefig(out_png, dpi=140)
    print(f"figure saved to {out_png}")


if __name__ == "__main__":
    main()
