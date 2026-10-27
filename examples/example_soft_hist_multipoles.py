"""Differentiable multipoles: soft histogramming in s *and* mu inside JAX.

With ``soft_mu=True`` the mu dimension of the (s, mu) scheme is soft-binned
with the same softmax + Gaussian kernel as the separations, so the full
2D histogram xi(s, mu) - and therefore the multipoles

    xi_ell(s_k) = sum_j xi(s_k, mu_j) (2 ell + 1) / nmu P_ell(mu_j),
    mu_j = (j + 1/2) / nmu          (FCFC's integration convention)

- become smooth functions of the catalogue coordinates.  This example
registers a ``jax.custom_vjp`` primitive for the raw 2D pair counts
(AA, AB, BB) whose backward pass calls ``soft_hist_vjp`` per pair count,
builds the Landy-Szalay estimator and the multipoles with plain JAX ops
(so those steps are differentiated natively), and then:

  1. checks ``jax.grad`` of a multipole loss against finite differences,
  2. recovers the two Alcock-Paczynski parameters (transverse and
     line-of-sight stretches; the only combinations the multipoles about
     the z axis can identify) from xi_0 / xi_2 alone, by Adam on the
     fully differentiable pipeline.

It saves ``examples/figures/soft_hist_multipoles.png``.  Requires ``jax``
(``pip install jax``).  Run from the root of the repository:

    python examples/example_soft_hist_multipoles.py
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
S_EDGES = np.arange(0, 191, 5, dtype=np.float64)
NS = len(S_EDGES) - 1
NMU = 10
SIGMA_S = 3.0
SIGMA_MU = 0.07
POLES = [0, 2]
# Alcock-Paczynski-like anisotropic stretch parameters: the multipoles of
# a rigidly translated periodic-box catalogue carry almost no information
# (GG is translation invariant, GR against uniform randoms is flat), while
# per-axis stretches reshape xi_0 / xi_2 strongly and are identifiable.
THETA_TRUE = np.array([0.04, -0.04])     # (alpha_perp - 1, alpha_par - 1)
CENTRE = np.array([200., 200., 200.])
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OK_MASK = None          # (ns, nmu) bool, fixed support mask (set in main)

FWD_KW = dict(label=['G', 'R'], bin=1, pair=['GG', 'GR', 'RR'],
              soft_hist=True, soft_sigma=SIGMA_S, soft_mu=True,
              soft_mu_sigma=SIGMA_MU, box=BOX)


def counts_np(gpos, rpos, wg, wr):
    """Raw soft 2D counts of the three pair types (forward, C)."""
    res = py_compute_cf([gpos, rpos], [wg, wr], S_EDGES, None, NMU, **FWD_KW)
    n = res['normalization']
    return tuple(res['pairs'][p] * n[p] for p in ('GG', 'GR', 'RR'))


@jax.custom_vjp
def soft_counts(gpos, rpos, wg, wr):
    """(GG, GR, RR) raw soft 2D counts, differentiable in the positions."""
    return tuple(jnp.asarray(c) for c in counts_np(gpos, rpos, wg, wr))


def _fwd(gpos, rpos, wg, wr):
    return soft_counts(gpos, rpos, wg, wr), (gpos, rpos, wg, wr)


def _bwd(res, g):
    gpos, rpos, wg, wr = res
    gg = np.zeros_like(np.asarray(gpos))
    gr = np.zeros_like(np.asarray(rpos))
    for pair, cot in zip(('GG', 'GR', 'RR'), g):
        out = soft_hist_vjp([np.asarray(gpos), np.asarray(rpos)],
                            [np.asarray(wg), np.asarray(wr)], S_EDGES,
                            None, NMU, pair=pair,
                            cotangent=np.asarray(cot), soft_sigma=SIGMA_S,
                            soft_mu=True, soft_mu_sigma=SIGMA_MU, bin=1,
                            box=BOX, label=['G', 'R'])
        for lab, gl in out['grads'].items():
            if lab == 'G':
                gg = gg + gl
            else:
                gr = gr + gl
    return jnp.asarray(gg), jnp.asarray(gr), jnp.zeros_like(wg), \
        jnp.zeros_like(wr)


soft_counts.defvjp(_fwd, _bwd)


def stretch(pos, theta):
    """AP stretch about the cube centre: (alpha_perp, alpha_par)."""
    a, b = theta
    return CENTRE + (pos - CENTRE) * jnp.array([1 + a, 1 + a, 1 + b])


def multipoles(gpos, rpos, wg, wr):
    """Landy-Szalay xi(s, mu) and its multipoles, in pure JAX."""
    ngg, ngr, nrr = soft_counts(gpos, rpos, wg, wr)
    nrm_g = wg.sum() ** 2 - (wg ** 2).sum()
    nrm_r = wr.sum() ** 2 - (wr ** 2).sum()
    nrm_x = wg.sum() * wr.sum()
    # FIXED mask (set in main from the trajectory endpoints) of (s, mu)
    # cells without random-pair support: they are kinematically forbidden
    # for a cubical geometry and make the estimator ill-conditioned
    # (pycorr-style ignore_nan).  Fixed, so the loss stays smooth.
    nrr_n = nrr / nrm_r
    xi = (ngg / nrm_g - 2 * ngr / nrm_x + nrr / nrm_r) / \
        jnp.where(OK_MASK, nrr_n, 1.0) * OK_MASK
    mu_c = (jnp.arange(NMU) + 0.5) / NMU
    outs = []
    for ell in POLES:
        pl = jnp.where(ell == 0, 1.0, 1.5 * mu_c ** 2 - 0.5)
        outs.append(jnp.sum(xi * pl[None, :] * (2 * ell + 1) / NMU, axis=1))
    return jnp.stack(outs)          # (nl, ns)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(101)

    g, _ = make_clustered_box(400, BOX, seed=11, n_centers=10)
    g = 150.0 + g / BOX * 100.0
    r, _ = make_clustered_box(400, BOX, seed=12, n_centers=10)
    r = 150.0 + r / BOX * 100.0
    wg = rng.uniform(0.8, 1.2, len(g))
    wr = rng.uniform(0.8, 1.2, len(r))

    # fixed support mask from the trajectory endpoints
    global OK_MASK
    nrm_r = wr.sum() ** 2 - (wr ** 2).sum()
    ok = (counts_np(np.asarray(g), np.asarray(r), wg, wr)[2] / nrm_r > 1e-4)
    ok &= (counts_np(stretch(np.asarray(g), jnp.asarray(THETA_TRUE)),
                     np.asarray(r), wg, wr)[2] / nrm_r > 1e-4)
    OK_MASK = jnp.asarray(ok)
    print(f"support mask: {int(ok.sum())}/{ok.size} (s, mu) cells retained")

    G = jnp.asarray(rng.normal(size=(len(POLES), NS)))

    def loss(theta):
        mp = multipoles(stretch(jnp.asarray(g), theta), jnp.asarray(r),
                        jnp.asarray(wg), jnp.asarray(wr))
        tgt = multipoles(stretch(jnp.asarray(g), jnp.asarray(THETA_TRUE)),
                         jnp.asarray(r), jnp.asarray(wg), jnp.asarray(wr))
        d = mp - tgt
        return 0.5 * jnp.sum(d * d)

    # ------------------------------------------------------------------
    # 1. gradient check of a linear functional of the multipoles
    # ------------------------------------------------------------------
    print("checking jax.grad of a multipole functional vs FD ...")

    def lin(theta):
        mp = multipoles(stretch(jnp.asarray(g), theta), jnp.asarray(r),
                        jnp.asarray(wg), jnp.asarray(wr))
        return jnp.sum(G * mp)

    th = jnp.zeros(2)
    grad = np.asarray(jax.grad(lin)(th))
    eps = 1e-5
    fd = np.array([(float(lin(th + eps * np.eye(2)[a])) -
                    float(lin(th - eps * np.eye(2)[a]))) / (2 * eps)
                   for a in range(2)])
    rel = np.abs(grad - fd) / np.abs(fd)
    print(f"  adjoint-based grad {grad}\n  finite differences {fd}\n"
          f"  max rel err {rel.max():.2e}")

    # ------------------------------------------------------------------
    # 2. recover the displacement from the multipoles
    # ------------------------------------------------------------------
    print(f"recovering AP stretches (true {THETA_TRUE}) from "
          f"xi_0, xi_2 ...")
    theta = jnp.zeros(2)
    m = v = jnp.zeros(2)
    n_iter = 150
    hist = []
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        val, gr = jax.value_and_grad(loss)(theta)
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        mh = m / (1 - 0.9 ** step)
        vh = v / (1 - 0.999 ** step)
        lr = 6e-3 * (1 - step / n_iter) ** 2 + 2e-5
        theta = theta - lr * mh / (jnp.sqrt(vh) + 1e-8)
        hist.append((float(val), *np.asarray(theta)))
        if step % 50 == 0 or step == 1:
            print(f"  step {step:3d}: loss {float(val):12.5f}  "
                  f"a_perp = {theta[0]:7.4f}  a_par = {theta[1]:7.4f}")
    dt = time.perf_counter() - t0
    hist = np.array(hist)
    print(f"  fit: a_perp = {theta[0]:.5f}, a_par = {theta[1]:.5f}  "
          f"[{dt:.0f} s]")

    mp_t = np.asarray(multipoles(stretch(jnp.asarray(g),
                                         jnp.asarray(THETA_TRUE)),
                                 jnp.asarray(r), jnp.asarray(wg),
                                 jnp.asarray(wr)))
    mp_0 = np.asarray(multipoles(jnp.asarray(g), jnp.asarray(r),
                                 jnp.asarray(wg), jnp.asarray(wr)))
    mp_f = np.asarray(multipoles(stretch(jnp.asarray(g), theta),
                                 jnp.asarray(r), jnp.asarray(wg),
                                 jnp.asarray(wr)))

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    cen = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])

    ax = axes[0, 0]
    idx = np.arange(2)
    ax.bar(idx - 0.19, fd, 0.38, label="finite differences",
           color=plt.get_cmap("viridis")(0.25))
    ax.bar(idx + 0.19, grad, 0.38, label="jax.grad + soft_hist_vjp",
           color=plt.get_cmap("viridis")(0.75))
    ax.set_xticks(idx)
    ax.set_xticklabels(["$\\alpha_\\perp - 1$",
                        "$\\alpha_\\parallel - 1$"] )
    ax.set_ylabel("$d\\sum G\\,\\xi_\\ell / d\\theta_a$")
    ax.set_title(f"Multipole gradient check (max rel err {rel.max():.1e})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    ax.semilogy(hist[:, 0], lw=1.5, color=plt.get_cmap("viridis")(0.25))
    ax.set_xlabel("Adam step")
    ax.set_ylabel("$\\frac{1}{2}|\\xi_\\ell(\\theta)-"
                  "\\xi_\\ell(\\theta_*)|^2$")
    ax.set_title("Fitting the stretch from $\\xi_0, \\xi_2$")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    for a, lab in enumerate(["$\\alpha_\\perp - 1$",
                             "$\\alpha_\\parallel - 1$"]):
        ax.plot(hist[:, 1 + a], lw=1.4,
                color=plt.get_cmap("viridis")(0.2 + 0.3 * a), label=lab)
        ax.axhline(THETA_TRUE[a], ls=":", color="0.4")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("AP stretch parameter")
    ax.set_title("Recovered AP parameters (dotted: truth)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)

    for k, ell in enumerate(POLES):
        ax = axes[1, 1]
        msk = cen <= 150          # drop the noisy edge bin (RR -> 0)
        ax.plot(cen[msk], mp_t[k][msk], color="k", lw=1.5 if k == 0 else 1.0,
                ls="-" if k == 0 else "--", label=f"target $\\xi_{ell}$")
        ax.plot(cen[msk], mp_0[k][msk], color="crimson", lw=1.0, ls=":",
                label=f"start $\\xi_{ell}$")
        ax.plot(cen[msk], mp_f[k][msk], "o", ms=3,
                color=plt.get_cmap("viridis")(0.25 + 0.5 * k),
                label=f"fitted $\\xi_{ell}$")
    ax.set_xlabel("$s$ [Mpc/h]")
    ax.set_ylabel("$\\xi_\\ell(s)$")
    ax.set_title("Multipoles along the fit")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "soft_hist_multipoles.png")
    fig.savefig(out_png, dpi=140)
    print(f"figure saved to {out_png}")


if __name__ == "__main__":
    main()
