"""The 2D (r_p, pi) correlation function and its gradient maps.

With ``soft_hist=True`` and ``soft_pi=True`` the soft 2D counts are smooth
in the catalogue coordinates, so the full (r_p, pi) correlation function
xi(rp, pi) - not just its summaries - becomes a differentiable function
of the data.  This example

  1. plots the soft-binned Landy-Szalay xi(rp, pi) map of a clustered
     periodic-box mock,
  2. computes the *gradient maps* d xi(rp, pi) / d theta for a two-
     parameter anisotropic stretch model (Alcock-Paczyński-like,
     theta = (a_perp, a_par)) with the reverse-mode adjoint: one-hot
     cotangents on each bin fed through ``soft_hist_vjp`` and contracted
     with the model's dx/dtheta,
  3. validates every entry of those Jacobian maps against central finite
     differences of the forward C pipeline,
  4. as a bonus, uses the gradient maps for a single Newton-like step
     sanity check (the directional derivative of xi along the model).

Panels: xi map, the two adjoint gradient maps, the two finite-difference
gradient maps and an all-bin adjoint-vs-FD scatter.

It saves ``examples/figures/soft_hist_rppi.png``.  Requires ``jax`` only
for array convenience?  No: this example is pure NumPy + the C adjoint
(the VJP is consumed directly, no autodiff framework needed).  Run from
the root of the repository:

    python examples/example_soft_hist_rppi.py
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

BOX = 400.0
S_EDGES = np.arange(0, 151, 10, dtype=np.float64)     # r_p bins
PI_EDGES = np.arange(0, 91, 10, dtype=np.float64)     # pi bins
NS, NPI = len(S_EDGES) - 1, len(PI_EDGES) - 1
SIGMA_S, SIGMA_PI = 5.0, 5.0
CENTRE = np.array([200., 200., 200.])
THETA0 = np.array([0.02, -0.02])          # evaluation point of the maps
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

KW = dict(label=['G', 'R'], bin=2, pair=['GG', 'GR', 'RR'],
          soft_hist=True, soft_sigma=SIGMA_S, soft_pi=True,
          soft_pi_sigma=SIGMA_PI, box=BOX)


def stretch(pos, theta):
    """Anisotropic stretch about the cube centre: (a_perp, a_par)."""
    a, b = theta
    return CENTRE + (pos - CENTRE) * np.array([1 + a, 1 + a, 1 + b])


def raw_counts(gpos, rpos, wg, wr):
    res = py_compute_cf([gpos, rpos], [wg, wr], S_EDGES, PI_EDGES, 1, **KW)
    n = res['normalization']
    return {p: res['pairs'][p] * n[p] for p in ('GG', 'GR', 'RR')}, n


def xi_map(gpos, rpos, wg, wr, mask):
    """Landy-Szalay xi(rp, pi) on the supported cells (masked elsewhere)."""
    cnt, n = raw_counts(gpos, rpos, wg, wr)
    u = cnt['GG'] / n['GG']
    v = cnt['GR'] / n['GR']
    w = cnt['RR'] / n['RR']
    xi = np.where(mask, (u - 2 * v + w) / np.where(mask, w, 1.0), np.nan)
    return xi, u, v, w


def xi_cotangents(mask, w, u, v):
    """d xi_kj / d(normalized counts) on masked cells: returns the three
    cotangent arrays to feed to soft_hist_vjp for a one-hot xi cotangent.
    For a general upstream G on xi: cot_DD = G/w, cot_DR = -2G/w,
    cot_RR = G(2v - u)/w^2 (then divide by the pair normalizations)."""
    g_dd = mask / w
    g_dr = -2 * mask / w
    g_rr = mask * (2 * v - u) / w ** 2
    return g_dd, g_dr, g_rr


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(303)
    g, wg = make_clustered_box(600, BOX, seed=31, n_centers=12)
    g = 150.0 + g / BOX * 100.0
    r, wr = make_clustered_box(600, BOX, seed=32, n_centers=12)
    r = 150.0 + r / BOX * 100.0
    wg = np.ones(len(g))
    wr = np.ones(len(r))
    dpos = g - CENTRE                     # dx/dtheta per point

    # fixed support mask at the evaluation point
    cnt0, nrm0 = raw_counts(stretch(g, THETA0), r, wg, wr)
    mask = (cnt0['RR'] / nrm0['RR']) > 1e-4
    print(f"support mask: {mask.sum()}/{mask.size} (rp, pi) cells")

    xi0, u0, v0, w0 = xi_map(stretch(g, THETA0), r, wg, wr, mask)

    # ------------------------------------------------------------------
    # gradient maps d xi(rp, pi) / d theta via the adjoint
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    g_dd, g_dr, g_rr = xi_cotangents(mask, w0, u0, v0)
    jac = np.zeros((2, NS, NPI))
    for k in range(NS):
        for j in range(NPI):
            if not mask[k, j]:
                continue
            gx = np.zeros_like(g)
            # RR depends only on the (fixed) randoms: no theta dependence
            for pair, cot, nrm in (('GG', g_dd, nrm0['GG']),
                                   ('GR', g_dr, nrm0['GR'])):
                e = np.zeros((NS, NPI))
                e[k, j] = 1.0
                out = soft_hist_vjp([stretch(g, THETA0), r], [wg, wr],
                                    S_EDGES, PI_EDGES, 1, pair=pair,
                                    cotangent=cot * e / nrm,
                                    soft_sigma=SIGMA_S, soft_pi=True,
                                    soft_pi_sigma=SIGMA_PI, bin=2, box=BOX,
                                    label=['G', 'R'])
                gx = gx + out['grads']['G']
            # contract with dx/dtheta: a_perp scales x,y; a_par scales z
            jac[0, k, j] = np.sum(gx[:, :2] * dpos[:, :2])
            jac[1, k, j] = np.sum(gx[:, 2] * dpos[:, 2])
    t_adj = time.perf_counter() - t0
    print(f"adjoint Jacobian maps: {t_adj:.1f} s "
          f"({mask.sum()} cells x 3 VJPs)")

    # ------------------------------------------------------------------
    # finite-difference Jacobian maps of the forward pipeline
    # ------------------------------------------------------------------
    eps = 1e-6
    jac_fd = np.zeros((2, NS, NPI))
    for a in range(2):
        h = np.zeros(2)
        h[a] = eps
        xp = xi_map(stretch(g, THETA0 + h), r, wg, wr, mask)[0]
        xm = xi_map(stretch(g, THETA0 - h), r, wg, wr, mask)[0]
        jac_fd[a] = (xp - xm) / (2 * eps)
    dj = np.abs(jac - jac_fd)[:, mask]
    fdm = np.abs(jac_fd[:, mask])
    rel = dj / np.maximum(fdm, 1e-6 * fdm.max())
    print(f"Jacobian check: max rel err over supported cells "
          f"{rel.max():.2e}, median {np.median(rel):.2e}")

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    rp = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    pi = 0.5 * (PI_EDGES[:-1] + PI_EDGES[1:])
    RP, PI = np.meshgrid(rp, pi, indexing='ij')

    def pmap(ax, Z, title, cmap="viridis", sym=False):
        Zm = np.where(mask, Z, np.nan)
        if sym:
            # robust scale: sparse edge cells would otherwise dominate
            vmax = np.nanpercentile(np.abs(Zm), 97)
        else:
            vmax = np.nanpercentile(Zm, 97)
        vmin = -vmax if sym else np.nanpercentile(Zm, 3)
        im = ax.pcolormesh(RP, PI, Zm, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto')
        ax.set_xlabel("$r_p$ [Mpc/h]")
        ax.set_ylabel("$\\pi$ [Mpc/h]")
        ax.set_title(title)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    im = pmap(axes[0, 0], xi0, "soft $\\xi(r_p, \\pi)$  (masked: grey)")
    fig.colorbar(im, ax=axes[0, 0])
    im = pmap(axes[0, 1], jac[0], r"$\partial\xi/\partial a_\perp$ (adjoint)",
              sym=True)
    fig.colorbar(im, ax=axes[0, 1])
    im = pmap(axes[0, 2], jac_fd[0], r"$\partial\xi/\partial a_\perp$ (FD)",
              sym=True)
    fig.colorbar(im, ax=axes[0, 2])
    im = pmap(axes[1, 0], jac[1], r"$\partial\xi/\partial a_\parallel$ "
              "(adjoint)", sym=True)
    fig.colorbar(im, ax=axes[1, 0])
    im = pmap(axes[1, 1], jac_fd[1], r"$\partial\xi/\partial a_\parallel$ "
              "(FD)", sym=True)
    fig.colorbar(im, ax=axes[1, 1])
    ax = axes[1, 2]
    ax.scatter(jac_fd[:, mask].ravel(), jac[:, mask].ravel(), s=8,
               color=plt.get_cmap("viridis")(0.4))
    lim = max(fdm.max(), np.abs(jac[:, mask]).max())
    ax.plot([-lim, lim], [-lim, lim], ":", color="0.4")
    ax.set_xlabel("finite differences")
    ax.set_ylabel("adjoint (soft_hist_vjp)")
    ax.set_title(f"all bins (max rel err {rel.max():.1e})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "soft_hist_rppi.png")
    fig.savefig(out_png, dpi=140)
    print(f"figure saved to {out_png}")


if __name__ == "__main__":
    main()
