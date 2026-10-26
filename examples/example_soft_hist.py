"""Soft histogramming vs the standard (hard) histogram method of FCFC.

pyfcfc exposes an optional pair counting method, ``soft_hist=True``:
instead of assigning every pair to a single separation bin, the weight of
the pair is spread over *all* separation bins with the softmax of a
Gaussian kernel of width ``soft_sigma`` centred on the bins,

    W_i = softmax_i(-(s - c_i)^2 / (2 * soft_sigma^2)).

The total pair weight inside [smin, smax) is conserved exactly, so the
soft histogram is a smoothed version of the hard one: sharp features are
broadened by ~soft_sigma, bin-to-bin noise from the hard bin edges
disappears, and the result is differentiable with respect to the pair
separations (useful e.g. for emulation / inference pipelines).

This example computes DD (and RR) pair counts for a clustered periodic-box
mock with the hard method and with three kernel widths, checks the weight
conservation, times both methods, and plots:

  1. the DD pair-count histograms (hard steps vs soft curves);
  2. the soft / hard ratio per bin;
  3. the correlation function xi(s) = DD/RR - 1 for both methods.

It saves ``examples/figures/soft_hist.png``.  This example does not
require pycorr.  Run from the root of the repository:

    python examples/example_soft_hist.py
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

from pyfcfc.boxes import py_compute_cf  # noqa: E402

BOX = 400.0
N_DATA = 8000
N_RAND = 8000
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def run(cats, wts, s_edges, soft_sigma=None, repeat=1, **kwargs):
    """Run py_compute_cf, optionally timing the best of `repeat` runs."""
    kw = dict(label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
              cf=['(DD - 2 * DR + RR) / RR'], box=BOX)
    kw.update(kwargs)
    if soft_sigma is not None:
        kw.update(soft_hist=True, soft_sigma=soft_sigma)
    t0 = time.perf_counter()
    for _ in range(repeat):
        res = py_compute_cf(cats, wts, s_edges, None, 1, **kw)
    return res, time.perf_counter() - t0


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Mock catalogues: clustered data + uniform randoms, mild weights.
    # ------------------------------------------------------------------
    data, w_d = make_clustered_box(N_DATA, BOX, seed=42, n_centers=25)
    rand, w_r = make_clustered_box(N_RAND, BOX, seed=7, n_centers=1,
                                   clump_frac=0.0)
    s_edges = np.arange(0, 85, 5, dtype=np.float64)   # ds = 5, 17 bins
    s_cen = 0.5 * (s_edges[:-1] + s_edges[1:])
    ds = s_edges[1] - s_edges[0]
    cats, wts = [data, rand], [w_d, w_r]

    # Kernel widths to compare: fractions of the bin size.
    frac = [0.25, 0.5, 1.0]
    sigmas = [f * ds for f in frac]

    # ------------------------------------------------------------------
    # Pair counts: hard method and soft method for every sigma.
    # ------------------------------------------------------------------
    hard, t_hard = run(cats, wts, s_edges, repeat=3)
    softs, t_softs = [], []
    for sig in sigmas:
        res, t = run(cats, wts, s_edges, soft_sigma=sig, repeat=3)
        softs.append(res)
        t_softs.append(t)

    def raw_counts(res, pair):
        return res['pairs'][pair] * res['normalization'][pair]

    dd_hard = raw_counts(hard, 'DD')
    rr_hard = raw_counts(hard, 'RR')
    xi_hard = hard['cf'][0]
    dd_soft = [raw_counts(r, 'DD') for r in softs]
    xi_soft = [r['cf'][0] for r in softs]

    # ------------------------------------------------------------------
    # Sanity checks: the softmax weights conserve the total pair weight.
    # ------------------------------------------------------------------
    print(f"catalogues: {N_DATA} data + {N_RAND} randoms, box {BOX:g}, "
          f"{len(s_cen)} bins of ds={ds:g}")
    print(f"DD total, hard: {dd_hard.sum():,.4f}")
    for f, dd in zip(frac, dd_soft):
        rel = abs(dd.sum() - dd_hard.sum()) / dd_hard.sum()
        print(f"DD total, soft (sigma={f:g}*ds): {dd.sum():,.4f} "
              f"(relative difference {rel:.2e})")
        assert rel < 1e-10, "soft histogramming must conserve the total weight"
    print(f"timing (best of 3): hard {t_hard * 1e3:.0f} ms, soft " +
          ", ".join(f"{t * 1e3:.0f} ms (sigma={f:g}*ds)"
                    for f, t in zip(frac, t_softs)))

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(frac)))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: the histograms.
    ax = axes[0]
    ax.step(np.repeat(s_edges, 2)[1:-1], np.repeat(dd_hard, 2),
            where='mid', color="0.3", lw=1.6, label="hard histogram (FCFC)")
    ax.plot(s_cen, dd_hard, "o", ms=3.5, color="0.3")
    for f, dd, c in zip(frac, dd_soft, colors):
        ax.plot(s_cen, dd, "-", marker="o", ms=3.5, lw=1.4, color=c,
                label=rf"soft, $\sigma={f:g}\,\Delta s$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$s\ [{\rm Mpc}/h$]")
    ax.set_ylabel(r"DD pair counts")
    ax.set_title("Soft vs hard histogram pair counts")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    # Panel 2: soft / hard ratio.
    ax = axes[1]
    ax.axhline(1.0, color="0.3", lw=1.2, ls="-", label="hard histogram")
    for f, dd, c in zip(frac, dd_soft, colors):
        ax.plot(s_cen, dd / dd_hard, "-", marker="o", ms=3.5, lw=1.4,
                color=c, label=rf"soft, $\sigma={f:g}\,\Delta s$")
    ax.set_xlabel(r"$s\ [{\rm Mpc}/h$]")
    ax.set_ylabel("soft / hard DD counts")
    ax.set_title("Ratio: kernel smoothing of the bin contents")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 3: the correlation function.
    ax = axes[2]
    ax.plot(s_cen, xi_hard, "o-", ms=3.5, lw=1.4, color="0.3",
            label="hard histogram")
    for f, xi, c in zip(frac, xi_soft, colors):
        ax.plot(s_cen, xi, "o-", ms=3.5, lw=1.4, color=c,
                label=rf"soft, $\sigma={f:g}\,\Delta s$")
    ax.set_xlabel(r"$s\ [{\rm Mpc}/h$]")
    ax.set_ylabel(r"$\xi(s)$")
    ax.set_title(r"$\xi(s)$, Landy-Szalay estimator")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "soft_hist.png")
    fig.savefig(out, dpi=140)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
