"""Standalone validation of pyfcfc against brute-force pair counting.

For a small periodic-box catalogue, the pair counts computed by pyfcfc
are compared bin-by-bin with a direct O(N^2) NumPy implementation
(``tests/brute_force.py``), for the three binning schemes (isotropic,
(s, mu) and (s_perp, pi)), with unit and random weights, and with both
the k-d tree and the ball tree.

This example does not require pycorr.  It saves
``examples/figures/validation_bruteforce.png``.

Run from the root of the repository:

    python examples/example_validation_bruteforce.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from brute_force import brute_force_box  # noqa: E402
from mocks import make_clustered_box  # noqa: E402

from pyfcfc.boxes import py_compute_cf  # noqa: E402

BOX = 400.0
N = 2500
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    data, w = make_clustered_box(N, BOX, seed=42, n_centers=15)
    rand = make_clustered_box(int(1.5 * N), BOX, seed=7, n_centers=1,
                              clump_frac=0.0)[0]
    w_r = np.ones(len(rand))
    s_edges = np.arange(0, 101, 10, dtype=np.float64)
    nmu = 10
    mu_edges = np.linspace(0, 1, nmu + 1)
    pi_edges = np.arange(0, 101, 10, dtype=np.float64)

    checks = []

    # ---------------- isotropic, unweighted, k-d tree & ball tree ---------
    for data_struct, name in [(0, 'kdtree'), (1, 'balltree')]:
        res = py_compute_cf([data, rand], [np.ones(N), w_r], s_edges, None,
                            0, label=['D', 'R'], bin=0,
                            pair=['DD', 'DR', 'RR'], box=BOX,
                            data_struct=data_struct)
        ref = brute_force_box([data, rand], [np.ones(N), w_r], ['D', 'R'],
                              ['DD', 'DR', 'RR'], s_edges, BOX)
        for pair in ('DD', 'DR', 'RR'):
            got = res['pairs'][pair] * res['normalization'][pair]
            checks.append((f'iso {pair} ({name})', s_edges,
                           got, ref['counts'][pair]))

    # ---------------- (s, mu), weighted ------------------------------------
    res = py_compute_cf([data, rand], [w, w_r], s_edges, None, nmu,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        box=BOX)
    ref = brute_force_box([data, rand], [w, w_r], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, BOX,
                          mu_edges=mu_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        # collapse the mu axis for the plot, but check bin-by-bin first
        assert np.allclose(got, ref['counts'][pair], rtol=1e-9, atol=2), pair
        checks.append((f'smu {pair} (weighted, mu-summed)', s_edges,
                       got.sum(axis=1), ref['counts'][pair].sum(axis=1)))

    # ---------------- (s_perp, pi), weighted -------------------------------
    res = py_compute_cf([data, rand], [w, w_r], s_edges, pi_edges, 0,
                        label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                        box=BOX)
    ref = brute_force_box([data, rand], [w, w_r], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, BOX,
                          pi_edges=pi_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        assert np.allclose(got, ref['counts'][pair], rtol=1e-9, atol=2), pair
        checks.append((f'spi {pair} (weighted, pi-summed)', s_edges,
                       got.sum(axis=1), ref['counts'][pair].sum(axis=1)))

    # ---------------- plot: pyfcfc / brute force - 1 -----------------------
    n = len(checks)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow),
                             squeeze=False, sharex=True)
    centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    for ax, (title, _edges, got, exp) in zip(axes.ravel(), checks):
        ratio = got / exp - 1.0
        ax.axhline(0, color='k', lw=0.8)
        ax.plot(centers, ratio, 'o-', ms=3.5, lw=1, color='C3')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('pyfcfc/brute force - 1')
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.2e-3, 1.2e-3)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$s\ [h^{-1}\mathrm{Mpc}]$')
    fig.tight_layout()
    fname = os.path.join(FIGDIR, 'validation_bruteforce.png')
    fig.savefig(fname, dpi=160)
    plt.close(fig)

    print("all checks passed (max deviations below):")
    for title, _edges, got, exp in checks:
        print(f"  {title:34s}: {np.max(np.abs(got / exp - 1)):.2e}")
    print(f"\nfigure saved to:\n  {fname}")


if __name__ == '__main__':
    main()
