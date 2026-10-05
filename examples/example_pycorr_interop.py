"""pyfcfc + pycorr interoperability: split randoms and pycorr states.

Workflow demonstrated here (the one used for real survey measurements):

1. split a large random catalogue into chunks and pair-count
   data-vs-chunk with pyfcfc, accumulating the results with
   ``pyfcfc.utils.add_pair_counts``;
2. convert the accumulated pair counts to a pycorr state with
   ``pyfcfc.utils.pairs_to_pycorr`` and save it to disk;
3. load it with ``pycorr.TwoPointCorrelationFunction.load`` and use the
   full pycorr API: estimators, rebinning, plotting;
4. compare the multipoles obtained from the loaded pycorr object with
   pyfcfc's own integration.

Saves ``examples/figures/pycorr_interop.png``.

Run from the root of the repository:

    python examples/example_pycorr_interop.py
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
from pyfcfc.utils import (add_pair_counts, compute_multipoles,  # noqa: E402
                          pairs_to_pycorr)

BOX = 1000.0
N_DATA, N_RAND, N_SPLIT = 30000, 90000, 4
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTDIR = os.path.join(FIGDIR, "states")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    s_edges = np.arange(0, 151, 5, dtype=np.float64)
    nmu = 48

    data, w_data = make_clustered_box(N_DATA, BOX, seed=42)
    rand, w_rand = make_clustered_box(N_RAND, BOX, seed=7, n_centers=1,
                                      clump_frac=0.0)

    # ------------------------------------------------------------------
    # 1. split-random pair counting with accumulation
    # ------------------------------------------------------------------
    total = None
    t0 = time.time()
    for i in range(N_SPLIT):
        sl = slice(i * len(rand) // N_SPLIT, (i + 1) * len(rand) // N_SPLIT)
        # DD only needs to be counted once (it does not involve randoms)
        pairs = ['DD', 'DR', 'RR'] if i == 0 else ['DR', 'RR']
        res = py_compute_cf([data, rand[sl]], [w_data, w_rand[sl]],
                            s_edges, None, nmu, label=['D', 'R'], bin=1,
                            pair=pairs, box=BOX)
        total = res if total is None else add_pair_counts(total, res)
        print(f"  split {i}: pairs {pairs} counted")
    t_acc = time.time() - t0
    print(f"split-random accumulation ({N_SPLIT} splits): {t_acc:.2f} s")

    # Landy-Szalay xi(s, mu) from the accumulated counts, and multipoles
    xi_smu = (total['pairs']['DD'] - 2 * total['pairs']['DR']
              + total['pairs']['RR']) / total['pairs']['RR']
    # use pycorr's integration scheme ('exact') for the comparison; with
    # FCFC's default midpoint rule the curves differ by O(dmu^2)
    mp_fcfc = compute_multipoles(xi_smu, [0, 2, 4], method='exact')
    s = total['s']

    # ------------------------------------------------------------------
    # 2-3. convert to a pycorr state, save, reload
    # ------------------------------------------------------------------
    state = pairs_to_pycorr(
        total, 'landyszalay',
        dict(DD='D1D2', DR=('D1R2', 'R1D2'), RR='R1R2'), box_size=BOX)
    fname = os.path.join(OUTDIR, "pyfcfc_accumulated.pkl.npy")
    np.save(fname, state)
    print(f"pycorr state saved to {fname}")

    from pycorr import TwoPointCorrelationFunction
    loaded = TwoPointCorrelationFunction.load(fname)
    print(f"loaded with pycorr: shape = {loaded.shape}, "
          f"mode = {loaded.mode}")

    mp_loaded = np.array([loaded(ell=ell) for ell in (0, 2, 4)])

    # rebin by 3 in s and 2 in mu, using the pycorr API
    rebinned = loaded[::3, ::2]
    s_rebin, mp0 = rebinned(ell=0, return_sep=True)
    mp_rebin = np.array([mp0] + [rebinned(ell=ell) for ell in (2, 4)])
    print(f"rebinned with pycorr: shape = {rebinned.shape}")

    # ------------------------------------------------------------------
    # 4. plot the comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for i, ell in enumerate((0, 2, 4)):
        ax = axes[i]
        ax.plot(s, s ** 2 * mp_fcfc[i], '-', color='C3', lw=1.4,
                label='pyfcfc counts, exact integration')
        ax.plot(s, s ** 2 * mp_loaded[i], '--', color='C0', lw=1.2,
                label='pycorr (loaded state)')
        ax.plot(s_rebin, s_rebin ** 2 * mp_rebin[i], ':', color='C2',
                lw=1.8, label='pycorr rebinned (x3, x2)')
        ax.set_title(f'$\\ell = {ell}$', fontsize=11)
        ax.set_xlabel(r'$s\ [h^{-1}\mathrm{Mpc}]$')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(r'$s^2 \xi_\ell(s)$')
            ax.legend(fontsize=8)
    fig.tight_layout()
    figname = os.path.join(FIGDIR, 'pycorr_interop.png')
    fig.savefig(figname, dpi=160)
    plt.close(fig)

    dev = np.max(np.abs(mp_loaded - mp_fcfc), axis=1)
    print("\nmax |pycorr(loaded) - pyfcfc integration| per multipole:",
          ", ".join(f"{d:.2e}" for d in dev))
    print(f"\nfigure saved to:\n  {figname}")


if __name__ == '__main__':
    main()
