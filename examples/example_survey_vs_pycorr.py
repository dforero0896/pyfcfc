"""Survey-like 2PCF with pyfcfc, cross-checked against pycorr.

pyfcfc.sky ingests {RA, Dec, redshift} catalogues and converts them to
comoving Cartesian coordinates internally (convert=True, with the
fiducial cosmology Omega_m = 0.31, Omega_Lambda = 0.69, w = -1,
c/H0 = 2997.92458 Mpc/h).  Its line-of-sight convention is the pair
midpoint, which is reproduced here in pycorr with ``los='midpoint'``
on externally converted coordinates.

Computes the Landy-Szalay multipoles xi_0/2/4(s) and the projected
correlation function w_p(s_perp) with both codes and saves a comparison
figure to ``examples/figures/survey_vs_pycorr.png``.

Run from the root of the repository:

    python examples/example_survey_vs_pycorr.py
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
from mocks import make_survey, rdd_to_xyz  # noqa: E402

from pyfcfc.sky import py_compute_cf  # noqa: E402
from pyfcfc.utils import compute_multipoles  # noqa: E402

OMEGA_M, OMEGA_L, EOS_W = 0.31, 0.69, -1.0
N_DATA, N_RAND = 30000, 60000
NTHREADS = int(os.environ.get("OMP_NUM_THREADS", "2"))
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    s_edges = np.arange(0, 121, 6, dtype=np.float64)
    pimax = 60.0
    npi = 12
    nmu = 40

    data_rdd, w_data, _ = make_survey(N_DATA, seed=43)
    rand_rdd, w_rand, _ = make_survey(N_RAND, seed=9, clustered=False)
    data_xyz = rdd_to_xyz(data_rdd, OMEGA_M, OMEGA_L, EOS_W)
    rand_xyz = rdd_to_xyz(rand_rdd, OMEGA_M, OMEGA_L, EOS_W)
    print(f"catalogues: N_data = {len(data_rdd)}, N_rand = {len(rand_rdd)}, "
          f"z in [0.7, 1.0]")

    # ---------------- pyfcfc: (RA, Dec, z) in, multipoles out -------------
    t0 = time.time()
    res = py_compute_cf([data_rdd, rand_rdd], [w_data, w_rand], s_edges,
                        None, nmu, label=['D', 'R'], bin=1,
                        pair=['DD', 'DR', 'RR'],
                        cf=['(DD - 2 * DR + RR) / RR'],
                        multipole=[0, 2, 4], convert=True,
                        omega_m=OMEGA_M, omega_l=OMEGA_L, eos_w=EOS_W)
    t_fc_smu = time.time() - t0

    t0 = time.time()
    res_wp = py_compute_cf([data_rdd, rand_rdd], [w_data, w_rand], s_edges,
                           np.linspace(0, pimax, npi + 1), 0,
                           label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                           cf=['(DD - 2 * DR + RR) / RR'], wp=True,
                           convert=True, omega_m=OMEGA_M,
                           omega_l=OMEGA_L, eos_w=EOS_W)
    t_fc_wp = time.time() - t0

    s = res['s']
    mp_fc = res['multipoles'][0]        # FCFC internal (midpoint rule)
    # pycorr-compatible integration of the same xi(s, mu)
    mp_fc_exact = compute_multipoles(res['cf'][0], [0, 2, 4],
                                     method='exact', ignore_nan=True)
    wp_fc = res_wp['projected'][0]

    # ---------------- pycorr on the converted coordinates -----------------
    from pycorr import TwoPointCorrelationFunction

    mu_edges = np.linspace(-1, 1, 2 * nmu + 1)
    t0 = time.time()
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=data_xyz.T, data_weights1=w_data,
        randoms_positions1=rand_xyz.T, randoms_weights1=w_rand,
        position_type='xyz', los='midpoint', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS, compute_sepsavg=False)
    t_pc_smu = time.time() - t0

    t0 = time.time()
    corr_wp = TwoPointCorrelationFunction(
        'rppi', (s_edges, np.linspace(-pimax, pimax, 2 * npi + 1)),
        data_positions1=data_xyz.T, data_weights1=w_data,
        randoms_positions1=rand_xyz.T, randoms_weights1=w_rand,
        position_type='xyz', los='midpoint', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS, compute_sepsavg=False)
    t_pc_wp = time.time() - t0

    mp_pc = np.array([corr(ell=ell) for ell in (0, 2, 4)])
    s_wp, wp_pc = corr_wp(mode='wp', return_sep=True)

    print("\ntimings (s):")
    print(f"  smu : pyfcfc {t_fc_smu:7.2f}   pycorr {t_pc_smu:7.2f}"
          f"   speedup x{t_pc_smu / max(t_fc_smu, 1e-9):.2f}")
    print(f"  wp  : pyfcfc {t_fc_wp:7.2f}   pycorr {t_pc_wp:7.2f}"
          f"   speedup x{t_pc_wp / max(t_fc_wp, 1e-9):.2f}")
    for i, ell in enumerate((0, 2, 4)):
        print(f"  max |diff| xi_{ell} (FCFC midpoint integration): "
              f"{np.max(np.abs(mp_fc[i] - mp_pc[i])):.3e}")
    for i, ell in enumerate((0, 2, 4)):
        print(f"  max |diff| xi_{ell} (exact integration)      : "
              f"{np.max(np.abs(mp_fc_exact[i] - mp_pc[i])):.3e}")
    print(f"  max |diff| wp   : {np.max(np.abs(wp_fc - wp_pc)):.3e}")

    # ------------------------------- plot ---------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.4), sharex=True,
                             gridspec_kw={'hspace': 0.08, 'wspace': 0.22},
                             layout='constrained')
    titles = [r'LS $\xi_0(s)$', r'LS $\xi_2(s)$', r'LS $\xi_4(s)$',
              r'LS $w_p(s_\perp)$']
    ylabels = [r'$\xi_0(s)$', r'$\xi_2(s)$', r'$\xi_4(s)$',
               r'$w_p(s_\perp)$']
    fc_curves = [mp_fc_exact[0], mp_fc_exact[1], mp_fc_exact[2], wp_fc]
    pc_curves = [mp_pc[0], mp_pc[1], mp_pc[2], wp_pc]
    xs = [s, s, s, s]
    for col in range(4):
        ax, axr = axes[0, col], axes[1, col]
        ax.plot(xs[col], pc_curves[col], '-', color='C0', lw=1.4,
                label='pycorr (midpoint LOS)')
        ax.plot(xs[col], fc_curves[col], '--', color='C3', lw=1.2,
                label='pyfcfc (RA,Dec,z; exact integration)')
        ax.set_title(titles[col], fontsize=11)
        ax.set_ylabel(ylabels[col])
        ax.grid(alpha=0.3)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (fc_curves[col] - pc_curves[col]) \
                / np.abs(pc_curves[col])
        axr.axhline(0, color='k', lw=0.8)
        axr.plot(xs[col], ratio, '.', color='C3', ms=4)
        axr.set_xlabel(r'$s\ [h^{-1}\mathrm{Mpc}]$')
        axr.set_ylabel('pyfcfc/pycorr - 1')
        axr.grid(alpha=0.3)
        good = ratio[np.isfinite(ratio)]
        lim = max(3 * (np.std(good) if good.size else 0.01), 0.02)
        axr.set_ylim(-lim, lim)
        if col == 0:
            ax.legend(fontsize=8, loc='upper right')
    fname = os.path.join(FIGDIR, 'survey_vs_pycorr.png')
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"\nfigure saved to:\n  {fname}")


if __name__ == '__main__':
    main()
