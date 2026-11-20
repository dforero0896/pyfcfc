"""Angular correlation function w(theta) with pyfcfc.

pyfcfc works on Cartesian positions and measures Euclidean separations.
The angular correlation function is the unit-sphere special case: for
objects placed at unit distance, the separation is the chord
``s = 2 sin(theta / 2)``.  ``pyfcfc.utils.compute_wtheta`` performs this
mapping (RA/Dec -> unit sphere, theta bins -> chord bins) and calls the
ordinary isotropic pair counter with the Landy-Szalay estimator.

This script emulates a narrow redshift slice of a survey, measures
w(theta), validates the weighted DD/DR/RR counts against an independent
O(N^2) NumPy counter, and saves a figure of ``theta^2 * w(theta)`` to
``examples/figures/angular_wtheta.png``.

Run from the root of the repository:

    python examples/example_angular_wtheta.py
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
from mocks import make_survey  # noqa: E402

from pyfcfc.utils import compute_wtheta, radec_to_unit_sphere  # noqa: E402

N_DATA, N_RAND = 120000, 240000
Z_RANGE = (0.7, 1.0)
Z_SLICE = (0.80, 0.85)
THETA_MIN, THETA_MAX, NTHETA = 0.05, 10.0, 20
NTHREADS = int(os.environ.get("OMP_NUM_THREADS", "2"))
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _brute_force_counts(unit_a, w_a, unit_b, w_b, edges):
    """Weighted, ordered pair counts in angular bins (O(N^2), chunked)."""
    counts = np.zeros(edges.size - 1)
    for start in range(0, len(unit_a), 512):
        stop = min(start + 512, len(unit_a))
        cosang = unit_a[start:stop] @ unit_b.T
        angles = np.arccos(np.clip(cosang, -1.0, 1.0))
        weights = w_a[start:stop, None] * w_b[None, :]
        counts += np.histogram(angles, bins=edges, weights=weights)[0]
    return counts


def _validate(res, data_rdd, w_data, rand_rdd, w_rand, n_data, n_rand,
              seed=7):
    """Compare pyfcfc pair counts with a brute-force counter on a subset."""
    rng = np.random.default_rng(seed)
    idata = rng.choice(len(data_rdd), n_data, replace=False)
    irand = rng.choice(len(rand_rdd), n_rand, replace=False)
    data_sub, w_data_sub = data_rdd[idata], w_data[idata]
    rand_sub, w_rand_sub = rand_rdd[irand], w_rand[irand]

    sub = compute_wtheta([data_sub, rand_sub], [w_data_sub, w_rand_sub],
                         res['theta_edges'])
    ud = radec_to_unit_sphere(data_sub[:, 0], data_sub[:, 1])
    ur = radec_to_unit_sphere(rand_sub[:, 0], rand_sub[:, 1])
    edges = np.deg2rad(res['theta_edges'])
    brute = {
        'DD': _brute_force_counts(ud, w_data_sub, ud, w_data_sub, edges),
        'DR': _brute_force_counts(ud, w_data_sub, ur, w_rand_sub, edges),
        'RR': _brute_force_counts(ur, w_rand_sub, ur, w_rand_sub, edges),
    }
    print(f"\nbrute-force validation on a subset "
          f"(N_data = {n_data}, N_rand = {n_rand}):")
    for key, expected in brute.items():
        got = sub['pairs'][key] * sub['normalization'][key]
        scale = np.maximum(np.abs(expected), 1.0)
        print(f"  {key}: max |pyfcfc - brute force| / max(count, 1) = "
              f"{np.max(np.abs(got - expected) / scale):.2e}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    data_rdd, w_data, _ = make_survey(N_DATA, seed=43, z_range=Z_RANGE)
    rand_rdd, w_rand, _ = make_survey(N_RAND, seed=9, z_range=Z_RANGE,
                                      clustered=False)

    # Emulate the usual analysis step: keep a narrow redshift bin and
    # measure the angular clustering of the projected catalogue.
    sel_data = (data_rdd[:, 2] >= Z_SLICE[0]) & (data_rdd[:, 2] < Z_SLICE[1])
    sel_rand = (rand_rdd[:, 2] >= Z_SLICE[0]) & (rand_rdd[:, 2] < Z_SLICE[1])
    data_rdd, w_data = data_rdd[sel_data], w_data[sel_data]
    rand_rdd, w_rand = rand_rdd[sel_rand], w_rand[sel_rand]
    print(f"redshift slice {Z_SLICE[0]} <= z < {Z_SLICE[1]}: "
          f"N_data = {len(data_rdd)}, N_rand = {len(rand_rdd)}")

    theta_edges = np.logspace(np.log10(THETA_MIN), np.log10(THETA_MAX),
                              NTHETA + 1)
    print(f"theta bins: {NTHETA} logarithmic bins in "
          f"[{THETA_MIN}, {THETA_MAX}] deg")

    t0 = time.time()
    res = compute_wtheta([data_rdd, rand_rdd], [w_data, w_rand], theta_edges,
                         label=['D', 'R'], pair=['DD', 'DR', 'RR'],
                         cf=['(DD - 2 * DR + RR) / RR'])
    t_fc = time.time() - t0
    print(f"pyfcfc w(theta): {t_fc:.2f} s")

    theta = res['theta']                    # degrees
    wtheta = res['wtheta'][0]

    _validate(res, data_rdd, w_data, rand_rdd, w_rand, 2000, 4000)

    # ------------------------------- plot ---------------------------------
    y = theta ** 2 * wtheta
    fig, ax = plt.subplots(figsize=(7.4, 5.2), layout='constrained')
    ax.axhline(0.0, color='0.6', lw=0.8, zorder=0)
    ax.plot(theta, y, 'o-', color='C0', lw=1.5, ms=5,
            label='pyfcfc (Landy–Szalay)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta\ [\mathrm{deg}]$')
    ax.set_ylabel(r'$\theta^2\,w(\theta)\ [\mathrm{deg}^2]$')
    ax.set_title(r'Angular correlation function, '
                 rf'${Z_SLICE[0]} \leq z < {Z_SLICE[1]}$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    fname = os.path.join(FIGDIR, 'angular_wtheta.png')
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"\nfigure saved to:\n  {fname}")


if __name__ == '__main__':
    main()
