"""A guided tour of every pyfcfc feature, in one script.

Each numbered section prints what it demonstrates, so reading the output
top-to-bottom is a complete tour of the API:

 1.  the result dictionary (keys, shapes, conventions)
 2.  the three binning schemes: isotropic, (s, mu), (s_perp, pi)
 3.  weights: float counting path vs exact integer path, normalizations
 4.  configuration files + keyword overrides
 5.  data structures: k-d tree vs ball tree
 6.  input flexibility: lists, float32, non-contiguous views
 7.  cross-correlation of two different tracers
 8.  multipoles & w_p: internal integration vs pyfcfc.utils (midpoint and
     exact schemes), ignore_nan for empty bins
 9.  the survey-like component: {RA, Dec, z} with cosmological conversion
 10. error handling and validation
 11. verbose output

Run from the root of the repository:

    python examples/example_features_tour.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from mocks import make_clustered_box  # noqa: E402
from pyfcfc.boxes import py_compute_cf, simd_info  # noqa: E402
from pyfcfc.sky import py_compute_cf as sky_compute_cf  # noqa: E402
from pyfcfc.utils import compute_multipoles, compute_wp  # noqa: E402


def header(n, title):
    print(f"\n{'=' * 72}\n{n}. {title}\n{'=' * 72}")


def main():
    print(f"pyfcfc build: SIMD = {simd_info()[1]}  "
          f"(OMP threads = {os.environ.get('OMP_NUM_THREADS', 'unset')})")

    BOX = 500.0
    data, w_data = make_clustered_box(4000, BOX, seed=1)
    rand, w_rand = make_clustered_box(6000, BOX, seed=2, n_centers=1,
                                      clump_frac=0.0)
    s_edges = np.arange(0, 101, 10, dtype=np.float64)

    # ------------------------------------------------------------------
    header(1, "The result dictionary")
    res = py_compute_cf([data, rand], [w_data, w_rand], s_edges, None, 20,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        cf=['(DD - 2 * DR + RR) / RR'], multipole=[0, 2],
                        box=BOX)
    for key in sorted(res):
        val = res[key]
        if isinstance(val, dict):
            desc = {k: (np.shape(v), getattr(np.asarray(v), 'dtype', None))
                    for k, v in list(val.items())[:3]}
            print(f"  {key:16s}: dict -> {desc} ...")
        elif isinstance(val, np.ndarray):
            print(f"  {key:16s}: array {val.shape} {val.dtype}")
        else:
            print(f"  {key:16s}: {val}")
    print("  raw pair counts = pairs[key] * normalization[key]; e.g. DD total"
          f" = {(res['pairs']['DD'] * res['normalization']['DD']).sum():.6g}"
          f" (norm = {res['normalization']['DD']:.6g})")

    # ------------------------------------------------------------------
    header(2, "The three binning schemes")
    pi_edges = np.arange(0, 101, 20, dtype=np.float64)
    for bin_type, pedges, nmu, name in [
            (0, None, 0, 'isotropic'),
            (1, None, 20, '(s, mu)'),
            (2, pi_edges, 0, '(s_perp, pi)')]:
        r = py_compute_cf([data], [w_data], s_edges, pedges, nmu,
                          label=['D'], bin=bin_type, pair=['DD'], box=BOX)
        extra = [k for k in ('mumin', 'pimin') if k in r['pairs']]
        print(f"  bin={bin_type} ({name:11s}): pairs['DD'] shape "
              f"{r['pairs']['DD'].shape}, edge arrays: {extra}")

    # ------------------------------------------------------------------
    header(3, "Weights: float vs exact integer counting path")
    ones = np.ones(len(data))
    r_w = py_compute_cf([data], [w_data], s_edges, None, 0, label=['D'],
                        bin=0, pair=['DD'], box=BOX)
    r_u = py_compute_cf([data], [ones], s_edges, None, 0, label=['D'],
                        bin=0, pair=['DD'], box=BOX)
    sw = w_data.sum()
    print(f"  weighted  : norm = {r_w['normalization']['DD']:.10g}"
          f"  (= (sum w)^2 - sum w^2 = {sw ** 2 - (w_data ** 2).sum():.10g})")
    print(f"  unweighted: norm = {r_u['normalization']['DD']:.10g}"
          f"  (= N (N-1)       = {len(data) * (len(data) - 1)})")
    print("  unweighted catalogs automatically use FCFC's exact integer"
          " counting path (weights of exactly 1 are detected)")

    # ------------------------------------------------------------------
    header(4, "Configuration files and keyword overrides")
    conf = os.path.join(tempfile.mkdtemp(), "tour.conf")
    with open(conf, 'w') as fh:
        fh.write(f"""CATALOG_LABEL   = [X, Y]
BOX_SIZE        = {BOX}
BINNING_SCHEME  = 0
PAIR_COUNT      = [XX, XY, YY]
CF_ESTIMATOR    = [(XX - 2 * XY + YY) / YY]
VERBOSE         = F
""")
    r = py_compute_cf([data, rand], [ones, np.ones(len(rand))], s_edges,
                      None, 0, conf=conf)
    print(f"  from file only          : labels {r['labels']}, pairs "
          f"{sorted(k for k in r['pairs'] if len(k) == 2)}")
    r = py_compute_cf([data, rand], [ones, np.ones(len(rand))], s_edges,
                      None, 0, conf=conf, pair=['XX'], label=['X', 'Y'],
                      cf=['XX / @@ - 1'])
    print(f"  kwargs override the file: pairs "
          f"{sorted(k for k in r['pairs'] if len(k) == 2)}")

    # ------------------------------------------------------------------
    header(5, "Data structures: k-d tree vs ball tree")
    r_kd = py_compute_cf([data], [ones], s_edges, None, 0, label=['D'],
                         bin=0, pair=['DD'], box=BOX, data_struct=0)
    r_bt = py_compute_cf([data], [ones], s_edges, None, 0, label=['D'],
                         bin=0, pair=['DD'], box=BOX, data_struct=1)
    same = np.array_equal(r_kd['pairs']['DD'], r_bt['pairs']['DD'])
    print(f"  identical counts with data_struct=0/1: {same}")

    # ------------------------------------------------------------------
    header(6, "Input flexibility")
    r_list = py_compute_cf([data.tolist()], [ones.tolist()],
                           s_edges.tolist(), None, 0, label=['D'], bin=0,
                           pair=['DD'], box=BOX)
    r_f32 = py_compute_cf([data.astype(np.float32)],
                          [ones.astype(np.float32)], s_edges, None, 0,
                          label=['D'], bin=0, pair=['DD'], box=BOX)
    view = np.asfortranarray(data)[:, :]     # non-contiguous in C order
    r_view = py_compute_cf([view], [ones], s_edges, None, 0, label=['D'],
                           bin=0, pair=['DD'], box=BOX)
    print(f"  python lists   : max|diff| = "
          f"{np.abs(r_list['pairs']['DD'] - r_u['pairs']['DD']).max():.2e}")
    print(f"  float32 inputs : max|diff| = "
          f"{np.abs(r_f32['pairs']['DD'] - r_u['pairs']['DD']).max():.2e}")
    print(f"  F-order view   : max|diff| = "
          f"{np.abs(r_view['pairs']['DD'] - r_u['pairs']['DD']).max():.2e}")

    # ------------------------------------------------------------------
    header(7, "Cross-correlation of two different tracers")
    tracer_b, w_b = make_clustered_box(3500, BOX, seed=9, n_centers=15)
    r = py_compute_cf([data, tracer_b], [ones, np.ones(len(tracer_b))],
                      s_edges, None, 0, label=['A', 'B'], bin=0,
                      pair=['AA', 'AB', 'BB'], cf=['AB / @@ - 1'], box=BOX)
    print(f"  pairs computed: {sorted(k for k in r['pairs'] if len(k) == 2)};"
          f"  norm(AB) = {r['normalization']['AB']:.6g} = N_A * N_B = "
          f"{len(data) * len(tracer_b)}")

    # ------------------------------------------------------------------
    header(8, "Multipoles & w_p: integration schemes and empty bins")
    nmu = 32
    r = py_compute_cf([data, rand], [ones, np.ones(len(rand))], s_edges,
                      None, nmu, label=['D', 'R'], bin=1,
                      pair=['DD', 'DR', 'RR'],
                      cf=['(DD - 2 * DR + RR) / RR'], multipole=[0, 2],
                      box=BOX)
    xi_smu = (r['pairs']['DD'] - 2 * r['pairs']['DR']
              + r['pairs']['RR']) / r['pairs']['RR']
    mp_mid = compute_multipoles(xi_smu, [0, 2])
    mp_exa = compute_multipoles(xi_smu, [0, 2], method='exact')
    print(f"  FCFC internal multipoles == utils midpoint rule: "
          f"{np.allclose(mp_mid, r['multipoles'][0], rtol=0, atol=1e-12)}")
    print(f"  midpoint vs exact (pycorr's scheme): max|diff| = "
          f"{np.abs(mp_mid - mp_exa).max():.2e}  (O(dmu^2), shrinks with "
          "nmu)")
    xi_nan = xi_smu.copy()
    xi_nan[0, : nmu // 2] = np.nan     # e.g. empty (s, mu) cells
    mp_nan = compute_multipoles(xi_nan, [0], ignore_nan=True)
    mp_raw = compute_multipoles(xi_nan, [0])
    print(f"  empty (s, mu) cells -> NaN propagates by default "
          f"(xi0[0] = {mp_raw[0][0]}); with ignore_nan=True the covered mu "
          f"range is rescaled: xi0[0] = {mp_nan[0][0]:.4f}")
    r_wp = py_compute_cf([data, rand], [ones, np.ones(len(rand))], s_edges,
                         pi_edges, 0, label=['D', 'R'], bin=2,
                         pair=['DD', 'DR', 'RR'],
                         cf=['(DD - 2 * DR + RR) / RR'], wp=True, box=BOX)
    wp_ext = compute_wp(r_wp['cf'][0], pi_edges)
    print(f"  w_p internal == utils.compute_wp: "
          f"{np.allclose(wp_ext, r_wp['projected'][0], rtol=0, atol=1e-12)}")

    # ------------------------------------------------------------------
    header(9, "Survey-like component: {RA, Dec, z} + cosmology")
    from mocks import make_survey
    rdd, w_s, _ = make_survey(4000, seed=5, z_range=(0.4, 0.55))
    rdd_r, w_sr, _ = make_survey(6000, seed=6, z_range=(0.4, 0.55),
                                 clustered=False)
    s_sky = np.arange(0, 151, 30, dtype=np.float64)
    r = sky_compute_cf([rdd, rdd_r], [w_s, w_sr], s_sky, None, 20,
                       label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'], multipole=[0],
                       convert=True, omega_m=0.31, omega_l=0.69, eos_w=-1)
    print(f"  converted on the fly (Om=0.31, OL=0.69, w=-1): xi0 range "
          f"[{r['multipoles'][0][0].min():.3f}, "
          f"{r['multipoles'][0][0].max():.3f}] over s in "
          f"[{r['s'][0]:.0f}, {r['s'][-1]:.0f}] Mpc/h")
    print("  the sky component uses the pair-midpoint line of sight "
          "(pycorr: los='midpoint')")

    # ------------------------------------------------------------------
    header(10, "Error handling and validation")
    for kwargs, exc, why in [
        (dict(sedges=[0., 10., 5.]), ValueError, "non-monotonic bin edges"),
        (dict(box=-1.), RuntimeError, "negative periodic box size"),
        (dict(pair=['DZ']), RuntimeError, "pair label without catalogue"),
        (dict(multipol=[0, 2]), ValueError, "typo in an option name"),
    ]:
        sedges = kwargs.pop('sedges', s_edges)
        call = dict(label=['D'], bin=0, pair=['DD'], box=BOX)
        call.update(kwargs)
        try:
            py_compute_cf([data], [ones], sedges, None, 0, **call)
            print(f"  {why}: NO ERROR (unexpected!)")
        except exc as e:
            print(f"  {why:38s} -> {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:60]}")

    # ------------------------------------------------------------------
    header(11, "Verbose output (FCFC's own log)")
    print("  calling with verbose=True ...")
    py_compute_cf([data], [ones], s_edges, None, 0, label=['D'], bin=0,
                  pair=['DD'], box=BOX, verbose=True)
    print("\ntour complete.")


if __name__ == '__main__':
    main()
