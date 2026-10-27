"""Validate the soft histogramming pair counting method (``soft_hist=True``).

With soft histogramming, the weight of every pair (1 or w1 * w2) is spread
over *all* separation bins with the softmax of a Gaussian kernel centred on
the bins,

    W_i = softmax_i(-(s - c_i)^2 / (2 * sigma^2)),

instead of being assigned to a single bin (``sigma = soft_sigma``; the mu
and pi dimensions of the 2D schemes stay hard-binned).  These tests check
the method against O(N^2) brute-force soft-histogram references, and its
consistency with the standard (hard) histogram method of the C library:

- bin-by-bin agreement with the brute-force references, for the three
  binning schemes, both tree types, unit and random weights, and auto and
  cross pairs (boxes and sky modules);
- conservation of the total pair weight (and, for the 2D schemes, of the
  weight in every mu / pi slab) with respect to the hard method;
- convergence to the hard histogram in the limit ``soft_sigma -> 0``;
- validation of the Python entry points.
"""

import numpy as np
import pytest

from brute_force import (brute_force_box, brute_force_soft_box,
                         brute_force_soft_sky, soft_gaussian_weights)
from conftest import assert_counts_close
from pyfcfc import sky
from pyfcfc.boxes import py_compute_cf


SIGMA = 2.5


def _raw(results, pair):
    """Un-normalized pair counts from a pyfcfc results dictionary."""
    return results['pairs'][pair] * results['normalization'][pair]


# ============================================================================
#  The softmax + Gaussian kernel weights themselves
# ============================================================================

def test_soft_weights_properties():
    """Kernel weights are positive, normalised, and peak on the right bin."""
    edges = np.arange(0, 51, 5, dtype=np.float64)
    s = np.array([2.5, 7.4, 26.0, 49.9])
    W = soft_gaussian_weights(s, edges, SIGMA)
    assert W.shape == (4, 10)
    assert np.all(W > 0)
    assert np.allclose(W.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    # the dominant bin is the one whose centre is nearest to s
    assert np.argmax(W, axis=1).tolist() == [0, 1, 5, 9]
    # a pair at a bin centre gets the largest weight in that bin
    Wc = soft_gaussian_weights(np.array([27.5]), edges, SIGMA)
    assert Wc[0, 5] == Wc.max()
    # tiny sigma -> one-hot on the containing bin
    Wt = soft_gaussian_weights(s, edges, 1e-4)
    assert np.allclose(Wt.max(axis=1), 1.0, atol=1e-12)


# ============================================================================
#  Periodic box: soft counts vs brute-force soft reference
# ============================================================================

@pytest.mark.parametrize("data_struct", [0, 1], ids=["kdtree", "balltree"])
def test_soft_box_iso(box_catalogs, data_struct):
    cat = box_catalogs
    data, rand, box = cat['data'][:900], cat['rand'][:900], cat['box']
    w1, w2 = cat['w_data'][:900], cat['w_rand'][:900]
    s_edges = np.arange(0, 81, 8, dtype=np.float64)

    res = py_compute_cf([data, rand], [w1, w2], s_edges, None, 1,
                        soft_hist=True, soft_sigma=SIGMA,
                        label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
                        box=box, data_struct=data_struct)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR', 'RR'], s_edges, box, SIGMA)
    for pair in ('DD', 'DR', 'RR'):
        assert np.isclose(res['normalization'][pair], ref['norm'][pair],
                          rtol=1e-12)
        assert_counts_close(_raw(res, pair), ref['counts'][pair],
                            rtol=1e-9, atol=1e-6, msg=f"box iso {pair}")


@pytest.mark.parametrize("weighted", [False, True])
def test_soft_box_smu(box_catalogs, weighted):
    cat = box_catalogs
    data, rand, box = cat['data'][:800], cat['rand'][:800], cat['box']
    if weighted:
        w1, w2 = cat['w_data'][:800], cat['w_rand'][:800]
    else:
        w1, w2 = np.ones(len(data)), np.ones(len(rand))
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    nmu = 6
    mu_edges = np.linspace(0, 1, nmu + 1)

    res = py_compute_cf([data, rand], [w1, w2], s_edges, None, nmu,
                        soft_hist=True, soft_sigma=SIGMA,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        box=box)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR', 'RR'], s_edges, box, SIGMA,
                               mu_edges=mu_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = _raw(res, pair)
        assert got.shape == (len(s_edges) - 1, nmu)
        assert_counts_close(got, ref['counts'][pair], rtol=1e-9, atol=1e-6,
                            msg=f"box smu {pair}")

    # The mu dimension stays hard-binned: every mu slab must contain the
    # same total weight as with the standard (hard) method.
    hard = py_compute_cf([data, rand], [w1, w2], s_edges, None, nmu,
                         label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                         box=box)
    for pair in ('DD', 'DR', 'RR'):
        assert_counts_close(_raw(res, pair).sum(axis=0),
                            _raw(hard, pair).sum(axis=0),
                            rtol=1e-9, atol=1e-6,
                            msg=f"box smu {pair}: mu-slab sums")


def test_soft_box_spi(box_catalogs):
    cat = box_catalogs
    data, rand, box = cat['data'][:800], cat['rand'][:800], cat['box']
    w1, w2 = cat['w_data'][:800], cat['w_rand'][:800]
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    pi_edges = np.arange(0, 81, 20, dtype=np.float64)

    res = py_compute_cf([data, rand], [w1, w2], s_edges, pi_edges, 1,
                        soft_hist=True, soft_sigma=SIGMA,
                        label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                        box=box)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR', 'RR'], s_edges, box, SIGMA,
                               pi_edges=pi_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = _raw(res, pair)
        assert got.shape == (len(s_edges) - 1, len(pi_edges) - 1)
        assert_counts_close(got, ref['counts'][pair], rtol=1e-9, atol=1e-6,
                            msg=f"box spi {pair}")


def test_soft_box_nonzero_smin(box_catalogs):
    """Soft counts with smin > 0 (the 'strict minimum' traversal path)."""
    cat = box_catalogs
    data, box = cat['data'][:900], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(20, 101, 10, dtype=np.float64)

    res = py_compute_cf([data], [w], s_edges, None, 1,
                        soft_hist=True, soft_sigma=SIGMA,
                        label=['D'], bin=0, pair=['DD'], box=box)
    ref = brute_force_soft_box([data], [w], ['D'], ['DD'], s_edges, box,
                               SIGMA)
    assert_counts_close(_raw(res, 'DD'), ref['counts']['DD'],
                        rtol=1e-9, atol=1e-6, msg="box iso smin>0")


# ============================================================================
#  Sky module: soft counts vs brute-force soft reference
# ============================================================================

def test_soft_sky_iso(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'][:700], cat['xyz_r'][:700]
    w, w_r = cat['w'][:700], cat['w_r'][:700]
    s_edges = np.arange(0, 201, 25, dtype=np.float64)

    res = sky.py_compute_cf([xyz, xyz_r], [w, w_r], s_edges, None, 1,
                            soft_hist=True, soft_sigma=10.0,
                            label=['D', 'R'], bin=0, pair=['DD', 'DR'])
    ref = brute_force_soft_sky([xyz, xyz_r], [w, w_r], ['D', 'R'],
                               ['DD', 'DR'], s_edges, 10.0)
    for pair in ('DD', 'DR'):
        assert_counts_close(_raw(res, pair), ref['counts'][pair],
                            rtol=1e-9, atol=1e-6, msg=f"sky iso {pair}")


def test_soft_sky_smu(sky_catalogs):
    cat = sky_catalogs
    xyz = cat['xyz'][:700]
    w = cat['w'][:700]
    s_edges = np.arange(0, 201, 25, dtype=np.float64)
    nmu = 4
    mu_edges = np.linspace(0, 1, nmu + 1)

    res = sky.py_compute_cf([xyz], [w], s_edges, None, nmu,
                            soft_hist=True, soft_sigma=10.0,
                            label=['D'], bin=1, pair=['DD'])
    ref = brute_force_soft_sky([xyz], [w], ['D'], ['DD'], s_edges, 10.0,
                               mu_edges=mu_edges)
    got = _raw(res, 'DD')
    assert got.shape == (len(s_edges) - 1, nmu)
    assert_counts_close(got, ref['counts']['DD'], rtol=1e-9, atol=1e-6,
                        msg="sky smu DD")


def test_soft_sky_spi(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'][:700], cat['xyz_r'][:700]
    w, w_r = np.ones(len(xyz)), np.ones(len(xyz_r))
    s_edges = np.arange(0, 151, 30, dtype=np.float64)
    pi_edges = np.arange(0, 151, 30, dtype=np.float64)

    res = sky.py_compute_cf([xyz, xyz_r], [w, w_r], s_edges, pi_edges, 1,
                            soft_hist=True, soft_sigma=8.0,
                            label=['D', 'R'], bin=2, pair=['DR'])
    ref = brute_force_soft_sky([xyz, xyz_r], [w, w_r], ['D', 'R'], ['DR'],
                               s_edges, 8.0, pi_edges=pi_edges)
    got = _raw(res, 'DR')
    assert got.shape == (len(s_edges) - 1, len(pi_edges) - 1)
    assert_counts_close(got, ref['counts']['DR'], rtol=1e-9, atol=1e-6,
                        msg="sky spi DR")


def test_soft_sky_smin_and_balltree(sky_catalogs):
    """Sky soft counts with smin > 0 and the ball tree."""
    cat = sky_catalogs
    xyz = cat['xyz'][:600]
    w = np.ones(len(xyz))
    s_edges = np.arange(50, 201, 25, dtype=np.float64)

    res = sky.py_compute_cf([xyz], [w], s_edges, None, 1,
                            soft_hist=True, soft_sigma=6.0, data_struct=1,
                            label=['D'], bin=0, pair=['DD'])
    ref = brute_force_soft_sky([xyz], [w], ['D'], ['DD'], s_edges, 6.0)
    assert_counts_close(_raw(res, 'DD'), ref['counts']['DD'],
                        rtol=1e-9, atol=1e-6, msg="sky iso smin>0 balltree")


# ============================================================================
#  Consistency with the hard histogram method of the C library
# ============================================================================

def test_soft_conserves_total_weight(box_catalogs):
    """Soft histogramming redistributes, but never creates or loses pairs."""
    cat = box_catalogs
    data, rand, box = cat['data'][:900], cat['rand'][:900], cat['box']
    w1, w2 = cat['w_data'][:900], cat['w_rand'][:900]
    s_edges = np.arange(0, 81, 8, dtype=np.float64)
    common = dict(label=['D', 'R'], pair=['DD', 'DR', 'RR'], box=box)

    hard = py_compute_cf([data, rand], [w1, w2], s_edges, None, 1,
                         bin=0, **common)
    soft = py_compute_cf([data, rand], [w1, w2], s_edges, None, 1,
                         bin=0, soft_hist=True, soft_sigma=SIGMA, **common)
    for pair in ('DD', 'DR', 'RR'):
        hs = _raw(hard, pair).sum()
        ss = _raw(soft, pair).sum()
        assert np.isclose(ss, hs, rtol=1e-11), \
            f"{pair}: soft total {ss} != hard total {hs}"


def test_soft_sigma_to_zero_matches_hard(box_catalogs):
    """In the sigma -> 0 limit the soft histogram converges to the hard one."""
    cat = box_catalogs
    data, box = cat['data'][:900], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(0, 81, 8, dtype=np.float64)
    ds = s_edges[1] - s_edges[0]
    common = dict(label=['D'], bin=0, pair=['DD'], box=box)

    hard = _raw(py_compute_cf([data], [w], s_edges, None, 1, **common), 'DD')
    soft = _raw(py_compute_cf([data], [w], s_edges, None, 1,
                              soft_hist=True, soft_sigma=1e-3 * ds,
                              **common), 'DD')
    # a few pairs sitting within ~1e-3*ds of a bin edge may split between
    # the two neighbouring bins
    assert_counts_close(soft, hard, rtol=1e-9, atol=1.0,
                        msg="sigma -> 0 limit")


def test_soft_default_sigma(box_catalogs):
    """The default kernel width is half the mean separation bin width."""
    cat = box_catalogs
    data, box = cat['data'][:600], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(0, 61, 6, dtype=np.float64)   # ds = 6 -> sigma = 3
    common = dict(label=['D'], bin=0, pair=['DD'], box=box)

    auto = _raw(py_compute_cf([data], [w], s_edges, None, 1,
                              soft_hist=True, **common), 'DD')
    explicit = _raw(py_compute_cf([data], [w], s_edges, None, 1,
                                  soft_hist=True, soft_sigma=3.0, **common),
                    'DD')
    assert np.allclose(auto, explicit, rtol=1e-12, atol=1e-9)


def test_soft_smoothing_effect(box_catalogs):
    """Soft histogramming spreads a monochromatic pair separation.

    Two tight clumps separated by exactly 24.3 give a delta-function
    hard histogram in the bin [20, 30) (pairs at s ~ 0 are excluded by
    smin > 0).  The Gaussian kernel must leak part of that weight into
    the neighbouring bins, broadening the histogram while conserving
    the total.
    """
    box = 400.0
    rng = np.random.default_rng(11)
    n = 400
    base = np.array([100.0, 100.0, 100.0])
    x = np.empty((n, 3))
    x[0::2] = base                                  # clump A
    x[1::2] = base + np.array([24.3, 0.0, 0.0])     # clump B
    x += rng.normal(0, 1e-6, x.shape)               # avoid degenerate splits
    w = np.ones(n)
    s_edges = np.arange(20, 61, 10, dtype=np.float64)
    common = dict(label=['D'], bin=0, pair=['DD'], box=box)

    hard = _raw(py_compute_cf([x], [w], s_edges, None, 1, **common), 'DD')
    soft = _raw(py_compute_cf([x], [w], s_edges, None, 1,
                              soft_hist=True, soft_sigma=3.0, **common), 'DD')
    cen = 0.5 * (s_edges[:-1] + s_edges[1:])

    # the hard histogram is a delta in the first bin, [20, 30)
    assert hard[1] == hard[2] == hard[3] == 0
    assert hard[0] > 0
    # the total weight is conserved ...
    assert np.isclose(soft.sum(), hard.sum(), rtol=1e-11)
    # ... but spread towards the neighbouring bin (nothing lies below
    # smin, so the kernel can only leak upwards)
    assert soft[0] < hard[0]
    assert soft[1] > 0
    # and the histogram is broader (larger second moment around 24.3)
    var_h = (hard * (cen - 24.3) ** 2).sum() / hard.sum()
    var_s = (soft * (cen - 24.3) ** 2).sum() / soft.sum()
    assert var_s > var_h


# ============================================================================
#  Python entry point validation
# ============================================================================

def test_soft_api_validation(box_catalogs):
    cat = box_catalogs
    data, box = cat['data'][:200], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(0, 61, 10, dtype=np.float64)
    common = dict(label=['D'], bin=0, pair=['DD'], box=box)

    def run(**extra):
        return py_compute_cf([data], [w], s_edges, None, 1, **common, **extra)

    # soft_sigma without soft_hist is rejected
    with pytest.raises(ValueError, match="soft_hist"):
        run(soft_sigma=2.0)
    # invalid kernel widths are rejected
    with pytest.raises(ValueError, match="positive finite"):
        run(soft_hist=True, soft_sigma=-1.0)
    with pytest.raises(ValueError, match="positive finite"):
        run(soft_hist=True, soft_sigma=np.inf)
    with pytest.raises(TypeError, match="must be a number"):
        run(soft_hist=True, soft_sigma="2.0")
    with pytest.raises(TypeError, match="must be a number"):
        run(soft_hist=True, soft_sigma=True)
    # the C side rejects invalid SOFT_SIGMA values passed via config keys;
    # SOFT_SIGMA = 0 through the config means "automatic" and must work
    with pytest.raises((ValueError, RuntimeError)):
        run(**{'soft-hist': True, 'soft-sigma': -2.0})
    res_auto = run(**{'soft-hist': True, 'soft-sigma': 0.0})
    assert np.all(np.isfinite(res_auto['pairs']['DD']))
    # unknown options are still caught, with helpful hints
    with pytest.raises(ValueError, match="soft_hist"):
        run(soft=True)
    with pytest.raises(ValueError, match="soft_sigma"):
        run(sigma=1.0)

    # results are finite fractional counts with the standard structure
    res = run(soft_hist=True, soft_sigma=1.5)
    got = res['pairs']['DD']
    assert got.dtype == np.float64
    assert np.all(np.isfinite(got))
    assert np.any(got % 1 != 0)          # fractional counts
    assert res['number']['D'] == len(data)
    assert np.isclose(res['normalization']['DD'],
                      len(data) * (len(data) - 1))


def test_soft_estimator_and_multipole_pipeline(box_catalogs):
    """CF estimators and multipoles work on top of soft pair counts."""
    cat = box_catalogs
    data, rand, box = cat['data'][:600], cat['rand'][:600], cat['box']
    w1, w2 = cat['w_data'][:600], cat['w_rand'][:600]
    s_edges = np.arange(20, 81, 10, dtype=np.float64)   # smin > 0: no empty
    nmu = 8                                             # RR cells -> no NaN
    common = dict(label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                  cf=['(DD - 2 * DR + RR) / RR'], multipole=[0, 2], box=box)

    res = py_compute_cf([data, rand], [w1, w2], s_edges, None, nmu,
                        soft_hist=True, soft_sigma=SIGMA, **common)
    assert res['cf'].shape == (1, len(s_edges) - 1, nmu)
    assert res['multipoles'].shape == (1, 2, len(s_edges) - 1)
    assert np.all(np.isfinite(res['multipoles']))


# ============================================================================
#  Soft mu binning (soft_mu=True): 2D softmax kernel over (s, mu)
# ============================================================================

def test_soft_mu_box_vs_bruteforce(box_catalogs):
    """Soft-soft (s, mu) counts against the brute-force outer product."""
    from brute_force import brute_force_soft_box
    cat = box_catalogs
    data, rand, box = cat['data'][:700], cat['rand'][:700], cat['box']
    w1, w2 = cat['w_data'][:700], cat['w_rand'][:700]
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    nmu = 4
    mu_edges = np.linspace(0, 1, nmu + 1)
    ss, sm = 3.0, 0.15

    res = py_compute_cf([data, rand], [w1, w2], s_edges, None, nmu,
                        soft_hist=True, soft_sigma=ss, soft_mu=True,
                        soft_mu_sigma=sm, label=['D', 'R'], bin=1,
                        pair=['DD', 'DR'], box=box)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR'], s_edges, box, ss,
                               mu_edges=mu_edges, soft_mu_sigma=sm)
    for pair in ('DD', 'DR'):
        assert_counts_close(_raw(res, pair), ref['counts'][pair],
                            rtol=1e-9, atol=1e-6, msg=f"box soft-mu {pair}")

    # total weight is conserved w.r.t. the hard method
    hard = py_compute_cf([data, rand], [w1, w2], s_edges, None, nmu,
                         label=['D', 'R'], bin=1, pair=['DD', 'DR'], box=box)
    for pair in ('DD', 'DR'):
        assert np.isclose(_raw(res, pair).sum(), _raw(hard, pair).sum(),
                          rtol=1e-11)


def test_soft_mu_sigma_to_zero_matches_hard(box_catalogs):
    cat = box_catalogs
    data, box = cat['data'][:600], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    nmu = 5
    common = dict(label=['D'], bin=1, pair=['DD'], box=box)
    hard = _raw(py_compute_cf([data], [w], s_edges, None, nmu, **common), 'DD')
    soft = _raw(py_compute_cf([data], [w], s_edges, None, nmu, soft_hist=True,
                              soft_sigma=1e-3 * 10, soft_mu=True,
                              soft_mu_sigma=1e-4, **common), 'DD')
    assert_counts_close(soft, hard, rtol=1e-9, atol=1.0,
                        msg="sigma_mu -> 0 limit")


def test_soft_mu_sky_vs_bruteforce(sky_catalogs):
    from brute_force import brute_force_soft_sky
    cat = sky_catalogs
    xyz = cat['xyz'][:600]
    w = cat['w'][:600]
    s_edges = np.arange(0, 201, 25, dtype=np.float64)
    nmu = 4
    mu_edges = np.linspace(0, 1, nmu + 1)
    ss, sm = 10.0, 0.15

    res = sky.py_compute_cf([xyz], [w], s_edges, None, nmu, soft_hist=True,
                            soft_sigma=ss, soft_mu=True, soft_mu_sigma=sm,
                            label=['D'], bin=1, pair=['DD'])
    ref = brute_force_soft_sky([xyz], [w], ['D'], ['DD'], s_edges, ss,
                               mu_edges=mu_edges, soft_mu_sigma=sm)
    assert_counts_close(_raw(res, 'DD'), ref['counts']['DD'], rtol=1e-9,
                        atol=1e-6, msg="sky soft-mu DD")


# ============================================================================
#  Soft pi binning (soft_pi=True): 2D softmax kernel over (s_perp, pi)
# ============================================================================

def test_soft_pi_box_vs_bruteforce(box_catalogs):
    from brute_force import brute_force_soft_box
    cat = box_catalogs
    data, rand, box = cat['data'][:700], cat['rand'][:700], cat['box']
    w1, w2 = cat['w_data'][:700], cat['w_rand'][:700]
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    pi_edges = np.arange(0, 81, 20, dtype=np.float64)
    ss, sp = 3.0, 6.0

    res = py_compute_cf([data, rand], [w1, w2], s_edges, pi_edges, 1,
                        soft_hist=True, soft_sigma=ss, soft_pi=True,
                        soft_pi_sigma=sp, label=['D', 'R'], bin=2,
                        pair=['DD', 'DR'], box=box)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR'], s_edges, box, ss,
                               pi_edges=pi_edges, soft_pi_sigma=sp)
    for pair in ('DD', 'DR'):
        assert_counts_close(_raw(res, pair), ref['counts'][pair],
                            rtol=1e-9, atol=1e-6, msg=f"box soft-pi {pair}")
    hard = py_compute_cf([data, rand], [w1, w2], s_edges, pi_edges, 1,
                         label=['D', 'R'], bin=2, pair=['DD', 'DR'], box=box)
    for pair in ('DD', 'DR'):
        assert np.isclose(_raw(res, pair).sum(), _raw(hard, pair).sum(),
                          rtol=1e-11)


def test_soft_pi_box_nonlinear_edges(box_catalogs):
    """Soft pi over non-linear bin edges uses the windowed softmax."""
    from brute_force import brute_force_soft_box
    cat = box_catalogs
    data, rand, box = cat['data'][:600], cat['rand'][:600], cat['box']
    w1, w2 = np.ones(len(data)), np.ones(len(rand))
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    pi_edges = np.array([0., 8., 20., 45., 80.])
    ss, sp = 3.0, 7.0
    res = py_compute_cf([data, rand], [w1, w2], s_edges, pi_edges, 1,
                        soft_hist=True, soft_sigma=ss, soft_pi=True,
                        soft_pi_sigma=sp, label=['D', 'R'], bin=2,
                        pair=['DR'], box=box)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'], ['DR'],
                               s_edges, box, ss, pi_edges=pi_edges,
                               soft_pi_sigma=sp)
    assert_counts_close(_raw(res, 'DR'), ref['counts']['DR'], rtol=1e-9,
                        atol=1e-6, msg="box soft-pi nonlinear")


def test_soft_pi_sky_vs_bruteforce(sky_catalogs):
    from brute_force import brute_force_soft_sky
    cat = sky_catalogs
    xyz = cat['xyz'][:600]
    w = cat['w'][:600]
    dmax = 250.0
    s_edges = np.linspace(0, dmax, 7)
    pi_edges = np.linspace(0, dmax, 5)
    ss, sp = 12.0, 12.0
    res = sky.py_compute_cf([xyz], [w], s_edges, pi_edges, 1, soft_hist=True,
                            soft_sigma=ss, soft_pi=True, soft_pi_sigma=sp,
                            label=['D'], bin=2, pair=['DD'])
    ref = brute_force_soft_sky([xyz], [w], ['D'], ['DD'], s_edges, ss,
                               pi_edges=pi_edges, soft_pi_sigma=sp)
    assert_counts_close(_raw(res, 'DD'), ref['counts']['DD'], rtol=1e-9,
                        atol=1e-6, msg="sky soft-pi DD")


def test_soft_pi_sigma_to_zero_matches_hard(box_catalogs):
    cat = box_catalogs
    data, box = cat['data'][:600], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    pi_edges = np.arange(0, 81, 20, dtype=np.float64)
    common = dict(label=['D'], bin=2, pair=['DD'], box=box)
    hard = _raw(py_compute_cf([data], [w], s_edges, pi_edges, 1, **common),
                'DD')
    soft = _raw(py_compute_cf([data], [w], s_edges, pi_edges, 1,
                              soft_hist=True, soft_sigma=1e-3,
                              soft_pi=True, soft_pi_sigma=1e-4, **common),
                'DD')
    assert_counts_close(soft, hard, rtol=1e-9, atol=1.0,
                        msg="sigma_pi -> 0 limit")


# ============================================================================
#  Scale-dependent kernel width: sigma(s) = f * s (logarithmic bins)
# ============================================================================

def test_soft_sigma_log_box(box_catalogs):
    """Logarithmic separation bins with sigma(s) = f * s."""
    from brute_force import brute_force_soft_box
    cat = box_catalogs
    data, rand, box = cat['data'][:800], cat['rand'][:800], cat['box']
    w1, w2 = cat['w_data'][:800], cat['w_rand'][:800]
    s_edges = np.geomspace(1.0, 80.0, 25)
    f = 0.06
    res = py_compute_cf([data, rand], [w1, w2], s_edges, None, 1,
                        soft_hist=True, soft_sigma=f, soft_sigma_log=True,
                        label=['D', 'R'], bin=0, pair=['DD', 'DR'], box=box)
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR'], s_edges, box, f, sigma_log=True)
    for pair in ('DD', 'DR'):
        assert_counts_close(_raw(res, pair), ref['counts'][pair],
                            rtol=1e-9, atol=1e-6, msg=f"log-sigma {pair}")
    hard = py_compute_cf([data, rand], [w1, w2], s_edges, None, 1,
                         label=['D', 'R'], bin=0, pair=['DD', 'DR'], box=box)
    for pair in ('DD', 'DR'):
        assert np.isclose(_raw(res, pair).sum(), _raw(hard, pair).sum(),
                          rtol=1e-11)


def test_soft_sigma_log_sky(sky_catalogs):
    from brute_force import brute_force_soft_sky
    cat = sky_catalogs
    xyz = cat['xyz'][:700]
    w = cat['w'][:700]
    s_edges = np.geomspace(5.0, 300.0, 20)
    f = 0.08
    res = sky.py_compute_cf([xyz], [w], s_edges, None, 1, soft_hist=True,
                            soft_sigma=f, soft_sigma_log=True,
                            label=['D'], bin=0, pair=['DD'])
    ref = brute_force_soft_sky([xyz], [w], ['D'], ['DD'], s_edges, f,
                               sigma_log=True)
    assert_counts_close(_raw(res, 'DD'), ref['counts']['DD'], rtol=1e-9,
                        atol=1e-6, msg="sky log-sigma DD")
