"""Validate the edge padding (``soft_pad``) of the soft histogram method.

Without padding, the tree traversal only counts pairs inside the binning
range ``[sedges[0], sedges[-1])`` while the Gaussian kernel has infinite
support: the truncated kernel tails bias the outermost bins (for a steep
xi(s), the last bin gains excess weight -- "leakage").  With
``soft_pad = z > 0`` the counting range is extended by z kernel widths
using buffer bins that continue the separation grid; the softmax
normalises over the real and buffer centres, the buffer-bin counts are
discarded, and the returned results refer to the original bins.

These tests check:

- the padding plan (buffer-bin counts, grid continuation, box clamp,
  lower-edge clamp at zero, default-kernel pinning);
- the padded wrapper against the equivalent manual run on the extended
  grid (all binning schemes, both modules, both tree types) and against
  the O(N^2) brute-force references;
- the API defaults (soft_pad=None == SOFT_PAD_DEFAULT, soft_pad=0 keeps
  the legacy behaviour exactly);
- weight conservation on the extended grid and convergence to the hard
  histogram as sigma -> 0;
- the result-dictionary slicing (counts, cf, multipoles, wp, bin arrays);
- the adjoint (VJP) with padding: finite-difference gradchecks, and
  equivalence with the manual extended-grid adjoint, including the
  weight and kernel-width cotangents;
- the input validation.
"""

import numpy as np
import pytest

from brute_force import brute_force_soft_box, brute_force_soft_sky
from conftest import assert_counts_close
from pyfcfc import sky
from pyfcfc._softpad import SOFT_PAD_DEFAULT, pad_cotangent, plan_padding
from pyfcfc.boxes import py_compute_cf, soft_hist_vjp

SIGMA = 2.5
PAD = 2.0


def _raw(results, pair):
    return results['pairs'][pair] * results['normalization'][pair]


def _max_edge(box):
    return 0.5 * float(box) * (1.0 - 1e-9)


# ============================================================================
#  The padding plan
# ============================================================================

def test_plan_padding_zero_is_identity():
    edges = np.arange(0, 61, 10, dtype=np.float64)
    ext, nlo, nhi, sig = plan_padding(edges, 0.0, SIGMA, False, None)
    assert ext is edges
    assert (nlo, nhi, sig) == (0, 0, None)


def test_plan_padding_linear_default_sigma():
    edges = np.arange(0, 61, 10, dtype=np.float64)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, None, False, None)
    # default sigma = 5 -> pad = 10 -> one buffer bin per side, but the
    # lower edge cannot go below zero
    assert sig == 5.0                      # pinned library default
    assert (nlo, nhi) == (0, 1)
    assert np.allclose(ext, [0, 10, 20, 30, 40, 50, 60, 70])


def test_plan_padding_linear_interior():
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, SIGMA, False, None)
    assert sig is None                     # explicit sigma: nothing pinned
    assert (nlo, nhi) == (1, 1)            # ceil(2*2.5/10) = 1
    assert np.allclose(ext[:2], [10, 20]) and np.allclose(ext[-2:], [100, 110])


def test_plan_padding_log_scale_free():
    edges = np.geomspace(5.0, 60.0, 13)
    du = np.log(edges[1] / edges[0])
    f = 0.5 * du                           # library default for sigma_log
    ext, nlo, nhi, sig = plan_padding(edges, PAD, f, True, None)
    assert (nlo, nhi) == (1, 1)            # 2f = one log bin
    # geometric continuation with the local edge spacing
    assert np.allclose(np.diff(np.log(ext)), du, rtol=1e-12)
    assert np.isclose(ext[0], edges[0] * np.exp(-du))
    assert np.isclose(ext[-1], edges[-1] * np.exp(du))


def test_plan_padding_nonuniform_local_spacing():
    # last bin much wider than the first: the continuation uses the local
    # spacing at each end
    edges = np.array([10., 12., 14., 16., 18., 25., 40.])
    ext, nlo, nhi, sig = plan_padding(edges, 1.0, 2.0, False, None)
    assert sig is None                     # explicit sigma: nothing pinned
    assert (nlo, nhi) == (1, 1)            # ceil(1*2/2) = ceil(1*2/15) = 1
    assert np.isclose(ext[0], 8.0)         # 10 - 2
    assert np.isclose(ext[-1], 55.0)       # 40 + 15
    # non-uniform bins: the default kernel width must be pinned from the
    # ORIGINAL grid (half the mean width = 2.5), not from the extended one
    ext, nlo, nhi, sig = plan_padding(edges, 1.0, None, False, None)
    assert np.isclose(sig, 2.5)
    assert (nlo, nhi) == (2, 1)            # ceil(2.5/2), ceil(2.5/15)
    assert np.allclose(ext[:3], [6., 8., 10.])


def test_plan_padding_box_clamp():
    edges = np.arange(140, 191, 10, dtype=np.float64)   # smax = 190
    # box = 400: only 10 Mpc/h of room below L/2 = 200, one bin fits
    ext, nlo, nhi, sig = plan_padding(edges, 3.0, 15.0, False, 200.0)
    assert nhi == 1                        # ceil(45/10)=5 wanted, 1 fits
    assert ext[-1] <= 200.0
    # no room at all -> upper padding silently dropped
    edges2 = np.arange(140, 200, 10, dtype=np.float64)  # smax = 190? no: 190
    ext2, nlo2, nhi2, _ = plan_padding(np.arange(150, 199, 12, dtype=float),
                                       3.0, 15.0, False, 200.0)
    assert ext2[-1] < 200.0


def test_plan_padding_sigma_log_needs_positive_smin_for_default():
    edges = np.arange(0, 61, 10, dtype=np.float64)
    with pytest.raises(ValueError, match="positive"):
        plan_padding(edges, PAD, None, True, None)
    # with an explicit f it works; no geometric continuation below 0
    ext, nlo, nhi, sig = plan_padding(edges, PAD, 0.2, True, None)
    assert nlo == 0 and nhi >= 1


# ============================================================================
#  Padded counts == manual run on the extended grid (sliced)
# ============================================================================

BOX_COMMON = dict(label=['D', 'R'], pair=['DD', 'DR', 'RR'])


def _manual_box(cat, edges, sigma, nmu, pedges, bintype, soft_kw,
                data_struct=0, pad=PAD):
    data, rand = cat['data'][:800], cat['rand'][:800]
    w1, w2 = cat['w_data'][:800], cat['w_rand'][:800]
    box = cat['box']
    kw = dict(soft_hist=True, soft_sigma=sigma, soft_pad=pad, bin=bintype,
              nmu_kw=None, box=box, data_struct=data_struct, **soft_kw)
    kw.pop('nmu_kw')
    wrapped = py_compute_cf([data, rand], [w1, w2], edges, pedges, nmu, **kw,
                            **BOX_COMMON)
    ext, nlo, nhi, sig = plan_padding(edges, pad, sigma,
                                      bool(soft_kw.get('soft_sigma_log')),
                                      _max_edge(box))
    kw2 = dict(kw)
    kw2['soft_pad'] = 0
    if sig is not None:
        kw2['soft_sigma'] = sig
    manual = py_compute_cf([data, rand], [w1, w2], ext, pedges, nmu, **kw2,
                           **BOX_COMMON)
    return wrapped, manual, nlo, len(edges) - 1


@pytest.mark.parametrize("data_struct", [0, 1], ids=["kdtree", "balltree"])
def test_pad_box_iso(box_catalogs, data_struct):
    cat = box_catalogs
    edges = np.arange(20, 101, 10, dtype=np.float64)
    wrapped, manual, nlo, ns = _manual_box(cat, edges, SIGMA, 1, None, 0,
                                           {}, data_struct)
    for pair in ('DD', 'DR', 'RR'):
        got = _raw(wrapped, pair)
        exp = _raw(manual, pair)[nlo:nlo + ns]
        assert got.shape == (ns,)
        assert np.allclose(got, exp, rtol=1e-12, atol=1e-9), pair


def test_pad_box_smu(box_catalogs):
    cat = box_catalogs
    edges = np.arange(20, 101, 10, dtype=np.float64)
    nmu = 6
    wrapped, manual, nlo, ns = _manual_box(cat, edges, SIGMA, nmu, None, 1, {})
    for pair in ('DD', 'DR', 'RR'):
        got = _raw(wrapped, pair)
        exp = _raw(manual, pair)[nlo:nlo + ns]
        assert got.shape == (ns, nmu)
        assert np.allclose(got, exp, rtol=1e-12, atol=1e-9), pair


def test_pad_box_spi(box_catalogs):
    cat = box_catalogs
    edges = np.arange(20, 101, 10, dtype=np.float64)
    pedges = np.arange(0, 41, 10, dtype=np.float64)
    wrapped, manual, nlo, ns = _manual_box(cat, edges, SIGMA, 1, pedges, 2, {})
    for pair in ('DD', 'DR', 'RR'):
        got = _raw(wrapped, pair)
        exp = _raw(manual, pair)[nlo:nlo + ns]
        assert got.shape == (ns, len(pedges) - 1)
        assert np.allclose(got, exp, rtol=1e-12, atol=1e-9), pair


def test_pad_box_sigma_log(box_catalogs):
    cat = box_catalogs
    edges = np.geomspace(25.0, 95.0, 9)
    f = 0.5 * np.log(edges[1] / edges[0])
    wrapped, manual, nlo, ns = _manual_box(
        cat, edges, f, 1, None, 0, dict(soft_sigma_log=True))
    for pair in ('DD', 'RR'):
        got = _raw(wrapped, pair)
        exp = _raw(manual, pair)[nlo:nlo + ns]
        assert np.allclose(got, exp, rtol=1e-12, atol=1e-9), pair


def test_pad_box_soft_mu(box_catalogs):
    cat = box_catalogs
    edges = np.arange(20, 101, 10, dtype=np.float64)
    nmu = 4
    wrapped, manual, nlo, ns = _manual_box(
        cat, edges, SIGMA, nmu, None, 1,
        dict(soft_mu=True, soft_mu_sigma=0.12))
    got = _raw(wrapped, 'DD')
    exp = _raw(manual, 'DD')[nlo:nlo + ns]
    assert got.shape == (ns, nmu)
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-9)


def test_pad_box_default_sigma(box_catalogs):
    """soft_sigma=None: the default kernel width must be pinned from the
    original grid (the buffer bins must not change it)."""
    cat = box_catalogs
    data, rand = cat['data'][:600], cat['rand'][:600]
    w1, w2 = cat['w_data'][:600], cat['w_rand'][:600]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    box = cat['box']
    common = dict(label=['D', 'R'], bin=0, pair=['DD'], box=box)
    wrapped = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                            soft_hist=True, soft_pad=PAD, **common)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, None, False,
                                      _max_edge(box))
    assert sig == 5.0
    manual = py_compute_cf([data, rand], [w1, w2], ext, None, 1,
                           soft_hist=True, soft_sigma=sig, soft_pad=0,
                           **common)
    ns = len(edges) - 1
    assert np.allclose(_raw(wrapped, 'DD'),
                       _raw(manual, 'DD')[nlo:nlo + ns], rtol=1e-12, atol=1e-9)


def test_pad_sky_iso(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'][:700], cat['xyz_r'][:700]
    w, w_r = cat['w'][:700], cat['w_r'][:700]
    edges = np.arange(25, 176, 25, dtype=np.float64)
    common = dict(label=['D', 'R'], bin=0, pair=['DD', 'DR'])
    wrapped = sky.py_compute_cf([xyz, xyz_r], [w, w_r], edges, None, 1,
                                soft_hist=True, soft_sigma=10.0,
                                soft_pad=PAD, **common)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, 10.0, False, None)
    manual = sky.py_compute_cf([xyz, xyz_r], [w, w_r], ext, None, 1,
                               soft_hist=True, soft_sigma=10.0, soft_pad=0,
                               **common)
    ns = len(edges) - 1
    for pair in ('DD', 'DR'):
        assert np.allclose(_raw(wrapped, pair),
                           _raw(manual, pair)[nlo:nlo + ns],
                           rtol=1e-12, atol=1e-9), pair


def test_pad_sky_spi(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'][:600], cat['xyz_r'][:600]
    w, w_r = np.ones(len(xyz)), np.ones(len(xyz_r))
    edges = np.arange(30, 151, 30, dtype=np.float64)
    pedges = np.arange(0, 121, 30, dtype=np.float64)
    common = dict(label=['D', 'R'], bin=2, pair=['DR'])
    wrapped = sky.py_compute_cf([xyz, xyz_r], [w, w_r], edges, pedges, 1,
                                soft_hist=True, soft_sigma=8.0,
                                soft_pad=PAD, **common)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, 8.0, False, None)
    manual = sky.py_compute_cf([xyz, xyz_r], [w, w_r], ext, pedges, 1,
                               soft_hist=True, soft_sigma=8.0, soft_pad=0,
                               **common)
    ns = len(edges) - 1
    got = _raw(wrapped, 'DR')
    exp = _raw(manual, 'DR')[nlo:nlo + ns]
    assert got.shape == (ns, len(pedges) - 1)
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-9)


# ============================================================================
#  Brute-force references on the extended grid
# ============================================================================

def test_pad_box_vs_bruteforce(box_catalogs):
    cat = box_catalogs
    data, rand = cat['data'][:700], cat['rand'][:700]
    w1, w2 = cat['w_data'][:700], cat['w_rand'][:700]
    box = cat['box']
    edges = np.arange(20, 101, 10, dtype=np.float64)
    res = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                        soft_hist=True, soft_sigma=SIGMA, soft_pad=PAD,
                        label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
                        box=box)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, SIGMA, False,
                                      _max_edge(box))
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'DR', 'RR'], ext, box, SIGMA)
    ns = len(edges) - 1
    for pair in ('DD', 'DR', 'RR'):
        assert_counts_close(_raw(res, pair),
                            ref['counts'][pair][nlo:nlo + ns],
                            rtol=1e-9, atol=1e-6, msg=f"padded box {pair}")


def test_pad_box_sigma_log_vs_bruteforce(box_catalogs):
    cat = box_catalogs
    data, rand = cat['data'][:700], cat['rand'][:700]
    w1, w2 = cat['w_data'][:700], cat['w_rand'][:700]
    box = cat['box']
    edges = np.geomspace(25.0, 95.0, 9)
    f = 0.5 * np.log(edges[1] / edges[0])
    res = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                        soft_hist=True, soft_sigma=f, soft_sigma_log=True,
                        soft_pad=PAD, label=['D', 'R'], bin=0,
                        pair=['DD', 'RR'], box=box)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, f, True, _max_edge(box))
    ref = brute_force_soft_box([data, rand], [w1, w2], ['D', 'R'],
                               ['DD', 'RR'], ext, box, f, sigma_log=True)
    ns = len(edges) - 1
    for pair in ('DD', 'RR'):
        assert_counts_close(_raw(res, pair),
                            ref['counts'][pair][nlo:nlo + ns],
                            rtol=1e-9, atol=1e-6, msg=f"padded log {pair}")


def test_pad_sky_vs_bruteforce(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'][:600], cat['xyz_r'][:600]
    w, w_r = cat['w'][:600], cat['w_r'][:600]
    edges = np.arange(25, 176, 25, dtype=np.float64)
    res = sky.py_compute_cf([xyz, xyz_r], [w, w_r], edges, None, 1,
                            soft_hist=True, soft_sigma=10.0, soft_pad=PAD,
                            label=['D', 'R'], bin=0, pair=['DD', 'DR'])
    ext, nlo, nhi, sig = plan_padding(edges, PAD, 10.0, False, None)
    ref = brute_force_soft_sky([xyz, xyz_r], [w, w_r], ['D', 'R'],
                               ['DD', 'DR'], ext, 10.0)
    ns = len(edges) - 1
    for pair in ('DD', 'DR'):
        assert_counts_close(_raw(res, pair),
                            ref['counts'][pair][nlo:nlo + ns],
                            rtol=1e-9, atol=1e-6, msg=f"padded sky {pair}")


# ============================================================================
#  Defaults, conservation, convergence, box clamp
# ============================================================================

def test_default_pad_equals_explicit(box_catalogs):
    cat = box_catalogs
    data, rand = cat['data'][:600], cat['rand'][:600]
    w1, w2 = cat['w_data'][:600], cat['w_rand'][:600]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    common = dict(soft_hist=True, soft_sigma=SIGMA, label=['D', 'R'],
                  bin=0, pair=['DD'], box=cat['box'])
    res_def = py_compute_cf([data, rand], [w1, w2], edges, None, 1, **common)
    res_exp = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                            soft_pad=SOFT_PAD_DEFAULT, **common)
    res_off = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                            soft_pad=0, **common)
    assert np.allclose(_raw(res_def, 'DD'), _raw(res_exp, 'DD'),
                       rtol=1e-12, atol=1e-9)
    assert not np.allclose(_raw(res_def, 'DD'), _raw(res_off, 'DD'))


def test_pad_conserves_weight_on_extended_grid(box_catalogs):
    """Softmax conserves the total pair weight over the *extended* grid;
    the real bins alone hold less (the rest sits in the buffer bins)."""
    cat = box_catalogs
    data, rand = cat['data'][:800], cat['rand'][:800]
    w1, w2 = cat['w_data'][:800], cat['w_rand'][:800]
    box = cat['box']
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ext, nlo, nhi, sig = plan_padding(edges, PAD, SIGMA, False,
                                      _max_edge(box))
    common = dict(label=['D', 'R'], bin=0, pair=['DD'], box=box)
    soft = py_compute_cf([data, rand], [w1, w2], ext, None, 1,
                         soft_hist=True, soft_sigma=SIGMA, soft_pad=0,
                         **common)
    hard = py_compute_cf([data, rand], [w1, w2], ext, None, 1, **common)
    s_soft = _raw(soft, 'DD').sum()
    s_hard = _raw(hard, 'DD').sum()
    assert np.isclose(s_soft, s_hard, rtol=1e-11)
    # ... and the wrapped call keeps only the real-bin share
    wrapped = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                            soft_hist=True, soft_sigma=SIGMA, soft_pad=PAD,
                            **common)
    assert _raw(wrapped, 'DD').sum() < s_soft


def test_pad_converges_to_hard_histogram(box_catalogs):
    cat = box_catalogs
    data, rand = cat['data'][:600], cat['rand'][:600]
    w1, w2 = cat['w_data'][:600], cat['w_rand'][:600]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ds = edges[1] - edges[0]
    common = dict(label=['D', 'R'], bin=0, pair=['DD'], box=cat['box'])
    soft = py_compute_cf([data, rand], [w1, w2], edges, None, 1,
                         soft_hist=True, soft_sigma=1e-3 * ds, soft_pad=PAD,
                         **common)
    hard = py_compute_cf([data, rand], [w1, w2], edges, None, 1, **common)
    assert_counts_close(_raw(soft, 'DD'), _raw(hard, 'DD').astype(float),
                        rtol=1e-9, atol=1e-6, msg="sigma->0 with padding")


def test_pad_box_clamp_integration(box_catalogs):
    """smax close to L/2: the upper padding is clamped (possibly to zero)
    instead of failing the FCFC validation."""
    cat = box_catalogs
    data = cat['data'][:800]
    w = np.ones(len(data))
    box = cat['box']
    edges = np.arange(140, 191, 10, dtype=np.float64)   # smax = 190, L/2=200
    common = dict(label=['D'], bin=0, pair=['DD'], box=box)
    res = py_compute_cf([data], [w], edges, None, 1, soft_hist=True,
                        soft_sigma=15.0, soft_pad=3.0, **common)
    ns = len(edges) - 1
    assert _raw(res, 'DD').shape == (ns,)
    assert np.all(np.isfinite(_raw(res, 'DD')))
    # equivalent manual run: the plan must have clamped nhi
    ext, nlo, nhi, sig = plan_padding(edges, 3.0, 15.0, False,
                                      _max_edge(box))
    assert ext[-1] <= 0.5 * box
    manual = py_compute_cf([data], [w], ext, None, 1, soft_hist=True,
                           soft_sigma=15.0, soft_pad=0, **common)
    assert np.allclose(_raw(res, 'DD'), _raw(manual, 'DD')[nlo:nlo + ns],
                       rtol=1e-12, atol=1e-9)


# ============================================================================
#  Results-dictionary slicing (cf estimators, multipoles, wp, bin arrays)
# ============================================================================

def test_pad_results_structure(box_catalogs):
    cat = box_catalogs
    data, rand = cat['data'][:600], cat['rand'][:600]
    w1, w2 = cat['w_data'][:600], cat['w_rand'][:600]
    edges = np.arange(20, 81, 10, dtype=np.float64)
    ns, nmu = len(edges) - 1, 8
    res = py_compute_cf([data, rand], [w1, w2], edges, None, nmu,
                        soft_hist=True, soft_sigma=SIGMA, soft_pad=PAD,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        cf=['(DD - 2 * DR + RR) / RR'], multipole=[0, 2],
                        box=cat['box'])
    assert res['cf'].shape == (1, ns, nmu)
    assert res['multipoles'].shape == (1, 2, ns)
    assert np.all(np.isfinite(res['multipoles']))
    assert res['pairs']['DD'].shape == (ns, nmu)
    assert np.allclose(res['pairs']['smin'], edges[:-1][:, None])
    assert np.allclose(res['pairs']['smax'], edges[1:][:, None])
    assert res['pairs']['mumin'].shape == (ns, nmu)
    assert np.allclose(res['s'], 0.5 * (edges[:-1] + edges[1:]))

    # wp on top of the (s_perp, pi) scheme
    pedges = np.arange(0, 41, 10, dtype=np.float64)
    res2 = py_compute_cf([data, rand], [w1, w2], edges, pedges, 1,
                         soft_hist=True, soft_sigma=SIGMA, soft_pad=PAD,
                         label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                         cf=['(DD - 2 * DR + RR) / RR'], wp=True,
                         box=cat['box'])
    assert res2['projected'].shape == (1, ns)
    assert np.all(np.isfinite(res2['projected']))
    assert res2['pairs']['pimin'].shape == (ns, len(pedges) - 1)


# ============================================================================
#  The adjoint with padding
# ============================================================================

def _fd_directional(loss, x, v, eps=1e-6):
    return (loss(x + eps * v) - loss(x - eps * v)) / (2 * eps)


def test_pad_vjp_gradcheck_box(box_catalogs):
    cat = box_catalogs
    d, r = cat['data'][:400], cat['rand'][:400]
    w1, w2 = cat['w_data'][:400], cat['w_rand'][:400]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ns = len(edges) - 1
    rng = np.random.default_rng(11)
    g = rng.normal(size=(ns,))
    common = dict(label=['A', 'B'], bin=0, box=cat['box'])
    fwd = dict(soft_hist=True, soft_sigma=SIGMA, soft_pad=PAD, pair=['AB'],
               **common)

    def loss(x):
        res = py_compute_cf([x, r], [w1, w2], edges, None, 1, **fwd)
        return float(np.sum(g * _raw(res, 'AB')))

    v = rng.normal(size=d.shape)
    fd = _fd_directional(loss, d, v)
    out = soft_hist_vjp([d, r], [w1, w2], edges, None, 1, pair='AB',
                        cotangent=g, soft_sigma=SIGMA, soft_pad=PAD,
                        bin=0, box=cat['box'])
    an = float(np.sum(out['grads']['A'] * v))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * max(1.0, abs(fd)))


def test_pad_vjp_gradcheck_box_smu(box_catalogs):
    cat = box_catalogs
    d, r = cat['data'][:400], cat['rand'][:400]
    w1, w2 = cat['w_data'][:400], cat['w_rand'][:400]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ns, nmu = len(edges) - 1, 4
    rng = np.random.default_rng(12)
    g = rng.normal(size=(ns, nmu))
    fwd = dict(soft_hist=True, soft_sigma=SIGMA, soft_pad=PAD, pair=['AB'],
               label=['A', 'B'], bin=1, box=cat['box'])

    def loss(x):
        res = py_compute_cf([x, r], [w1, w2], edges, None, nmu, **fwd)
        return float(np.sum(g * _raw(res, 'AB')))

    v = rng.normal(size=d.shape)
    fd = _fd_directional(loss, d, v)
    out = soft_hist_vjp([d, r], [w1, w2], edges, None, nmu, pair='AB',
                        cotangent=g, soft_sigma=SIGMA, soft_pad=PAD,
                        bin=1, box=cat['box'])
    an = float(np.sum(out['grads']['A'] * v))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * max(1.0, abs(fd)))


def test_pad_vjp_gradcheck_sky(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'][:400], cat['xyz_r'][:400]
    w, w_r = cat['w'][:400], cat['w_r'][:400]
    edges = np.arange(25, 176, 25, dtype=np.float64)
    ns = len(edges) - 1
    rng = np.random.default_rng(13)
    g = rng.normal(size=(ns,))
    fwd = dict(soft_hist=True, soft_sigma=10.0, soft_pad=PAD, pair=['AB'],
               label=['A', 'B'], bin=0)

    def loss(x):
        res = sky.py_compute_cf([x, xyz_r], [w, w_r], edges, None, 1, **fwd)
        return float(np.sum(g * _raw(res, 'AB')))

    v = rng.normal(size=xyz.shape)
    fd = _fd_directional(loss, xyz, v)
    out = sky.soft_hist_vjp([xyz, xyz_r], [w, w_r], edges, None, 1,
                            pair='AB', cotangent=g, soft_sigma=10.0,
                            soft_pad=PAD, bin=0)
    an = float(np.sum(out['grads']['A'] * v))
    assert np.isclose(an, fd, rtol=1e-4, atol=1e-7 * max(1.0, abs(fd)))


def test_pad_vjp_equals_manual_extension(box_catalogs):
    """The wrapper's adjoint equals the adjoint on the manually extended
    grid with zero cotangents on the buffer bins (coords, weights and
    kernel width)."""
    cat = box_catalogs
    d, r = cat['data'][:500], cat['rand'][:500]
    w1, w2 = cat['w_data'][:500], cat['w_rand'][:500]
    box = cat['box']
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ns = len(edges) - 1
    rng = np.random.default_rng(14)
    g = rng.normal(size=(ns,))

    out = soft_hist_vjp([d, r], [w1, w2], edges, None, 1, pair='AB',
                        cotangent=g, soft_sigma=SIGMA, soft_pad=PAD,
                        grad_wts=True, grad_sigma=True, bin=0, box=box,
                        label=['A', 'B'])

    ext, nlo, nhi, sig = plan_padding(edges, PAD, SIGMA, False,
                                      _max_edge(box))
    g_ext = pad_cotangent(g, nlo, len(ext) - 1)
    out2 = soft_hist_vjp([d, r], [w1, w2], ext, None, 1, pair='AB',
                         cotangent=g_ext, soft_sigma=SIGMA, soft_pad=0,
                         grad_wts=True, grad_sigma=True, bin=0, box=box,
                         label=['A', 'B'])

    scale = np.abs(out2['grads']['A']).max() + 1.0
    for lab in ('A', 'B'):
        assert np.allclose(out['grads'][lab], out2['grads'][lab],
                           rtol=1e-9, atol=1e-9 * scale)
        assert np.allclose(out['grad_wts'][lab], out2['grad_wts'][lab],
                           rtol=1e-9, atol=1e-9 * scale)
    assert np.isclose(out['grad_sigma'], out2['grad_sigma'], rtol=1e-9,
                      atol=1e-9 * abs(out2['grad_sigma']))


def test_pad_vjp_normalized_flag(box_catalogs):
    cat = box_catalogs
    d, r = cat['data'][:400], cat['rand'][:400]
    w1, w2 = cat['w_data'][:400], cat['w_rand'][:400]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ns = len(edges) - 1
    rng = np.random.default_rng(15)
    g = rng.normal(size=(ns,))
    res = py_compute_cf([d, r], [w1, w2], edges, None, 1, soft_hist=True,
                        soft_sigma=SIGMA, soft_pad=PAD, label=['A', 'B'],
                        bin=0, pair=['AB'], box=cat['box'])
    norm = res['normalization']['AB']
    o_raw = soft_hist_vjp([d, r], [w1, w2], edges, None, 1, pair='AB',
                          cotangent=g / norm, soft_pad=PAD,
                          soft_sigma=SIGMA, bin=0, box=cat['box'],
                          label=['A', 'B'])
    o_nrm = soft_hist_vjp([d, r], [w1, w2], edges, None, 1, pair='AB',
                          cotangent=g, normalized=True, soft_pad=PAD,
                          soft_sigma=SIGMA, bin=0, box=cat['box'],
                          label=['A', 'B'])
    for lab in ('A', 'B'):
        assert np.allclose(o_raw['grads'][lab], o_nrm['grads'][lab],
                           rtol=1e-12, atol=1e-14)


# ============================================================================
#  Validation
# ============================================================================

def test_pad_api_validation(box_catalogs):
    cat = box_catalogs
    data = cat['data'][:200]
    w = np.ones(len(data))
    edges = np.arange(0, 61, 10, dtype=np.float64)
    common = dict(label=['D'], bin=0, pair=['DD'], box=cat['box'])

    def run(**extra):
        return py_compute_cf([data], [w], edges, None, 1, **common, **extra)

    with pytest.raises(ValueError, match="non-negative"):
        run(soft_hist=True, soft_pad=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        run(soft_hist=True, soft_pad=np.nan)
    with pytest.raises(TypeError, match="must be a number"):
        run(soft_hist=True, soft_pad="2.0")
    with pytest.raises(TypeError, match="must be a number"):
        run(soft_hist=True, soft_pad=True)
    with pytest.raises(ValueError, match="soft_hist"):
        run(soft_pad=2.0)
    # a configuration file hides the box size / kernel options from the
    # wrapper: explicit padding must be rejected
    with pytest.raises(ValueError, match="configuration file"):
        run(soft_hist=True, soft_pad=2.0, conf='does-not-matter.conf')
    # soft_pad=0 keeps working everywhere
    res = run(soft_hist=True, soft_sigma=1.5, soft_pad=0)
    assert np.all(np.isfinite(res['pairs']['DD']))


def test_pad_vjp_validation(box_catalogs):
    cat = box_catalogs
    data = cat['data'][:200]
    w = np.ones(len(data))
    edges = np.arange(0, 61, 10, dtype=np.float64)
    ns = len(edges) - 1
    g = np.ones(ns)
    with pytest.raises(ValueError, match="non-negative"):
        soft_hist_vjp([data], [w], edges, None, 1, pair='DD', cotangent=g,
                      soft_sigma=1.5, soft_pad=-2.0, bin=0, box=cat['box'],
                      label=['D'])
    with pytest.raises(ValueError, match="configuration file"):
        soft_hist_vjp([data], [w], edges, None, 1, pair='DD', cotangent=g,
                      soft_sigma=1.5, soft_pad=2.0, bin=0, label=['D'],
                      conf='does-not-matter.conf')


# ============================================================================
#  The JAX interface (skipped when JAX is not installed)
# ============================================================================

def test_pad_jax_module(box_catalogs):
    pjax = pytest.importorskip("pyfcfc.jax")
    import jax
    import jax.numpy as jnp
    cat = box_catalogs
    d, r = cat['data'][:300], cat['rand'][:300]
    w1, w2 = cat['w_data'][:300], cat['w_rand'][:300]
    edges = np.arange(20, 101, 10, dtype=np.float64)
    ns = len(edges) - 1

    kw = dict(module="boxes", s_edges=edges, pairs=("AB",),
              labels=["A", "B"], soft_sigma=SIGMA, soft_pad=PAD,
              bin=0, box=cat['box'])
    c_pad = pjax.soft_pair_counts([d, r], [w1, w2], **kw)
    c_off = pjax.soft_pair_counts([d, r], [w1, w2],
                                  **{**kw, 'soft_pad': 0.0})
    ref = py_compute_cf([d, r], [w1, w2], edges, None, 1, soft_hist=True,
                        soft_sigma=SIGMA, soft_pad=PAD, label=['A', 'B'],
                        bin=0, pair=['AB'], box=cat['box'])
    assert np.allclose(np.asarray(c_pad["AB"]), _raw(ref, 'AB'),
                       rtol=1e-12, atol=1e-9)
    assert not np.allclose(np.asarray(c_pad["AB"]), np.asarray(c_off["AB"]))

    # gradients flow through the padded forward
    g = jnp.asarray(np.random.default_rng(16).normal(size=(ns,)))

    def loss(x):
        return jnp.sum(g * pjax.soft_pair_counts([x, r], [w1, w2],
                                                 **kw)["AB"])

    gx = jax.grad(loss)(jnp.asarray(d))
    assert np.all(np.isfinite(np.asarray(gx)))
