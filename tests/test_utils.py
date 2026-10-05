"""Tests for pyfcfc.utils: pair-count accumulation, integration helpers,
and the pycorr state conversion."""

import numpy as np
import pytest

from pyfcfc.boxes import py_compute_cf
from pyfcfc.utils import (add_pair_counts, compute_multipoles, compute_wp,
                          pairs_to_pycorr)


@pytest.fixture
def box_results(rng):
    """pyfcfc results (s, mu) for a data/random pair in a periodic box."""
    box = 500.
    n_d, n_r = 1500, 2000
    data = rng.uniform(0, box, (n_d, 3))
    rand = rng.uniform(0, box, (n_r, 3))
    w_d = rng.uniform(0.5, 1.5, n_d)
    w_r = rng.uniform(0.5, 1.5, n_r)
    s_edges = np.arange(0, 121, 15, dtype=np.float64)
    nmu = 8
    res = py_compute_cf([data, rand], [w_d, w_r], s_edges, None, nmu,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        box=box)
    return res, dict(box=box, s_edges=s_edges, nmu=nmu,
                     n_d=n_d, n_r=n_r, w_d=w_d, w_r=w_r,
                     data=data, rand=rand)


def test_add_pair_counts_first_argument_none(box_results):
    res, meta = box_results
    # None or empty first argument returns a copy of the second
    out = add_pair_counts(None, res)
    assert np.array_equal(out['pairs']['DD'], res['pairs']['DD'])
    out = add_pair_counts({}, res)
    assert np.array_equal(out['pairs']['RR'], res['pairs']['RR'])
    # the input must not be modified by later additions
    dd_before = res['pairs']['DD'].copy()
    add_pair_counts(out, res)
    assert np.array_equal(res['pairs']['DD'], dd_before)


def test_add_pair_counts_doubling(box_results):
    """Adding a result to itself doubles the normalizations and leaves
    the normalized counts unchanged."""
    res, meta = box_results
    out = add_pair_counts(res, res)
    for key in ('DD', 'DR', 'RR'):
        assert np.isclose(out['normalization'][key],
                          2 * res['normalization'][key], rtol=1e-12)
        assert np.allclose(out['pairs'][key], res['pairs'][key], rtol=1e-12)
    for lab in res['labels']:
        assert out['number'][lab] == 2 * res['number'][lab]
        assert np.isclose(out['weighted_number'][lab],
                          2 * res['weighted_number'][lab], rtol=1e-12)


def test_add_pair_counts_matches_combined_run(rng):
    """Accumulating two random-catalogue halves:

    - DD is identical (the data catalogue is the same);
    - DR pair counts are exact, since the splits partition the D x R pairs;
    - the RR normalization is the sum of the split normalizations (the raw
      sums miss the cross-split pairs by construction); the *normalized*
      RR density remains statistically consistent with the full run.
    """
    box = 500.
    data = rng.uniform(0, box, (1200, 3))
    rand = rng.uniform(0, box, (1600, 3))
    w_d, w_r = np.ones(len(data)), np.ones(len(rand))
    s_edges = np.arange(0, 101, 25, dtype=np.float64)
    nmu = 6

    def run(rpos, rw):
        return py_compute_cf([data, rpos], [w_d, rw], s_edges, None, nmu,
                             label=['D', 'R'], bin=1,
                             pair=['DD', 'DR', 'RR'], box=box)

    half = len(rand) // 2
    r1 = run(rand[:half], w_r[:half])
    r2 = run(rand[half:], w_r[half:])
    full = run(rand, w_r)
    comb = add_pair_counts(r1, r2)

    n1, n2 = half, len(rand) - half
    assert np.isclose(comb['normalization']['RR'],
                      n1 * (n1 - 1) + n2 * (n2 - 1), rtol=1e-12)
    assert np.isclose(comb['normalization']['RR'],
                      r1['normalization']['RR'] + r2['normalization']['RR'],
                      rtol=1e-12)
    assert np.isclose(comb['normalization']['DR'],
                      full['normalization']['DR'], rtol=1e-12)

    assert np.allclose(comb['pairs']['DD'], full['pairs']['DD'], rtol=1e-12)

    got = comb['pairs']['DR'] * comb['normalization']['DR']
    exp = full['pairs']['DR'] * full['normalization']['DR']
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-6)

    # normalized RR density statistically consistent with the full run
    assert np.isclose(comb['pairs']['RR'].sum(), full['pairs']['RR'].sum(),
                      rtol=5e-2)

    # stale estimator outputs must be dropped
    assert 'cf' not in comb and 'multipoles' not in comb


def test_add_pair_counts_checks_binning(box_results):
    res, meta = box_results
    other = py_compute_cf([meta['data']], [meta['w_d']],
                          np.arange(0, 101, 10, dtype=np.float64), None,
                          meta['nmu'], label=['D'], bin=1, pair=['DD'],
                          box=meta['box'])
    with pytest.raises(ValueError):
        add_pair_counts(res, other)


def test_compute_multipoles_against_scipy():
    """The midpoint-rule integration must equal an explicit summation."""
    from scipy.special import eval_legendre

    rng = np.random.default_rng(5)
    ns, nmu = 7, 41
    xi = rng.normal(0, 1, (ns, nmu))
    mu = (np.arange(nmu) + 0.5) / nmu
    poles = [0, 2, 4]
    got = compute_multipoles(xi, poles)
    for i, ell in enumerate(poles):
        expected = (2 * ell + 1) * np.sum(
            xi * eval_legendre(ell, mu)[None, :], axis=1) / nmu
        assert np.allclose(got[i], expected, rtol=1e-12, atol=1e-14)
    assert got.shape == (3, ns)

    # the 'exact' scheme uses the per-bin Legendre integrals (pycorr's
    # convention)
    from scipy.special import legendre as legendre_poly
    mu_edges = np.linspace(0, 1, nmu + 1)
    got_exact = compute_multipoles(xi, poles, method='exact')
    for i, ell in enumerate(poles):
        integ = legendre_poly(ell).integ()(mu_edges)
        w = (2 * ell + 1) * (integ[1:] - integ[:-1])
        expected = np.sum(xi * w[None, :], axis=1)
        assert np.allclose(got_exact[i], expected, rtol=1e-12, atol=1e-14)
    # the two schemes agree up to O(dmu^2)
    assert np.max(np.abs(got_exact - got)) < 5e-3

    with pytest.raises(ValueError):
        compute_multipoles(xi, poles, method='trapezoid')
    with pytest.raises(ValueError):
        compute_multipoles(xi, poles, mu_edges=np.linspace(0, 1, nmu))

    # ignore_nan: NaN mu-bins are skipped and the result is rescaled by
    # the covered mu range
    xi_nan = np.ones((ns, nmu))
    xi_nan[:, : nmu // 2] = np.nan
    mp_nan = compute_multipoles(xi_nan, [0], ignore_nan=True)
    assert np.allclose(mp_nan[0], 1.0, rtol=1e-12)
    wp_nan = compute_wp(xi_nan, np.arange(nmu + 1, dtype=np.float64),
                        ignore_nan=True)
    # xi = 1 everywhere -> w_p = 2 * pimax (the NaN bins are rescaled away)
    assert np.allclose(wp_nan, 2 * nmu, rtol=1e-12)
    # without ignore_nan, NaNs propagate
    assert np.all(np.isnan(compute_multipoles(xi_nan, [0])[0]))

    # xi = P_2(mu) must give exactly xi_2 = 1 (orthonormality, midpoint
    # rule is exact for polynomials up to degree 2*nmu-1... here just
    # check the leading behaviour)
    # the midpoint rule has O(dmu^2) error: with nmu bins on [0, 1],
    # integrating P_2 against itself gives 1 up to ~1/(3 nmu^2)
    xi2 = np.tile(eval_legendre(2, mu), (ns, 1))
    got2 = compute_multipoles(xi2, [0, 2])
    assert np.allclose(got2[1], 1.0, rtol=3e-3)
    assert np.allclose(got2[0], 0.0, atol=1e-3)


def test_compute_wp():
    xi = np.ones((5, 10))
    pi_edges = np.arange(0, 101, 10, dtype=np.float64)
    wp = compute_wp(xi, pi_edges)
    # wp = 2 * sum(xi * dpi) = 2 * 100 for xi = 1
    assert np.allclose(wp, 200.0)
    with pytest.raises(ValueError):
        compute_wp(xi, pi_edges[:-1])
    with pytest.raises(ValueError):
        compute_wp(np.ones(10), pi_edges)


def test_pairs_to_pycorr_structure(box_results):
    """The state dictionary has the structure pycorr expects."""
    res, meta = box_results
    state = pairs_to_pycorr(res, 'landyszalay',
                            dict(DD='D1D2', DR=('D1R2', 'R1D2'),
                                 RR='R1R2'), box_size=meta['box'])
    assert state['name'] == 'landyszalay'
    for name in ['D1D2', 'D1R2', 'R1D2', 'R1R2']:
        assert name in state
        st = state[name]
        assert st['mode'] == 'smu'
        ns, nmu = meta['s_edges'].size - 1, meta['nmu']
        assert st['wcounts'].shape == (ns, 2 * nmu)
        assert st['ncounts'].shape == (ns, 2 * nmu)
        assert st['ncounts'].dtype == np.int64
        assert len(st['edges'][1]) == 2 * nmu + 1
        assert np.allclose(st['edges'][1], np.linspace(-1, 1, 2 * nmu + 1))
        # mirror symmetry
        assert np.allclose(st['wcounts'][:, :nmu],
                           st['wcounts'][:, nmu:][:, ::-1], rtol=1e-12)
    # normalizations are carried over
    assert np.isclose(state['D1D2']['wnorm'], res['normalization']['DD'])
    assert np.isclose(state['D1D2']['size1'], res['weighted_number']['D'])
    assert np.isclose(state['D1D2']['size2'], res['weighted_number']['D'])
    assert np.isclose(state['D1R2']['size1'], res['weighted_number']['D'])
    assert np.isclose(state['D1R2']['size2'], res['weighted_number']['R'])


@pytest.mark.pycorr
def test_pairs_to_pycorr_loads(box_results, tmp_path):
    """pycorr can load the converted state and evaluate estimators."""
    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction

    res, meta = box_results
    state = pairs_to_pycorr(res, 'landyszalay',
                            dict(DD='D1D2', DR=('D1R2', 'R1D2'),
                                 RR='R1R2'), box_size=meta['box'])
    fname = str(tmp_path / "state.pkl.npy")
    np.save(fname, state)
    loaded = TwoPointCorrelationFunction.load(fname)

    # the loaded estimator reproduces the xi(s, mu) from the pair counts
    xi_expected = (res['pairs']['DD'] - 2 * res['pairs']['DR']
                   + res['pairs']['RR']) / res['pairs']['RR']
    xi_loaded = loaded.corr[:, meta['nmu']:]
    assert np.allclose(xi_loaded, xi_expected, rtol=1e-10, atol=1e-12)

    # slicing/rebinning works
    rebinned = loaded[::2, ::2]
    assert rebinned.shape == (xi_expected.shape[0] // 2,
                              xi_expected.shape[1])


@pytest.mark.pycorr
def test_pairs_to_pycorr_natural_analytic_rr(box_results):
    """For the natural estimator, analytic RR counts are generated."""
    pytest.importorskip("pycorr")
    res, meta = box_results
    state = pairs_to_pycorr(res, 'natural', dict(DD='D1D2'),
                            box_size=meta['box'])
    assert 'R1R2' in state
    assert state['R1R2']['name'] == 'analytic'
    # the analytic RR total must equal size1 * (size1 - 1)-ish normalization
    rr = state['R1R2']
    frac = rr['wcounts'].sum() / rr['wnorm']
    assert 0 < frac < 1
