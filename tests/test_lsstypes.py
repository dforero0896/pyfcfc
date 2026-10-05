"""Tests for the optional lsstypes conversion (pyfcfc.external).

Skipped when lsstypes is not installed.
"""

import numpy as np
import pytest

lsstypes = pytest.importorskip("lsstypes",
                               reason="lsstypes is not installed")

from pyfcfc.boxes import py_compute_cf  # noqa: E402
from pyfcfc.external import (to_lsstypes, to_lsstypes_correlation,  # noqa: E402
                             to_lsstypes_counts)
from pyfcfc.utils import add_pair_counts, compute_multipoles, compute_wp  # noqa: E402

BOX = 800.0
NMU = 24


@pytest.fixture(scope="module")
def box_results(rng):
    from mocks import make_clustered_box
    data, w = make_clustered_box(12000, BOX, seed=5)
    rand, wr = make_clustered_box(24000, BOX, seed=6, n_centers=1,
                                  clump_frac=0.0)
    s_edges = np.arange(0, 121, 10, dtype=np.float64)
    res = py_compute_cf([data, rand], [w, wr], s_edges, None, NMU,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        box=BOX)
    res_iso = py_compute_cf([data, rand], [w, wr], s_edges, None, 0,
                            label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
                            box=BOX)
    pi_edges = np.arange(0, 81, 10, dtype=np.float64)
    res_spi = py_compute_cf([data, rand], [w, wr], s_edges, pi_edges, 0,
                            label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                            box=BOX)
    return dict(data=data, w=w, rand=rand, wr=wr, s_edges=s_edges,
                pi_edges=pi_edges, res=res, res_iso=res_iso, res_spi=res_spi)


def _xi(res):
    return (res['pairs']['DD'] - 2 * res['pairs']['DR']
            + res['pairs']['RR']) / res['pairs']['RR']


def test_counts_leaves(box_results):
    """Count2 leaves carry the mirrored counts, norms and sizes."""
    res = box_results['res']
    counts = to_lsstypes_counts(res)
    assert set(counts) == {'DD', 'DR', 'RD', 'RR'}
    dd = counts['DD']
    assert list(dd.coords()) == ['s', 'mu']
    assert dd.values('norm')[0, 0] == res['normalization']['DD']
    # positive mu half of the mirrored counts = pyfcfc ordered counts / 2
    half = dd.values('counts')[:, NMU:]
    expected = 0.5 * res['pairs']['DD'] * res['normalization']['DD']
    assert np.allclose(half, expected, rtol=1e-12, atol=1e-6)
    # mirror symmetry
    assert np.allclose(dd.values('counts')[:, :NMU],
                       dd.values('counts')[:, NMU:][:, ::-1])
    assert counts['DD'].attrs['size1'] == res['weighted_number']['D']
    assert counts['DR'].attrs['size2'] == res['weighted_number']['R']
    # RD is the reversed ordering of DR: same mirrored array
    assert np.allclose(counts['RD'].values('counts'),
                       counts['DR'].values('counts'))


def test_correlation_value(box_results):
    """The Landy-Szalay xi(s, mu) of the converted object matches pyfcfc."""
    res = box_results['res']
    corr = to_lsstypes_correlation(res, 'landyszalay')
    assert isinstance(corr, lsstypes.Count2Correlation)
    xi_full = corr.value()
    assert xi_full.shape == (len(box_results['s_edges']) - 1, 2 * NMU)
    assert np.allclose(xi_full[:, NMU:], _xi(res), rtol=1e-10, atol=1e-12)
    # mirror symmetry of xi(s, mu) for this LOS convention
    assert np.allclose(xi_full[:, :NMU], xi_full[:, NMU:][:, ::-1])

    # an explicit estimator formula gives the same result
    corr2 = to_lsstypes_correlation(res, '(DD - DR - RD + RR) / RR')
    assert np.allclose(corr2.value(), xi_full, rtol=1e-12, atol=1e-14)


def test_poles_match_exact_integration(box_results):
    """Count2CorrelationPoles values == utils.compute_multipoles(exact)."""
    res = box_results['res']
    poles = to_lsstypes(res, ells=[0, 2, 4])
    assert isinstance(poles, lsstypes.Count2CorrelationPoles)
    assert poles.ells == [0, 2, 4]
    expected = compute_multipoles(_xi(res), [0, 2, 4], method='exact')
    for ill, ell in enumerate((0, 2, 4)):
        pole = poles.get(ell)
        assert pole.ell == ell
        assert np.allclose(pole.values('value'), expected[ill],
                           rtol=1e-10, atol=1e-12)
        assert np.allclose(pole.s, res['s'], rtol=0, atol=1e-12)


def test_wedges_and_wp(box_results):
    """Wedge and w_p projections work on the converted objects."""
    res = box_results['res']
    wedges = to_lsstypes(res, project='wedges',
                         wedges=[(0.0, 0.5), (0.5, 1.0)])
    assert isinstance(wedges, lsstypes.Count2CorrelationWedges)

    res_spi = box_results['res_spi']
    wp = to_lsstypes(res_spi, project='wp')
    assert isinstance(wp, lsstypes.types.Count2CorrelationWp)
    expected = compute_wp(_xi(res_spi), box_results['pi_edges'])
    assert np.allclose(wp.values('value'), expected, rtol=1e-10, atol=1e-12)


def test_isotropic_binned(box_results):
    res_iso = box_results['res_iso']
    binned = to_lsstypes(res_iso, project='binned')
    assert isinstance(binned, lsstypes.types.Count2CorrelationBinned)
    assert np.allclose(binned.values('value'), _xi(res_iso),
                       rtol=1e-10, atol=1e-12)


def test_natural_analytic_rr(box_results):
    """'natural' with synthesized analytic RR == FCFC's DD / @@ - 1."""
    from mocks import make_clustered_box
    data = make_clustered_box(12000, BOX, seed=5)[0]
    w = np.ones(len(data))
    res = py_compute_cf([data], [w], box_results['s_edges'], None, NMU,
                        label=['D'], bin=1, pair=['DD'],
                        cf=['DD / @@ - 1'], box=BOX)
    corr = to_lsstypes_correlation(res, 'natural', box_size=BOX)
    assert np.allclose(corr.value()[:, NMU:], res['cf'][0],
                       rtol=1e-10, atol=1e-12)
    # without box_size and without RR counts, a clear error is raised
    with pytest.raises(ValueError):
        to_lsstypes_correlation(res, 'natural')


def test_custom_pair_mapping(box_results):
    """Non D/R catalogue labels can be mapped onto estimator names."""
    data, w = box_results['data'], box_results['w']
    rand, wr = box_results['rand'], box_results['wr']
    res = py_compute_cf([data, rand], [w, wr], box_results['s_edges'],
                        None, NMU, label=['A', 'B'], bin=1,
                        pair=['AA', 'AB', 'BB'], box=BOX)
    corr = to_lsstypes_correlation(
        res, 'landyszalay',
        pair_mapping={'AA': 'DD', 'AB': ('DR', 'RD'), 'BB': 'RR'})
    ref = to_lsstypes_correlation(box_results['res'], 'landyszalay')
    assert np.allclose(corr.value(), ref.value(), rtol=1e-12, atol=1e-14)


@pytest.mark.pycorr
def test_against_lsstypes_from_pycorr(box_results):
    """The conversion agrees with lsstypes.external.from_pycorr."""
    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction
    from lsstypes.external import from_pycorr

    res = box_results['res']
    s_edges = box_results['s_edges']
    corr_pycorr = TwoPointCorrelationFunction(
        'smu', (s_edges, np.linspace(-1, 1, 2 * NMU + 1)),
        data_positions1=box_results['data'].T,
        data_weights1=box_results['w'],
        randoms_positions1=box_results['rand'].T,
        randoms_weights1=box_results['wr'],
        position_type='xyz', boxsize=BOX, los='z',
        estimator='landyszalay', engine='corrfunc', nthreads=2,
        compute_sepsavg=False)
    ref = from_pycorr(corr_pycorr)
    got = to_lsstypes_correlation(res, 'landyszalay')
    scale = np.max(np.abs(ref.value()))
    assert np.max(np.abs(got.value() - ref.value())) < 1e-3 * scale + 1e-3


def test_sum_of_unweighted_splits_matches_add_pair_counts(rng):
    """For unweighted, equal-size splits, lsstypes.sum reproduces the
    exact raw-count accumulation of pyfcfc.utils.add_pair_counts."""
    from mocks import make_clustered_box
    data = make_clustered_box(8000, BOX, seed=11)[0]
    rand = make_clustered_box(16000, BOX, seed=12, n_centers=1,
                              clump_frac=0.0)[0]
    w, wr = np.ones(len(data)), np.ones(len(rand))
    s_edges = np.arange(0, 101, 20, dtype=np.float64)
    nmu = 12

    def run(rpos):
        return py_compute_cf([data, rpos], [w, wr[:len(rpos)]], s_edges,
                             None, nmu, label=['D', 'R'], bin=1,
                             pair=['DD', 'DR', 'RR'], box=BOX)

    half = len(rand) // 2
    r1, r2 = run(rand[:half]), run(rand[half:])
    comb = add_pair_counts(r1, r2)
    tot = lsstypes.sum([to_lsstypes_correlation(r1), to_lsstypes_correlation(r2)])
    assert np.allclose(tot.value()[:, nmu:],
                       to_lsstypes_correlation(comb).value()[:, nmu:],
                       rtol=1e-9, atol=1e-10)


def test_write_read_roundtrip(box_results, tmp_path):
    poles = to_lsstypes(box_results['res'], ells=[0, 2])
    fname = str(tmp_path / "poles.pkl")
    lsstypes.write(fname, poles)
    back = lsstypes.read(fname)
    assert isinstance(back, lsstypes.Count2CorrelationPoles)
    for ell in (0, 2):
        assert np.allclose(back.get(ell).values('value'),
                           poles.get(ell).values('value'),
                           rtol=0, atol=0)


def test_replica_covariance(box_results):
    """lsstypes' replica algebra (mean/cov) works on converted poles."""
    res = box_results['res']
    reps = []
    rng = np.random.default_rng(0)
    # cheap replicas: bootstrap-resample the pair counts' noise is not
    # needed; just perturb weights to get distinct realizations
    for seed in range(4):
        w = rng.uniform(0.8, 1.2, len(box_results['data']))
        r = py_compute_cf([box_results['data'], box_results['rand']],
                          [w, box_results['wr']], box_results['s_edges'],
                          None, NMU, label=['D', 'R'], bin=1,
                          pair=['DD', 'DR', 'RR'], box=BOX)
        reps.append(to_lsstypes_correlation(r).project('poles', ells=[0]))
    mean = lsstypes.mean(reps)
    cov = lsstypes.cov(reps)
    assert mean.get(0).values('value').shape == (len(res['s']),)
    assert np.shape(cov.value()) == (len(res['s']), len(res['s']))
