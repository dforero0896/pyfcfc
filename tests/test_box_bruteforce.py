"""Validate periodic-box pair counts against brute-force O(N^2) counting."""

import numpy as np
import pytest

from brute_force import brute_force_box
from pyfcfc.boxes import py_compute_cf


def _run(cats, wts, s_edges, pedges, nmu, bin_type, labels, pairs,
         box, data_struct=0, **extra):
    kwargs = dict(label=labels, bin=bin_type, pair=list(pairs), box=box,
                  data_struct=data_struct)
    kwargs.update(extra)
    return py_compute_cf(cats, wts, s_edges, pedges, nmu, **kwargs)


@pytest.mark.parametrize("data_struct", [0, 1], ids=["kdtree", "balltree"])
def test_iso_unweighted(box_catalogs, data_struct):
    """Isotropic pair counts, unit weights: exact integer match."""
    cat = box_catalogs
    data, rand, box = cat['data'], cat['rand'], cat['box']
    n1, n2 = len(data), len(rand)
    w1, w2 = np.ones(n1), np.ones(n2)
    s_edges = np.arange(0, 101, 10, dtype=np.float64)

    res = _run([data, rand], [w1, w2], s_edges, None, 0, 0,
               ['D', 'R'], ['DD', 'DR', 'RR'], box, data_struct)
    ref = brute_force_box([data, rand], [w1, w2], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, box)

    assert res['normalization']['DD'] == n1 * (n1 - 1)
    assert res['normalization']['DR'] == n1 * n2
    assert res['normalization']['RR'] == n2 * (n2 - 1)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        assert np.allclose(got, ref['counts'][pair], rtol=0, atol=2), \
            f"{pair}: pyfcfc vs brute force mismatch"


@pytest.mark.parametrize("weighted", [False, True])
def test_smu_weighted(box_catalogs, weighted):
    """(s, mu) pair counts with and without weights."""
    cat = box_catalogs
    data, rand, box = cat['data'], cat['rand'], cat['box']
    w1 = cat['w_data'] if weighted else np.ones(len(data))
    w2 = cat['w_rand'] if weighted else np.ones(len(rand))
    s_edges = np.arange(0, 101, 20, dtype=np.float64)
    nmu = 8
    mu_edges = np.linspace(0, 1, nmu + 1)

    res = _run([data, rand], [w1, w2], s_edges, None, nmu, 1,
               ['D', 'R'], ['DD', 'DR', 'RR'], box)
    ref = brute_force_box([data, rand], [w1, w2], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, box,
                          mu_edges=mu_edges)

    if weighted:
        sw = w1.sum()
        assert np.isclose(res['normalization']['DD'],
                          sw * sw - (w1 ** 2).sum(), rtol=1e-12)
        assert np.isclose(res['normalization']['DR'], sw * w2.sum(),
                          rtol=1e-12)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        rtol = 1e-9 if weighted else 0
        assert np.allclose(got, ref['counts'][pair], rtol=rtol, atol=2), \
            f"{pair}: pyfcfc vs brute force mismatch"


def test_spi_weighted(box_catalogs):
    """(s_perp, pi) pair counts with weights."""
    cat = box_catalogs
    data, rand, box = cat['data'], cat['rand'], cat['box']
    w1, w2 = cat['w_data'], cat['w_rand']
    s_edges = np.arange(0, 81, 20, dtype=np.float64)
    pi_edges = np.arange(0, 81, 20, dtype=np.float64)

    res = _run([data, rand], [w1, w2], s_edges, pi_edges, 0, 2,
               ['D', 'R'], ['DD', 'DR', 'RR'], box)
    ref = brute_force_box([data, rand], [w1, w2], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, box,
                          pi_edges=pi_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        assert np.allclose(got, ref['counts'][pair], rtol=1e-9, atol=2), \
            f"{pair}: pyfcfc vs brute force mismatch"


@pytest.mark.parametrize("edges", [
    np.arange(10, 101, 15, dtype=np.float64),          # smin > 0
    10 ** np.linspace(0.7, 2, 9),                       # logarithmic
    np.array([0, 1, 3, 7, 15, 31, 63, 100.]),           # irregular
], ids=["smin>0", "log", "irregular"])
def test_nonlinear_bins(box_catalogs, edges):
    """Non-uniform binning exercises the hybrid lookup table."""
    cat = box_catalogs
    data, box = cat['data'], cat['box']
    w = cat['w_data']
    nmu = 6

    res = _run([data], [w], edges, None, nmu, 1, ['D'], ['DD'], box)
    ref = brute_force_box([data], [w], ['D'], ['DD'], edges, box,
                          mu_edges=np.linspace(0, 1, nmu + 1))
    got = res['pairs']['DD'] * res['normalization']['DD']
    assert np.allclose(got, ref['counts']['DD'], rtol=1e-9, atol=2)

    # the smin > 0 case also in the isotropic scheme
    res_iso = _run([data], [w], edges, None, 0, 0, ['D'], ['DD'], box)
    ref_iso = brute_force_box([data], [w], ['D'], ['DD'], edges, box)
    got_iso = res_iso['pairs']['DD'] * res_iso['normalization']['DD']
    assert np.allclose(got_iso, ref_iso['counts']['DD'], rtol=1e-9, atol=2)


def test_fractional_width_bins(box_catalogs):
    """Bin edges with fractional widths use the rescaled integer table."""
    cat = box_catalogs
    data, box = cat['data'], cat['box']
    w = np.ones(len(data))
    edges = np.arange(0, 50.5, 2.5)

    res = _run([data], [w], edges, None, 0, 0, ['D'], ['DD'], box)
    ref = brute_force_box([data], [w], ['D'], ['DD'], edges, box)
    got = res['pairs']['DD'] * res['normalization']['DD']
    assert np.allclose(got, ref['counts']['DD'], rtol=0, atol=2)


def test_multipoles_match_external_integration(box_catalogs):
    """FCFC's internal multipole/wp integration matches pyfcfc.utils."""
    from pyfcfc.utils import compute_multipoles, compute_wp

    cat = box_catalogs
    data, rand, box = cat['data'], cat['rand'], cat['box']
    w1, w2 = np.ones(len(data)), np.ones(len(rand))
    s_edges = np.arange(0, 101, 10, dtype=np.float64)
    nmu = 48

    res = py_compute_cf([data, rand], [w1, w2], s_edges, None, nmu,
                        label=['D', 'R'], bin=1,
                        pair=['DD', 'DR', 'RR'],
                        cf=['(DD - 2 * DR + RR) / RR'],
                        multipole=[0, 2, 4], box=box)
    ext = compute_multipoles(res['cf'][0], [0, 2, 4])
    assert res['poles'] == [0, 2, 4]
    assert np.allclose(ext, res['multipoles'][0], rtol=1e-10, atol=1e-12)

    pi_edges = np.arange(0, 101, 10, dtype=np.float64)
    res3 = py_compute_cf([data, rand], [w1, w2], s_edges, pi_edges, 0,
                         label=['D', 'R'], bin=2,
                         pair=['DD', 'DR', 'RR'],
                         cf=['(DD - 2 * DR + RR) / RR'], wp=True, box=box)
    ext_wp = compute_wp(res3['cf'][0], pi_edges)
    assert np.allclose(ext_wp, res3['projected'][0], rtol=1e-10, atol=1e-10)


def test_xi_of_uniform_random_is_small(box_catalogs):
    """DD / @@ - 1 for a uniform random catalogue must be ~0."""
    cat = box_catalogs
    rand, box = cat['rand'], cat['box']
    w = np.ones(len(rand))
    s_edges = np.arange(0, 101, 10, dtype=np.float64)

    res = py_compute_cf([rand], [w], s_edges, None, 0, label=['R'], bin=0,
                        pair=['RR'], cf=['RR / @@ - 1'], box=box)
    xi = res['cf'][0]
    # per-bin statistical scatter: sigma = 1/sqrt(expected pair count)
    n = len(rand)
    shell = 4. / 3. * np.pi * (s_edges[1:] ** 3 - s_edges[:-1] ** 3)
    expected = n * (n - 1) * shell / box ** 3
    sigma = 1. / np.sqrt(expected)
    assert np.all(np.abs(xi) < 4.5 * sigma), \
        f"xi deviates from 0: {xi / sigma}"
    assert np.all(np.isfinite(xi))


@pytest.mark.parametrize("bin_type,pedges", [
    (0, None), (1, None), (2, np.arange(0, 81, 20, dtype=np.float64))],
    ids=["iso", "smu", "spi"])
def test_all_requested_pairs_computed_regardless_of_estimator(
        box_catalogs, bin_type, pedges):
    """Pair counts are the main product: every entry of `pair` must be
    evaluated and returned even when the CF estimator references only a
    subset of them (or none at all).

    The isotropic case also passes ``nmu > 1`` on purpose: together with
    an `@@` estimator this used to overflow the analytic-RR buffer
    (nmu was not clamped outside the (s, mu) scheme), corrupting memory.
    """
    cat = box_catalogs
    data, rand, box = cat['data'], cat['rand'], cat['box']
    w1, w2 = np.ones(len(data)), np.ones(len(rand))
    s_edges = np.arange(0, 101, 20, dtype=np.float64)
    nmu = 4
    kwargs = dict(label=['D', 'R'], bin=bin_type, pair=['DD', 'DR', 'RR'],
                  box=box)
    # estimator references only DD
    res_cf = py_compute_cf([data, rand], [w1, w2], s_edges, pedges, nmu,
                           cf=['DD / @@ - 1'], **kwargs)
    # no estimator at all
    res_no = py_compute_cf([data, rand], [w1, w2], s_edges, pedges, nmu,
                           **kwargs)

    for res in (res_cf, res_no):
        assert sorted(k for k in res['pairs'] if len(k) == 2) == \
            ['DD', 'DR', 'RR']
        assert sorted(res['normalization']) == ['DD', 'DR', 'RR']
        for pair in ('DD', 'DR', 'RR'):
            total = (res['pairs'][pair] * res['normalization'][pair]).sum()
            assert total > 0, f"{pair} counts are empty"
    # the estimator must not change the counts themselves
    for pair in ('DD', 'DR', 'RR'):
        assert np.array_equal(res_cf['pairs'][pair],
                              res_no['pairs'][pair]), pair
        assert res_cf['normalization'][pair] == \
            res_no['normalization'][pair], pair
    # ... while the estimator products are present only when requested
    assert 'cf' in res_cf and 'cf' not in res_no


def test_pair_counts_without_estimator_respect_binning(box_catalogs):
    """The binning scheme must be honoured even without a CF estimator."""
    cat = box_catalogs
    data, box = cat['data'], cat['box']
    w = np.ones(len(data))
    s_edges = np.arange(0, 101, 20, dtype=np.float64)
    nmu = 5

    res = py_compute_cf([data], [w], s_edges, None, nmu, label=['D'],
                        bin=1, pair=['DD'], box=box)
    assert res['pairs']['DD'].shape == (len(s_edges) - 1, nmu)
    assert 'mumin' in res['pairs']
