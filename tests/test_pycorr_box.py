"""Cross-validate pyfcfc against pycorr (Corrfunc engine) in a periodic box.

These tests run the *same* catalogues through both codes, so the
differences are purely algorithmic (binning conventions at the edges,
multipole integration schemes) rather than statistical.

Skipped when pycorr or Corrfunc are not installed.
"""

import copy

import numpy as np
import pytest

from conftest import assert_curves_close

pycorr = pytest.importorskip("pycorr", reason="pycorr is not installed")
pytest.importorskip("Corrfunc", reason="Corrfunc (pycorr engine) missing")

from pycorr import TwoPointCorrelationFunction  # noqa: E402

from pyfcfc.boxes import py_compute_cf  # noqa: E402
from pyfcfc.utils import (add_pair_counts, compute_multipoles,  # noqa: E402
                          pairs_to_pycorr)

NTHREADS = 2
BOX = 1000.0
MAPPING = dict(DD='D1D2', DR=('D1R2', 'R1D2'), RR='R1R2')


def _from_state(state):
    """Build a pycorr estimator from a state dict (without mutating it)."""
    return TwoPointCorrelationFunction.from_state(copy.deepcopy(state))


@pytest.fixture(scope="module")
def box_data(rng):
    """A larger, mildly clustered catalogue + randoms in a 1 Gpc/h box."""
    n_data, n_rand = 24000, 36000
    n_clump = n_data // 2
    n_centers = 40
    centers = rng.uniform(0, BOX, (n_centers, 3))
    assign = rng.integers(0, n_centers, n_clump)
    clumpy = centers[assign] + rng.normal(0, 20.0, (n_clump, 3))
    uniform_part = rng.uniform(0, BOX, (n_data - n_clump, 3))
    data = np.concatenate([clumpy, uniform_part], axis=0) % BOX
    rand = rng.uniform(0, BOX, (n_rand, 3))
    w_data = rng.uniform(0.8, 1.2, n_data)
    w_rand = rng.uniform(0.8, 1.2, n_rand)
    return data, rand, w_data, w_rand


def _smu_edges(nmu):
    return np.arange(0, 161, 10.), np.linspace(-1, 1, 2 * nmu + 1)


def _assert_multipoles_close(fc_mp, corr, ells, msg, rtol=5e-3, floor=3e-3):
    """Compare multipoles with a tolerance that scales with the local
    monopole amplitude.

    corrfunc bins floating-point mu while FCFC bins the squared mu with
    integer (rescaled) arithmetic, so a tiny fraction of pairs land in
    neighbouring mu bins in one code or the other.  The impact on a
    multipole is proportional to xi at that scale, hence the tolerance.
    """
    # The residual difference between the codes is the integration
    # scheme (midpoint rule vs exact per-bin Legendre integrals): it is
    # O(dmu^2) times the xi(s, mu) variation within a bin, which at low
    # signal is set by the shot noise of the (s, mu) cells, not by the
    # local xi_0.  A tolerance proportional to the *global* xi_0 amplitude
    # covers both regimes while still catching any real bug (which would
    # be O(1)).
    amp = np.abs(corr(ell=0)).max()
    for i, ell in enumerate(ells):
        exp = corr(ell=ell)
        diff = np.abs(np.asarray(fc_mp[i]) - exp)
        allowed = floor * amp + rtol * np.abs(exp)
        bad = diff > allowed
        assert not bad.any(), (
            f"{msg} xi_{ell}: {int(bad.sum())} bins differ; max diff "
            f"{diff.max():.4g}, max allowed {allowed.max():.4g}")



def test_paircounts_smu_vs_pycorr(box_data):
    """Raw (s, mu) pair counts agree with Corrfunc through pycorr."""
    data, rand, w_data, w_rand = box_data
    s_edges, mu_edges = _smu_edges(32)
    nmu = 32

    fc = py_compute_cf([data, rand], [w_data, w_rand], s_edges, None, nmu,
                       label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                       box=BOX)
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=data.T, data_weights1=w_data,
        randoms_positions1=rand.T, randoms_weights1=w_rand,
        position_type='xyz', boxsize=BOX, los='z', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    # compare via the pycorr-state conversion
    state = pairs_to_pycorr(fc, 'landyszalay', MAPPING, box_size=BOX)
    loaded = _from_state(state)

    for name in ['D1D2', 'R1R2']:
        got = getattr(loaded, name).wcounts
        exp = getattr(corr, name).wcounts
        assert got.shape == exp.shape, name
        scale = exp.max()
        assert np.max(np.abs(got - exp)) < 2e-3 * scale + 2, name
        assert np.isclose(getattr(loaded, name).wnorm,
                          getattr(corr, name).wnorm, rtol=1e-10), name

    # the reversible cross counts are mapped onto both D1R2 and R1D2;
    # their sum must match pycorr's
    got = loaded.D1R2.wcounts + loaded.R1D2.wcounts
    exp = corr.D1R2.wcounts + corr.R1D2.wcounts
    assert np.max(np.abs(got - exp)) < 2e-3 * exp.max() + 2


def test_multipoles_natural_vs_pycorr(box_data):
    """xi_0/2/4 with the natural estimator (analytic RR) match pycorr."""
    data, rand, w_data, w_rand = box_data
    s_edges, mu_edges = _smu_edges(48)
    nmu = 48

    fc = py_compute_cf([data], [w_data], s_edges, None, nmu, label=['D'],
                       bin=1, pair=['DD'], cf=['DD / @@ - 1'],
                       multipole=[0, 2, 4], box=BOX)
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=data.T, data_weights1=w_data,
        position_type='xyz', boxsize=BOX, los='z', estimator='natural',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    sep_c, _ = corr(ell=0, return_sep=True)
    assert np.allclose(sep_c, fc['s'], rtol=0, atol=1e-9)
    # FCFC's internal (midpoint-rule) integration: agreement up to the
    # O(dmu^2) difference of the integration schemes
    _assert_multipoles_close(fc['multipoles'][0], corr, [0, 2, 4],
                             msg="natural")
    # with pycorr's exact Legendre integration applied to FCFC's own
    # xi(s, mu), the multipoles must match to machine precision
    mp_exact = compute_multipoles(fc['cf'][0], [0, 2, 4], method='exact')
    for i, ell in enumerate([0, 2, 4]):
        assert np.allclose(mp_exact[i], corr(ell=ell), rtol=1e-9,
                           atol=1e-11), f"natural xi_{ell}, exact integration"


def test_multipoles_landyszalay_vs_pycorr(box_data):
    """xi_0/2/4 with the Landy-Szalay estimator match pycorr."""
    data, rand, w_data, w_rand = box_data
    s_edges, mu_edges = _smu_edges(48)
    nmu = 48

    fc = py_compute_cf([data, rand], [w_data, w_rand], s_edges, None, nmu,
                       label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'],
                       multipole=[0, 2, 4], box=BOX)
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=data.T, data_weights1=w_data,
        randoms_positions1=rand.T, randoms_weights1=w_rand,
        position_type='xyz', boxsize=BOX, los='z', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    sep, _ = corr(ell=0, return_sep=True)
    assert np.allclose(sep, fc['s'], rtol=0, atol=1e-9)
    _assert_multipoles_close(fc['multipoles'][0], corr, [0, 2, 4], msg="LS")

    # external integration of the pyfcfc pair counts must reproduce
    # FCFC's internal multipoles (midpoint rule), and match pycorr to
    # machine precision with the exact integration scheme
    xi_smu = (fc['pairs']['DD'] - 2 * fc['pairs']['DR']
              + fc['pairs']['RR']) / fc['pairs']['RR']
    mp_ext = compute_multipoles(xi_smu, [0, 2, 4])
    assert np.allclose(mp_ext, fc['multipoles'][0], rtol=1e-10, atol=1e-12)
    mp_exact = compute_multipoles(xi_smu, [0, 2, 4], method='exact')
    for i, ell in enumerate([0, 2, 4]):
        assert np.allclose(mp_exact[i], corr(ell=ell), rtol=1e-9,
                           atol=1e-11), f"LS xi_{ell}, exact integration"


def test_2d_xi_rppi_vs_pycorr(box_data):
    """The 2D correlation function xi(s_perp, pi) matches pycorr bin-by-bin.

    pyfcfc stores pi >= 0 only; pycorr's estimator corr array spans the
    mirrored full range, so the positive half is compared.
    """
    data, rand, w_data, w_rand = box_data
    s_edges = np.arange(0, 121, 10.)
    pimax = 60.
    npi = 12
    pi_edges = np.linspace(0, pimax, npi + 1)

    fc = py_compute_cf([data, rand], [w_data, w_rand], s_edges, pi_edges, 0,
                       label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'], box=BOX)
    corr = TwoPointCorrelationFunction(
        'rppi', (s_edges, np.linspace(-pimax, pimax, 2 * npi + 1)),
        data_positions1=data.T, data_weights1=w_data,
        randoms_positions1=rand.T, randoms_weights1=w_rand,
        position_type='xyz', boxsize=BOX, los='z', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS, compute_sepsavg=False)

    xi2d = fc['cf'][0]
    assert xi2d.shape == (len(s_edges) - 1, npi)
    xi2d_pc = corr.corr[:, npi:]
    # pair counts are identical between the codes, so the estimator
    # ratios agree to floating-point precision
    assert np.allclose(xi2d, xi2d_pc, rtol=1e-8, atol=1e-10)
    # and the mirrored negative half equals the positive one
    assert np.allclose(corr.corr[:, :npi], xi2d_pc[:, ::-1],
                       rtol=1e-8, atol=1e-10)


def test_wp_vs_pycorr(box_data):
    """The projected correlation function w_p matches pycorr."""
    data, rand, w_data, w_rand = box_data
    s_edges = np.arange(0, 121, 10.)
    pimax = 80.
    npi = 16
    pi_edges = np.linspace(0, pimax, npi + 1)

    fc = py_compute_cf([data, rand], [w_data, w_rand], s_edges, pi_edges, 0,
                       label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'], wp=True, box=BOX)
    corr = TwoPointCorrelationFunction(
        'rppi', (s_edges, np.linspace(-pimax, pimax, 2 * npi + 1)),
        data_positions1=data.T, data_weights1=w_data,
        randoms_positions1=rand.T, randoms_weights1=w_rand,
        position_type='xyz', boxsize=BOX, los='z', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    sep, wp_c = corr(mode='wp', return_sep=True)
    assert np.allclose(sep, fc['s'], rtol=0, atol=1e-9)
    assert_curves_close(fc['projected'][0], wp_c, rtol=1e-2,
                        msg="w_p (pyfcfc vs pycorr)")


def test_iso_s_mode_vs_pycorr(box_data):
    """Isotropic xi(s) in a periodic box matches pycorr's 's' mode."""
    data, rand, w_data, w_rand = box_data
    s_edges = np.arange(0, 161, 10.)

    fc = py_compute_cf([data, rand], [w_data, w_rand], s_edges, None, 0,
                       label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'], box=BOX)
    corr = TwoPointCorrelationFunction(
        's', (s_edges,),
        data_positions1=data.T, data_weights1=w_data,
        randoms_positions1=rand.T, randoms_weights1=w_rand,
        position_type='xyz', boxsize=BOX, estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)
    sep, xi_c = corr(return_sep=True)
    assert np.allclose(sep, fc['s'], rtol=0, atol=1e-9)
    assert_curves_close(fc['cf'][0], xi_c, rtol=1e-2,
                        msg="xi(s) (pyfcfc vs pycorr)")

    # also compare the converted state's wcounts (mode 's': auto counts
    # are halved because pycorr stores unordered auto pairs)
    state = pairs_to_pycorr(fc, 'landyszalay', MAPPING, box_size=BOX)
    loaded = _from_state(state)
    for name in ['D1D2', 'R1R2']:
        exp = getattr(corr, name).wcounts.ravel()
        got = getattr(loaded, name).wcounts.ravel()
        assert np.max(np.abs(got - exp)) < 2e-3 * exp.max() + 2, name


def test_pycorr_state_roundtrip_and_rebin(box_data, tmp_path):
    """Save the converted state, reload with pycorr, and rebin."""
    data, rand, w_data, w_rand = box_data
    s_edges, mu_edges = _smu_edges(32)
    nmu = 32

    fc = py_compute_cf([data, rand], [w_data, w_rand], s_edges, None, nmu,
                       label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                       box=BOX)
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=data.T, data_weights1=w_data,
        randoms_positions1=rand.T, randoms_weights1=w_rand,
        position_type='xyz', boxsize=BOX, los='z', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    state = pairs_to_pycorr(fc, 'landyszalay', MAPPING, box_size=BOX)
    fname = str(tmp_path / "pyfcfc_state.pkl.npy")
    np.save(fname, state)
    loaded = TwoPointCorrelationFunction.load(fname)

    # rebin by 2 in s and 4 in mu, then compare the multipoles
    loaded2 = loaded[::2, ::4]
    corr2 = corr[::2, ::4]
    for ell in [0, 2]:
        s_a, xi_a = loaded2(ell=ell, return_sep=True)
        s_b, xi_b = corr2(ell=ell, return_sep=True)
        assert np.allclose(s_a, s_b, rtol=0, atol=1e-9)
        assert_curves_close(xi_a, xi_b, rtol=1e-2,
                            msg=f"rebinned xi_{ell} (pyfcfc vs pycorr)")


def test_split_random_accumulation(box_data):
    """add_pair_counts over random splits reproduces the full measurement."""
    data, rand, w_data, w_rand = box_data
    s_edges = np.arange(0, 121, 20.)
    nmu = 8

    def run(rpos, rw):
        return py_compute_cf([data, rpos], [w_data, rw], s_edges, None, nmu,
                             label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                             box=BOX)

    full = run(rand, w_rand)

    n_split = 3
    total = None
    for i in range(n_split):
        sl = slice(i * len(rand) // n_split, (i + 1) * len(rand) // n_split)
        piece = run(rand[sl], w_rand[sl])
        total = piece if total is None else add_pair_counts(total, piece)

    # DD is identical (same data catalogue)
    assert np.allclose(total['pairs']['DD'], full['pairs']['DD'], rtol=1e-12)
    # DR is exact: the splits partition the D x R pair set
    got = total['pairs']['DR'] * total['normalization']['DR']
    exp = full['pairs']['DR'] * full['normalization']['DR']
    assert np.allclose(got, exp, rtol=1e-9, atol=2)
    assert np.isclose(total['normalization']['DR'],
                      full['normalization']['DR'], rtol=1e-12)
    # RR: the accumulated normalization is the sum of the split ones;
    # the normalized density remains statistically consistent
    assert total['normalization']['RR'] < full['normalization']['RR']
    assert np.isclose(total['pairs']['RR'].sum(), full['pairs']['RR'].sum(),
                      rtol=5e-2)

    # xi0 from the accumulated counts matches the full-run xi0 within
    # statistical fluctuations
    def xi0(res):
        xi = (res['pairs']['DD'] - 2 * res['pairs']['DR']
              + res['pairs']['RR']) / res['pairs']['RR']
        return compute_multipoles(xi, [0])[0]

    assert_curves_close(xi0(total), xi0(full), rtol=5e-2, atol_frac=5e-3,
                        msg="split-accumulated xi0")
