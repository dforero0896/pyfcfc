"""Cross-validate the survey-like pyfcfc component against pycorr.

FCFC's survey-like pair counting uses the midpoint line-of-sight
convention, so pycorr is run with ``los='midpoint'`` on Cartesian
coordinates converted with the same fiducial cosmology
(Omega_m = 0.31, Omega_Lambda = 0.69, w = -1, c/H0 = 2997.92458 Mpc/h).
"""

import copy

import numpy as np
import pytest

from conftest import assert_curves_close

pycorr = pytest.importorskip("pycorr", reason="pycorr is not installed")
pytest.importorskip("Corrfunc", reason="Corrfunc (pycorr engine) missing")

from pycorr import TwoPointCorrelationFunction  # noqa: E402

from pyfcfc.sky import py_compute_cf  # noqa: E402
from pyfcfc.utils import pairs_to_pycorr  # noqa: E402

NTHREADS = 2
MAPPING = dict(DD='D1D2', DR=('D1R2', 'R1D2'), RR='R1R2')


def test_sky_paircounts_vs_pycorr(sky_catalogs):
    """(s, mu) counts with coordinate conversion match pycorr/corrfunc."""
    cat = sky_catalogs
    s_edges = np.arange(0, 151, 15.)
    nmu = 20
    mu_edges = np.linspace(-1, 1, 2 * nmu + 1)

    fc = py_compute_cf([cat['rdd'], cat['rdd_r']], [cat['w'], cat['w_r']],
                       s_edges, None, nmu, label=['D', 'R'], bin=1,
                       pair=['DD', 'DR', 'RR'], convert=True,
                       omega_m=cat['omega_m'], omega_l=cat['omega_l'],
                       eos_w=cat['eos_w'])
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=cat['xyz'].T, data_weights1=cat['w'],
        randoms_positions1=cat['xyz_r'].T, randoms_weights1=cat['w_r'],
        position_type='xyz', los='midpoint', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    state = pairs_to_pycorr(fc, 'landyszalay', MAPPING)
    loaded = TwoPointCorrelationFunction.from_state(copy.deepcopy(state))

    for name in ['D1D2', 'R1R2']:
        got = getattr(loaded, name).wcounts
        exp = getattr(corr, name).wcounts
        assert got.shape == exp.shape, name
        assert np.max(np.abs(got - exp)) < 2e-3 * exp.max() + 3, name
        assert np.isclose(getattr(loaded, name).wnorm,
                          getattr(corr, name).wnorm, rtol=1e-10), name
    got = loaded.D1R2.wcounts + loaded.R1D2.wcounts
    exp = corr.D1R2.wcounts + corr.R1D2.wcounts
    assert np.max(np.abs(got - exp)) < 2e-3 * exp.max() + 3


def test_sky_multipoles_landyszalay_vs_pycorr(sky_catalogs):
    """Landy-Szalay multipoles from (RA, Dec, z) match pycorr."""
    cat = sky_catalogs
    s_edges = np.arange(0, 151, 15.)
    nmu = 40
    mu_edges = np.linspace(-1, 1, 2 * nmu + 1)

    fc = py_compute_cf([cat['rdd'], cat['rdd_r']], [cat['w'], cat['w_r']],
                       s_edges, None, nmu, label=['D', 'R'], bin=1,
                       pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'],
                       multipole=[0, 2, 4], convert=True,
                       omega_m=cat['omega_m'], omega_l=cat['omega_l'],
                       eos_w=cat['eos_w'])
    corr = TwoPointCorrelationFunction(
        'smu', (s_edges, mu_edges),
        data_positions1=cat['xyz'].T, data_weights1=cat['w'],
        randoms_positions1=cat['xyz_r'].T, randoms_weights1=cat['w_r'],
        position_type='xyz', los='midpoint', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    sep, _ = corr(ell=0, return_sep=True)
    assert np.allclose(sep, fc['s'], rtol=0, atol=1e-9)
    for i, ell in enumerate([0, 2, 4]):
        assert_curves_close(fc['multipoles'][0, i], corr(ell=ell),
                            rtol=1e-2,
                            msg=f"sky LS xi_{ell} (pyfcfc vs pycorr)")
    # With pycorr's exact Legendre integration scheme applied to FCFC's
    # own xi(s, mu), the remaining differences come only from the tiny
    # (relative ~1e-8) differences between FCFC's spline-based comoving
    # distance integration and the reference quadrature used to feed
    # pycorr, which move a few pairs across bin edges.
    from pyfcfc.utils import compute_multipoles
    xi_smu = fc['cf'][0]
    # some (s, mu) cells may have zero random pairs at these small
    # catalogue sizes -> NaN; ignore_nan matches pycorr's behaviour
    mp_exact = compute_multipoles(xi_smu, [0, 2, 4], method='exact',
                                  ignore_nan=True)
    for i, ell in enumerate([0, 2, 4]):
        assert_curves_close(mp_exact[i], corr(ell=ell), rtol=2e-3,
                            atol_frac=1e-3,
                            msg=f"sky xi_{ell}, exact integration")


def test_sky_wp_vs_pycorr(sky_catalogs):
    """Projected correlation function for survey-like data matches."""
    cat = sky_catalogs
    s_edges = np.arange(0, 121, 20.)
    pimax = 100.
    npi = 10
    pi_edges = np.linspace(0, pimax, npi + 1)

    fc = py_compute_cf([cat['rdd'], cat['rdd_r']], [cat['w'], cat['w_r']],
                       s_edges, pi_edges, 0, label=['D', 'R'], bin=2,
                       pair=['DD', 'DR', 'RR'],
                       cf=['(DD - 2 * DR + RR) / RR'], wp=True,
                       convert=True, omega_m=cat['omega_m'],
                       omega_l=cat['omega_l'], eos_w=cat['eos_w'])
    corr = TwoPointCorrelationFunction(
        'rppi', (s_edges, np.linspace(-pimax, pimax, 2 * npi + 1)),
        data_positions1=cat['xyz'].T, data_weights1=cat['w'],
        randoms_positions1=cat['xyz_r'].T, randoms_weights1=cat['w_r'],
        position_type='xyz', los='midpoint', estimator='landyszalay',
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=False)

    sep, wp_c = corr(mode='wp', return_sep=True)
    assert np.allclose(sep, fc['s'], rtol=0, atol=1e-9)
    assert_curves_close(fc['projected'][0], wp_c, rtol=2e-2,
                        msg="sky w_p (pyfcfc vs pycorr)")
    # the 2D xi(s_perp, pi) itself (positive pi half of pycorr's range)
    assert_curves_close(fc['cf'][0], corr.corr[:, npi:], rtol=2e-3,
                        atol_frac=1e-3, msg="sky 2D xi(rp, pi)")


def test_sky_split_randoms_workflow(sky_catalogs, tmp_path):
    """The classic survey workflow: split randoms, accumulate with
    add_pair_counts, convert to a pycorr state, reload and integrate.

    Mirrors the legacy ``test/pyfcfc-v-pycorr.py`` script of the original
    port, with quantitative checks:

    - DR counts accumulate exactly (the splits partition D x R);
    - the RR normalization is the sum of the split normalizations;
    - multipoles from the reloaded state match those of a single full
      run within the statistical noise of the missing cross-split RR
      pairs.
    """
    import copy

    from mocks import make_survey
    from pyfcfc.utils import add_pair_counts, pairs_to_pycorr
    from pycorr import TwoPointCorrelationFunction

    cat = sky_catalogs
    rdd, w, _ = make_survey(6000, seed=21, z_range=(0.4, 0.55))
    rdd_r, w_r, _ = make_survey(24000, seed=22, z_range=(0.4, 0.55),
                                clustered=False)
    s_edges = np.arange(0, 151, 30.)
    nmu = 10

    def run(rr, wr):
        return py_compute_cf([rdd, rr], [w, wr], s_edges, None, nmu,
                             label=['D', 'R'], bin=1,
                             pair=['DD', 'DR', 'RR'], convert=True,
                             omega_m=cat['omega_m'], omega_l=cat['omega_l'],
                             eos_w=cat['eos_w'])

    half = len(rdd_r) // 2
    r1, r2 = run(rdd_r[:half], w_r[:half]), run(rdd_r[half:], w_r[half:])
    total = add_pair_counts(r1, r2)
    full = run(rdd_r, w_r)

    # DR accumulates exactly
    got = total['pairs']['DR'] * total['normalization']['DR']
    exp = full['pairs']['DR'] * full['normalization']['DR']
    assert np.allclose(got, exp, rtol=1e-9, atol=2)
    assert np.isclose(total['normalization']['DR'],
                      full['normalization']['DR'], rtol=1e-12)
    # RR normalization is the sum of the split normalizations
    assert np.isclose(total['normalization']['RR'],
                      r1['normalization']['RR'] + r2['normalization']['RR'],
                      rtol=1e-12)

    # convert, save, reload, integrate
    state = pairs_to_pycorr(total, 'landyszalay', MAPPING)
    fname = str(tmp_path / 'sky_split_state.pkl.npy')
    np.save(fname, state)
    loaded = TwoPointCorrelationFunction.load(fname)
    assert loaded.mode == 'smu'
    assert loaded.D1D2.los_type == 'midpoint'

    state_full = pairs_to_pycorr(full, 'landyszalay', MAPPING)
    loaded_full = TwoPointCorrelationFunction.from_state(
        copy.deepcopy(state_full))
    for ell in (0, 2):
        a = loaded(ell=ell)
        b = loaded_full(ell=ell)
        amp = np.abs(b).max()
        assert np.max(np.abs(a - b)) < 0.15 * amp + 1e-3, \
            f"xi_{ell} from split-accumulated state deviates"
