"""Validate survey-like (sky) pair counts against brute-force counting.

FCFC's survey-like component defines the line-of-sight direction as the
midpoint of each pair, i.e. ``pi = |(r1 - r2) . (r1 + r2)| / |r1 + r2|``.
The brute-force reference uses the same definition.
"""

import numpy as np
import pytest

from brute_force import brute_force_sky, comoving_distance
from pyfcfc.sky import py_compute_cf


def test_sky_iso_unweighted(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'], cat['xyz_r']
    w1, w2 = np.ones(len(xyz)), np.ones(len(xyz_r))
    s_edges = np.arange(0, 201, 50, dtype=np.float64)

    res = py_compute_cf([xyz, xyz_r], [w1, w2], s_edges, None, 0,
                        label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
                        convert=False)
    ref = brute_force_sky([xyz, xyz_r], [w1, w2], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        assert np.allclose(got, ref['counts'][pair], rtol=0, atol=2), \
            f"{pair}: pyfcfc vs brute force mismatch"


def test_sky_smu_weighted(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'], cat['xyz_r']
    w1, w2 = cat['w'], cat['w_r']
    s_edges = np.arange(0, 201, 50, dtype=np.float64)
    nmu = 8
    mu_edges = np.linspace(0, 1, nmu + 1)

    res = py_compute_cf([xyz, xyz_r], [w1, w2], s_edges, None, nmu,
                        label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
                        cf=['(DD - 2 * DR + RR) / RR'], multipole=[0, 2],
                        convert=False)
    ref = brute_force_sky([xyz, xyz_r], [w1, w2], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, mu_edges=mu_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        assert np.allclose(got, ref['counts'][pair], rtol=1e-9, atol=2), \
            f"{pair}: pyfcfc vs brute force mismatch"

    # multipoles of a random-vs-random-like sky patch: xi0 dominated by
    # the integral constraint, but everything must be finite
    assert np.all(np.isfinite(res['multipoles']))


def test_sky_spi_weighted(sky_catalogs):
    cat = sky_catalogs
    xyz, xyz_r = cat['xyz'], cat['xyz_r']
    w1, w2 = cat['w'], cat['w_r']
    s_edges = np.arange(0, 151, 50, dtype=np.float64)
    pi_edges = np.arange(0, 151, 50, dtype=np.float64)

    res = py_compute_cf([xyz, xyz_r], [w1, w2], s_edges, pi_edges, 0,
                        label=['D', 'R'], bin=2, pair=['DD', 'DR', 'RR'],
                        convert=False)
    ref = brute_force_sky([xyz, xyz_r], [w1, w2], ['D', 'R'],
                          ['DD', 'DR', 'RR'], s_edges, pi_edges=pi_edges)
    for pair in ('DD', 'DR', 'RR'):
        got = res['pairs'][pair] * res['normalization'][pair]
        assert np.allclose(got, ref['counts'][pair], rtol=1e-9, atol=2), \
            f"{pair}: pyfcfc vs brute force mismatch"


def test_coordinate_conversion_matches_quadrature(sky_catalogs):
    """convert=True with (RA, Dec, z) inputs must reproduce the counts
    obtained from externally converted Cartesian coordinates."""
    cat = sky_catalogs
    s_edges = np.arange(0, 151, 30, dtype=np.float64)
    nmu = 6

    res_rdd = py_compute_cf(
        [cat['rdd']], [cat['w']], s_edges, None, nmu, label=['D'], bin=1,
        pair=['DD'], convert=True, omega_m=cat['omega_m'],
        omega_l=cat['omega_l'], eos_w=cat['eos_w'])
    res_xyz = py_compute_cf(
        [cat['xyz']], [cat['w']], s_edges, None, nmu, label=['D'], bin=1,
        pair=['DD'], convert=False)

    c1 = res_rdd['pairs']['DD'] * res_rdd['normalization']['DD']
    c2 = res_xyz['pairs']['DD'] * res_xyz['normalization']['DD']
    # tiny differences are allowed: FCFC integrates distances with a
    # cubic spline (relative error ~ 1e-8), which can move a few pairs
    # across bin boundaries
    assert np.allclose(c1, c2, rtol=1e-6, atol=3)


def test_comoving_distance_convention(sky_catalogs):
    """FCFC distances use c/H0 = 2997.92458 Mpc/h with the given (Om, OL, w).

    Verified indirectly: converting a catalogue at fixed redshift z0 must
    place all objects at the expected comoving radius (pairs between
    objects at the same z have pi ~ 0 for the midpoint LOS only if the
    radius is right; here we simply check the pair counts of a thin shell
    against brute force using the reference conversion)."""
    rng = np.random.default_rng(11)
    n = 800
    z0 = 0.5
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    z = np.full(n, z0) * (1 + rng.normal(0, 1e-3, n))
    rdd = np.stack([ra, dec, z], axis=1)
    w = np.ones(n)

    s_edges = np.arange(0, 101, 25, dtype=np.float64)
    res = py_compute_cf([rdd], [w], s_edges, None, 0, label=['D'], bin=0,
                        pair=['DD'], convert=True, omega_m=0.31,
                        omega_l=0.69, eos_w=-1)
    xyz = np.stack([
        comoving_distance(z, 0.31, 0.69) * np.cos(np.radians(dec))
        * np.cos(np.radians(ra)),
        comoving_distance(z, 0.31, 0.69) * np.cos(np.radians(dec))
        * np.sin(np.radians(ra)),
        comoving_distance(z, 0.31, 0.69) * np.sin(np.radians(dec))], axis=1)
    ref = brute_force_sky([xyz], [w], ['D'], ['DD'], s_edges)
    got = res['pairs']['DD'] * res['normalization']['DD']
    assert np.allclose(got, ref['counts']['DD'], rtol=1e-6, atol=3)
