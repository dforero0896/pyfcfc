"""Shared fixtures and helpers for the pyfcfc test suite."""

import os
import sys

import numpy as np
import pytest

# make the example helpers (mock catalogues, ...) importable from tests
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)


def pytest_configure(config):
    config.addinivalue_line("markers", "pycorr: requires pycorr + Corrfunc")
    config.addinivalue_line("markers", "slow: slower tests")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20231004)


@pytest.fixture(scope="session")
def box_catalogs():
    """Two catalogues in a 400 Mpc/h periodic box.

    ``data`` is mildly clustered (Gaussian clumps on top of a uniform
    component), ``random`` is uniform.  Both come with non-trivial
    weights.  Uses a private generator so that the catalogues do not
    depend on the test execution order.
    """
    rng = np.random.default_rng(20231004)
    box = 400.0
    n_data, n_rand = 2500, 3000

    # clustered catalogue: half uniform, half in clumps
    n_clump = n_data // 2
    n_centers = 25
    centers = rng.uniform(0, box, (n_centers, 3))
    assign = rng.integers(0, n_centers, n_clump)
    clumpy = centers[assign] + rng.normal(0, 12.0, (n_clump, 3))
    uniform_part = rng.uniform(0, box, (n_data - n_clump, 3))
    data = np.concatenate([clumpy, uniform_part], axis=0) % box

    rand = rng.uniform(0, box, (n_rand, 3))

    w_data = rng.uniform(0.5, 1.5, n_data)
    w_rand = rng.uniform(0.5, 1.5, n_rand)

    return {
        'box': box,
        'data': data, 'w_data': w_data,
        'rand': rand, 'w_rand': w_rand,
    }


@pytest.fixture(scope="session")
def sky_catalogs():
    """A small survey-like catalogue: {RA, Dec, z} + Cartesian version.

    Uses a private generator (order-independent).
    """
    rng = np.random.default_rng(4242)
    from brute_force import comoving_distance

    n_data, n_rand = 2000, 2500
    ra = rng.uniform(140, 220, n_data)
    dec = np.degrees(np.arcsin(rng.uniform(-0.3, 0.3, n_data)))
    z = rng.uniform(0.4, 0.55, n_data)
    rdd = np.stack([ra, dec, z], axis=1)
    w = rng.uniform(0.5, 1.5, n_data)

    ra_r = rng.uniform(140, 220, n_rand)
    dec_r = np.degrees(np.arcsin(rng.uniform(-0.3, 0.3, n_rand)))
    z_r = rng.uniform(0.4, 0.55, n_rand)
    rdd_r = np.stack([ra_r, dec_r, z_r], axis=1)
    w_r = rng.uniform(0.5, 1.5, n_rand)

    # matching Cartesian coordinates (omega_m = 0.31, omega_l = 0.69, w = -1)
    def to_xyz(ra, dec, z):
        chi = comoving_distance(z, 0.31, 0.69, -1.0)
        cd = np.cos(np.radians(dec))
        return np.stack([chi * cd * np.cos(np.radians(ra)),
                         chi * cd * np.sin(np.radians(ra)),
                         chi * np.sin(np.radians(dec))], axis=1)

    return {
        'rdd': rdd, 'w': w,
        'rdd_r': rdd_r, 'w_r': w_r,
        'xyz': to_xyz(ra, dec, z),
        'xyz_r': to_xyz(ra_r, dec_r, z_r),
        'omega_m': 0.31, 'omega_l': 0.69, 'eos_w': -1.0,
    }


def assert_counts_close(got, expected, rtol=1e-9, atol=2.0, msg=""):
    """Compare pair counts allowing a couple of boundary pairs to move."""
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert got.shape == expected.shape, \
        f"{msg}: shapes {got.shape} != {expected.shape}"
    diff = np.abs(got - expected)
    tol = atol + rtol * np.abs(expected)
    bad = diff > tol
    assert not bad.any(), (
        f"{msg}: {bad.sum()} bins differ; max abs diff = {diff.max():.6g}, "
        f"max expected count = {np.abs(expected).max():.6g}")


def assert_curves_close(got, expected, rtol=5e-3, atol_frac=2e-3, msg=""):
    """Compare correlation-function curves.

    ``atol_frac`` is an absolute tolerance relative to the peak amplitude
    of ``expected``, useful for multipoles that oscillate around zero.
    Bins where ``expected`` is not finite are skipped, but NaNs in
    ``got`` where ``expected`` is finite are treated as failures.
    """
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert got.shape == expected.shape, \
        f"{msg}: shapes {got.shape} != {expected.shape}"
    finite_exp = np.isfinite(expected)
    assert np.all(np.isfinite(got[finite_exp])), \
        f"{msg}: NaN/inf values where the reference is finite"
    m = finite_exp & np.isfinite(got)
    if not m.any():
        return
    scale = max(np.abs(expected[m]).max(), 1e-12)
    atol = atol_frac * scale
    diff = np.abs(got[m] - expected[m])
    bad = diff > (atol + rtol * np.abs(expected[m]))
    assert not bad.any(), (
        f"{msg}: {bad.sum()} bins differ; max abs diff = {diff.max():.6g} "
        f"(amplitude {scale:.6g})")
