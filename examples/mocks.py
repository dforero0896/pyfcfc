"""Mock catalogue generators shared by the pyfcfc examples."""

import numpy as np


def make_clustered_box(n, box_size, seed=42, n_centers=40, clump_frac=0.5,
                       clump_scale=20.0, weighted=True):
    """A periodic-box catalogue: uniform component + Gaussian clumps.

    Returns
    -------
    pos : (n, 3) float64 array
    w : (n,) float64 array of weights (ones if ``weighted`` is False)
    """
    rng = np.random.default_rng(seed)
    n_clump = int(n * clump_frac)
    centers = rng.uniform(0, box_size, (n_centers, 3))
    assign = rng.integers(0, n_centers, n_clump)
    clumpy = centers[assign] + rng.normal(0, clump_scale, (n_clump, 3))
    uniform = rng.uniform(0, box_size, (n - n_clump, 3))
    pos = np.concatenate([clumpy, uniform], axis=0) % box_size
    if weighted:
        w = rng.uniform(0.8, 1.2, n)
    else:
        w = np.ones(n)
    return np.ascontiguousarray(pos), w


def make_survey(n, seed=43, ra_range=(140., 220.), dec_max=25.,
                z_range=(0.7, 1.0), clustered=True, weighted=True):
    """A toy survey catalogue in {RA, Dec, z} with FKP-like weights.

    When ``clustered``, angular Gaussian clumps are added on top of a
    uniform sky distribution, so that the correlation function has a
    non-trivial signal.

    Returns
    -------
    rdd : (n, 3) float64 array with columns {RA [deg], Dec [deg], z}
    w : (n,) float64 array of weights
    nz : (n,) float64 array, mock number density (for FKP weights)
    """
    rng = np.random.default_rng(seed)
    ra_min, ra_max = ra_range
    sin_dec_max = np.sin(np.radians(dec_max))

    n_clump = int(n * 0.5) if clustered else 0
    n_centers = 30
    ra_c = rng.uniform(ra_min, ra_max, n_centers)
    dec_c = np.degrees(np.arcsin(rng.uniform(-sin_dec_max, sin_dec_max,
                                             n_centers)))
    z_c = rng.uniform(z_range[0], z_range[1], n_centers)

    ra = np.empty(n); dec = np.empty(n); z = np.empty(n)
    if n_clump:
        assign = rng.integers(0, n_centers, n_clump)
        # ~2 deg angular scatter, ~40 Mpc/h radial scatter
        ra[:n_clump] = ra_c[assign] + rng.normal(0, 2.0, n_clump)
        dec[:n_clump] = dec_c[assign] + rng.normal(0, 2.0, n_clump)
        z[:n_clump] = z_c[assign] + rng.normal(0, 0.012, n_clump)
    ra[n_clump:] = rng.uniform(ra_min, ra_max, n - n_clump)
    dec[n_clump:] = np.degrees(np.arcsin(
        rng.uniform(-sin_dec_max, sin_dec_max, n - n_clump)))
    z[n_clump:] = rng.uniform(z_range[0], z_range[1], n - n_clump)

    ra = np.mod(ra - ra_min, ra_max - ra_min) + ra_min
    dec = np.clip(dec, -dec_max, dec_max)
    z = np.clip(z, z_range[0], z_range[1] - 1e-6)

    # mock n(z) and FKP-like weights with P0 = 1e4 (Mpc/h)^3
    nz = 1e-4 * (1.0 + 0.5 * np.sin(6.0 * (z - z_range[0])))
    if weighted:
        w = 1.0 / (1.0 + 1e4 * nz)
    else:
        w = np.ones(n)
    rdd = np.ascontiguousarray(np.stack([ra, dec, z], axis=1))
    return rdd, w, nz


def rdd_to_xyz(rdd, omega_m=0.31, omega_l=0.69, w=-1.0):
    """Convert {RA, Dec, z} to comoving Cartesian coordinates (Mpc/h).

    Uses the same fiducial convention as FCFC: c/H0 = 2997.92458 Mpc/h.
    This matches what pyfcfc.sky does internally with convert=True, and
    is used to feed identical coordinates to pycorr in the examples.
    """
    from scipy.integrate import cumulative_trapezoid

    ra, dec, z = rdd[:, 0], rdd[:, 1], rdd[:, 2]
    zgrid = np.linspace(0, z.max() * 1.0001 + 1e-6, 200001)

    def inv_E(zz):
        return 1.0 / np.sqrt(omega_m * (1 + zz) ** 3
                             + (1 - omega_m - omega_l) * (1 + zz) ** 2
                             + omega_l * (1 + zz) ** (3 * (1 + w)))

    chi_grid = 2997.92458 * cumulative_trapezoid(inv_E(zgrid), zgrid,
                                                 initial=0.0)
    chi = np.interp(z, zgrid, chi_grid)
    cd = np.cos(np.radians(dec))
    xyz = np.stack([chi * cd * np.cos(np.radians(ra)),
                    chi * cd * np.sin(np.radians(ra)),
                    chi * np.sin(np.radians(dec))], axis=1)
    return np.ascontiguousarray(xyz)
