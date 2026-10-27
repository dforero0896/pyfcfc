"""Brute-force (O(N^2)) reference pair counters for validating pyfcfc.

These implementations follow the FCFC / pyfcfc conventions:

- separations are computed with the periodic minimum-image convention
  (boxes) or directly (sky);
- bins are left-closed, right-open;
- for the (s, mu) and (s_perp, pi) schemes, mu (pi) is the *absolute*
  value of the cosine (parallel separation), binned in [0, 1] ([0, pmax]);
- auto-correlation pairs are counted twice (ordered pairs, i != j),
  cross-correlation pairs once for every (i, j) combination;
- pair counts are the sums of the products of the weights;
- pairs with exactly mu == 1 are dropped by FCFC (WITH_MU_ONE disabled);
  such pairs are measure zero for random floating point coordinates, and
  are dropped here as well.

The returned counts are *raw* (unnormalized).  Divide by the
normalization returned alongside to compare with ``results['pairs']``.
"""

import numpy as np


def _pair_iterations(n1, n2, autocorr):
    """Yield (slice_i, slice_j) index blocks for pair counting."""
    if autocorr:
        for i in range(n1):
            if i + 1 < n1:
                yield i, np.arange(i + 1, n1)
    else:
        for i in range(n1):
            yield i, np.arange(n2)


def _bin2d(d_first, d_second, edges_first, edges_second):
    """Histogram squared distances into (first, second) bins."""
    i = np.searchsorted(edges_first, d_first, side='right') - 1
    j = np.searchsorted(edges_second, d_second, side='right') - 1
    valid = (i >= 0) & (i < len(edges_first) - 1) & \
            (j >= 0) & (j < len(edges_second) - 1)
    return i[valid], j[valid], valid


def brute_force_box(cats, wts, labels, pairs, s_edges, box_size,
                    mu_edges=None, pi_edges=None):
    """Count pairs in a periodic box, exactly following FCFC conventions.

    Parameters
    ----------
    cats : list of (N, 3) arrays
    wts : list of (N,) arrays
    labels : list of str, catalogue labels
    pairs : list of str, e.g. ['DD', 'DR', 'RR']
    s_edges : separation (or s_perp) bin edges
    box_size : float or 3-array
    mu_edges : edges in [0, 1] for the (s, mu) scheme (None otherwise)
    pi_edges : edges in [0, pmax] for the (s_perp, pi) scheme

    Returns
    -------
    dict
        ``counts[pair]``: raw pair counts, shape (ns,), (ns, nmu) or
        (ns, np); ``norm[pair]``: the normalization (ordered-pair
        convention).
    """
    box = np.broadcast_to(np.asarray(box_size, dtype=np.float64), (3,))
    cats = [np.asarray(c, dtype=np.float64) for c in cats]
    wts = [np.asarray(w, dtype=np.float64) for w in wts]
    idx = {lab: i for i, lab in enumerate(labels)}
    ns = len(s_edges) - 1
    s2_edges = np.asarray(s_edges, dtype=np.float64) ** 2

    if mu_edges is not None:
        mu_edges = np.asarray(mu_edges, dtype=np.float64)
        nmu = len(mu_edges) - 1
        mu2_edges = mu_edges ** 2
        shape = (ns, nmu)
    elif pi_edges is not None:
        pi_edges = np.asarray(pi_edges, dtype=np.float64)
        npi = len(pi_edges) - 1
        pi2_edges = pi_edges ** 2
        shape = (ns, npi)
    else:
        shape = (ns,)

    counts = {}
    norm = {}
    for pair in pairs:
        a, b = idx[pair[0]], idx[pair[1]]
        autocorr = a == b
        out = np.zeros(shape, dtype=np.float64)
        pa, pb = cats[a], cats[b]
        wa, wb = wts[a], wts[b]
        n1, n2 = len(pa), len(pb)

        for i, js in _pair_iterations(n1, n2 if not autocorr else n1,
                                      autocorr):
            d = pa[i] - pb[js]
            d -= box * np.round(d / box)
            s2 = np.einsum('ij,ij->i', d, d)
            wprod = wa[i] * wb[js]
            if autocorr:
                wprod = 2.0 * wprod   # ordered-pair convention
            if mu_edges is not None:
                # mu^2 = dz^2 / s^2, pairs with s == 0 land in the first bin
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu2 = np.where(s2 > 0, d[:, 2] ** 2 / s2, 0.0)
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0]) & (mu2 < 1.0)
                ii, jj, _ = _bin2d(s2[keep], mu2[keep], s2_edges, mu2_edges)
                np.add.at(out, (ii, jj), wprod[keep])
            elif pi_edges is not None:
                pi2 = d[:, 2] ** 2
                s_perp2 = s2 - pi2
                keep = (s_perp2 < s2_edges[-1]) & (s_perp2 >= s2_edges[0]) \
                    & (pi2 < pi2_edges[-1]) & (pi2 >= pi2_edges[0])
                ii, jj, _ = _bin2d(s_perp2[keep], pi2[keep],
                                   s2_edges, pi2_edges)
                np.add.at(out, (ii, jj), wprod[keep])
            else:
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0])
                ii = np.searchsorted(s2_edges, s2[keep], side='right') - 1
                np.add.at(out, ii, wprod[keep])

        counts[pair] = out
        if autocorr:
            sw = wa.sum()
            norm[pair] = sw * sw - (wa ** 2).sum()
        else:
            norm[pair] = wa.sum() * wb.sum()

    return {'counts': counts, 'norm': norm}


def brute_force_sky(cats, wts, labels, pairs, s_edges,
                    mu_edges=None, pi_edges=None):
    """Count pairs for survey-like (Cartesian) data with a midpoint LOS.

    FCFC's survey-like pair counting defines the line-of-sight direction
    as that of the midpoint of each pair, i.e.
    ``pi = |(r1 - r2) . (r1 + r2)| / |r1 + r2|`` and ``mu = pi / s``.
    Conventions (double counting of auto pairs, bin edges, ...) are as in
    :func:`brute_force_box`.
    """
    cats = [np.asarray(c, dtype=np.float64) for c in cats]
    wts = [np.asarray(w, dtype=np.float64) for w in wts]
    idx = {lab: i for i, lab in enumerate(labels)}
    ns = len(s_edges) - 1
    s2_edges = np.asarray(s_edges, dtype=np.float64) ** 2

    if mu_edges is not None:
        mu_edges = np.asarray(mu_edges, dtype=np.float64)
        nmu = len(mu_edges) - 1
        shape = (ns, nmu)
    elif pi_edges is not None:
        pi_edges = np.asarray(pi_edges, dtype=np.float64)
        npi = len(pi_edges) - 1
        pi2_edges = pi_edges ** 2
        shape = (ns, npi)
    else:
        shape = (ns,)

    counts = {}
    norm = {}
    for pair in pairs:
        a, b = idx[pair[0]], idx[pair[1]]
        autocorr = a == b
        out = np.zeros(shape, dtype=np.float64)
        pa, pb = cats[a], cats[b]
        wa, wb = wts[a], wts[b]
        n1, n2 = len(pa), len(pb)

        for i, js in _pair_iterations(n1, n2 if not autocorr else n1,
                                      autocorr):
            d = pa[i] - pb[js]
            m = pa[i] + pb[js]
            s2 = np.einsum('ij,ij->i', d, d)
            m2 = np.einsum('ij,ij->i', m, m)
            # pi^2 = (d . m)^2 / |m|^2
            with np.errstate(divide='ignore', invalid='ignore'):
                pi2 = np.where(m2 > 0,
                               np.einsum('ij,ij->i', d, m) ** 2 / m2, 0.0)
            wprod = wa[i] * wb[js]
            if autocorr:
                wprod = 2.0 * wprod
            if mu_edges is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu2 = np.where(s2 > 0, pi2 / s2, 0.0)
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0]) & (mu2 < 1.0)
                mu2_edges = np.asarray(mu_edges, dtype=np.float64) ** 2
                ii, jj, _ = _bin2d(s2[keep], mu2[keep], s2_edges, mu2_edges)
                np.add.at(out, (ii, jj), wprod[keep])
            elif pi_edges is not None:
                s_perp2 = s2 - pi2
                keep = (s_perp2 < s2_edges[-1]) & (s_perp2 >= s2_edges[0]) \
                    & (pi2 < pi2_edges[-1]) & (pi2 >= pi2_edges[0])
                ii, jj, _ = _bin2d(s_perp2[keep], pi2[keep],
                                   s2_edges, pi2_edges)
                np.add.at(out, (ii, jj), wprod[keep])
            else:
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0])
                ii = np.searchsorted(s2_edges, s2[keep], side='right') - 1
                np.add.at(out, ii, wprod[keep])

        counts[pair] = out
        if autocorr:
            sw = wa.sum()
            norm[pair] = sw * sw - (wa ** 2).sum()
        else:
            norm[pair] = wa.sum() * wb.sum()

    return {'counts': counts, 'norm': norm}


def comoving_distance(z, omega_m=0.31, omega_l=0.69, w=-1.0, h=1.0):
    """Radial comoving distance in Mpc/h, c/H0 = 2997.92458 Mpc/h.

    Matches the fiducial conversion performed by FCFC's ``cnvt_coord``
    (dark-energy equation of state w, flat or non-flat via omega_l).
    """
    from scipy.integrate import cumulative_trapezoid

    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    zs = np.concatenate(([0.0], np.sort(np.unique(z))))
    # integrate on a very fine grid (trapezoid error ~ relative 1e-10)
    zgrid = np.linspace(0, zs[-1], 200001)

    def inv_E(zz):
        om = omega_m * (1 + zz) ** 3
        ok = (1.0 - omega_m - omega_l) * (1 + zz) ** 2   # curvature term
        de = omega_l * (1 + zz) ** (3 * (1 + w))
        return 1.0 / np.sqrt(om + ok + de)

    chi_grid = 2997.92458 / h * cumulative_trapezoid(inv_E(zgrid), zgrid,
                                                     initial=0.0)
    chi = np.interp(zs, zgrid, chi_grid)
    out = np.interp(z, zs, chi)
    return out


def soft_gaussian_weights(s, s_edges, sigma, sigma_log=False):
    """Softmax Gaussian-kernel weights for soft histogramming.

    For each separation ``s``, the pair weight is distributed over *all*
    separation bins as

        W_i = softmax_i(-(s - c_i)^2 / (2 sigma^2)),

    with c_i the bin centres, evaluated in the numerically stable form
    (the minimum logit is subtracted before exponentiating).  This is
    exactly what pyfcfc's soft histogramming (``soft_hist=True``)
    computes in C.

    Parameters
    ----------
    s : (N,) array of separations
    s_edges : (ns + 1,) array of bin edges
    sigma : Gaussian kernel width (same units as ``s``)

    Returns
    -------
    (N, ns) array of weights; every row sums to 1.
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    s_edges = np.asarray(s_edges, dtype=np.float64)
    centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    if sigma_log:
        # scale-dependent width sigma(s) = sigma * s
        x = (s[:, None] - centers[None, :]) ** 2 / \
            (2.0 * (float(sigma) * s) ** 2)[:, None]
    else:
        x = (s[:, None] - centers[None, :]) ** 2 / (2.0 * float(sigma) ** 2)
    x -= x.min(axis=1, keepdims=True)
    e = np.exp(-x)
    return e / e.sum(axis=1, keepdims=True)


def brute_force_soft_box(cats, wts, labels, pairs, s_edges, box_size, sigma,
                         mu_edges=None, pi_edges=None, soft_mu_sigma=None,
                         soft_pi_sigma=None, sigma_log=False):
    """Soft-histogram reference pair counts in a periodic box.

    Conventions are exactly those of :func:`brute_force_box` (minimum
    image, left-closed/right-open bins, mu == 1 pairs dropped, auto
    pairs counted twice, raw counts), but the separation (or s_perp)
    dimension is soft-binned with :func:`soft_gaussian_weights`, while
    the mu (pi) dimension remains hard-binned.

    Returns
    -------
    dict
        ``counts[pair]``: raw soft pair counts, shape (ns,), (ns, nmu)
        or (ns, np); ``norm[pair]``: the normalization.
    """
    box = np.broadcast_to(np.asarray(box_size, dtype=np.float64), (3,))
    cats = [np.asarray(c, dtype=np.float64) for c in cats]
    wts = [np.asarray(w, dtype=np.float64) for w in wts]
    idx = {lab: i for i, lab in enumerate(labels)}
    s_edges = np.asarray(s_edges, dtype=np.float64)
    ns = len(s_edges) - 1
    s2_edges = s_edges ** 2

    if mu_edges is not None:
        mu2_edges = np.asarray(mu_edges, dtype=np.float64) ** 2
        shape = (ns, len(mu2_edges) - 1)
    elif pi_edges is not None:
        pi2_edges = np.asarray(pi_edges, dtype=np.float64) ** 2
        shape = (ns, len(pi2_edges) - 1)
    else:
        shape = (ns,)

    counts = {}
    norm = {}
    for pair in pairs:
        a, b = idx[pair[0]], idx[pair[1]]
        autocorr = a == b
        out = np.zeros(shape, dtype=np.float64)
        pa, pb = cats[a], cats[b]
        wa, wb = wts[a], wts[b]
        n1, n2 = len(pa), len(pb)

        for i, js in _pair_iterations(n1, n2 if not autocorr else n1,
                                      autocorr):
            d = pa[i] - pb[js]
            d -= box * np.round(d / box)
            s2 = np.einsum('ij,ij->i', d, d)
            wprod = wa[i] * wb[js]
            if autocorr:
                wprod = 2.0 * wprod   # ordered-pair convention
            if mu_edges is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu2 = np.where(s2 > 0, d[:, 2] ** 2 / s2, 0.0)
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0]) & (mu2 < 1.0)
                W = soft_gaussian_weights(np.sqrt(s2[keep]), s_edges, sigma,
                                          sigma_log=sigma_log)
                if soft_mu_sigma is not None:
                    WM = soft_gaussian_weights(np.sqrt(mu2[keep]),
                                               np.asarray(mu_edges),
                                               soft_mu_sigma)
                    out += np.einsum('k,ki,kj->ij', wprod[keep], W, WM)
                else:
                    jj = np.searchsorted(mu2_edges, mu2[keep],
                                         side='right') - 1
                    np.add.at(out.T, jj, wprod[keep, None] * W)
            elif pi_edges is not None:
                pi2 = d[:, 2] ** 2
                s_perp2 = s2 - pi2
                keep = (s_perp2 < s2_edges[-1]) & (s_perp2 >= s2_edges[0]) \
                    & (pi2 < pi2_edges[-1]) & (pi2 >= pi2_edges[0])
                W = soft_gaussian_weights(np.sqrt(s_perp2[keep]), s_edges,
                                          sigma, sigma_log=sigma_log)
                if soft_pi_sigma is not None:
                    WP = soft_gaussian_weights(np.sqrt(pi2[keep]),
                                               np.asarray(pi_edges),
                                               soft_pi_sigma)
                    out += np.einsum('k,ki,kj->ij', wprod[keep], W, WP)
                else:
                    jj = np.searchsorted(pi2_edges, pi2[keep],
                                         side='right') - 1
                    np.add.at(out.T, jj, wprod[keep, None] * W)
            else:
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0])
                W = soft_gaussian_weights(np.sqrt(s2[keep]), s_edges, sigma,
                                          sigma_log=sigma_log)
                out += (wprod[keep, None] * W).sum(axis=0)

        counts[pair] = out
        if autocorr:
            sw = wa.sum()
            norm[pair] = sw * sw - (wa ** 2).sum()
        else:
            norm[pair] = wa.sum() * wb.sum()

    return {'counts': counts, 'norm': norm}


def brute_force_soft_sky(cats, wts, labels, pairs, s_edges, sigma,
                         mu_edges=None, pi_edges=None, soft_mu_sigma=None,
                         soft_pi_sigma=None, sigma_log=False):
    """Soft-histogram reference pair counts for survey-like data.

    Conventions are exactly those of :func:`brute_force_sky` (midpoint
    line of sight, mu == 1 pairs dropped, auto pairs counted twice, raw
    counts), with the separation (or s_perp) dimension soft-binned via
    :func:`soft_gaussian_weights` and the mu (pi) dimension hard-binned.
    """
    cats = [np.asarray(c, dtype=np.float64) for c in cats]
    wts = [np.asarray(w, dtype=np.float64) for w in wts]
    idx = {lab: i for i, lab in enumerate(labels)}
    s_edges = np.asarray(s_edges, dtype=np.float64)
    ns = len(s_edges) - 1
    s2_edges = s_edges ** 2

    if mu_edges is not None:
        mu2_edges = np.asarray(mu_edges, dtype=np.float64) ** 2
        shape = (ns, len(mu2_edges) - 1)
    elif pi_edges is not None:
        pi2_edges = np.asarray(pi_edges, dtype=np.float64) ** 2
        shape = (ns, len(pi2_edges) - 1)
    else:
        shape = (ns,)

    counts = {}
    norm = {}
    for pair in pairs:
        a, b = idx[pair[0]], idx[pair[1]]
        autocorr = a == b
        out = np.zeros(shape, dtype=np.float64)
        pa, pb = cats[a], cats[b]
        wa, wb = wts[a], wts[b]
        n1, n2 = len(pa), len(pb)

        for i, js in _pair_iterations(n1, n2 if not autocorr else n1,
                                      autocorr):
            d = pa[i] - pb[js]
            m = pa[i] + pb[js]
            s2 = np.einsum('ij,ij->i', d, d)
            m2 = np.einsum('ij,ij->i', m, m)
            with np.errstate(divide='ignore', invalid='ignore'):
                pi2 = np.where(m2 > 0,
                               np.einsum('ij,ij->i', d, m) ** 2 / m2, 0.0)
            wprod = wa[i] * wb[js]
            if autocorr:
                wprod = 2.0 * wprod
            if mu_edges is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu2 = np.where(s2 > 0, pi2 / s2, 0.0)
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0]) & (mu2 < 1.0)
                W = soft_gaussian_weights(np.sqrt(s2[keep]), s_edges, sigma,
                                          sigma_log=sigma_log)
                if soft_mu_sigma is not None:
                    WM = soft_gaussian_weights(np.sqrt(mu2[keep]),
                                               np.asarray(mu_edges),
                                               soft_mu_sigma)
                    out += np.einsum('k,ki,kj->ij', wprod[keep], W, WM)
                else:
                    jj = np.searchsorted(mu2_edges, mu2[keep],
                                         side='right') - 1
                    np.add.at(out.T, jj, wprod[keep, None] * W)
            elif pi_edges is not None:
                s_perp2 = s2 - pi2
                keep = (s_perp2 < s2_edges[-1]) & (s_perp2 >= s2_edges[0]) \
                    & (pi2 < pi2_edges[-1]) & (pi2 >= pi2_edges[0])
                W = soft_gaussian_weights(np.sqrt(s_perp2[keep]), s_edges,
                                          sigma, sigma_log=sigma_log)
                if soft_pi_sigma is not None:
                    WP = soft_gaussian_weights(np.sqrt(pi2[keep]),
                                               np.asarray(pi_edges),
                                               soft_pi_sigma)
                    out += np.einsum('k,ki,kj->ij', wprod[keep], W, WP)
                else:
                    jj = np.searchsorted(pi2_edges, pi2[keep],
                                         side='right') - 1
                    np.add.at(out.T, jj, wprod[keep, None] * W)
            else:
                keep = (s2 < s2_edges[-1]) & (s2 >= s2_edges[0])
                W = soft_gaussian_weights(np.sqrt(s2[keep]), s_edges, sigma,
                                          sigma_log=sigma_log)
                out += (wprod[keep, None] * W).sum(axis=0)

        counts[pair] = out
        if autocorr:
            sw = wa.sum()
            norm[pair] = sw * sw - (wa ** 2).sum()
        else:
            norm[pair] = wa.sum() * wb.sum()

    return {'counts': counts, 'norm': norm}
