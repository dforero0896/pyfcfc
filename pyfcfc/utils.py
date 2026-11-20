"""Utilities to manipulate ``pyfcfc`` results.

This module provides

- :func:`add_pair_counts`, to combine results computed from different
  random-catalog splits;
- :func:`compute_multipoles` and :func:`compute_wp`, to integrate
  correlation functions into multipoles or the projected correlation
  function;
- :func:`pairs_to_pycorr`, to convert ``pyfcfc`` pair counts into a
  `pycorr <https://github.com/cosmodesi/pycorr>`_ state dictionary, so that
  the pair counts can be manipulated (estimators, rebinning, saving, ...)
  with the pycorr API;
- :func:`compute_wtheta`, to measure the angular correlation function
  w(theta) from RA/Dec catalogues, using the unit-sphere reduction of the
  ordinary 3-D pair counting (with :func:`radec_to_unit_sphere`,
  :func:`theta_to_chord` and :func:`chord_to_theta` as helpers).

Conventions
-----------
``pyfcfc`` pair counts follow the FCFC conventions:

- auto pairs are counted twice (ordered pairs), so that, e.g., the total
  number of pairs of a catalogue with N objects (unit weights) is
  N (N - 1), and the normalization is sum(w)^2 - sum(w^2), matching pycorr;
- cross pairs are counted once for every combination of the two catalogues,
  with normalization sum(w1) * sum(w2);
- for the (s, mu) and (s_perp, pi) binning schemes, only mu >= 0 (or
  pi >= 0) is binned, with mu (pi) the absolute value of the cosine
  (parallel separation).

pycorr stores (s, mu) and (r_p, pi) counts over the full, symmetric ranges
mu in [-1, 1] and pi in [-pi_max, pi_max], such that the sum of the counts
over all bins equals the pair normalization (ordered pairs).  For auto pairs
each unordered pair appears once in the positive half and once (mirrored) in
the negative half, while for cross pairs the two halves are statistically
(but not exactly) equal.  :func:`pairs_to_pycorr` applies this
mirror + halve transformation, which is exact for auto pairs and a (very
good) statistical approximation for cross pairs.  In the isotropic mode
('s') the conventions of FCFC and pycorr coincide directly.
"""

import numpy as np

__all__ = ['add_pair_counts', 'compute_multipoles', 'compute_wp',
           'pairs_to_pycorr', 'compute_wtheta', 'radec_to_unit_sphere',
           'theta_to_chord', 'chord_to_theta']


def _pair_keys(results):
    """Iterate over the pair-count keys of a pyfcfc results dictionary."""
    skip = {'smin', 'smax', 'mumin', 'mumax', 'pimin', 'pimax'}
    for key in results['pairs']:
        if key in skip:
            continue
        yield key


def _check_same_binning(results_a, results_b):
    """Check that two results dictionaries share the same binning."""
    for key in ('smin', 'smax'):
        a = np.asarray(results_a['pairs'][key])
        b = np.asarray(results_b['pairs'][key])
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError("the two results have different separation bins;"
                             " they cannot be combined")
    for key in ('mumin', 'pimin'):
        if key in results_a['pairs'] or key in results_b['pairs']:
            a = np.asarray(results_a['pairs'].get(key, []))
            b = np.asarray(results_b['pairs'].get(key, []))
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError("the two results have different second-axis"
                                 " bins; they cannot be combined")


def add_pair_counts(results_a, results_b, repeat_missing=True):
    """Add the pair counts of two ``pyfcfc`` result dictionaries.

    This is typically used to combine measurements performed against
    different splits of a random catalogue.  The pair counts are combined
    as normalization-weighted averages, and the normalizations are summed,
    which is equivalent to summing the raw counts of the two measurements.

    The result is a new dictionary; the inputs are not modified.  Keys that
    depend on a correlation function estimator (``cf``, ``multipoles``,
    ``projected``, ``poles``) are dropped from the output, since they are
    stale after the combination; recompute them from the combined pair
    counts (see e.g. :func:`compute_multipoles`).

    Parameters
    ----------
    results_a, results_b : dict
        Result dictionaries returned by ``py_compute_cf``.  ``results_a``
        may be ``None`` or empty, in which case a copy of ``results_b``
        is returned.
    repeat_missing : bool, default=True
        If a pair count (or catalogue) is present in ``results_a`` but not
        in ``results_b``, double the ``results_a`` value (as if the same
        measurement had been performed twice).  If False, raise an error
        instead.

    Returns
    -------
    dict
        The combined results.
    """
    if results_a is None or len(results_a) == 0:
        return {**results_b, 'pairs': dict(results_b['pairs']),
                'normalization': dict(results_b['normalization']),
                'number': dict(results_b['number']),
                'weighted_number': dict(results_b['weighted_number'])}

    _check_same_binning(results_a, results_b)

    new = {}
    new['labels'] = list(results_a['labels'])
    for name in ('number', 'weighted_number'):
        merged = {}
        for lab in new['labels']:
            in_a = lab in results_a[name]
            in_b = lab in results_b[name]
            if in_a and in_b:
                merged[lab] = results_a[name][lab] + results_b[name][lab]
            elif in_a:
                if not repeat_missing:
                    raise KeyError(f"label {lab!r} missing from results_b")
                merged[lab] = 2 * results_a[name][lab]
            elif in_b:
                if not repeat_missing:
                    raise KeyError(f"label {lab!r} missing from results_a")
                merged[lab] = 2 * results_b[name][lab]
        new[name] = merged

    new['normalization'] = {}
    new['pairs'] = {}
    # bin edges and other metadata arrays
    for key, val in results_a['pairs'].items():
        if key not in _pair_keys(results_a):
            new['pairs'][key] = np.array(val, copy=True)

    for key in _pair_keys(results_a):
        norm_a = results_a['normalization'][key]
        cnt_a = np.asarray(results_a['pairs'][key]) * norm_a
        if key in results_b['pairs'] and key in results_b['normalization']:
            norm_b = results_b['normalization'][key]
            cnt_a = cnt_a + np.asarray(results_b['pairs'][key]) * norm_b
            norm_a = norm_a + norm_b
        else:
            if not repeat_missing:
                raise KeyError(f"pair count {key!r} missing from results_b")
            cnt_a = 2 * cnt_a
            norm_a = 2 * norm_a
        new['normalization'][key] = norm_a
        new['pairs'][key] = cnt_a / norm_a

    new['s'] = np.array(results_a['s'], copy=True)
    return new


def compute_multipoles(correlation_function, poles, mu_edges=None,
                       method='midpoint', ignore_nan=False):
    r"""Integrate xi(s, mu) into Legendre multipoles.

    Two integration schemes are available:

    - ``method='midpoint'`` (default): the midpoint rule,

      .. math::

          \xi_\ell(s) = (2\ell + 1) \sum_j \xi(s, \mu_j)\,
                        P_\ell(\mu_j)\, \Delta\mu_j

      with :math:`\mu_j` the centers of the mu bins.  This matches the
      internal multipole integration of FCFC (uniform bins on [0, 1]).

    - ``method='exact'``: the exact per-bin integral of the Legendre
      polynomial, i.e. xi is treated as piecewise constant over each bin.
      This is the scheme used by pycorr; it agrees with the midpoint rule
      up to O(dmu^2), and with it the multipoles computed from pyfcfc
      pair counts match pycorr's to machine precision.

    Only mu >= 0 is stored by FCFC (mu being the absolute cosine), and
    both schemes assume xi(s, -mu) = xi(s, mu), as is the case for the
    periodic box (line of sight along z) and for the midpoint
    line-of-sight convention.

    Parameters
    ----------
    correlation_function : array
        xi(s, mu), of shape (ns, nmu).
    poles : sequence of int
        Orders of the Legendre polynomials to evaluate.
    mu_edges : array, optional
        Edges of the mu bins (length nmu + 1); defaults to ``nmu``
        uniform bins spanning [0, 1].
    method : str, default='midpoint'
        Integration scheme: 'midpoint' (FCFC's) or 'exact' (pycorr's).
    ignore_nan : bool, default=False
        If True, mu bins where xi is NaN (e.g. bins without random pairs)
        are ignored and the normalization is rescaled by the covered mu
        range, like pycorr's ``ignore_nan`` option.  If False, NaN bins
        propagate into the multipoles.

    Returns
    -------
    array
        Multipoles, of shape (len(poles), ns).
    """
    from scipy.special import eval_legendre, legendre

    xi = np.asarray(correlation_function, dtype=np.float64)
    if xi.ndim != 2:
        raise ValueError(f"expected xi(s, mu) with 2 dimensions, got shape "
                         f"{xi.shape}")
    nmu = xi.shape[1]
    if mu_edges is None:
        mu_edges = np.linspace(0., 1., nmu + 1)
    else:
        mu_edges = np.asarray(mu_edges, dtype=np.float64)
        if mu_edges.shape != (nmu + 1,):
            raise ValueError(f"mu_edges must have shape ({nmu + 1},), got "
                             f"{mu_edges.shape}")
        if np.any(np.diff(mu_edges) <= 0):
            raise ValueError("mu_edges must be strictly increasing")
    dmu = np.diff(mu_edges)
    total = mu_edges[-1] - mu_edges[0]

    if method == 'midpoint':
        mu = mu_edges[:-1] + 0.5 * dmu
        base_w = [(2 * ell + 1) * eval_legendre(ell, mu) * dmu
                  for ell in poles]
    elif method == 'exact':
        base_w = []
        for ell in poles:
            integ = legendre(ell).integ()(mu_edges)
            base_w.append((2 * ell + 1) * (integ[1:] - integ[:-1]))
    else:
        raise ValueError(f"method must be 'midpoint' or 'exact', got "
                         f"{method!r}")

    poles = list(poles)
    multipoles = np.empty((len(poles), xi.shape[0]), dtype=np.float64)
    if ignore_nan:
        # Skip NaN bins and rescale by the covered mu range, as pycorr's
        # `ignore_nan` option does.
        valid = ~np.isnan(xi)
        xi_safe = np.where(valid, xi, 0.0)
        covered = (dmu[None, :] * valid).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = total / covered
        for i in range(len(poles)):
            num = (xi_safe * base_w[i][None, :]).sum(axis=1)
            multipoles[i, :] = num / total * frac
    else:
        for i in range(len(poles)):
            w = base_w[i] / total
            multipoles[i, :] = (xi * w[None, :]).sum(axis=1)
    return multipoles


def compute_wp(correlation_function, pi_edges, ignore_nan=False):
    r"""Integrate xi(s_perp, pi) into the projected correlation function.

    .. math::

        w_p(s_\perp) = 2 \sum_j \xi(s_\perp, \pi_j) \Delta\pi_j

    where the factor 2 accounts for the symmetric range of pi.

    Parameters
    ----------
    correlation_function : array
        xi(s_perp, pi), of shape (ns, np).
    pi_edges : array
        Edges of the pi bins used for the pair counting (length np + 1),
        covering pi >= 0 only.
    ignore_nan : bool, default=False
        If True, pi bins where xi is NaN are ignored and the result is
        rescaled by the covered pi range (like pycorr's ``ignore_nan``).

    Returns
    -------
    array
        w_p(s_perp), of shape (ns,).
    """
    xi = np.asarray(correlation_function, dtype=np.float64)
    pi_edges = np.asarray(pi_edges, dtype=np.float64)
    if xi.ndim != 2:
        raise ValueError(f"expected xi(s_perp, pi) with 2 dimensions, got "
                         f"shape {xi.shape}")
    if pi_edges.shape != (xi.shape[1] + 1,):
        raise ValueError(f"pi_edges must have length {xi.shape[1] + 1} "
                         f"(np + 1), got {pi_edges.shape}")
    dpi = np.diff(pi_edges)
    if ignore_nan:
        valid = ~np.isnan(xi)
        xi_safe = np.where(valid, xi, 0.0)
        covered = (dpi[None, :] * valid).sum(axis=1)
        total = pi_edges[-1] - pi_edges[0]
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = total / covered
        return 2. * (xi_safe * dpi[None, :]).sum(axis=1) * frac
    return 2. * (xi * dpi[None, :]).sum(axis=1)


def radec_to_unit_sphere(ra, dec):
    """Convert right ascension and declination to unit-sphere coordinates.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination, in degrees, of shape (N,).

    Returns
    -------
    array
        Cartesian coordinates on the unit sphere, of shape (N, 3).
    """
    ra = np.deg2rad(np.asarray(ra, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec, dtype=np.float64))
    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec must have the same shape, got "
                         f"{ra.shape} and {dec.shape}")
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra),
                            cos_dec * np.sin(ra),
                            np.sin(dec)))


def theta_to_chord(theta):
    r"""Angular separation to chord length on the unit sphere.

    The Euclidean distance between two unit vectors separated by an angle
    :math:`\theta` is the chord :math:`s = 2\sin(\theta/2)`.  This is the
    separation that ``pyfcfc`` measures for unit-sphere coordinates, so
    angular bins are passed to it as chord bins.

    Parameters
    ----------
    theta : array_like
        Angular separations, in radians.

    Returns
    -------
    array
        Chord lengths, in the same shape as ``theta``.
    """
    return 2. * np.sin(0.5 * np.asarray(theta, dtype=np.float64))


def chord_to_theta(chord):
    """Inverse of :func:`theta_to_chord`: chord length to angle, in radians."""
    chord = np.asarray(chord, dtype=np.float64)
    if np.any(chord < 0.) or np.any(chord > 2.):
        raise ValueError("chord lengths on the unit sphere must be in [0, 2]")
    return 2. * np.arcsin(0.5 * chord)


def compute_wtheta(data_cats, data_wts, theta_edges, degrees=True, **kwargs):
    r"""Angular correlation function w(theta) from RA/Dec catalogues.

    ``pyfcfc`` only measures Euclidean separations between Cartesian
    positions, but the angular correlation function is the unit-sphere
    special case: for objects at unit distance the separation is the chord
    :math:`s = 2\sin(\theta/2)`.  This function converts RA/Dec to
    unit-sphere Cartesian coordinates, bins the chord length with the
    ordinary isotropic pair counting (``bin=0``), and converts the bin
    centres back to angles.  The sphere is internally rescaled to a
    maximum chord of one, which leaves the geometry and the pair counts
    unchanged but avoids a small-separation limit of the FCFC bin lookup
    table; returned separations are converted back to chord lengths on the
    unit sphere.

    The catalogues are expected as {RA, Dec} or {RA, Dec, z}; RA and Dec
    must be in degrees, and the (optional) third column is ignored, so a
    redshift slice can be passed without reformatting it.  Coordinate
    conversion is always disabled (no cosmology is involved).

    Parameters
    ----------
    data_cats : sequence of array_like
        Catalogues with shape (N, 2) or (N, 3); the first two columns are
        RA and Dec in degrees.
    data_wts : sequence of array_like
        Weights of the catalogues, each of shape (N,).
    theta_edges : array_like
        Angular bin edges, strictly increasing and non-negative, in
        degrees if ``degrees=True`` and in radians otherwise.  The edges
        must not exceed 180 deg (360 deg for a full circle in radians).
    degrees : bool, default=True
        Whether ``theta_edges`` (and the returned angles) are in degrees.
    **kwargs
        Further options passed to :func:`pyfcfc.sky.py_compute_cf`, e.g.
        ``pair`` or ``cf``.  The ``bin`` and ``convert`` options are set by
        this function and must not be given.  For two catalogues the
        defaults are the Landy-Szalay estimator with labels 'D' and 'R';
        for a single catalogue the natural DD counts are returned.

    Returns
    -------
    dict
        The result dictionary of ``py_compute_cf``, with the angular bin
        centres in ``theta`` (in the requested unit), the bin edges in
        ``theta_edges``, and ``wtheta`` as an alias of ``cf``.  The
        separation ``s`` returned by ``py_compute_cf`` is the chord length.
    """
    if not isinstance(degrees, bool):
        raise ValueError("degrees must be a boolean")
    theta_edges = np.asarray(theta_edges, dtype=np.float64)
    if theta_edges.ndim != 1 or theta_edges.size < 2:
        raise ValueError("theta_edges must be a one-dimensional array with "
                         "at least two entries")
    if not np.all(np.isfinite(theta_edges)):
        raise ValueError("theta_edges must be finite")
    theta_max = 180. if degrees else np.pi
    if theta_edges[0] < 0. or theta_edges[-1] > theta_max:
        raise ValueError(f"theta_edges must lie in [0, {theta_max}]")
    if np.any(np.diff(theta_edges) <= 0.):
        raise ValueError("theta_edges must be strictly increasing")
    if 'bin' in kwargs and kwargs['bin'] != 0:
        raise ValueError("compute_wtheta only supports the isotropic binning "
                         "scheme (bin=0)")
    if kwargs.get('convert', False):
        raise ValueError("compute_wtheta works on the unit sphere and does "
                         "not support convert=True")

    theta_rad = np.deg2rad(theta_edges) if degrees else theta_edges
    chord_edges = theta_to_chord(theta_rad)

    # FCFC rescales separations internally for its bin lookup table, and the
    # resulting table becomes unreliable when the largest separation is much
    # smaller than unity (the rescaled squared distance overflows the 32-bit
    # bin index, and pairs pile up in one of the last bins).  The angular
    # geometry is scale free, so normalise the unit sphere to a maximum
    # chord of one: coordinates and chord edges are multiplied by the same
    # constant, which leaves all pair counts unchanged.  This keeps the
    # internal separations O(1) for any angular range.
    scale = 1. / chord_edges[-1]
    chord_scaled = chord_edges * scale

    cats = [np.asarray(cat) for cat in data_cats]
    if len(cats) != len(data_wts):
        raise ValueError(f"got {len(cats)} catalogues but "
                         f"{len(data_wts)} weight arrays")
    unit_cats = []
    for i, cat in enumerate(cats):
        if cat.ndim != 2 or cat.shape[1] not in (2, 3):
            raise ValueError(f"catalogue {i} must have shape (N, 2) or "
                             f"(N, 3), got {cat.shape}")
        unit_cats.append(radec_to_unit_sphere(cat[:, 0], cat[:, 1]) * scale)

    # Labels and default estimator.  Use friendly D/R labels for the
    # usual data+random case; otherwise fall back to the pyfcfc defaults.
    ncat = len(unit_cats)
    labels = kwargs.pop('label', None)
    if labels is None:
        labels = ['D', 'R'] if ncat == 2 else \
            [chr(ord('A') + i) for i in range(ncat)]
    elif isinstance(labels, str):
        labels = list(labels)
    if len(labels) != ncat:
        raise ValueError(f"got {len(labels)} labels for {ncat} catalogues")

    if ncat == 1:
        default_pair = [labels[0] * 2]
        default_cf = [labels[0] * 2]
    elif ncat == 2:
        dd, dr, rr = labels[0] * 2, labels[0] + labels[1], labels[1] * 2
        default_pair = [dd, dr, rr]
        default_cf = [f'({dd} - 2 * {dr} + {rr}) / {rr}']
    else:
        default_pair = None
        default_cf = None
    pair = kwargs.pop('pair', default_pair)
    cf = kwargs.pop('cf', default_cf)
    options = dict(kwargs)
    if pair is not None:
        options['pair'] = pair
    if cf is not None:
        options['cf'] = cf
    options.setdefault('verbose', False)

    # An explicit soft histogram kernel width is expressed in the units of
    # the separation edges, which we have just rescaled.
    if (options.get('soft_hist') and options.get('soft_sigma') is not None
            and not options.get('soft_sigma_log')):
        options['soft_sigma'] = float(options['soft_sigma']) * scale

    from pyfcfc.sky import py_compute_cf

    results = py_compute_cf(unit_cats, data_wts, chord_scaled, None, 1,
                            label=labels, bin=0, convert=False, **options)
    # Report separations in the original chord units.
    if 's' in results:
        results['s'] = np.asarray(results['s'], dtype=np.float64) / scale
    pairs = results.get('pairs', {})
    for key in ('smin', 'smax'):
        if key in pairs:
            pairs[key] = np.asarray(pairs[key], dtype=np.float64) / scale
    results['theta_edges'] = theta_edges
    results['theta'] = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    results['theta_edges_chord'] = chord_edges
    if 'cf' in results:
        results['wtheta'] = results['cf']
    return results


def _full_range_edges(half_edges):
    """Mirror positive edges [0, ..., max] to [-max, ..., 0, ..., max]."""
    half_edges = np.asarray(half_edges, dtype=np.float64)
    return np.concatenate((-half_edges[::-1][:-1], half_edges))


def pairs_to_pycorr(results, estimator_name, pair_mapping, box_size=None,
                    los_type=None):
    """Convert ``pyfcfc`` pair counts to a ``pycorr`` state dictionary.

    The returned dictionary can be saved with :func:`numpy.save` (with a
    ``.pkl.npy``-style suffix) and loaded with
    ``pycorr.TwoPointCorrelationFunction.load``, which gives access to the
    pycorr estimators, rebinning, plotting and saving tools.

    Parameters
    ----------
    results : dict
        Result dictionary returned by ``py_compute_cf``.  The binning
        scheme is inferred from its content: (s, mu) bins if ``mumin`` is
        present, (s_perp, pi) bins if ``pimin`` is present, isotropic
        otherwise.
    estimator_name : str
        Name of the pycorr estimator, e.g. 'natural', 'davispeebles',
        'landyszalay'.
    pair_mapping : dict
        Mapping from ``pyfcfc`` pair labels (e.g. 'DD', 'DR', 'RR') to
        pycorr attribute names.  Reversible pairs (counted by ``pyfcfc``
        once with absolute mu) must be mapped to *both* pycorr names, e.g.
        ``dict(DD='D1D2', DR=('D1R2', 'R1D2'), RR='R1R2')``.
    box_size : float or array, optional
        Side length(s) of the periodic box, for pair counts measured in a
        periodic box.  If set, the analytic random-random pair counts are
        generated when needed by the estimator (e.g. 'natural') and not
        present in ``pair_mapping``.
    los_type : str, optional
        Line-of-sight convention stored in the pycorr state.  Defaults to
        'z' if ``box_size`` is set (periodic box, line of sight along z),
        and 'midpoint' otherwise (the convention of FCFC for survey-like
        data).

    Returns
    -------
    dict
        A pycorr state dictionary, with ``estimator_name`` under the 'name'
        key and one entry per mapped pair count.
    """
    pairs = results['pairs']
    smin = np.asarray(pairs['smin'])
    smax = np.asarray(pairs['smax'])
    if smin.ndim == 2:
        s_edges = np.append(smin[:, 0], smax[-1, 0])
    elif smin.ndim == 1:
        s_edges = np.append(smin, smax[-1])
    else:
        raise ValueError(f"unexpected shape {smin.shape} for pairs['smin']")

    if 'mumin' in pairs:
        mode = 'smu'
        mu_edges = np.append(np.asarray(pairs['mumin'])[0, :],
                             np.asarray(pairs['mumax'])[0, -1])
        second_edges = _full_range_edges(mu_edges)
        edges = (s_edges, second_edges)
    elif 'pimin' in pairs:
        mode = 'rppi'
        pi_edges = np.append(np.asarray(pairs['pimin'])[0, :],
                             np.asarray(pairs['pimax'])[0, -1])
        second_edges = _full_range_edges(pi_edges)
        edges = (s_edges, second_edges)
    else:
        mode = 's'
        second_edges = None
        edges = (s_edges,)

    if los_type is None:
        los_type = 'z' if box_size is not None else 'midpoint'

    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    if second_edges is not None:
        second_centers = 0.5 * (second_edges[:-1] + second_edges[1:])

    if box_size is not None:
        box_size = np.atleast_1d(np.asarray(box_size, dtype=np.float64))
        if box_size.size == 1:
            box_size = np.repeat(box_size, 3)

    estimator_state = {'name': estimator_name}

    for key in _pair_keys(results):
        if key not in pair_mapping:
            continue
        counts = np.asarray(pairs[key], dtype=np.float64) \
            * results['normalization'][key]
        autocorr = key[0] == key[1]

        if mode in ('smu', 'rppi'):
            if counts.ndim != 2:
                raise ValueError(f"pair counts {key!r} must be 2-dimensional"
                                 f" for mode {mode!r}, got shape "
                                 f"{counts.shape}")
            # Mirror the |mu| (|pi|) counts to the full range: each half
            # holds half of the ordered pairs (exactly so for auto pairs).
            counts = 0.5 * counts
            wcounts = np.concatenate((counts[:, ::-1], counts), axis=1)
            # pycorr stores `seps' as full meshgrid-shaped arrays
            seps = [np.repeat(s_centers[:, None], second_centers.size,
                              axis=1),
                    np.repeat(second_centers[None, :], s_centers.size,
                              axis=0)]
        else:
            if counts.ndim != 1:
                raise ValueError(f"pair counts {key!r} must be 1-dimensional"
                                 f" for mode 's', got shape {counts.shape}")
            # In mode 's', pycorr stores the counts over the full pair
            # set (sum of wcounts = wnorm), exactly like FCFC's ordered
            # auto pairs and full cross pairs: no rescaling is needed.
            wcounts = counts
            seps = [s_centers]

        state = {}
        state['name'] = 'base'
        state['autocorr'] = int(autocorr)
        state['is_reversible'] = 1
        state['seps'] = [np.array(s, dtype=np.float64) for s in seps]
        state['ncounts'] = np.rint(wcounts).astype(np.int64)
        state['wcounts'] = wcounts
        state['wnorm'] = float(results['normalization'][key])
        state['size1'] = float(results['weighted_number'][key[0]])
        state['size2'] = float(results['weighted_number'][key[1]])
        state['edges'] = edges
        state['mode'] = mode
        state['bin_type'] = 'auto'
        state['boxsize'] = box_size
        state['los_type'] = los_type
        state['compute_sepsavg'] = [False] * len(edges)
        state['weight_attrs'] = {}
        state['selection_attrs'] = {}
        state['attrs'] = {}
        state['dtype'] = np.dtype(np.float64)

        mapping = pair_mapping[key]
        if isinstance(mapping, str):
            estimator_state[mapping] = state
        else:
            for name in mapping:
                estimator_state[name] = state

    # For the 'natural' estimator in a periodic box, the random-random
    # pair counts can be computed analytically.
    if 'R1R2' not in estimator_state and estimator_name == 'natural':
        if box_size is None:
            raise ValueError("the 'natural' estimator requires either R1R2 "
                             "pair counts in `pair_mapping', or `box_size' "
                             "for the analytic random-random counts")
        from pycorr import AnalyticTwoPointCounter
        dstate = estimator_state['D1D2']
        analytic = AnalyticTwoPointCounter(mode, edges, box_size,
                                           size1=dstate['size1'], size2=None,
                                           los='z')
        estimator_state['R1R2'] = analytic.__getstate__()

    return estimator_state
