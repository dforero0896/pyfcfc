"""Optional converters between pyfcfc results and external frameworks.

Currently supported:

- `lsstypes <https://github.com/adematti/lsstypes>`_: the pair counts of a
  pyfcfc result dictionary are converted into :class:`lsstypes.Count2`
  leaves, assembled into a :class:`lsstypes.Count2Correlation` (which
  knows the Landy-Szalay / natural estimators), and optionally projected
  onto multipoles (:class:`lsstypes.Count2CorrelationPoles`), wedges or
  :math:`w_p`.  This gives access to lsstypes' machinery: replication
  (sum/mean/cov over catalogue replicas), plotting, and serialisation.

All functions raise an informative :class:`ImportError` when lsstypes is
not installed; nothing else in pyfcfc depends on it.
"""

import numpy as np

__all__ = ['to_lsstypes_counts', 'to_lsstypes_correlation', 'to_lsstypes']


def _import_lsstypes():
    try:
        import lsstypes
    except ImportError as exc:
        raise ImportError(
            "this utility requires the optional dependency `lsstypes'; "
            "install it with:\n    pip install git+https://github.com/"
            "adematti/lsstypes") from exc
    return lsstypes


def _pair_keys(results):
    skip = {'smin', 'smax', 'mumin', 'mumax', 'pimin', 'pimax'}
    return [key for key in results['pairs'] if key not in skip]


def _default_pair_mapping(results):
    """Map pyfcfc pair labels onto lsstypes estimator names.

    Auto pairs keep their name ('DD' -> 'DD'); cross pairs are mapped to
    both orderings ('DR' -> ('DR', 'RD')), as required by the Landy-Szalay
    estimator.  This default assumes the catalogues are labelled D/R (or
    D/S...); pass an explicit ``pair_mapping`` otherwise, e.g.
    ``{'AA': 'DD', 'AB': ('DR', 'RD'), 'BB': 'RR'}``.
    """
    mapping = {}
    for key in _pair_keys(results):
        if key[0] == key[1]:
            mapping[key] = key
        else:
            mapping[key] = (key, key[::-1])
    return mapping


def _mode_and_edges(results):
    pairs = results['pairs']
    smin = np.asarray(pairs['smin'])
    smax = np.asarray(pairs['smax'])
    if smin.ndim == 2:
        s_edges = np.append(smin[:, 0], smax[-1, 0])
    else:
        s_edges = np.append(smin, smax[-1])
    if 'mumin' in pairs:
        mumin = np.asarray(pairs['mumin'])
        mumax = np.asarray(pairs['mumax'])
        half = np.append(mumin[0, :], mumax[0, -1])
        return 'smu', s_edges, half
    if 'pimin' in pairs:
        pimin = np.asarray(pairs['pimin'])
        pimax = np.asarray(pairs['pimax'])
        half = np.append(pimin[0, :], pimax[0, -1])
        return 'rppi', s_edges, half
    return 's', s_edges, None


def _full_grid(half_edges):
    """Mirror positive edges/centers onto the full symmetric range."""
    full_edges = np.concatenate((-half_edges[::-1][:-1], half_edges))
    centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    return full_edges, centers


def to_lsstypes_counts(results, pair_mapping=None, attrs=None):
    """Convert pyfcfc pair counts into :class:`lsstypes.Count2` leaves.

    Parameters
    ----------
    results : dict
        Result dictionary returned by ``py_compute_cf``.
    pair_mapping : dict, optional
        Mapping from pyfcfc pair labels to lsstypes count names; see
        :func:`_default_pair_mapping` for the default.  A pair mapped to
        two names (e.g. ``('DR', 'RD')``) produces two identical
        :class:`lsstypes.Count2` leaves (pyfcfc counts |mu|, so the two
        orderings carry the same information).
    attrs : dict, optional
        Extra attributes attached to every leaf.

    Returns
    -------
    dict
        ``{count_name: lsstypes.Count2}``.
    """
    lsstypes = _import_lsstypes()
    mode, s_edges, half = _mode_and_edges(results)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    if mode == 'smu':
        coord_names = ['s', 'mu']
    elif mode == 'rppi':
        coord_names = ['rp', 'pi']
    else:
        coord_names = ['s']

    if mode != 's':
        full_edges, second_centers = _full_grid(half)
        s_coord = s_centers                      # broadcast by lsstypes
        second_coord = second_centers
    else:
        full_edges = None
        s_coord = s_centers
        second_coord = None

    mapping = pair_mapping if pair_mapping is not None \
        else _default_pair_mapping(results)

    counts = {}
    for key, names in mapping.items():
        raw = np.asarray(results['pairs'][key], dtype=np.float64) \
            * results['normalization'][key]
        if mode != 's':
            # mirror the |mu| (|pi|) counts onto the full range; each half
            # carries half of the (ordered) pairs, as in pycorr
            raw = 0.5 * raw
            raw = np.concatenate((raw[:, ::-1], raw), axis=1)
        norm = np.full(raw.shape, results['normalization'][key],
                       dtype=np.float64)
        kw = {coord_names[0]: s_coord,
              f'{coord_names[0]}_edges':
                  np.column_stack([s_edges[:-1], s_edges[1:]])}
        if mode != 's':
            kw[coord_names[1]] = second_coord
            kw[f'{coord_names[1]}_edges'] = \
                np.column_stack([full_edges[:-1], full_edges[1:]])
        leaf_attrs = {'size1': results['weighted_number'][key[0]],
                      'size2': results['weighted_number'][key[1]]}
        if attrs:
            leaf_attrs.update(attrs)
        names = (names,) if isinstance(names, str) else tuple(names)
        for name in names:
            counts[name] = lsstypes.Count2(counts=raw, norm=norm,
                                           coords=coord_names,
                                           attrs=dict(leaf_attrs), **kw)
    return counts


def _analytic_rr_count(lsstypes, results, mode, s_edges, half, box_size):
    """Analytic random-random counts for a periodic box (natural estimator).

    The expected (ordered) pair counts are ``norm * dV / V``, with ``norm``
    the normalization of the DD pair counts (so that the estimator is
    consistent for weighted catalogues), ``dV`` the shell volume, and, for
    the (s, mu) scheme, the uniform mu fraction of each bin.
    """
    box = np.broadcast_to(np.asarray(box_size, dtype=np.float64), (3,))
    volume = box[0] * box[1] * box[2]
    shell = 4. / 3. * np.pi * (s_edges[1:] ** 3 - s_edges[:-1] ** 3)
    auto = _auto_key(results)
    norm = results['normalization'][auto]
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    if mode == 's':
        raw = norm * shell / volume
        kw = {'s': s_centers,
              's_edges': np.column_stack([s_edges[:-1], s_edges[1:]])}
        coord_names = ['s']
    elif mode == 'smu':
        full_edges, mu_centers = _full_grid(half)
        dmu = np.diff(full_edges)
        raw = norm * (shell / volume)[:, None] * (dmu / 2.)[None, :]
        kw = {'s': s_centers,
              's_edges': np.column_stack([s_edges[:-1], s_edges[1:]]),
              'mu': mu_centers,
              'mu_edges': np.column_stack([full_edges[:-1], full_edges[1:]])}
        coord_names = ['s', 'mu']
    else:
        raise NotImplementedError(
            "analytic random-random counts are only implemented for the "
            "isotropic and (s, mu) binning schemes; provide RR pair counts "
            "for (s_perp, pi) binning")
    size1 = results['weighted_number'][auto[0]]
    return lsstypes.Count2(counts=raw, norm=np.full(raw.shape, norm),
                           coords=coord_names,
                           attrs={'size1': size1, 'size2': size1}, **kw)


def _auto_key(results):
    """Name of an auto pair count present in the results ('DD'-like)."""
    auto = [k for k in _pair_keys(results) if k[0] == k[1]]
    if not auto:
        raise ValueError("no auto pair count found in the results; cannot "
                         "build the analytic random-random counts")
    return auto[0]


def to_lsstypes_correlation(results, estimator='landyszalay',
                            pair_mapping=None, box_size=None, attrs=None):
    """Convert pyfcfc pair counts into a :class:`lsstypes.Count2Correlation`.

    Parameters
    ----------
    results : dict
        Result dictionary returned by ``py_compute_cf``.
    estimator : str, default='landyszalay'
        Estimator name ('landyszalay', 'natural') or an explicit formula
        in terms of the pair-count names, e.g. ``'(DD - 2 * DR + RR) / RR'``
        (evaluated by lsstypes on the normalized counts).
    pair_mapping : dict, optional
        See :func:`to_lsstypes_counts`.
    box_size : float or array, optional
        Periodic box side(s); required to synthesize analytic
        random-random counts for the 'natural' estimator when the results
        do not contain RR pair counts.
    attrs : dict, optional
        Extra attributes for the leaves.

    Returns
    -------
    lsstypes.Count2Correlation
    """
    lsstypes = _import_lsstypes()
    counts = to_lsstypes_counts(results, pair_mapping=pair_mapping,
                                attrs=attrs)
    if estimator == 'natural' and 'RR' not in counts:
        if box_size is None:
            raise ValueError("the 'natural' estimator requires RR pair "
                             "counts in the results, or `box_size' to "
                             "synthesize analytic random-random counts")
        mode, s_edges, half = _mode_and_edges(results)
        counts['RR'] = _analytic_rr_count(lsstypes, results, mode, s_edges,
                                          half, box_size)
    return lsstypes.Count2Correlation(estimator=estimator, **counts)


def to_lsstypes(results, estimator='landyszalay', project='poles',
                ells=(0, 2, 4), pair_mapping=None, box_size=None,
                attrs=None, **project_kwargs):
    """Convert pyfcfc pair counts into lsstypes correlation objects.

    Parameters
    ----------
    results : dict
        Result dictionary returned by ``py_compute_cf``.
    estimator : str, default='landyszalay'
        Estimator used by :class:`lsstypes.Count2Correlation`.
    project : str or None, default='poles'
        Projection applied to the correlation function: 'poles'
        (multipoles, :class:`lsstypes.Count2CorrelationPoles`), 'wedges',
        'wp' (projected correlation function, requires (s_perp, pi)
        binning), 'binned' (isotropic), or None to return the
        un-projected :class:`lsstypes.Count2Correlation`.
    ells : sequence of int, default=(0, 2, 4)
        Multipole orders for ``project='poles'``.
    pair_mapping, box_size, attrs :
        See :func:`to_lsstypes_correlation`.
    **project_kwargs :
        Extra arguments for :meth:`lsstypes.Count2Correlation.project`
        (e.g. ``ignore_nan=True``).

    Returns
    -------
    lsstypes.Count2CorrelationPoles / Count2CorrelationWedges /
    Count2CorrelationWp / Count2CorrelationBinned / Count2Correlation
    """
    correlation = to_lsstypes_correlation(
        results, estimator=estimator, pair_mapping=pair_mapping,
        box_size=box_size, attrs=attrs)
    if project is None:
        return correlation
    if project == 'poles':
        return correlation.project('poles', ells=list(ells),
                                   **project_kwargs)
    return correlation.project(project, **project_kwargs)
