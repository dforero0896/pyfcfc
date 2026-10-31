"""Edge padding ("buffer bins") for the soft histogramming pair counts.

Why padding is needed
---------------------
With soft histogramming, the weight of a pair with separation ``s`` is
spread over the bins with the softmax of a Gaussian kernel of width
``sigma`` (``sigma(s) = f * s`` for the scale-free kernel).  The Gaussian
has infinite support, but the tree traversal only counts pairs inside the
binning range ``[sedges[0], sedges[-1])``: the kernel tails that reach
beyond the outermost edges are truncated.  For the outermost bins the
estimator therefore averages ``xi(s)`` over a *half* kernel: mass leaking
out of the last bin towards larger separations (where ``xi`` is smaller
for a steep correlation function) is not compensated by mass leaking in
from beyond ``sedges[-1]``, because those pairs are never counted.  The
result is a positive bias (excess power) in the last separation bin and a
smaller one in the first, growing with the steepness of ``xi`` and with
the kernel width (see ``leak_repro/analytic_demo.py``: e.g. +8.7% excess
in the last bin for a ``xi ~ s^-1.8`` power law with the default kernel
``f = 0.5 * dln s``, reduced to +0.7% by padding with two kernel widths).

How padding works
-----------------
``soft_pad = z`` extends the counting range by ``z`` kernel widths beyond
the outermost edges,

* ``sedges[-1] -> sedges[-1] * exp(z f)``   (scale-free kernel, log bins),
* ``sedges[-1] -> sedges[-1] + z sigma``    (constant kernel width),

and symmetrically at the lower edge.  The extension is realised with
*buffer bins*: continuation bins appended to the separation grid, with
centres on the natural continuation of the grid (geometric for
log-spaced bins, arithmetic otherwise, using the local spacing at each
edge).  The softmax therefore normalises over the real *and* the buffer
centres, exactly like an infinite virtual continuation of the grid, so
pairs beyond the original edges contribute their proper (Gaussian-decayed)
weight to the outermost real bins, while pairs near the edges lose the
spurious boost from the truncated normalisation.  The buffer-bin counts
are discarded from the results; the real bins are returned with their
original edges.

The padding is quantised to whole continuation bins: the number of buffer
bins per side is ``ceil(z * sigma_edge / spacing)``, at least one when
``z > 0``.  ``soft_pad = 0`` disables padding and restores the exact
previous behaviour (softmax over the real bins only, pairs in
``[sedges[0], sedges[-1])``).

Cost
----
Padding enumerates the pairs in the buffer shells as well.  At the upper
edge their number grows with the enclosed volume, ~ ``exp(3 z f) - 1``
relative to the pairs inside ``sedges[-1]`` for log-spaced bins (e.g.
+86% for the default kernel with one buffer bin per side at 0.2 dex bin
width, +16% at 0.05 dex), and ~ ``((smax + z sigma)/smax)^3 - 1`` for
linear bins.  Lower-edge padding is cheap (the volume below
``sedges[0]`` is tiny).  In a periodic box the extended maximum edge is
clamped to stay below half the box size (FCFC's minimum-image limit), so
setups binning close to ``L/2`` get reduced or no upper padding.
"""

import numpy as np

__all__ = ["SOFT_PAD_DEFAULT", "plan_padding", "slice_results",
           "pad_cotangent"]

#: Default edge padding (in kernel widths) of the soft histogram method.
SOFT_PAD_DEFAULT = 2.0


def resolve_sigma(edges, soft_sigma, soft_sigma_log):
    """Kernel width used for the padding plan.

    Returns the user value, or the library default (half the mean bin
    width, computed from the *original* edges exactly as the C setup
    does) when `soft_sigma` is None.
    """
    ns = len(edges) - 1
    if soft_sigma is not None:
        return float(soft_sigma)
    if soft_sigma_log:
        return 0.5 * float(np.log(edges[-1] / edges[0])) / ns
    return 0.5 * float(edges[-1] - edges[0]) / ns


def plan_padding(edges, soft_pad, soft_sigma, soft_sigma_log,
                 max_edge=None):
    """Extended separation grid with buffer bins for soft-hist padding.

    Parameters
    ----------
    edges : ndarray
        Original separation (or s_perp) bin edges, length ns + 1.
    soft_pad : float
        Padding in kernel widths (``z``); 0 disables padding.
    soft_sigma : float or None
        Kernel width (None: library default, half the mean bin width;
        for `soft_sigma_log`, the fraction ``f`` with
        ``sigma(s) = f * s``).
    soft_sigma_log : bool
        Whether the scale-free kernel is used.
    max_edge : float or None
        Upper limit for the extended maximum edge (half the periodic box
        size for the boxes module); None means no limit.

    Returns
    -------
    (edges_ext, nlo, nhi, sigma) :
        Extended edges (identical to `edges` when no padding applies),
        the numbers of buffer bins below and above, and the kernel width
        to pass explicitly to the C library when `soft_sigma` was None
        (pinning the default to the *original* grid; None otherwise).
    """
    edges = np.asarray(edges, dtype=np.float64)
    ns = edges.size - 1
    if not (soft_pad > 0):
        return edges, 0, 0, None

    if soft_sigma_log and edges[0] <= 0 and soft_sigma is None:
        # The default f is undefined without a positive first edge (and no
        # geometric continuation exists below zero).
        raise ValueError(
            "soft_sigma_log with soft_pad requires positive sedges[0] "
            "(or an explicit soft_sigma)")
    sig = resolve_sigma(edges, soft_sigma, soft_sigma_log)
    # When soft_sigma was None, pin the library default explicitly: the
    # C code would otherwise recompute it from the *extended* grid, which
    # differs for non-uniform bins.
    sigma = None if soft_sigma is not None else sig

    if soft_sigma_log:
        if edges[0] <= 0:
            nlo = 0            # no geometric continuation below zero
        else:
            dlo = float(np.log(edges[1] / edges[0]))
            nlo = max(int(np.ceil(soft_pad * sig / dlo - 1e-12)), 0)
        dhi = float(np.log(edges[-1] / edges[-2]))
        nhi = max(int(np.ceil(soft_pad * sig / dhi - 1e-12)), 0)
        if max_edge is not None and nhi > 0:
            room = np.log(max_edge / edges[-1]) / dhi
            nhi = min(nhi, max(int(np.floor(room + 1e-12)), 0))
        lo = edges[0] * np.exp(-dlo * np.arange(nlo, 0, -1)) if nlo else \
            np.empty(0)
        hi = edges[-1] * np.exp(dhi * np.arange(1, nhi + 1)) if nhi else \
            np.empty(0)
    else:
        dlo = float(edges[1] - edges[0])
        dhi = float(edges[-1] - edges[-2])
        # The lower edge must stay non-negative (FCFC requirement, and
        # no pairs exist below zero anyway).
        nlo = max(int(np.ceil(soft_pad * sig / dlo - 1e-12)), 0)
        nlo = min(nlo, max(int(np.floor(edges[0] / dlo + 1e-12)), 0))
        nhi = max(int(np.ceil(soft_pad * sig / dhi - 1e-12)), 0)
        if max_edge is not None and nhi > 0:
            room = (max_edge - edges[-1]) / dhi
            nhi = min(nhi, max(int(np.floor(room + 1e-12)), 0))
        lo = (edges[0] - dlo * np.arange(nlo, 0, -1)) if nlo else \
            np.empty(0)
        hi = (edges[-1] + dhi * np.arange(1, nhi + 1)) if nhi else \
            np.empty(0)

    if nlo == 0 and nhi == 0:
        return edges, 0, 0, None
    edges_ext = np.concatenate([lo, edges, hi]).astype(np.float64)
    return edges_ext, nlo, nhi, sigma


def slice_results(results, nlo, ns):
    """Remove the buffer bins from a py_compute_cf results dictionary.

    Every array with a separation axis (pair counts, correlation
    functions, multipoles, wp, bin edges and centres) is sliced back to
    the `ns` real bins starting at offset `nlo`; the other entries
    (labels, normalizations, particle numbers, poles) are scalars or
    catalogue-level and are kept as they are.  Modified in place and
    returned for convenience.
    """
    if nlo == 0 and ns == len(results['s']):
        return results
    sl = slice(nlo, nlo + ns)
    pairs = results['pairs']
    for key in list(pairs):
        pairs[key] = pairs[key][sl]
    if 'cf' in results:
        results['cf'] = results['cf'][:, sl]
    if 'multipoles' in results:
        results['multipoles'] = results['multipoles'][:, :, sl]
    if 'projected' in results:
        results['projected'] = results['projected'][:, sl]
    results['s'] = results['s'][sl]
    return results


def pad_cotangent(cot, nlo, ns_ext):
    """Embed a cotangent array of the real bins in the extended grid.

    The buffer bins get zero cotangents: they are discarded from the
    forward counts, so no upstream gradient flows through them (they do
    still enter the softmax normalisation of the real-bin weights, which
    the C adjoint replays exactly).
    """
    if cot.ndim == 1:
        out = np.zeros((ns_ext,), dtype=np.float64)
        out[nlo:nlo + cot.shape[0]] = cot
    else:
        out = np.zeros((ns_ext, cot.shape[1]), dtype=np.float64)
        out[nlo:nlo + cot.shape[0]] = cot
    return out
