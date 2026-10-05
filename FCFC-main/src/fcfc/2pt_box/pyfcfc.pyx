#cython: language_level=3
#cython: boundscheck=False
# cython: initializedcheck=False
"""
pyfcfc Cython bindings for FCFC_2PT_BOX (2PCF for periodic simulation boxes).

This module exposes a single function, ``py_compute_cf``, which evaluates
pair counts and correlation functions from in-memory (NumPy) catalogues.
The FCFC C structures are treated as opaque here: every access goes through
the accessor functions declared in ``pyfcfc_helpers.h``, which guarantees
that the Python bindings cannot desynchronise from the C structure layouts.
"""

import numpy as np

from libc.stdlib cimport malloc, free

cdef extern from "define.h":
    cdef int FCFC_BIN_ISO
    cdef int FCFC_BIN_SMU
    cdef int FCFC_BIN_SPI

cdef extern from "eval_cf.h":
    ctypedef struct CF:
        pass
    ctypedef struct DATA:
        pass
    void cf_destroy(CF *cf) nogil

cdef extern from "pyfcfc_helpers.h":
    const char *pyfcfc_arg0() nogil
    int pyfcfc_simd_level() nogil
    const char *pyfcfc_simd_name() nogil

    DATA *pyfcfc_data_alloc(int ncat) nogil
    int pyfcfc_data_fill(DATA *dat, int idx, size_t n,
                         const double *xyz, const double *w) nogil
    int pyfcfc_data_fill_f(DATA *dat, int idx, size_t n,
                           const float *xyz, const float *w) nogil
    void pyfcfc_data_free(DATA *dat, int ncat) nogil

    int pyfcfc_cf_ncat(const CF *cf) nogil
    int pyfcfc_cf_ns(const CF *cf) nogil
    int pyfcfc_cf_np(const CF *cf) nogil
    int pyfcfc_cf_nmu(const CF *cf) nogil
    int pyfcfc_cf_npc(const CF *cf) nogil
    int pyfcfc_cf_ncf(const CF *cf) nogil
    int pyfcfc_cf_nl(const CF *cf) nogil
    int pyfcfc_cf_bintype(const CF *cf) nogil
    int pyfcfc_cf_has_mp(const CF *cf) nogil
    int pyfcfc_cf_has_wp(const CF *cf) nogil
    char pyfcfc_cf_label(const CF *cf, int i) nogil
    size_t pyfcfc_cf_data_n(const CF *cf, int i) nogil
    double pyfcfc_cf_data_wt(const CF *cf, int i) nogil
    int pyfcfc_cf_pc_idx(const CF *cf, int i, int j) nogil
    double pyfcfc_cf_norm(const CF *cf, int i) nogil
    int pyfcfc_cf_pole(const CF *cf, int i) nogil
    void pyfcfc_cf_sbin_raw(const CF *cf, double *dst) nogil
    void pyfcfc_cf_pbin_raw(const CF *cf, double *dst) nogil
    void pyfcfc_cf_ncnt(const CF *cf, int i, double *dst) nogil
    void pyfcfc_cf_cfval(const CF *cf, int i, double *dst) nogil
    void pyfcfc_cf_mp(const CF *cf, int i, double *dst) nogil
    void pyfcfc_cf_wp(const CF *cf, int i, double *dst) nogil

    CF *compute_cf(int argc, char *argv[], DATA *dat, int ncat,
                   double *sbins, int ns, double *pbins, int np_,
                   int nmu) nogil


def simd_info():
    """Return the SIMD instruction set compiled into this build.

    Returns
    -------
    (int, str)
        The FCFC SIMD level (0 = none, 1 = AVX, 2 = AVX2, 3 = AVX512) and
        a human-readable name.  Rebuild with ``PYFCFC_WITH_SIMD=1`` to
        enable the vectorised counting kernels.
    """
    return pyfcfc_simd_level(), (<bytes> pyfcfc_simd_name()).decode('ascii')


# ============================================================================
#  Python-side helpers
# ============================================================================

# Valid FCFC configuration options for this component: the command-line
# long option names (with '-' replaced by '_' when passed as keywords).
_OPTIONS = frozenset(['bin', 'box', 'cf', 'cf-output', 'conf', 'data-struct', 'label', 'mp-output', 'multipole', 'out-format', 'overwrite', 'pair', 'pair-output', 'verbose', 'weight', 'wp', 'wp-output'])

# Friendly hints for frequent mistakes.
_OPTION_HINTS = {
    'nthreads': ("the number of OpenMP threads is set through the "
                 "OMP_NUM_THREADS environment variable"),
    'num_threads': ("the number of OpenMP threads is set through the "
                    "OMP_NUM_THREADS environment variable"),
    'box_size': "the periodic box size option is called 'box'",
    'boxsize': "the periodic box size option is called 'box'",
    'mu_num': "the number of mu bins is the positional argument `nmu'",
    'nmu': "the number of mu bins is the positional argument `nmu'",
    's_min': "separation bins are passed positionally as `sedges'",
    's_max': "separation bins are passed positionally as `sedges'",
    's_step': "separation bins are passed positionally as `sedges'",
    'pi_min': "pi bins are passed positionally as `pedges'",
    'pi_max': "pi bins are passed positionally as `pedges'",
    'pi_step': "pi bins are passed positionally as `pedges'",
    'poles': "the multipole orders option is called 'multipole'",
    'ells': "the multipole orders option is called 'multipole'",
    'estimator': "correlation function estimators are passed via 'cf'",
    'positions': "catalogues are passed positionally as `data_cats'",
    'weights': "weights are passed positionally as `data_wts'",
    'convert': "coordinate conversion only exists in the survey-like "
               "module (pyfcfc.sky)",
    'omega_m': "cosmology parameters only exist in the survey-like module "
               "(pyfcfc.sky)",
}


def _validate_options(kwargs):
    """Raise on unknown FCFC options instead of silently ignoring them."""
    import difflib
    for key in kwargs:
        dashed = str(key).strip().replace('_', '-')
        if dashed in _OPTIONS:
            continue
        msg = f"unknown FCFC option: {key!r}"
        close = difflib.get_close_matches(dashed, sorted(_OPTIONS), n=3)
        if close:
            pretty = ', '.join(repr(c.replace('-', '_')) for c in close)
            msg += f" (did you mean {pretty}?)"
        hint = _OPTION_HINTS.get(str(key))
        if hint:
            msg += f" -- note: {hint}"
        msg += f"; valid options are: {', '.join(sorted(o.replace('-', '_') for o in _OPTIONS))}"
        raise ValueError(msg)


def _format_option(val):
    """Format a Python object as an FCFC command-line option value."""
    if isinstance(val, (bool, np.bool_)):
        return 'T' if val else 'F'
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple, np.ndarray)):
        return '[' + ', '.join(_format_option(v) for v in val) + ']'
    if isinstance(val, (int, float, np.integer, np.floating)):
        return str(val)
    raise TypeError(f"cannot convert keyword argument of type "
                    f"{type(val).__name__} to an FCFC option")


def _validate_labels(labels, ncat):
    """Validate and normalize the catalogue labels."""
    if labels is None:
        if ncat > 26:
            raise ValueError("at most 26 catalogs are supported; please pass "
                             "explicit (unique uppercase) labels")
        return [chr(ord('A') + i) for i in range(ncat)]
    if isinstance(labels, str):
        labels = list(labels)
    labels = [str(lab).strip().upper() for lab in labels]
    if len(labels) != ncat:
        raise ValueError(f"got {len(labels)} labels for {ncat} catalogs")
    for lab in labels:
        if len(lab) != 1 or not ('A' <= lab <= 'Z'):
            raise ValueError(f"invalid catalog label: {lab!r} (labels must be "
                             "single uppercase letters)")
    if len(set(labels)) != ncat:
        raise ValueError(f"catalog labels must be unique, got {labels}")
    return labels


cdef dict _extract_results(CF *cf):
    """Copy the results out of the CF structure and destroy it."""
    cdef int i, idx
    cdef int ncat = pyfcfc_cf_ncat(cf)
    cdef int ns = pyfcfc_cf_ns(cf)
    cdef int nmu = pyfcfc_cf_nmu(cf)
    cdef int npi = pyfcfc_cf_np(cf)
    cdef int npc = pyfcfc_cf_npc(cf)
    cdef int ncf = pyfcfc_cf_ncf(cf)
    cdef int nl = pyfcfc_cf_nl(cf)
    cdef int bintype = pyfcfc_cf_bintype(cf)
    cdef size_t ntot = ns
    cdef double[::1] mv1
    cdef double[:, ::1] mv2
    cdef double[:, ::1] mv3

    if bintype == FCFC_BIN_SMU:
        ntot *= nmu
    elif bintype == FCFC_BIN_SPI:
        ntot *= npi

    results = {}
    labels = []
    for i in range(ncat):
        labels.append(bytes([<unsigned char> pyfcfc_cf_label(cf, i)]
                            ).decode('ascii'))
    results['labels'] = labels
    results['number'] = {labels[i]: pyfcfc_cf_data_n(cf, i)
                         for i in range(ncat)}
    results['weighted_number'] = {labels[i]: pyfcfc_cf_data_wt(cf, i)
                                  for i in range(ncat)}

    # Bin edges in input units.
    sedges = np.empty(ns + 1, dtype=np.float64)
    mv1 = sedges
    pyfcfc_cf_sbin_raw(cf, &mv1[0])
    smin = sedges[:ns].copy()
    smax = sedges[1:].copy()

    pairs = {}
    if bintype == FCFC_BIN_SMU:
        mu_min = np.arange(nmu, dtype=np.float64) / nmu
        mu_max = mu_min + 1.0 / nmu
        pairs['smin'] = np.ascontiguousarray(np.tile(smin, (nmu, 1)).T)
        pairs['smax'] = np.ascontiguousarray(np.tile(smax, (nmu, 1)).T)
        pairs['mumin'] = np.tile(mu_min, (ns, 1))
        pairs['mumax'] = np.tile(mu_max, (ns, 1))
    elif bintype == FCFC_BIN_SPI:
        pedges = np.empty(npi + 1, dtype=np.float64)
        mv1 = pedges
        pyfcfc_cf_pbin_raw(cf, &mv1[0])
        pmin = pedges[:npi].copy()
        pmax = pedges[1:].copy()
        pairs['smin'] = np.ascontiguousarray(np.tile(smin, (npi, 1)).T)
        pairs['smax'] = np.ascontiguousarray(np.tile(smax, (npi, 1)).T)
        pairs['pimin'] = np.tile(pmin, (ns, 1))
        pairs['pimax'] = np.tile(pmax, (ns, 1))
    else:
        pairs['smin'] = smin
        pairs['smax'] = smax

    # Normalized pair counts.
    normalization = {}
    buf = np.empty(ntot, dtype=np.float64)
    for idx in range(npc):
        i0 = pyfcfc_cf_pc_idx(cf, idx, 0)
        i1 = pyfcfc_cf_pc_idx(cf, idx, 1)
        if not (0 <= i0 < ncat and 0 <= i1 < ncat):
            # Cannot happen with the current C code (every requested pair
            # is evaluated, or read from a PAIR_COUNT_FILE); guard against
            # future regressions that would silently return empty counts.
            raise RuntimeError(
                f"pair count {idx} was not evaluated by FCFC; please "
                "report this as a pyfcfc bug")
        key = labels[i0] + labels[i1]
        normalization[key] = pyfcfc_cf_norm(cf, idx)
        mv1 = buf
        pyfcfc_cf_ncnt(cf, idx, &mv1[0])
        if bintype == FCFC_BIN_SMU:
            pairs[key] = np.ascontiguousarray(buf.reshape(nmu, ns).T)
        elif bintype == FCFC_BIN_SPI:
            pairs[key] = np.ascontiguousarray(buf.reshape(npi, ns).T)
        else:
            pairs[key] = buf.copy()
    results['normalization'] = normalization
    results['pairs'] = pairs

    # Correlation functions evaluated by FCFC (if any).
    if ncf > 0:
        cf_buf = np.empty((ncf, ntot), dtype=np.float64)
        mv2 = cf_buf
        for idx in range(ncf):
            pyfcfc_cf_cfval(cf, idx, &mv2[idx, 0])
        if bintype == FCFC_BIN_SMU:
            results['cf'] = np.ascontiguousarray(
                cf_buf.reshape(ncf, nmu, ns).transpose(0, 2, 1))
        elif bintype == FCFC_BIN_SPI:
            results['cf'] = np.ascontiguousarray(
                cf_buf.reshape(ncf, npi, ns).transpose(0, 2, 1))
        else:
            results['cf'] = cf_buf

    # Bin centers.
    results['s'] = 0.5 * (smin + smax)

    # Multipoles (shape: ncf x nl x ns) and poles.
    if pyfcfc_cf_has_mp(cf) and ncf > 0:
        mp = np.empty((ncf, nl, ns), dtype=np.float64)
        mp_buf = np.empty(nl * ns, dtype=np.float64)
        mv1 = mp_buf
        for idx in range(ncf):
            pyfcfc_cf_mp(cf, idx, &mv1[0])
            mp[idx] = mp_buf.reshape(nl, ns)
        results['multipoles'] = mp
        results['poles'] = [pyfcfc_cf_pole(cf, i) for i in range(nl)]

    # Projected correlation functions (shape: ncf x ns).
    if pyfcfc_cf_has_wp(cf) and ncf > 0:
        wp = np.empty((ncf, ns), dtype=np.float64)
        mv3 = wp
        for idx in range(ncf):
            pyfcfc_cf_wp(cf, idx, &mv3[idx, 0])
        results['projected'] = wp

    cf_destroy(cf)
    return results


# ============================================================================
#  Main entry point
# ============================================================================

def py_compute_cf(data_cats, data_wts, sedges, pedges=None, nmu=1, **kwargs):
    """Evaluate pair counts and correlation functions with FCFC.

    Parameters
    ----------
    data_cats : sequence of array_like
        Cartesian coordinates of the input catalogues, each with shape
        (N, 3) (positions are assumed to be periodic-box coordinates, with the
        z axis along the line of sight).  Arrays are copied internally; the caller's
        data are never modified.  float32 inputs are supported (they are
        used natively only if *all* positions and weights are float32,
        otherwise everything is promoted to float64).
    data_wts : sequence of array_like
        Weights of the input catalogues, each with shape (N,).  Use an
        array of ones for unweighted catalogues.
    sedges : array_like
        Edges of the separation (or s_perp) bins, length ns + 1, strictly
        increasing and non-negative.
    pedges : array_like, optional
        Edges of the pi bins, required by the (s_perp, pi) binning scheme
        (``bin=2``).
    nmu : int, optional
        Number of mu bins in [0, 1), used by the (s, mu) binning scheme
        (``bin=1``).
    **kwargs :
        FCFC configuration options, following the command-line option
        names of FCFC with ``-`` replaced by ``_`` (e.g. ``box=1000``,
        ``pair=['DD', 'DR', 'RR']``, ``cf=['(DD - 2*DR + RR) / RR']``,
        ``multipole=[0, 2, 4]``, ``verbose=False``).  Python booleans are
        converted to 'T'/'F', and sequences to FCFC array literals.

    Returns
    -------
    dict
        Results dictionary with the keys:

        - ``labels``: the catalogue labels;
        - ``number`` / ``weighted_number``: per catalogue, the number of
          objects and the sum of the weights;
        - ``normalization``: per pair count, the normalization factor
          (for auto pairs: sum(w)^2 - sum(w^2), matching pycorr;
          for cross pairs: sum(w1) * sum(w2));
        - ``pairs``: normalized pair counts, keyed by the concatenation of
          the two catalogue labels, plus the bin edge arrays ``smin`` /
          ``smax`` (and ``mumin`` / ``mumax`` or ``pimin`` / ``pimax``);
          pair counts have shape (ns, nmu) or (ns, np) or (ns,), with the
          convention that auto pairs are counted twice (i.e. ordered
          pairs) and mu (or pi) is the absolute value;
        - ``s``: centers of the separation (or s_perp) bins;
        - ``cf`` (if estimators were requested): array of correlation
          functions with shape (ncf, ns[, nmu or np]);
        - ``multipoles`` and ``poles`` (if multipoles were requested):
          multipole moments of ``cf`` with shape (ncf, nl, ns);
        - ``projected`` (if the projected correlation function was
          requested): w_p with shape (ncf, ns).
    """
    cdef size_t i
    cdef int e
    cdef CF *cf = NULL
    cdef DATA *dat = NULL
    cdef char **argv = NULL
    cdef int argc
    cdef int c_ncat
    cdef int c_nmu
    cdef bint dat_owned_by_cf = False

    # ------------------------------------------------------------------
    # Validate and normalize the inputs.
    # ------------------------------------------------------------------
    _validate_options(kwargs)

    ncat = len(data_cats)
    if ncat != len(data_wts):
        raise ValueError(f"got {ncat} catalogs but {len(data_wts)} weight "
                         "arrays")
    if ncat < 1:
        raise ValueError("at least one catalog is required")

    if 'label' in kwargs:
        kwargs['label'] = _validate_labels(kwargs['label'], ncat)
    elif 'conf' not in kwargs:
        # No configuration file: default to A, B, C, ... labels.  When a
        # configuration file is given, its CATALOG_LABEL entries are used.
        kwargs['label'] = _validate_labels(None, ncat)
    # FCFC is a command-line tool with verbose defaults; as a library,
    # the default is to be quiet.
    kwargs.setdefault('verbose', False)
    if ('pair' not in kwargs) and ('conf' not in kwargs):
        raise ValueError("no pairs to count: please pass the `pair' keyword "
                         "argument (e.g. pair=['DD', 'DR', 'RR']), or a "
                         "configuration file via `conf'")
    if ('multipole' in kwargs or 'wp' in kwargs) and \
            ('cf' not in kwargs and 'conf' not in kwargs):
        raise ValueError("multipoles and projected correlation functions "
                         "are integrated from a CF estimator: please pass "
                         "`cf' (e.g. cf=['DD']) or a configuration file "
                         "together with `multipole'/`wp'")

    all_f32 = True
    cats = []
    for c in data_cats:
        c = np.asarray(c)
        if c.dtype != np.float32:
            all_f32 = False
        cats.append(c)
    wts = []
    for w in data_wts:
        w = np.asarray(w)
        if w.dtype != np.float32:
            all_f32 = False
        wts.append(w)
    cdef int use_float = 1 if all_f32 else 0
    dtype = np.float32 if all_f32 else np.float64

    pos_list = []
    wt_list = []
    for i in range(ncat):
        p = np.ascontiguousarray(cats[i], dtype=dtype)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"catalog {i}: positions must have shape (N, 3),"
                             f" got {cats[i].shape}")
        w = np.ascontiguousarray(wts[i], dtype=dtype).reshape(-1)
        if w.shape[0] != p.shape[0]:
            raise ValueError(f"catalog {i}: got {p.shape[0]} positions but "
                             f"{w.shape[0]} weights")
        if p.shape[0] < 1:
            raise ValueError(f"catalog {i}: empty catalog")
        pos_list.append(p)
        wt_list.append(w)

    # Enable the (exact and faster) integer counting path for catalogues
    # whose weights are all exactly 1, unless the user set `weight'
    # explicitly.  FCFC treats the literal '1' as "unweighted".
    if 'weight' not in kwargs:
        kwargs['weight'] = ['1' if np.all(wt == 1.0) else 'w'
                            for wt in wt_list]

    sedges_arr = np.array(sedges, dtype=np.float64, order='C').reshape(-1)
    if sedges_arr.size < 2:
        raise ValueError("sedges must contain at least 2 edges")
    if not np.all(np.isfinite(sedges_arr)):
        raise ValueError("sedges must be finite")
    if np.any(np.diff(sedges_arr) <= 0):
        raise ValueError("sedges must be strictly increasing")
    if sedges_arr[0] < 0:
        raise ValueError("sedges must be non-negative")
    cdef int ns = sedges_arr.size - 1
    cdef double[::1] sedges_mv = sedges_arr

    cdef double *pedges_ptr = NULL
    cdef int npi = 0
    pedges_arr = None
    cdef double[::1] pedges_mv
    if pedges is not None:
        pedges_arr = np.array(pedges, dtype=np.float64,
                              order='C').reshape(-1)
        if pedges_arr.size < 2:
            raise ValueError("pedges must contain at least 2 edges")
        if not np.all(np.isfinite(pedges_arr)):
            raise ValueError("pedges must be finite")
        if np.any(np.diff(pedges_arr) <= 0):
            raise ValueError("pedges must be strictly increasing")
        if pedges_arr[0] < 0:
            raise ValueError("pedges must be non-negative")
        pedges_mv = pedges_arr
        pedges_ptr = &pedges_mv[0]
        npi = pedges_arr.size - 1

    nmu = int(nmu)
    if nmu < 1:
        nmu = 1

    # ------------------------------------------------------------------
    # Copy the catalogues into the C data structures.
    # ------------------------------------------------------------------
    dat = pyfcfc_data_alloc(ncat)
    if dat is NULL:
        raise MemoryError("failed to allocate memory for the input catalogs")
    cdef const double[:, ::1] pos_d
    cdef const float[:, ::1] pos_f
    cdef const double[::1] wt_d
    cdef const float[::1] wt_f
    try:
        for i in range(ncat):
            if use_float:
                pos_f = pos_list[i]
                wt_f = wt_list[i]
                e = pyfcfc_data_fill_f(dat, i, pos_f.shape[0],
                                       &pos_f[0, 0], &wt_f[0])
            else:
                pos_d = pos_list[i]
                wt_d = wt_list[i]
                e = pyfcfc_data_fill(dat, i, pos_d.shape[0],
                                     &pos_d[0, 0], &wt_d[0])
            if e:
                raise MemoryError(
                    f"failed to copy catalog {i} into C memory")

        # --------------------------------------------------------------
        # Translate the keyword arguments to command-line options.
        # --------------------------------------------------------------
        arg_bytes = []
        for key, val in kwargs.items():
            key = str(key).strip().replace('_', '-')
            arg_bytes.append(
                (f"--{key}={_format_option(val)}").encode('utf-8'))
        argc = <int> len(arg_bytes) + 1
        argv = <char **> malloc(sizeof(char *) * argc)
        if argv is NULL:
            raise MemoryError("failed to allocate the argument list")
        argv[0] = <char *> pyfcfc_arg0()
        for i in range(len(arg_bytes)):
            # `arg_bytes' is kept alive until after the call below, so the
            # buffers remain valid while the C code parses them.
            argv[i + 1] = <char *> arg_bytes[i]

        # --------------------------------------------------------------
        # Run the pair counting (releasing the GIL).  `compute_cf' takes
        # over the ownership of `dat' whether it succeeds or fails.
        # --------------------------------------------------------------
        c_ncat = ncat
        c_nmu = nmu
        with nogil:
            cf = compute_cf(argc, argv, dat, c_ncat,
                            &sedges_mv[0], ns, pedges_ptr, npi, c_nmu)
        dat_owned_by_cf = True
        if cf is NULL:
            raise RuntimeError("FCFC failed to evaluate the correlation "
                               "function; see the error message above")
        return _extract_results(cf)
    finally:
        if argv is not NULL:
            free(argv)
        if not dat_owned_by_cf:
            pyfcfc_data_free(dat, ncat)
