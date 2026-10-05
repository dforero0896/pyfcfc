"""Tests of the Python API: validation, memory-safety regressions, dtypes,
determinism, and result structure."""

import os

import numpy as np
import pytest

from pyfcfc.boxes import py_compute_cf as box_compute
from pyfcfc.sky import py_compute_cf as sky_compute


@pytest.fixture
def small(rng):
    n = 1200
    box = 300.
    pos = rng.uniform(0, box, (n, 3))
    w = np.ones(n)
    return pos, w, box


# ----------------------------------------------------------------------
# Regressions for bugs of the original port
# ----------------------------------------------------------------------

def test_edges_not_mutated(small):
    """The caller's bin-edge arrays must never be modified (the C code
    used to rescale the Python buffers in place)."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)     # integer table
    p_edges = np.arange(0, 61, 30, dtype=np.float64)
    log_edges = 10 ** np.linspace(0, 1.5, 7)            # hybrid table
    s_copy, p_copy, l_copy = s_edges.copy(), p_edges.copy(), log_edges.copy()

    box_compute([pos], [w], s_edges, None, 0, label=['D'], bin=0,
                pair=['DD'], box=box)
    box_compute([pos], [w], s_edges, p_edges, 0, label=['D'], bin=2,
                pair=['DD'], box=box)
    box_compute([pos], [w], log_edges, None, 4, label=['D'], bin=1,
                pair=['DD'], box=box)

    assert np.array_equal(s_edges, s_copy)
    assert np.array_equal(p_edges, p_copy)
    assert np.array_equal(log_edges, l_copy)


def test_repeated_calls_identical(small):
    """Calling twice with the same arrays must give identical results."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    kwargs = dict(label=['D'], bin=1, pair=['DD'], box=box,
                  cf=['DD / @@ - 1'], multipole=[0, 2])
    r1 = box_compute([pos], [w], s_edges, None, 8, **kwargs)
    r2 = box_compute([pos], [w], s_edges, None, 8, **kwargs)
    assert np.array_equal(r1['pairs']['DD'], r2['pairs']['DD'])
    assert np.array_equal(r1['multipoles'], r2['multipoles'])


def test_positions_not_mutated(small):
    """Input catalogues must not be modified either."""
    pos, w, box = small
    pos_copy, w_copy = pos.copy(), w.copy()
    # log-spaced bins force a non-unit coordinate rescaling internally
    box_compute([pos], [w], 10 ** np.linspace(0, 1.5, 7), None, 0,
                label=['D'], bin=0, pair=['DD'], box=box)
    assert np.array_equal(pos, pos_copy)
    assert np.array_equal(w, w_copy)


def test_weighted_normalization_convention(rng):
    """Auto-pair normalization must be (sum w)^2 - sum w^2 (pycorr's)."""
    n = 2000
    box = 300.
    pos = rng.uniform(0, box, (n, 3))
    w = rng.uniform(0.2, 2.0, n)
    res = box_compute([pos], [w], np.arange(0, 61, 20.), None, 0,
                      label=['D'], bin=0, pair=['DD'], box=box)
    expected = w.sum() ** 2 - (w ** 2).sum()
    assert np.isclose(res['normalization']['DD'], expected, rtol=1e-12)
    assert np.isclose(res['weighted_number']['D'], w.sum(), rtol=1e-12)
    assert res['number']['D'] == n


# ----------------------------------------------------------------------
# Input validation & conversions
# ----------------------------------------------------------------------

def test_accepts_lists_and_dtypes(small):
    """Positions/weights/edges may be lists, float32, ints, transposed
    views, etc.; they are converted internally."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    ref = box_compute([pos], [w], s_edges, None, 0, label=['D'], bin=0,
                      pair=['DD'], box=box)

    # python lists
    r_list = box_compute([pos.tolist()], [w.tolist()], s_edges.tolist(),
                         None, 0, label=['D'], bin=0, pair=['DD'], box=box)
    assert np.allclose(r_list['pairs']['DD'], ref['pairs']['DD'], rtol=1e-12)

    # float32 inputs (native float32 path)
    r_f32 = box_compute([pos.astype(np.float32)], [w.astype(np.float32)],
                        s_edges, None, 0, label=['D'], bin=0, pair=['DD'],
                        box=box)
    assert np.allclose(r_f32['pairs']['DD'], ref['pairs']['DD'], rtol=1e-6)

    # non-contiguous view
    padded = np.zeros((len(pos), 5))
    padded[:, 1:4] = pos
    view = padded[:, 1:4]
    assert not view.flags['C_CONTIGUOUS']
    r_view = box_compute([view], [w], s_edges, None, 0, label=['D'], bin=0,
                         pair=['DD'], box=box)
    assert np.array_equal(r_view['pairs']['DD'], ref['pairs']['DD'])

    # integer positions get promoted to float64
    r_int = box_compute([np.floor(pos).astype(np.int32)], [w.astype(int)],
                        s_edges, None, 0, label=['D'], bin=0, pair=['DD'],
                        box=box)
    assert np.all(np.isfinite(r_int['pairs']['DD']))


def test_bool_kwargs(small):
    """Python booleans are translated to FCFC's 'T'/'F'."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    r1 = box_compute([pos], [w], s_edges, None, 0, label=['D'], bin=0,
                     pair=['DD'], box=box, verbose=False)
    r2 = box_compute([pos], [w], s_edges, None, 0, label=['D'], bin=0,
                     pair=['DD'], box=box, verbose=True)
    assert np.array_equal(r1['pairs']['DD'], r2['pairs']['DD'])


def test_configuration_file_labels_and_overrides(small, tmp_path):
    """Labels and pairs may come entirely from a configuration file, and
    keyword arguments override the file (FCFC precedence rules)."""
    pos, w, box = small
    conf = tmp_path / "tour.conf"
    conf.write_text(
        f"CATALOG_LABEL   = [X, Y]\n"
        f"BOX_SIZE        = {box}\n"
        f"BINNING_SCHEME  = 0\n"
        f"PAIR_COUNT      = [XX, XY]\n"
        f"VERBOSE         = F\n")
    s_edges = np.arange(0, 61, 30, dtype=np.float64)
    res = box_compute([pos, pos], [w, w], s_edges, None, 0, conf=str(conf))
    assert res['labels'] == ['X', 'Y']
    assert 'XX' in res['pairs'] and 'XY' in res['pairs']
    # kwargs override the file
    res = box_compute([pos, pos], [w, w], s_edges, None, 0, conf=str(conf),
                      pair=['XY'], label=['X', 'Y'], cf=['XY / @@ - 1'])
    assert sorted(k for k in res['pairs'] if len(k) == 2) == ['XY']
    assert res['cf'].shape == (1, len(s_edges) - 1)

    # an estimator referencing pair counts that are not computed is a
    # configuration error, and must raise
    with pytest.raises(RuntimeError):
        box_compute([pos, pos], [w, w], s_edges, None, 0, conf=str(conf),
                    pair=['XY'], label=['X', 'Y'], cf=['XX / @@ - 1'])


def test_configuration_file_drives_binning_and_multipoles(small, tmp_path):
    """Binning scheme, estimator and multipoles can come from the file
    (as in the legacy test/test.py, which ran conf-driven ell/wp/iso)."""
    pos, w, box = small
    conf = tmp_path / "smu.conf"
    conf.write_text(
        f"CATALOG_LABEL   = [D]\n"
        f"BOX_SIZE        = {box}\n"
        f"BINNING_SCHEME  = 1\n"
        f"PAIR_COUNT      = [DD]\n"
        f"CF_ESTIMATOR    = [DD / @@ - 1]\n"
        f"MULTIPOLE       = [0, 2]\n"
        f"VERBOSE         = F\n")
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    nmu = 6
    res = box_compute([pos], [w], s_edges, None, nmu, conf=str(conf))
    assert res['pairs']['DD'].shape == (len(s_edges) - 1, nmu)
    assert res['multipoles'].shape == (1, 2, len(s_edges) - 1)
    assert res['poles'] == [0, 2]
    # identical to the equivalent keyword-driven call
    ref = box_compute([pos], [w], s_edges, None, nmu, label=['D'], bin=1,
                      pair=['DD'], cf=['DD / @@ - 1'], multipole=[0, 2],
                      box=box)
    assert np.allclose(res['multipoles'], ref['multipoles'], rtol=0,
                       atol=0)


def test_default_labels(small):
    """Labels default to A, B, C, ... when not provided."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 30, dtype=np.float64)
    res = box_compute([pos, pos], [w, w], s_edges, None, 0, bin=0,
                      pair=['AB'], box=box)
    assert res['labels'] == ['A', 'B']
    assert 'AB' in res['pairs']


@pytest.mark.parametrize("s_edges", [
    [0., 10., 5.],            # not increasing
    [10.],                    # single edge
    [0., 10., np.nan],        # non-finite
    [0., 10., np.inf],        # non-finite
    [-5., 10.],               # negative
])
def test_invalid_sedges_raise(small, s_edges):
    pos, w, box = small
    with pytest.raises(ValueError):
        box_compute([pos], [w], s_edges, None, 0,
                    label=['D'], bin=0, pair=['DD'], box=box)


def test_spi_requires_pedges(small):
    """bin=2 without pi edges must fail cleanly with a RuntimeError."""
    pos, w, box = small
    with pytest.raises(RuntimeError):
        box_compute([pos], [w], np.arange(0, 61, 15.), None, 0,
                    label=['D'], bin=2, pair=['DD'], box=box)


def test_shape_mismatch_raises(small):
    pos, w, box = small
    with pytest.raises(ValueError):
        box_compute([pos[:, :2]], [w], np.arange(0, 61, 15.), None, 0,
                    label=['D'], bin=0, pair=['DD'], box=box)
    with pytest.raises(ValueError):
        box_compute([pos], [w[:-1]], np.arange(0, 61, 15.), None, 0,
                    label=['D'], bin=0, pair=['DD'], box=box)
    with pytest.raises(ValueError):
        box_compute([pos], [w], np.arange(0, 61, 15.), None, 0,
                    label=['D', 'R'], bin=0, pair=['DD'], box=box)


def test_multipole_without_estimator_raises(small):
    """Multipoles/wp need a CF estimator: fail loudly, not silently."""
    pos, w, box = small
    with pytest.raises(ValueError, match="CF estimator"):
        box_compute([pos], [w], np.arange(0, 61, 15.), None, 8,
                    label=['D'], bin=1, pair=['DD'], box=box,
                    multipole=[0, 2])
    with pytest.raises(ValueError, match="CF estimator"):
        box_compute([pos], [w], np.arange(0, 61, 15.),
                    np.arange(0, 61, 15.), 0, label=['D'], bin=2,
                    pair=['DD'], box=box, wp=True)


def test_missing_pair_kwarg_raises(small):
    pos, w, box = small
    with pytest.raises(ValueError):
        box_compute([pos], [w], np.arange(0, 61, 15.), None, 0,
                    label=['D'], bin=0, box=box)


def test_bad_label_raises(small):
    pos, w, box = small
    with pytest.raises(ValueError):
        box_compute([pos], [w], np.arange(0, 61, 15.), None, 0,
                    label=['dd'], bin=0, pair=['DD'], box=box)
    with pytest.raises(ValueError):
        box_compute([pos, pos], [w, w], np.arange(0, 61, 15.), None, 0,
                    label=['D', 'D'], bin=0, pair=['DD'], box=box)


def test_invalid_config_raises_runtime(small):
    """Errors inside the C code must surface as Python exceptions
    (and not crash or leak)."""
    pos, w, box = small
    with pytest.raises(RuntimeError):
        # pair label 'X' does not correspond to any catalogue label
        box_compute([pos], [w], np.arange(0, 61, 15.), None, 0,
                    label=['D'], bin=0, pair=['XX'], box=box)


def test_empty_catalog_raises(small):
    pos, w, box = small
    with pytest.raises(ValueError):
        box_compute([pos[:0]], [w[:0]], np.arange(0, 61, 15.), None, 0,
                    label=['D'], bin=0, pair=['DD'], box=box)


# ----------------------------------------------------------------------
# Result structure
# ----------------------------------------------------------------------

def test_result_shapes_and_keys(small):
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    ns = len(s_edges) - 1

    # isotropic, two estimators
    res = box_compute([pos, pos], [w, w * 2], s_edges, None, 0,
                      label=['A', 'B'], bin=0, pair=['AA', 'AB', 'BB'],
                      cf=['AA / @@ - 1', 'AB / @@ - 1'], box=box)
    assert res['pairs']['AA'].shape == (ns,)
    assert res['pairs']['smin'].shape == (ns,)
    assert res['cf'].shape == (2, ns)
    assert res['s'].shape == (ns,)
    assert 'multipoles' not in res

    # smu with multipoles
    nmu = 12
    res = box_compute([pos], [w], s_edges, None, nmu, label=['A'], bin=1,
                      pair=['AA'], cf=['AA / @@ - 1'], multipole=[0, 2],
                      box=box)
    assert res['pairs']['AA'].shape == (ns, nmu)
    assert res['pairs']['mumin'].shape == (ns, nmu)
    assert res['cf'].shape == (1, ns, nmu)
    assert res['multipoles'].shape == (1, 2, ns)
    assert res['poles'] == [0, 2]

    # spi with wp
    p_edges = np.arange(0, 61, 30, dtype=np.float64)
    res = box_compute([pos], [w], s_edges, p_edges, 0, label=['A'], bin=2,
                      pair=['AA'], cf=['AA / @@ - 1'], wp=True, box=box)
    assert res['pairs']['AA'].shape == (ns, len(p_edges) - 1)
    assert res['cf'].shape == (1, ns, len(p_edges) - 1)
    assert res['projected'].shape == (1, ns)


def test_multipoles_of_isotropic_xi(small):
    """For isotropic-ish data xi0 ~ xi(s) and higher poles are small."""
    rng = np.random.default_rng(3)
    n = 4000
    box = 300.
    pos = rng.uniform(0, box, (n, 3))
    w = np.ones(n)
    s_edges = np.arange(0, 81, 10, dtype=np.float64)
    res = box_compute([pos], [w], s_edges, None, 64, label=['D'], bin=1,
                      pair=['DD'], cf=['DD / @@ - 1'], multipole=[0, 2, 4],
                      box=box)
    mp = res['multipoles'][0]
    # random catalogue: all poles consistent with zero within noise
    assert np.max(np.abs(mp[0])) < 0.05
    assert np.max(np.abs(mp[1])) < 0.05
    assert np.max(np.abs(mp[2])) < 0.05


# ----------------------------------------------------------------------
# Sky module
# ----------------------------------------------------------------------

def test_sky_defaults_quiet_and_consistent(sky_catalogs):
    """The sky module works both with and without coordinate conversion."""
    cat = sky_catalogs
    s_edges = np.arange(0, 101, 25, dtype=np.float64)

    res_xyz = sky_compute([cat['xyz']], [cat['w']], s_edges, None, 0,
                          label=['D'], bin=0, pair=['DD'], convert=False)
    res_rdd = sky_compute([cat['rdd']], [cat['w']], s_edges, None, 0,
                          label=['D'], bin=0, pair=['DD'], convert=True,
                          omega_m=cat['omega_m'], omega_l=cat['omega_l'],
                          eos_w=cat['eos_w'])
    c1 = res_xyz['pairs']['DD'] * res_xyz['normalization']['DD']
    c2 = res_rdd['pairs']['DD'] * res_rdd['normalization']['DD']
    # FCFC's spline-based distance integration agrees with the reference
    # quadrature to ~1e-8, so only bin-boundary pairs may move
    assert np.allclose(c1, c2, rtol=1e-6, atol=3)


def test_float32_path(small):
    """The float32 code path gives the same results as float64."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    r64 = box_compute([pos], [w], s_edges, None, 6, label=['D'], bin=1,
                      pair=['DD'], box=box)
    r32 = box_compute([pos.astype(np.float32)], [w.astype(np.float32)],
                      s_edges, None, 6, label=['D'], bin=1, pair=['DD'],
                      box=box)
    assert np.allclose(r32['pairs']['DD'], r64['pairs']['DD'], rtol=1e-6)


# ----------------------------------------------------------------------
# Optional file outputs (the remaining file I/O of the C code)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("fmt", [0, 1], ids=["binary", "ascii"])
def test_pair_output_write_and_readback(small, tmp_path, fmt):
    """PAIR_COUNT_FILE outputs are written, and a second run reads the
    pair counts back from those files instead of recomputing them."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    pout = [str(tmp_path / f"pc_{fmt}_{i}") for i in range(3)]
    r1 = box_compute([pos, pos], [w, w], s_edges, None, 4, label=['D', 'R'],
                     bin=1, pair=['DD', 'DR', 'RR'], box=box,
                     pair_output=pout, out_format=fmt, overwrite=2)
    assert all(os.path.exists(p) for p in pout)

    # overwrite=1 keeps existing pair-count files: FCFC reads them back
    r2 = box_compute([pos, pos], [w, w], s_edges, None, 4, label=['D', 'R'],
                     bin=1, pair=['DD', 'DR', 'RR'], box=box,
                     pair_output=pout, out_format=fmt, overwrite=1)
    rtol = 0 if fmt == 0 else 1e-9   # ascii output is text-precision
    for k in ('DD', 'DR', 'RR'):
        assert np.allclose(r2['pairs'][k], r1['pairs'][k], rtol=rtol,
                           atol=rtol), k


def test_cf_and_multipole_output_files(small, tmp_path):
    """CF_OUTPUT_FILE / MULTIPOLE_FILE are written when requested."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    cfout = str(tmp_path / "cf.out")
    mpout = str(tmp_path / "mp.out")
    box_compute([pos], [w], s_edges, None, 4, label=['D'], bin=1,
                pair=['DD'], cf=['DD / @@ - 1'], multipole=[0, 2], box=box,
                cf_output=[cfout], mp_output=[mpout], overwrite=2)
    for p in (cfout, mpout):
        assert os.path.exists(p) and os.path.getsize(p) > 0


# ----------------------------------------------------------------------
# Runtime behaviour: GIL release and memory hygiene
# ----------------------------------------------------------------------

def test_gil_released_during_counting():
    """The pair counting must run with the GIL released, so other
    Python threads keep running during the computation."""
    import threading

    rng = np.random.default_rng(17)
    n = 60000
    box = 800.
    pos = rng.uniform(0, box, (n, 3))
    w = np.ones(n)
    s_edges = np.arange(0, 151, 5, dtype=np.float64)

    counter = [0]
    stop = [False]

    def spin():
        while not stop[0]:
            counter[0] += 1

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        box_compute([pos], [w], s_edges, None, 0, label=['D'], bin=0,
                    pair=['DD'], box=box)
    finally:
        stop[0] = True
        thread.join(timeout=5)
    # with the GIL released, the spinning thread executes a huge number
    # of iterations; if the GIL were held by the C code it would be ~0
    assert counter[0] > 1000, f"counter thread only ran {counter[0]} times"


def _rss_kb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1])
    except OSError:
        pytest.skip("no /proc on this platform")
    pytest.skip("no VmRSS on this platform")


@pytest.mark.slow
def test_no_gross_memory_leak(small):
    """Repeated successful and failing calls must not leak memory.

    The failing calls exercise the C error paths (configuration errors),
    where the ownership of the input data transfers back and forth.
    """
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)

    for _ in range(5):   # warm up allocator arenas
        box_compute([pos], [w], s_edges, None, 4, label=['D'], bin=1,
                    pair=['DD'], box=box)
    base = _rss_kb()

    for _ in range(200):
        box_compute([pos], [w], s_edges, None, 4, label=['D'], bin=1,
                    pair=['DD'], box=box)
    for _ in range(100):
        with pytest.raises(RuntimeError):
            # unknown pair label: fails inside the C configuration checks
            box_compute([pos], [w], s_edges, None, 4, label=['D'], bin=1,
                        pair=['XX'], box=box)
    # unknown configuration options are rejected up-front in Python
    for _ in range(100):
        with pytest.raises(ValueError):
            box_compute([pos], [w], s_edges, None, 4, label=['D'], bin=1,
                        pair=['DD'], box=box, not_an_option=3)
    after = _rss_kb()
    growth_mb = (after - base) / 1024.
    assert growth_mb < 50, f"RSS grew by {growth_mb:.1f} MB"


def test_unknown_option_raises(small):
    """Typos in configuration keywords must raise, not be ignored."""
    pos, w, box = small
    s_edges = np.arange(0, 61, 15, dtype=np.float64)
    base = dict(label=['D'], bin=0, pair=['DD'], box=box)

    with pytest.raises(ValueError, match="unknown FCFC option"):
        box_compute([pos], [w], s_edges, None, 0, datastruk=1, **base)
    with pytest.raises(ValueError, match="did you mean"):
        box_compute([pos], [w], s_edges, None, 0, multipol=[0, 2], **base)
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        box_compute([pos], [w], s_edges, None, 0, nthreads=4, **base)
    with pytest.raises(ValueError, match="called 'box'"):
        box_compute([pos], [w], s_edges, None, 0, box_size=box,
                    label=['D'], bin=0, pair=['DD'])
    # the survey-like module rejects the periodic box option
    with pytest.raises(ValueError, match="survey-like"):
        sky_compute([pos], [w], s_edges, None, 0, box=box, label=['D'],
                    bin=0, pair=['DD'])
    # a valid call still works
    res = box_compute([pos], [w], s_edges, None, 0, **base)
    assert np.all(np.isfinite(res['pairs']['DD']))
