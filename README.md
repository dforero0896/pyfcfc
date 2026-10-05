# pyfcfc — Python bindings for the Fast Correlation Function Calculator

`pyfcfc` wraps the two pair-counting components of the C package
[FCFC](https://github.com/cheng-zhao/FCFC) (by Cheng Zhao) so that they can
be driven entirely from Python with in-memory NumPy catalogues:

| module         | FCFC component | data                                  | line of sight            |
|----------------|----------------|---------------------------------------|--------------------------|
| `pyfcfc.boxes` | `FCFC_2PT_BOX` | periodic-box Cartesian coordinates    | box z axis               |
| `pyfcfc.sky`   | `FCFC_2PT`     | Cartesian comoving coordinates, or {RA, Dec, redshift} with on-the-fly cosmological conversion | pair midpoint            |

Pair counts, correlation functions, multipoles and projected correlation
functions are computed with k-d tree or ball tree dual-tree traversal,
OpenMP parallelisation and (optionally) SIMD vectorisation, and returned
as plain NumPy arrays in a Python dictionary.

The package also ships utilities (`pyfcfc.utils`) to accumulate
split-random measurements, integrate correlation functions, and convert
results into [pycorr](https://github.com/cosmodesi/pycorr) states, so that
pair counting can be done with (fast) FCFC while the analysis uses the
pycorr API (estimators, rebinning, covariance tools, saving...). The
optional `pyfcfc.external` module converts results into
[lsstypes](https://github.com/adematti/lsstypes) containers (`Count2`,
`Count2Correlation`, `Count2CorrelationPoles`, ...), giving access to
replica covariances and serialisation.

Compared to pycorr (Corrfunc engine), `pyfcfc` pair counts and correlation
functions agree **to machine precision** on identical inputs (see
`examples/`); the wall-clock speedup is typically 1.5–2.5x on the counting
step (and larger with SIMD enabled).

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The `py_compute_cf` function](#the-py_compute_cf-function)
  - [Configuration options](#configuration-options)
  - [Result dictionary](#result-dictionary)
- [Conventions](#conventions)
  - [Pair counts and normalizations](#pair-counts-and-normalizations)
  - [Multipole integration: midpoint vs exact](#multipole-integration-midpoint-vs-exact)
- [Working with pycorr](#working-with-pycorr)
- [Accumulating split-random measurements](#accumulating-split-random-measurements)
- [Tests](#tests)
- [Examples](#examples)
- [Gotchas and notes](#gotchas-and-notes)
- [Changes with respect to the original port](#changes-with-respect-to-the-original-port)
- [Acknowledgements](#acknowledgements)

---

## Installation

Requirements: a C compiler with OpenMP support (gcc/clang), Python ≥ 3.9,
NumPy, SciPy; Cython and NumPy at build time (handled automatically by
`pip`).

```bash
pip install .            # from the repository root
# or, for development:
pip install -e .
# or with the Makefile:
make
```

A one-shot script that installs the system build dependencies, creates a
virtual environment with all (optional) extras and installs pyfcfc in
editable mode is provided:

```bash
bash scripts/setup_dev_env.sh [VENV_DIR]
```

**SIMD acceleration.** FCFC's vectorised counting kernels are enabled with

```bash
PYFCFC_WITH_SIMD=1 pip install . --no-build-isolation
```

This implies `-march=native`: build on a machine whose instruction set
(AVX2/AVX512) is available at run time. Compilation of the SIMD kernels is
memory hungry (a couple of GB of RAM); `PYFCFC_NO_LTO=1` and
`PYFCFC_EXTRA_CFLAGS="-O2 -mno-avx512f"` can reduce the footprint (the
latter restricts the build to AVX2).

Other build environment variables: `PYFCFC_MARCH_NATIVE=1` (native code
generation without SIMD), `PYFCFC_NO_LTO=1` (disable link-time
optimisation), `PYFCFC_EXTRA_CFLAGS="..."` (extra compiler flags).

Optional dependencies for the extras:

- `pycorr` (+ `Corrfunc`, its default engine) for the interoperability
  utilities: `pip install git+https://github.com/cosmodesi/pycorr.git`
  and `pip install git+https://github.com/cosmodesi/Corrfunc@desi`;
- `matplotlib` to run the plotting examples.

The number of OpenMP threads is read from `OMP_NUM_THREADS` at run time.

## Quick start

```python
import numpy as np
from pyfcfc.boxes import py_compute_cf

data = ...   # (N_d, 3) float array, periodic-box coordinates
rand = ...   # (N_r, 3) float array
w_d, w_rand = np.ones(len(data)), np.ones(len(rand))

s_edges = np.arange(0, 151, 5.)
results = py_compute_cf(
    [data, rand], [w_d, w_rand],
    s_edges,          # separation bin edges
    None,             # pi bin edges (only for bin=2)
    48,               # number of mu bins in [0, 1) (bin=1)
    label=['D', 'R'],                 # catalogue labels
    bin=1,                            # binning scheme: 0 iso, 1 (s,mu), 2 (s_perp,pi)
    pair=['DD', 'DR', 'RR'],          # pair counts to evaluate
    cf=['(DD - 2 * DR + RR) / RR'],   # correlation function estimator(s)
    multipole=[0, 2, 4],              # Legendre multipoles to integrate
    box=1000.,                        # periodic box side(s)
)
s, xi0, xi2, xi4 = results['s'], *results['multipoles'][0]
```

Survey-like data, converting {RA, Dec, z} internally:

```python
from pyfcfc.sky import py_compute_cf

results = py_compute_cf(
    [rdd_data, rdd_rand], [w_data, w_rand],
    s_edges, None, 48,
    label=['D', 'R'], bin=1, pair=['DD', 'DR', 'RR'],
    cf=['(DD - 2 * DR + RR) / RR'], multipole=[0, 2, 4],
    convert=True, omega_m=0.31, omega_l=0.69, eos_w=-1.0,
)
```

Configuration files (`conf=...`) are still supported (catalogue/position
entries are ignored, since the data come from Python); keyword arguments
override the file, exactly as command-line options do in FCFC.

## The `py_compute_cf` function

```python
py_compute_cf(data_cats, data_wts, sedges, pedges=None, nmu=1, **kwargs)
```

- `data_cats`: sequence of catalogues, each convertible to an (N, 3)
  array. Non-contiguous arrays, lists and integer/float32 inputs are
  converted internally (a copy is made; **your arrays are never
  modified**). float32 is used natively only when *all* positions and
  weights are float32; otherwise everything is promoted to float64.
- `data_wts`: sequence of per-object weights, each of length N. Pass
  arrays of ones for unweighted catalogues (this automatically enables
  FCFC's faster, exact integer counting path).
- `sedges`: separation (or s_perp) bin edges; strictly increasing,
  non-negative, finite. For the box component, `smax < box/2`.
- `pedges`: pi bin edges, required for `bin=2`.
- `nmu`: number of mu bins in [0, 1), used for `bin=1`.
- `**kwargs`: FCFC configuration options (below). Python booleans are
  translated to `'T'`/`'F'`, sequences to FCFC array literals
  (`[0, 2, 4]`), and unknown options raise `ValueError` with suggestions
  instead of being silently ignored.

### How configuration parsing works (and what is left of the parameter files)

All options reach FCFC through a single libcfg parameter table, whether
they come from keyword arguments or from a configuration file: the
keywords are translated to command-line-style ``--key=value`` strings
(which take priority), and an optional ``conf=`` file is read afterwards
with lower priority, exactly as in the FCFC CLI.  Since the catalogues
and the bin edges always come from the function arguments, the upstream
file-based inputs were removed from the table (`CATALOG`, `POSITION`,
`ASCII_*`, `SELECTION`, `SEP_BIN_*`, `MU_BIN_NUM`, `PI_BIN_*`), as were
the CLI-only ``--help``/``--version``/``--template`` callbacks and the
FITS/HDF5 catalogue readers (excluded from the build).

What remains of the file machinery, and stays supported:

- the ``conf=`` overlay itself (any option in the table below);
- the optional outputs ``pair_output``, ``cf_output``, ``mp_output``,
  ``wp_output`` (with ``out_format`` and ``overwrite``); when a
  ``PAIR_COUNT_FILE`` already exists, FCFC reads the pair counts from it
  instead of recomputing them (tested: write/read round-trip);
- ``cmvdst_file`` (sky component): a {redshift, comoving distance} table
  replacing the internal distance integration;
- the ``verbose=True`` configuration summary.

### Configuration options

| option (keyword)        | `boxes` | `sky` | meaning |
|-------------------------|:-------:|:-----:|---------|
| `label`                 | ✔ | ✔ | single-uppercase-letter catalogue labels (defaults to A, B, C, ...) |
| `pair`                  | ✔ | ✔ | pair counts to evaluate, e.g. `['DD','DR','RR']` |
| `bin`                   | ✔ | ✔ | binning scheme: 0 = isotropic, 1 = (s, mu), 2 = (s_perp, pi) |
| `cf`                    | ✔ | ✔ | correlation function estimator expression(s), e.g. `['(DD - 2*DR + RR)/RR']`; `'@@'` denotes the analytic random-random counts (box) |
| `multipole`             | ✔ | ✔ | multipole orders to integrate (requires `bin=1` and `cf`) |
| `wp`                    | ✔ | ✔ | compute the projected correlation function (requires `bin=2` and `cf`) |
| `box`                   | ✔ |   | periodic box side length(s) |
| `convert`               |   | ✔ | convert {RA, Dec, z} to comoving Cartesian coordinates |
| `omega_m`, `omega_l`, `eos_w` |  | ✔ | fiducial cosmology for the conversion |
| `cmvdst_err`, `cmvdst_file` | | ✔ | accuracy / lookup table for the distance integration |
| `data_struct`           | ✔ | ✔ | 0 = k-d tree (default), 1 = ball tree |
| `weight`                | ✔ | ✔ | per-catalogue weight flags; set automatically (`'1'` when all weights are exactly 1, enabling the integer path) |
| `pair_output`, `cf_output`, `mp_output`, `wp_output` | ✔ | ✔ | optional files where FCFC writes its results |
| `out_format`, `overwrite` | ✔ | ✔ | output file options |
| `verbose`               | ✔ | ✔ | print FCFC's configuration summary and progress (default: False) |

Errors inside the C code (bad configuration, missing pairs, ...) raise
Python exceptions (`RuntimeError`) and never crash the interpreter; all
allocated memory is released on both success and failure paths.

### Result dictionary

| key                | shape / content |
|--------------------|-----------------|
| `labels`           | list of catalogue labels |
| `number`           | dict label → number of objects |
| `weighted_number`  | dict label → sum of weights (equals `number` for unit weights) |
| `normalization`    | dict pair → pair-count normalization (see conventions) |
| `pairs`            | dict pair → normalized pair counts, shape (ns,), (ns, nmu) or (ns, np); plus bin edge arrays `smin`, `smax` and (`mumin`, `mumax`) or (`pimin`, `pimax`) |
| `s`                | (ns,) centers of the separation (s_perp) bins |
| `cf`               | (ncf, ns[, nmu or np]) estimator values, if `cf` was given |
| `multipoles`       | (ncf, nl, ns) multipole moments, if `multipole` was given |
| `poles`            | list of the multipole orders computed |
| `projected`        | (ncf, ns) projected correlation function w_p, if `wp=True` |

The pair-count arrays are normalized counts; multiply by
`normalization[pair]` to recover the raw (weighted) pair counts.

## Conventions

### Pair counts are always computed

Every pair listed in ``pair`` (or ``PAIR_COUNT`` in a configuration file)
is evaluated and returned in ``results['pairs']``, **whether or not a CF
estimator references it**: the pair counts are the main product of the
code, and estimators/multipoles/projections are optional extras computed
from them (here, or externally with `pyfcfc.utils`, pycorr or lsstypes).
Passing a dummy estimator such as ``cf=['DD']`` is therefore *not* needed
(for historical reasons the original port required an estimator to keep
the requested binning scheme; this was fixed in 0.2.0 and is covered by
regression tests). The only pairs not counted from scratch are those with
an existing ``PAIR_COUNT_FILE`` output, which FCFC reads instead.

### Pair counts and normalizations

- Auto pairs (e.g. `DD`) are counted **twice** per unordered pair
  (ordered-pair convention), and the normalization is
  `(Σw)² − Σw²` — the same convention as pycorr's `wnorm`.
- Cross pairs (e.g. `DR`) are counted once per (i, j) combination, with
  normalization `Σw₁ × Σw₂`.
- For `bin=1` and `bin=2`, only `mu ≥ 0` (`pi ≥ 0`) is binned, with mu
  (pi) the **absolute value** of the cosine (parallel separation).
- Bins are left-closed, right-open; pairs with exactly `mu == 1` are
  dropped (as in FCFC compiled without `WITH_MU_ONE`).
- The box component uses the periodic minimum-image metric; the sky
  component defines the line of sight as the pair **midpoint** direction
  (equivalent to pycorr's `los='midpoint'`).

### Multipole integration: midpoint vs exact

FCFC integrates multipoles with the midpoint rule,
`ξ_ℓ(s) = (2ℓ+1) Σ_j ξ(s, μ_j) P_ℓ(μ_j) Δμ`. pycorr instead uses the
exact integral of the Legendre polynomial over each bin. The two agree up
to O(Δμ²): for ℓ = 0 they coincide (the difference cancels), while for
ℓ = 2, 4 the difference can reach ~10⁻² at small scales when ξ is large —
this is a *scheme* difference, not a counting error (the raw pair counts
of both codes agree to machine precision).

`pyfcfc.utils.compute_multipoles` supports both schemes:

```python
from pyfcfc.utils import compute_multipoles
mp_mid   = compute_multipoles(xi_smu, [0, 2, 4])                    # FCFC's scheme
mp_exact = compute_multipoles(xi_smu, [0, 2, 4], method='exact')    # pycorr's scheme
```

so you can reproduce either code's numbers exactly from pyfcfc pair
counts (both also accept `ignore_nan=True` to skip empty bins, as pycorr
does).

## Working with pycorr

```python
from pyfcfc.utils import pairs_to_pycorr
from pycorr import TwoPointCorrelationFunction

state = pairs_to_pycorr(
    results,                                  # pyfcfc result dict
    'landyszalay',                            # pycorr estimator name
    dict(DD='D1D2', DR=('D1R2', 'R1D2'), RR='R1R2'),   # label mapping
    box_size=1000.,                           # periodic box (optional)
)
np.save('counts.pkl.npy', state)
tpcf = TwoPointCorrelationFunction.load('counts.pkl.npy')
s, xi0 = tpcf(ell=0, return_sep=True)
rebinned = tpcf[::2, ::4]                     # pycorr rebinning works
```

Notes:

- Reversible pair counts must be mapped to **both** pycorr names (as for
  `DR` above); pyfcfc's |mu| counts are mirrored to pycorr's full
  [-1, 1] range (exactly for auto pairs, statistically for cross pairs).
- For the `natural` estimator in a periodic box, the analytic
  random-random counts are generated automatically when `box_size` is
  given.
- The pycorr state records `los_type='z'` for box measurements and
  `'midpoint'` for sky measurements, matching FCFC's conventions.
- In `rppi` mode, pycorr stores symmetric pi edges [-pimax, pimax]; the
  conversion mirrors pyfcfc's pi ≥ 0 counts accordingly.

## Working with lsstypes (optional)

[lsstypes](https://github.com/adematti/lsstypes) provides container types
for LSS measurements (replica algebra, covariances, serialisation). The
optional module `pyfcfc.external` converts pyfcfc results into them:

```bash
pip install git+https://github.com/adematti/lsstypes   # optional extra
```

```python
from pyfcfc.external import to_lsstypes, to_lsstypes_correlation, to_lsstypes_counts

counts = to_lsstypes_counts(results)              # {'DD': Count2, 'DR': ..., 'RD': ..., 'RR': ...}
corr = to_lsstypes_correlation(results, 'landyszalay')   # Count2Correlation
poles = to_lsstypes(results, ells=[0, 2, 4])      # Count2CorrelationPoles
wp = to_lsstypes(results_spi, project='wp')       # Count2CorrelationWp
```

- `project` accepts 'poles' (default), 'wedges', 'wp', 'binned', or None
  (returns the un-projected `Count2Correlation`);
- multipoles use lsstypes' exact per-bin Legendre integration, i.e.
  `compute_multipoles(..., method='exact')` on the pyfcfc side;
- for the 'natural' estimator in a periodic box without RR counts, the
  analytic random-random counts are synthesized (same convention as
  FCFC's `@@`);
- catalogue labels other than D/R are handled with `pair_mapping`, e.g.
  `{'AA': 'DD', 'AB': ('DR', 'RD'), 'BB': 'RR'}`;
- replica tools work out of the box: `lsstypes.mean/cov/sum` over
  converted objects, and `lsstypes.write/read` (`.pkl`, `.h5`, `.txt`).
  Note that `lsstypes.sum` weights replicas by the DD normalization,
  while `pyfcfc.utils.add_pair_counts` accumulates each pair type with
  its own normalization (exact raw-count addition); the two coincide for
  unweighted, equal-size replicas.

See `notebooks/04_lsstypes.ipynb` for a complete walkthrough and
`tests/test_lsstypes.py` (11 tests, skipped when lsstypes is absent).

## Accumulating split-random measurements

```python
from pyfcfc.utils import add_pair_counts

total = None
for chunk in random_chunks:
    res = py_compute_cf([data, chunk.pos], [w_data, chunk.w], ...)
    total = res if total is None else add_pair_counts(total, res)
```

`add_pair_counts` combines the raw pair counts exactly (it is a
normalization-weighted average of the normalized counts, with summed
normalizations), returns a **new** dictionary, and drops stale
estimator-dependent keys (`cf`, `multipoles`, ...), which must be
recomputed from the combined counts.

## Tests

```bash
pip install pytest
python -m pytest tests -v          # 61 tests
```

The suite covers, among other things:

- **exact agreement with brute-force O(N²) counting** (periodic box and
  survey-like data; iso/(s, mu)/(s_perp, pi); linear, logarithmic and
  irregular bins; unit and random weights; k-d and ball trees);
- **cross-validation against pycorr/Corrfunc** on identical catalogues
  (pair counts, natural and Landy-Szalay estimators, multipoles, w_p,
  isotropic mode, state conversion + save/load + rebinning);
- API robustness: input validation and conversion, error propagation,
  immutability of caller arrays, determinism, dtype handling;
- runtime hygiene: GIL release during counting and absence of gross
  memory leaks on success and failure paths.

Tests that need pycorr/Corrfunc are skipped automatically when those are
not installed.

## Examples and notebooks

`examples/README.md` contains the guided tour of the examples, a
feature-coverage matrix and the benchmarking guide. In short:

| script | content | figure |
|--------|---------|--------|
| `examples/example_features_tour.py` | **start here**: every feature in one commented script (result dict, binning schemes, weights, conf files, trees, dtypes, cross-pairs, integrations, sky, errors, verbose) | — |
| `examples/example_validation_bruteforce.py` | pyfcfc vs brute-force counting, all binning schemes | `figures/validation_bruteforce.png` |
| `examples/example_box_vs_pycorr.py` | periodic box: xi(s), xi_0/2/4, the 2D xi(s_perp, pi), w_p, natural & LS estimators, pyfcfc vs pycorr + timings | `figures/box_vs_pycorr.png`, `figures/box_vs_pycorr_natural.png`, `figures/box_vs_pycorr_2d.png` |
| `examples/example_survey_vs_pycorr.py` | survey-like data from {RA, Dec, z}: multipoles and w_p vs pycorr (midpoint LOS) | `figures/survey_vs_pycorr.png` |
| `examples/example_pycorr_interop.py` | split-random accumulation → pycorr state → save/load/rebin | `figures/pycorr_interop.png` |
| `examples/example_benchmark.py` | performance study vs pycorr: sizes, binning schemes, thread scaling, SIMD reporting | `figures/benchmark.png`, `figures/benchmark_results.csv` |

Run them all with `make examples` (or individually with
`python examples/example_*.py`; matplotlib and, for the pycorr ones,
pycorr + Corrfunc are required).

Interactive versions are provided as executed Jupyter notebooks in
`notebooks/`:

| notebook | content |
|----------|---------|
| `01_quickstart_box.ipynb` | result dictionary, xi(s), multipoles, w_p, conventions |
| `02_survey_and_pycorr.ipynb` | survey-like data, midpoint LOS, pycorr cross-check, state save/load/rebin |
| `03_benchmarking.ipynb` | how to benchmark properly (sizes, threads, SIMD) with a small in-notebook run |
| `04_lsstypes.ipynb` | conversion to lsstypes containers: counts, estimators, poles, replica covariance, serialisation |

Typical output of the box example (40k data / 80k randoms, 2 threads):

```
max |pyfcfc - pycorr|:
  xi(s)  : 6.5e-13
  multipoles, exact per-bin Legendre integration: xi_0 1.8e-13, xi_2 9.1e-13, xi_4 1.2e-12
  wp     : 1.6e-11
```

## Gotchas and notes

- `multipole` (and `wp`) require a `cf` estimator: multipoles are
  integrated from the estimator's xi(s, mu), as in FCFC. If you only need
  pair counts but still want the correct binning, pass a trivial
  estimator such as `cf=['DD']`.
- For `bin=1`, pass `nmu ≥ 2`; pycorr comparisons need linear mu bins.
  `nmu` is ignored (clamped to 1) for `bin=0` and `bin=2`, and `pedges`
  is ignored unless `bin=2`.
- The `s`-mode and multipole results are deterministic for unit weights
  (integer counting); with non-trivial weights, repeated runs can differ
  at the ~1e-15 relative level due to OpenMP reduction ordering.
- Empty (s, mu) cells produce NaN in `cf` when the random-random counts
  vanish there; use `ignore_nan=True` in the integration helpers or mask
  them.
- Catalogue labels must be unique uppercase letters (at most 26
  catalogues).
- The vendored `FCFC-main/etc/*.conf` files are the *upstream CLI*
  templates: several of their entries (`CATALOG`, `POSITION`, ...) do not
  apply to library mode (the data come from Python), and their
  `CATALOG_LABEL` is empty, so pass `label=` (and `pair=`, `bin=`, ...)
  explicitly or copy the template and fill them in. Unknown entries in a
  configuration file are reported by libcfg and otherwise ignored.

## Repository layout

| path | content |
|------|---------|
| `pyfcfc/` | the Python package: `boxes` and `sky` Cython extensions, `utils`, optional `external` |
| `FCFC-main/` | vendored FCFC C sources (upstream **v1.0.1** with selected **v1.1.0** fixes backported and the Python-binding modifications described below), plus the new `pyfcfc_helpers.{c,h}` accessor layer |
| `tests/` | pytest suite (brute-force and pycorr cross-validation, API robustness, lsstypes) |
| `examples/` | runnable scripts + `figures/` outputs (see `examples/README.md`) |
| `notebooks/` | executed Jupyter notebooks |
| `scripts/` | `setup_dev_env.sh` environment bootstrap |
| `setup.py`, `pyproject.toml`, `Makefile` | build & packaging |

## Changes with respect to the original port

Version 0.2.0 is a substantial hardening of the original wrapper. The
most important fixes:

**Correctness bugs**

1. *Input arrays were mutated in place.* `cf_setup` rescaled the
   separation/pi bin edges **inside the caller's NumPy buffers** (the C
   code aliased them), silently corrupting any array reused across
   calls. Bins are now deep-copied on entry.
2. *Uninitialized variable in the auto-pair normalization.* The hack that
   switched to pycorr's `(Σw)² − Σw²` normalization read an uninitialized
   accumulator (`wt2sum`), i.e. undefined behaviour. Replaced by the
   clean upstream FCFC v1.1.0 implementation (a `w2` field computed when
   the trees are built).
3. *`weighted_number` reported the number of objects, not the sum of
   weights*, breaking normalizations and the pycorr state conversion for
   weighted catalogues.
4. *`w_p` used rescaled pi bins* (upstream FCFC bug fixed in v1.1.0,
   backported): `dpi` is now taken from the unrescaled bins.
5. *The multipole orders (`poles`) and catalogue labels were read from
   freed memory* (the configuration structure is destroyed before the
   results are extracted on the Python side). Both are deep-copied now.
6. *Dangling pointers in the command-line argument array*: the bytes
   objects backing `argv` were freed when the helper function returned,
   before the C parser ran. They are now kept alive for the duration of
   the call (and `argv` is freed afterwards).
7. *Memory leaks / ownership hazards*: the initial `cf_init()` result was
   leaked on every call; input catalogues leaked on configuration
   errors; the survey-like weight-buffer branch could leak/overwrite
   buffers. Ownership of the input data is now explicit and covered by
   tests.
8. *Linear-bin detection* compared only the first and last bin widths
   with `==` (and mutated the configuration structure); it is now a
   tolerant check over all widths, and nonlinear bins with coincidental
   first/last widths can no longer be misclassified.
9. *Backported upstream v1.1.0 tree fix* (`gather_nodes` dropping shallow
   leaves), which can lose pairs in some tree shapes.
10. *SIMD padding*: with SIMD enabled, FCFC's kernels read a few elements
    past the end of the coordinate/weight arrays; the Python-provided
    buffers were unpadded (potential out-of-bounds reads). They are now
    padded exactly like FCFC's own input routines.
11. *`pairs_to_pycorr`*: the `rppi` conversion used one-sided pi edges
    without the mirror+halve transformation (wrong normalization in
    pycorr), the isotropic mode halved counts that should not be halved,
    `iso` states crashed on 1-D bin arrays, and `los_type` was recorded
    as `firstpoint` although FCFC uses the midpoint convention. All
    fixed, and validated against native pycorr counts.

**Robustness and usability**

- The Cython layer no longer duplicates the C structure layouts (a
  silent-corruption hazard): all access goes through an accessor layer
  (`pyfcfc_helpers.c`) that treats `CF`/`DATA` as opaque.
- Full input validation with informative Python exceptions: shapes,
  dtypes, monotonic/finite/non-negative bin edges, label consistency,
  unknown configuration keywords (with typo suggestions).
- Python booleans, lists, tuples and NumPy arrays are accepted as
  keyword values; `weight` is set automatically so that unweighted
  catalogues use the exact integer counting path.
- The GIL is released during pair counting; the library is quiet by
  default (`verbose=True` restores FCFC's progress output).
- Result arrays are plain contiguous NumPy arrays with deterministic
  shapes (`cf` is no longer squeezed), and `poles` is returned.
- Modern packaging: `pyproject.toml`, no `numpy.compat`/`distutils`
  imports (works with NumPy 2 and Cython 3), no hardcoded system paths,
  optional SIMD/march/LTO flags, stale generated C sources removed.
- Dead code from the CLI era was pruned: `read_bins` (file-based bin
  edges), the `usage`/`version`/`conf_template` callbacks with their
  stale texts, the commented-out `conf_print` blocks, and the FITS/HDF5
  catalogue readers (excluded from the build); `read_ascii.c` stays
  because it backs `cmvdst_file`.
- Pair counts read back from an existing `PAIR_COUNT_FILE` now get their
  catalogue indices resolved like computed ones; previously their labels
  were left unset (out-of-bounds label reads in the original port), so
  the results dictionary could not name them correctly.
- The number of mu bins is now forced to 1 outside the (s, mu) scheme
  (upstream enforced this in the configuration parser, which the Python
  binding bypasses): passing `nmu > 1` with `bin=0` used to overflow the
  analytic-RR buffer whenever an estimator with `@@` was requested,
  corrupting memory; `pi` edges are likewise ignored unless `bin=2`.
- Pair counts requested via `pair` are always evaluated and returned,
  even when no CF estimator references them (the estimator only controls
  the optional `cf`/`multipoles`/`projected` products); a defensive check
  now raises if FCFC ever returns an unevaluated pair.
- Repository hygiene: the legacy `test/` scripts and artefacts of the
  original port (NERSC-dependent scripts, stale pair-count/`.pkl.npy`/
  `.png` outputs, `command_line_help.txt`, duplicated `*_original.conf`
  files) and the `cython_debug/` and `main.zip` leftovers were removed;
  their functionality is superseded by `tests/`, `examples/` and
  `notebooks/`.

## Acknowledgements

FCFC is written by Cheng Zhao and distributed under the MIT licence
(see `LICENSE.txt`); this wrapper inherits it. If you use FCFC in
research, please cite the FCFC paper (Zhao et al.). pycorr is developed
by the DESI collaboration (cosmodesi/pycorr).
