# pyfcfc examples

Each script is self-contained (it generates its own mock catalogues),
prints a short quantitative summary and — where relevant — saves figures
to `examples/figures/`. They can be run from anywhere:

```bash
python examples/example_features_tour.py        # ~5 s
python examples/example_validation_bruteforce.py  # ~10 s
python examples/example_soft_hist.py          # ~20 s
python examples/example_soft_hist_jax.py      # ~40 s (needs jax)
python examples/example_soft_hist_fit.py      # ~40 s
python examples/example_soft_hist_multipoles.py  # ~60 s (needs jax)
python examples/example_soft_hist_wp.py         # ~50 s (needs jax)
python examples/example_soft_hist_rppi.py       # ~60 s
python examples/example_box_vs_pycorr.py        # ~30 s (needs pycorr)
python examples/example_survey_vs_pycorr.py     # ~15 s (needs pycorr)
python examples/example_pycorr_interop.py       # ~10 s (needs pycorr)
python examples/example_benchmark.py            # ~1-10 min, see below
```

(`make examples` runs the first five in sequence.)

Interactive versions of the examples, plus a benchmarking walkthrough and
an lsstypes-interoperability tour, are provided as executed Jupyter
notebooks in `../notebooks/` (see the table in the main README).

## Guided tour

| # | script | what it teaches |
|---|--------|-----------------|
| 1 | `example_features_tour.py` | **Every feature in one place**, with commented output: result dictionary, the three binning schemes, weights and the integer counting path, configuration files + keyword overrides, k-d vs ball tree, input flexibility (lists/float32/views), cross-correlations, multipole & w_p integration schemes, `ignore_nan`, the survey-like component, error handling, verbose logs. |
| 2 | `example_validation_bruteforce.py` | Why you can trust the numbers: bin-by-bin comparison against an independent O(N²) NumPy pair counter (agreement at machine precision), figure `figures/validation_bruteforce.png`. |
| 3 | `example_box_vs_pycorr.py` | Periodic-box science products vs pycorr on identical data: ξ(s), ξ₀/₂/₄ (Landy–Szalay, weighted), natural estimator with analytic RR (unweighted), w_p; timings and ratio panels; figures `figures/box_vs_pycorr.png`, `figures/box_vs_pycorr_natural.png`. |
| 4 | `example_survey_vs_pycorr.py` | Survey-like data: {RA, Dec, z} in, cosmological conversion inside pyfcfc, midpoint line-of-sight comparison with pycorr; multipoles and w_p; figure `figures/survey_vs_pycorr.png`. |
| 5 | `example_pycorr_interop.py` | The production workflow: split-random accumulation (`add_pair_counts`), conversion to a pycorr state (`pairs_to_pycorr`), save/load with pycorr, rebinning; figure `figures/pycorr_interop.png`. |
| 6 | `example_benchmark.py` | Performance: pyfcfc vs pycorr timings (see below); figure `figures/benchmark.png`, table `figures/benchmark_results.csv`. |
| 7 | `example_soft_hist.py` | The optional soft histogramming method (softmax + Gaussian kernel pair distribution) against the standard hard histogram of the C library: pair counts, soft/hard ratios, xi(s), weight-conservation checks and timings; figure `figures/soft_hist.png`. |
| 8 | `example_soft_hist_jax.py` | The soft histogram as a JAX primitive: `jax.custom_vjp` wrapping the C forward and adjoint, gradient validation against finite differences, and an Alcock-Paczyński-like stretch recovery by Adam on `jax.grad`; figure `figures/soft_hist_jax.png`. |
| 9 | `example_soft_hist_fit.py` | Realistic reverse-mode use case without any autodiff framework: recover a 3-parameter rigid displacement of the galaxy catalogue (one shift per axis) by Adam on the soft cross-histogram matching loss, gradient from `soft_hist_vjp`; gradient check vs finite differences and the rigid-translation null direction; figure `figures/soft_hist_fit.png`. |
| 10 | `example_soft_hist_multipoles.py` | Differentiable multipoles: soft binning in s *and* mu makes xi(s, mu) - and hence xi_0/xi_2 - smooth; `jax.custom_vjp` on the 2D counts, JAX-side Legendre integration, AP stretch recovery from the multipoles; figure `figures/soft_hist_multipoles.png`. |
| 11 | `example_soft_hist_wp.py` | Differentiable projected correlation function: soft s *and* pi bins make w_p smooth; `jax.custom_vjp` on the 2D counts, JAX-side Landy-Szalay + pi integration, transverse stretch recovered from w_p; figure `figures/soft_hist_wp.png`. |
| 12 | `example_soft_hist_rppi.py` | The full 2D xi(r_p, pi) map as a differentiable object: adjoint Jacobian maps d xi/d theta for an anisotropic stretch model (one-hot cotangents through `soft_hist_vjp`), validated bin-by-bin against finite differences; figure `figures/soft_hist_rppi.png`. |

## Feature coverage matrix

| feature | tour | validation | box | survey | interop | benchmark |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| isotropic binning (`bin=0`) | ✔ | ✔ | ✔ | | | ✔ |
| (s, μ) binning (`bin=1`) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| (s, π) binning (`bin=2`) + `wp` | ✔ | ✔ | ✔ | ✔ | | ✔ |
| 2D ξ(s⊥, π) estimator vs pycorr | | | ✔ | ✔ | | |
| multipoles (`multipole=`) | ✔ | | ✔ | ✔ | | ✔ |
| weights / integer path | ✔ | ✔ | ✔ | ✔ | ✔ | optional |
| configuration file + overrides | ✔ | | | | | |
| k-d tree / ball tree (`data_struct`) | ✔ | ✔ | | | | |
| float32 / lists / views | ✔ | | | | | |
| cross-correlation of two tracers | ✔ | | | | | |
| `cf` estimators incl. `@@` | ✔ | | ✔ | ✔ | ✔ | ✔ |
| `ignore_nan` integration | ✔ | | | | | |
| survey conversion (`convert`, cosmology) | ✔ | | | ✔ | | |
| error handling / validation | ✔ | | | | | |
| `verbose` logs | ✔ | | | | | |
| `add_pair_counts` (split randoms) | | | | | ✔ | |
| `pairs_to_pycorr` + save/load/rebin | | | | | ✔ | |
| GIL release / threading | (see `tests/test_api.py`) | | | | | ✔ (thread scan) |

## Benchmarking

```bash
# quick default (2 sizes, current OMP_NUM_THREADS)
python examples/example_benchmark.py

# a serious run on a big machine:
OMP_NUM_THREADS=16 python examples/example_benchmark.py \
    --sizes 1e5,5e5,1e6,5e6 --repeats 3

# thread scaling (each thread count runs in its own process):
python examples/example_benchmark.py --threads 1,2,4,8,16 --sizes 5e5,2e6

# weighted (floating-point counting path):
python examples/example_benchmark.py --weighted
```

What is timed: the **complete measurement** (pair counting + estimator +
multipole/w_p integration) for each binning scheme, best of `--repeats`
runs, on identical uniform catalogues (N_rand = 2 N_data) in a
1000 Mpc/h box. pycorr runs its four Landy–Szalay counters (D1D2, D1R2,
R1D2, R1R2) while pyfcfc evaluates three pair counts (DD, DR, RR) — this
is what each code does in normal use.

**SIMD.** The script prints the SIMD level compiled into your pyfcfc
build (`pyfcfc.boxes.simd_info()`). To benchmark the vectorised kernels
on an AVX-capable CPU:

```bash
PYFCFC_WITH_SIMD=1 pip install -e . --no-build-isolation   # rebuild
python examples/example_benchmark.py ...
```

(The SIMD kernels add AVX2/AVX512 pair-counting loops; expect roughly a
further factor ~2 on top of the non-SIMD speedup, as reported in the
pyfcfc README.)

Outputs: `figures/benchmark_results.csv` (all timings, with SIMD level
and thread count) and `figures/benchmark.png` (timings vs N and speedup,
or thread-scaling curves when `--threads` is given).
