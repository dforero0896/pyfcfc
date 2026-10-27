/*******************************************************************************
* 2pt_box/soft_hist.c: this file is part of the FCFC program.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

*******************************************************************************/

/*============================================================================*\
  Soft histogramming pair counts (optional pair counting method), and its
  adjoint (reverse-mode derivative) for automatic differentiation.

  Forward: instead of assigning each pair to exactly one separation bin
  (the standard "hard" histogram of the counting kernels in
  `metric_common.c`), the weight of the pair (1 or w1 * w2) is distributed
  over *all* separation bins with normalised Gaussian-kernel weights
  computed with a numerically stable softmax:

    x_i = (s - c_i)^2 / (2 * sigma^2)                                (1)
    W_i = exp(-(x_i - min_j x_j)) / sum_k exp(-(x_k - min_j x_j))    (2)
    cnt_i += w_pair * W_i                                            (3)

  where s is the pair separation, c_i are the separation bin centres, and
  sigma is the kernel width (SOFT_SIGMA, in input units; the default 0 means
  "half the mean bin width").  Because the softmax weights sum to one, the
  total weight of all pairs with smin <= s < smax is conserved exactly;
  only their distribution among the bins changes.  The second dimensions
  of the (s, mu) and (s_perp, pi) schemes remain hard-binned, exactly as in
  the standard kernels.

  The dual-tree traversal, the node pruning criteria, the periodic wrap
  treatment and the per-pair range checks below replicate the scalar hard
  histogram kernels (`dual_tree.c` + `metric_kdtree.c` / `metric_balltree.c`
  / `metric_common.c`) with runtime instead of compile-time branches on the
  binning scheme, so the two methods operate on exactly the same set of
  pairs and their histograms are directly comparable.

  Adjoint (VJP): the map from the pair separations to the histogram is
  smooth, which is the point of the method.  With W the (window-
  renormalised) softmax weights of a pair of weight w and cotangents g_i
  on the bins, the chain rule through (2)-(3) gives, with cbar = sum_k W_k
  c_k and xbar = sum_k W_k x_k,

    dW_i/ds   = W_i * (c_i - cbar) / sigma^2                         (4)
    s_bar     = w * sum_i g_i dW_i/ds
              = (w / sigma^2) * sum_i g_i W_i (c_i - cbar)           (5)
    dsigma    = (2 w / sigma) * sum_i g_i W_i (x_i - xbar)           (6)
    dw_pair   = sum_i g_i W_i                                        (7)

  and s_bar is chained to the catalogue coordinates through the scheme-
  specific separation (s, or s_perp for the (s_perp, pi) scheme, with the
  periodic minimum image where applicable; the discrete choices - peak bin,
  kernel window, mu / pi bin - are held constant, as any autodiff engine
  does for non-differentiable index selections).  `soft_hist_vjp` replays
  the forward traversal (same pairs, same windows, same weights) and
  accumulates the coordinate, weight and kernel-width cotangents, so it can
  be hooked directly into a reverse-mode autodiff engine.

  Parallelisation and performance: the counting is parallelised with
  OpenMP exactly like the hard kernels (seed node pairs distributed over
  the threads with a dynamic schedule, thread-private count buffers with
  cache-line aligned bases and padded strides, reduction at the end; the
  adjoint accumulates into shared buffers with OpenMP atomics).  For
  linear separation bins the Gaussian kernel vector is generated from the
  peak bin outwards with an exact multiplicative recurrence (the log-
  ratios of a Gaussian over equispaced centres form an arithmetic
  progression, hence two exp() calls per pair and one multiplication per
  visited bin), and each walk stops once the weights fall below
  SOFT_WEIGHT_CUTOFF, so the per-pair cost is O(min(ns, sigma / delta))
  instead of O(ns); non-linear bins fall back to the direct stable softmax
  over an adaptive window.  The kernels remain scalar (no SIMD); soft
  counts are fractional, hence always accumulated as doubles; and MPI-
  distributed counting is not supported (the pyfcfc Python bindings do not
  use MPI).
\*============================================================================*/

#include "define.h"
#include "eval_cf.h"
#include "soft_hist.h"
#include "kdtree.h"
#include "balltree.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef OMP
#include <omp.h>
#endif

/*============================================================================*@\
                  Definitions for the soft histogram kernels
\*============================================================================*/

/* Spatial relationship between two nodes (as `fcfc_node_dist_e`, which is
 * private to `count_func.c`). */
typedef enum {
  SOFT_NODE_DIST_IN,            /* node separation inside distance range */
  SOFT_NODE_DIST_OUT,           /* node separation outside distance range */
  SOFT_NODE_DIST_X,             /* node separation crosses distance range */
  SOFT_NODE_DIST_X_WRAP         /* need periodic wrap check for each point */
} soft_node_dist_e;

/* Configuration for the soft histogram kernels.  All lengths are in the
 * internal (rescaled) units of the CF structure. */
typedef struct {
  int bintype;                  /* binning scheme: iso, smu, or spi */
  int ns;                       /* number of separation (s_perp) bins */
  int np;                       /* number of pi bins (spi) */
  int nmu2;                     /* squared number of mu bins (smu) */
  bool smin0;                   /* indicate s2min == 0 */
  bool pmin0;                   /* indicate pmin == 0 (spi) */
  bool strict_single;           /* no IN shortcut for single nodes */
  real s2min;                   /* minimum squared s (s_perp) of interest */
  real s2max;                   /* maximum squared s (s_perp) of interest */
  real smin;                    /* minimum s (s_perp) of interest */
  real smax;                    /* maximum s (s_perp) of interest */
  real pmin;                    /* minimum pi of interest (spi) */
  real pmax;                    /* maximum pi of interest (spi) */
  real bsize[3];                /* side lengths of the periodic box */
  real hsize[3];                /* half side lengths of the periodic box */
  const real *pbin;             /* edges of pi bins (spi) */
  const uint8_t *mutab;         /* lookup table for mu bins (smu) */
  real *scen;                   /* centres of separation (s_perp) bins */
  double inv2sig2;              /* 1 / (2 * sigma^2) of the kernel */
  double sigma;                 /* kernel width (internal units) */
  bool lin;                     /* separation bins are linear */
  real delta;                   /* linear bin width */
  real scen0;                   /* first bin centre */
  double gg;                    /* delta^2 / sigma^2 (log-ratio step) */
  double iq;                    /* exp(-gg): geometric step of ratios */
  bool soft_s;                  /* soft (softmax) binning of separations */
  bool sigma_log;               /* kernel width proportional to s: f * s */
  double f;                     /* the fraction f of sigma(s) = f * s */
  bool soft_mu;                 /* soft (softmax) binning of mu (smu) */
  int nmu;                      /* number of mu bins (smu) */
  double sigma_mu;              /* mu kernel width (dimensionless) */
  double inv2sig2_mu;           /* 1 / (2 * sigma_mu^2) */
  double gg_mu;                 /* mu log-ratio step */
  double iq_mu;                 /* exp(-gg_mu) */
  bool soft_pi;                 /* soft (softmax) binning of pi (spi) */
  double sigma_pi;              /* pi kernel width (internal units) */
  double inv2sig2_pi;           /* 1 / (2 * sigma_pi^2) */
  double gg_pi;                 /* pi log-ratio step (linear pi bins) */
  double iq_pi;                 /* exp(-gg_pi) */
  bool lin_pi;                  /* pi bins are linear */
  real pi_c0;                   /* first pi centre */
  real pi_delta;                /* pi bin width */
  real *pcen;                   /* centres of pi bins (spi) */
  const real *s2bin;            /* edges of squared separation bins */
  int nsec;                     /* max(nmu, np): 2nd-dim scratch width */
} SOFT_CONF;

/* Work descriptor shared by the forward and adjoint traversals. */
typedef struct {
  const SOFT_CONF *conf;
  int adjoint;                  /* 0: forward counts, 1: VJP */
  double *cnt;                  /* forward: pair count accumulation buffer */
  const double *g;              /* adjoint: cotangents on the bins (ntot) */
  double *gx1, *gx2;            /* adjoint: coordinate cotangents (internal
                                 * units, 3 components per object, shared
                                 * between threads, accumulated atomically) */
  double *gw1, *gw2;            /* adjoint: weight cotangents (or NULL) */
  double *gsig;                 /* adjoint: kernel width cotangent (or NULL) */
  double *gmus;                 /* adjoint: mu kernel width cotangent (or NULL) */
  double *gpi;                  /* adjoint: pi kernel width cotangent (or NULL) */
  double wscale;                /* adjoint: ordered-pair factor (2 for auto) */
} SOFT_JOB;

#ifdef OMP
#define SOFT_ATOMIC_ADD(p, v)                                           \
  _Pragma("omp atomic") *(p) += (v)
#else
#define SOFT_ATOMIC_ADD(p, v)   (*(p) += (v))
#endif

/******************************************************************************
Function `soft_conf_init`:
  Initialise the soft histogram configuration from the CF structure.
Arguments:
  * `sc`:       structure for soft histogram configurations;
  * `cf`:       structure for correlation function configurations.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
static int soft_conf_init(SOFT_CONF *sc, const CF *cf) {
  if (!sc || !cf) return FCFC_ERR_ARG;
  memset(sc, 0, sizeof *sc);

  sc->bintype = cf->bintype;
  sc->ns = cf->ns;
  sc->np = cf->np;
  sc->s2min = cf->s2bin[0];
  sc->s2max = cf->s2bin[cf->ns];
  sc->smin = (real) sqrt((double) sc->s2min);
  sc->smax = (real) sqrt((double) sc->s2max);
  sc->smin0 = (sc->s2min == 0);
  sc->pmin0 = true;

  if (cf->bintype == FCFC_BIN_SPI) {
    if (!cf->pbin) {
      P_ERR("pi bins are not initialised for soft histogramming\n");
      return FCFC_ERR_CFG;
    }
    sc->pmin = cf->pbin[0];
    sc->pmax = cf->pbin[cf->np];
    sc->pmin0 = (sc->pmin == 0);
    sc->pbin = cf->pbin;
  }
  else if (cf->bintype == FCFC_BIN_SMU) {
    if (!cf->mutab) {
      P_ERR("mu bins are not initialised for soft histogramming\n");
      return FCFC_ERR_CFG;
    }
    sc->nmu2 = cf->nmu * cf->nmu;
    sc->mutab = cf->mutab;
  }

  /* The single-node traversal follows the hard kernels: the IN shortcut is
   * only used when no minimum separation (or pi) is requested. */
  sc->strict_single = (!sc->smin0) ||
      (cf->bintype == FCFC_BIN_SPI && !sc->pmin0);

  for (int i = 0; i < 3; i++) {
    sc->bsize[i] = cf->bsize[i];
    sc->hsize[i] = cf->bsize[i] * 0.5;
  }

  sc->soft_s = (cf->soft != 0);
  sc->sigma_log = (cf->soft_sigma_log != 0);
  sc->f = (double) cf->soft_sigma;
  const double sig = (double) cf->soft_sigma;
  if (sc->soft_s) {
    if (!(sig > 0) || !isfinite(sig)) {
      P_ERR("invalid Gaussian kernel width for soft histogramming: %g\n",
          sig);
      return FCFC_ERR_CFG;
    }
    sc->inv2sig2 = 0.5 / (sig * sig);
    sc->sigma = sig;
  }
  sc->s2bin = cf->s2bin;
  sc->nmu = cf->nmu;
  sc->nsec = cf->np > cf->nmu ? cf->np : cf->nmu;
  sc->soft_mu = (cf->soft_mu != 0) && (cf->bintype == FCFC_BIN_SMU);
  sc->soft_pi = (cf->soft_pi != 0) && (cf->bintype == FCFC_BIN_SPI);
  if (cf->bintype == FCFC_BIN_SPI) {
    sc->pbin = cf->pbin;
    if (!(sc->pcen = malloc(sizeof(real) * (size_t) cf->np))) {
      P_ERR("failed to allocate memory for soft histogramming\n");
      return FCFC_ERR_MEMORY;
    }
    for (int j = 0; j < cf->np; j++)
      sc->pcen[j] = (sc->pbin[j] + sc->pbin[j + 1]) * 0.5;
    if (sc->soft_pi) {
      const double spi = (double) cf->soft_pi_sigma;
      if (!(spi > 0) || !isfinite(spi)) {
        P_ERR("invalid Gaussian kernel width for soft pi histogramming: "
            "%g\n", spi);
        return FCFC_ERR_CFG;
      }
      sc->sigma_pi = spi;
      sc->inv2sig2_pi = 0.5 / (spi * spi);
      const real pmean = (sc->pbin[cf->np] - sc->pbin[0]) / cf->np;
      sc->lin_pi = true;
      for (int j = 0; j < cf->np; j++) {
        const real wj = sc->pbin[j + 1] - sc->pbin[j];
        if (REAL_ABS(wj - pmean) > pmean * (real) 1e-13) {
          sc->lin_pi = false;
          break;
        }
      }
      if (sc->lin_pi && pmean > 0) {
        sc->pi_c0 = sc->pcen[0];
        sc->pi_delta = pmean;
        sc->gg_pi = (double) pmean * (double) pmean * sc->inv2sig2_pi * 2.0;
        sc->iq_pi = exp(-sc->gg_pi);
      }
      else sc->lin_pi = false;
    }
  }
  if (sc->soft_mu) {
    const double smu = (double) cf->soft_mu_sigma;
    if (!(smu > 0) || !isfinite(smu)) {
      P_ERR("invalid Gaussian kernel width for soft mu histogramming: %g\n",
          smu);
      return FCFC_ERR_CFG;
    }
    sc->sigma_mu = smu;
    sc->inv2sig2_mu = 0.5 / (smu * smu);
    const double dm = 1.0 / cf->nmu;
    sc->gg_mu = dm * dm / (smu * smu);
    sc->iq_mu = exp(-sc->gg_mu);
  }

  if (!(sc->scen = malloc(sizeof(real) * (size_t) cf->ns))) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    return FCFC_ERR_MEMORY;
  }
  for (int i = 0; i < cf->ns; i++)
    sc->scen[i] = (cf->sbin[i] + cf->sbin[i + 1]) * 0.5;

  /* For linear bins the Gaussian kernel weights can be generated with a
   * multiplicative recurrence (see `soft_weights`): precompute its step. */
  sc->lin = true;
  const real dmean = (cf->sbin[cf->ns] - cf->sbin[0]) / cf->ns;
  for (int i = 0; i < cf->ns; i++) {
    const real wi = cf->sbin[i + 1] - cf->sbin[i];
    if (REAL_ABS(wi - dmean) > dmean * (real) 1e-13) {
      sc->lin = false;
      break;
    }
  }
  if (sc->lin && dmean > 0) {
    sc->delta = dmean;
    sc->scen0 = sc->scen[0];
    sc->gg = (double) dmean * (double) dmean * sc->inv2sig2 * 2.0;
    sc->iq = exp(-sc->gg);
  }
  else sc->lin = false;

  return 0;
}

/******************************************************************************
Function `soft_conf_destroy`:
  Release memory allocated for the soft histogram configuration.
Arguments:
  * `sc`:       structure for soft histogram configurations.
******************************************************************************/
static void soft_conf_destroy(SOFT_CONF *sc) {
  if (sc) {
    if (sc->scen) free(sc->scen);
    if (sc->pcen) free(sc->pcen);
    sc->scen = sc->pcen = NULL;
  }
}


/*============================================================================*\
                    Functions for the soft histogram update
\*============================================================================*/

/******************************************************************************
Function `soft_bin_search`:
  Binary search for the bin of a value: the largest k with edges[k] <= v,
  assuming edges[0] <= v < edges[n] (as guaranteed by the range checks).
******************************************************************************/
static inline int soft_bin_search(const real *edges, const int n,
    const real v) {
  int lo = 0, hi = n;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (v < edges[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}

/* Relative weight below which the kernel walk is stopped: the missing
 * tail mass is renormalised away (the pair weight is conserved exactly),
 * and the induced shape error is far below double precision significance. */
#define SOFT_WEIGHT_CUTOFF      1e-13

/* Kernel half-window (in units of sigma) used by the non-linear-bin
 * fallback: bins farther away carry less than exp(-K^2 / 2) ~ 2e-16 of
 * the peak weight, far below double precision significance. */
#define SOFT_WINDOW_SIGMA       8.5

/******************************************************************************
Function `soft_weights`:
  Unnormalised Gaussian-kernel (softmax) weights of a pair with separation
  `s`, written to `scratch[*lo .. *hi]` over an adaptive window of bins.
  Both the forward and the adjoint kernels use this function, so the
  backward pass replays exactly the weights of the forward pass.
Arguments:
  * `conf`:     structure for soft histogram configurations;
  * `s`:        separation (or s_perp) of the pair;
  * `scratch`:  thread-private scratch array with at least `ns` elements;
  * `lo`, `hi`: inclusive bounds of the kernel window (output).
Return:
  The sum of the returned weights (>= 1).
******************************************************************************/
static inline double soft_weights_recip(const int n, const double c0,
    const double delta, const double inv2sig2, const double iq,
    const double gg, const real s, double *scratch, int *lo, int *hi) {
  /* Nearest centre (the softmax peak). */
  const double ds = (double) s;
  int m = (int) floor((ds - c0) / delta + 0.5);
  if (m < 0) m = 0;
  else if (m >= n) m = n - 1;

  /* Upward and downward weight ratios at the peak, and their geometric
   * step: the log-ratios of a Gaussian over equispaced centres form an
   * arithmetic progression. */
  const double u = ds - (c0 + (double) m * delta);
  const double md = (2.0 * u - delta) * delta * inv2sig2;   /* -D_m */
  double rup = exp(md);                 /* w_{i+1} / w_i  (<= 1) */
  double rdn = exp(-md - gg);           /* w_{i-1} / w_i  (<= 1) */

  double w = 1.0, sum = 1.0;
  scratch[m] = w;
  int ilo = m, ihi = m;
  for (int i = m + 1; i < n; i++) {     /* towards larger values */
    w *= rup;
    if (w < SOFT_WEIGHT_CUTOFF) break;
    scratch[i] = w;
    sum += w;
    ihi = i;
    rup *= iq;
  }
  w = 1.0;
  for (int i = m - 1; i >= 0; i--) {    /* towards smaller values */
    w *= rdn;
    if (w < SOFT_WEIGHT_CUTOFF) break;
    scratch[i] = w;
    sum += w;
    ilo = i;
    rdn *= iq;
  }
  *lo = ilo;
  *hi = ihi;
  return sum;
}

/******************************************************************************
Function `soft_weights`:
  Unnormalised Gaussian-kernel (softmax) weights of a pair with separation
  `s`, written to `scratch[*lo .. *hi]` over an adaptive window of bins.
  Both the forward and the adjoint kernels use this function, so the
  backward pass replays exactly the weights of the forward pass.
Arguments:
  * `conf`:     structure for soft histogram configurations;
  * `s`:        separation (or s_perp) of the pair;
  * `scratch`:  thread-private scratch array with at least `ns` elements;
  * `lo`, `hi`: inclusive bounds of the kernel window (output).
Return:
  The sum of the returned weights (>= 1).
******************************************************************************/
static inline double soft_weights_gen(const real *cen, const int n,
    const double inv2sig2, const double sigma, const real s,
    double *scratch, int *lo, int *hi);

static inline double soft_weights(const SOFT_CONF *conf, const real s,
    double *scratch, int *lo, int *hi) {
  const int ns = conf->ns;

  if (conf->sigma_log) {
    /* scale-free kernel: sigma(s) = f * s (for logarithmic bins) */
    const double sl = (double) s * conf->f;
    return soft_weights_gen(conf->scen, ns, 0.5 / (sl * sl), sl, s,
        scratch, lo, hi);
  }
  if (conf->lin) {
    return soft_weights_recip(ns, (double) conf->scen0,
        (double) conf->delta, conf->inv2sig2, conf->iq, conf->gg, s,
        scratch, lo, hi);
  }

  /* Non-linear bins: direct stable softmax over an adaptive window of
   * bins within SOFT_WINDOW_SIGMA kernel widths of s. */
  const double k = conf->inv2sig2;
  const double dsv = (double) s;
  const double wlo = dsv - SOFT_WINDOW_SIGMA * conf->sigma;
  const double whi = dsv + SOFT_WINDOW_SIGMA * conf->sigma;
  int lo0 = 0, hi0 = ns - 1, a0 = 0, a1 = ns - 1;
  while (hi0 - lo0 > 1) {                 /* first centre >= wlo */
    const int mid = (lo0 + hi0) >> 1;
    if ((double) conf->scen[mid] < wlo) lo0 = mid;
    else hi0 = mid;
  }
  a0 = ((double) conf->scen[lo0] >= wlo) ? lo0 : hi0;
  if (a0 > ns - 1) a0 = ns - 1;
  lo0 = a0; hi0 = ns - 1;
  while (hi0 - lo0 > 1) {                 /* last centre <= whi */
    const int mid = (lo0 + hi0) >> 1;
    if ((double) conf->scen[mid] <= whi) lo0 = mid;
    else hi0 = mid;
  }
  a1 = ((double) conf->scen[hi0] <= whi) ? hi0 : lo0;
  if (a1 < a0) a1 = a0;

  double xmin = DBL_MAX;
  for (int i = a0; i <= a1; i++) {
    const double d = dsv - (double) conf->scen[i];
    const double x = d * d * k;
    scratch[i] = x;
    if (x < xmin) xmin = x;
  }
  double sum = 0;
  for (int i = a0; i <= a1; i++) {
    /* Clamp the exponent to avoid denormal results. */
    double a = xmin - scratch[i];
    if (a < -40.0) a = -40.0;
    const double e = exp(a);
    scratch[i] = e;
    sum += e;
  }
  *lo = a0;
  *hi = a1;
  return sum;
}

/******************************************************************************
Function `soft_weights_gen`:
  Windowed stable softmax over an arbitrary (sorted) centre array.
******************************************************************************/
static inline double soft_weights_gen(const real *cen, const int n,
    const double inv2sig2, const double sigma, const real s,
    double *scratch, int *lo, int *hi) {
  const double k = inv2sig2;
  const double dsv = (double) s;
  const double wlo = dsv - SOFT_WINDOW_SIGMA * sigma;
  const double whi = dsv + SOFT_WINDOW_SIGMA * sigma;
  int lo0 = 0, hi0 = n - 1, a0 = 0, a1 = n - 1;
  while (hi0 - lo0 > 1) {
    const int mid = (lo0 + hi0) >> 1;
    if ((double) cen[mid] < wlo) lo0 = mid;
    else hi0 = mid;
  }
  a0 = ((double) cen[lo0] >= wlo) ? lo0 : hi0;
  if (a0 > n - 1) a0 = n - 1;
  lo0 = a0; hi0 = n - 1;
  while (hi0 - lo0 > 1) {
    const int mid = (lo0 + hi0) >> 1;
    if ((double) cen[mid] <= whi) lo0 = mid;
    else hi0 = mid;
  }
  a1 = ((double) cen[hi0] <= whi) ? hi0 : lo0;
  if (a1 < a0) a1 = a0;
  double xmin = DBL_MAX;
  for (int i = a0; i <= a1; i++) {
    const double d = dsv - (double) cen[i];
    const double x = d * d * k;
    scratch[i] = x;
    if (x < xmin) xmin = x;
  }
  double sum = 0;
  for (int i = a0; i <= a1; i++) {
    /* Clamp the exponent to avoid denormal results. */
    double a = xmin - scratch[i];
    if (a < -40.0) a = -40.0;
    const double e = exp(a);
    scratch[i] = e;
    sum += e;
  }
  *lo = a0;
  *hi = a1;
  return sum;
}

/******************************************************************************
Function `soft_weights_pi`:
  The same as `soft_weights` for the pi dimension of the (s_perp, pi)
  scheme: the pi bin centres come from the (un-squared) pi edges; the
  multiplicative recurrence applies when those bins are linear, else the
  windowed stable softmax does.
******************************************************************************/
static inline double soft_weights_pi(const SOFT_CONF *conf, const real pi,
    double *scratch, int *lo, int *hi) {
  if (conf->lin_pi) {
    return soft_weights_recip(conf->np, (double) conf->pi_c0,
        (double) conf->pi_delta, conf->inv2sig2_pi, conf->iq_pi,
        conf->gg_pi, pi, scratch, lo, hi);
  }
  return soft_weights_gen(conf->pcen, conf->np, conf->inv2sig2_pi,
      conf->sigma_pi, pi, scratch, lo, hi);
}

/******************************************************************************
Function `soft_weights_mu`:
  The same as `soft_weights` for the mu dimension: the mu bins of FCFC are
  always linear on [0, 1), with centres (j + 1/2) / nmu, so the
  multiplicative recurrence always applies.
******************************************************************************/
static inline double soft_weights_mu(const SOFT_CONF *conf, const real mu,
    double *scratch, int *lo, int *hi) {
  const double dm = 1.0 / conf->nmu;
  return soft_weights_recip(conf->nmu, 0.5 * dm, dm, conf->inv2sig2_mu,
      conf->iq_mu, conf->gg_mu, mu, scratch, lo, hi);
}

/******************************************************************************
Function `hard_s_bin`:
  Index of the squared-separation bin of `s2` (left-closed, right-open),
  used when the separations are hard-binned (SOFT_HIST off) while the mu
  dimension is soft-binned.
******************************************************************************/
static inline int hard_s_bin(const SOFT_CONF *conf, const real s2) {
  const real *e = conf->s2bin;
  int lo = 0, hi = conf->ns;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (s2 < e[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}

/******************************************************************************
Function `soft_bin_pair`:
  Forward: soft-bin a pair of the periodic-box component, given the
  coordinate differences (already periodic-wrapped or shifted where
  necessary).  The range checks and the mu / pi binning replicate the
  scalar hard kernels (`compute_dist_hist_scalar` and
  `compute_dist_wrap_hist_scalar`).
******************************************************************************/
static inline void soft_bin_pair(const SOFT_CONF *conf,
    real dx, real dy, real dz, const real wt,
    double *cnt, double *scratch) {
  double *scr_s = scratch;
  double *scr_mu = scratch + conf->ns;   /* mu (smu) or pi (spi) weights */
  const bool soft_s = conf->soft_s;
  const bool soft_mu = conf->soft_mu;
  bool pi_soft = false;
  real pi2nd = 0;
  real s2;
  size_t off = 0;
  int mlo = 0, mhi = -1;
  double msum = 1.0;
  size_t j0 = 0;

  if (conf->bintype == FCFC_BIN_SPI) {
    const real pi = REAL_ABS(dz);                 /* pi, a.k.a. s_par */
    if (pi >= conf->pmax || pi < conf->pmin) return;
    s2 = dx * dx + dy * dy;                       /* s_perp squared */
    if (s2 >= conf->s2max || s2 < conf->s2min) return;
    if (conf->soft_pi) {
      pi_soft = true;
      pi2nd = pi;
      msum = soft_weights_pi(conf, pi, scr_mu, &mlo, &mhi);
    }
    else {
      j0 = (size_t) soft_bin_search(conf->pbin, conf->np, pi);
      off = j0 * (size_t) conf->ns;
    }
  }
  else {
    const real dz2 = dz * dz;
    s2 = dx * dx + dy * dy + dz2;                 /* s squared */
    if (s2 >= conf->s2max || s2 < conf->s2min) return;
    if (conf->bintype == FCFC_BIN_SMU) {
      const real mu2 = (s2 < REAL_EPS) ? 0 : dz2 / s2;
      if (soft_mu) {
        if (mu2 >= (real) 1) return;              /* mu == 1 pairs dropped */
        msum = soft_weights_mu(conf, (real) sqrt((double) mu2), scr_mu,
            &mlo, &mhi);
      }
      else {
        const int pidx = (int) (mu2 * conf->nmu2);
        if (pidx >= conf->nmu2) return;           /* mu == 1 pairs dropped */
        j0 = (size_t) conf->mutab[pidx];
        off = j0 * (size_t) conf->ns;
      }
    }
  }

  /* Spread the pair weight over the (window of) separation bins, times
   * the mu weights when mu is soft-binned. */
  if (pi_soft) {
    if (soft_s) {
      int slo, shi;
      const double ssum2 = soft_weights(conf, (real) sqrt((double) s2),
          scr_s, &slo, &shi);
      const double f = (double) wt / (ssum2 * msum);
      for (int j = mlo; j <= mhi; j++) {
        const double wm = scr_mu[j] * f;
        const size_t o = (size_t) j * conf->ns;
        for (int i = slo; i <= shi; i++) cnt[o + i] += scr_s[i] * wm;
      }
    }
    else {
      const size_t sb2 = (size_t) hard_s_bin(conf, s2);
      const double f = (double) wt / msum;
      for (int j = mlo; j <= mhi; j++)
        cnt[(size_t) j * conf->ns + sb2] += scr_mu[j] * f;
    }
    return;
  }
  if (soft_mu) {
    if (soft_s) {
      int slo, shi;
      const double ssum = soft_weights(conf, (real) sqrt((double) s2),
          scr_s, &slo, &shi);
      const double f = (double) wt / (ssum * msum);
      for (int j = mlo; j <= mhi; j++) {
        const double wm = scr_mu[j] * f;
        const size_t o = off + (size_t) j * conf->ns;
        for (int i = slo; i <= shi; i++) cnt[o + i] += scr_s[i] * wm;
      }
    }
    else {
      const size_t sb = (size_t) hard_s_bin(conf, s2);
      const double f = (double) wt / msum;
      for (int j = mlo; j <= mhi; j++)
        cnt[off + (size_t) j * conf->ns + sb] += scr_mu[j] * f;
    }
    return;
  }
  if (soft_s) {
    int slo, shi;
    const double ssum = soft_weights(conf, (real) sqrt((double) s2),
        scr_s, &slo, &shi);
    const double f = (double) wt / ssum;
    for (int i = slo; i <= shi; i++) cnt[off + i] += scr_s[i] * f;
  }
  else {
    cnt[off + (size_t) hard_s_bin(conf, s2)] += (double) wt;
  }
}

/******************************************************************************
Function `soft_bin_pair_adj`:
  Adjoint (VJP) of `soft_bin_pair` with respect to the pair separation and
  the pair weight: contract the bin cotangents `g` with the kernel weight
  derivatives (eqs. 4-7 of the file header) and accumulate the resulting
  coordinate, weight and kernel-width cotangents.  The discrete choices
  (range checks, mu / pi bin, kernel window) are replayed identically to
  the forward pass and are treated as constants.
Arguments:
  * `conf`:     structure for soft histogram configurations;
  * `dx,dy,dz`: coordinate differences between the two points (wrapped or
                shifted as in the forward pass);
  * `x1,y1,z1`: coordinates of the first point (internal units);
  * `x2,y2,z2`: coordinates of the second point (internal units);
  * `w1,w2`:    weights of the two points (1 if unweighted);
  * `i1,i2`:    indices of the two points in their catalogues;
  * `job`:      work descriptor with the cotangent buffers;
  * `scratch`:  thread-private scratch array with at least `ns` elements.
******************************************************************************/
static inline void soft_bin_pair_adj(const SOFT_CONF *conf,
    real dx, real dy, real dz,
    const real x1, const real y1, const real z1,
    const real x2, const real y2, const real z2,
    const real w1, const real w2, const size_t i1, const size_t i2,
    const SOFT_JOB *job, double *scratch) {
  double *scr_s = scratch;
  double *scr_mu = scratch + conf->ns;   /* mu (smu) or pi (spi) weights */
  double *gs = scratch + (size_t) conf->ns + conf->nsec;
  double *gm = scratch + 2 * (size_t) conf->ns + conf->nsec;
  const bool soft_s = conf->soft_s;
  const bool soft_mu = conf->soft_mu;
  bool pi_soft = false;
  real pi2nd = 0;
  const int ns = conf->ns;
  real s2;
  size_t off = 0;
  int slo = 0, shi = -1, mlo = 0, mhi = -1;
  double ssum = 1.0, msum = 1.0;
  size_t sb = 0, j0 = 0;
  real mu = 0;

  if (conf->bintype == FCFC_BIN_SPI) {
    const real pi = REAL_ABS(dz);
    if (pi >= conf->pmax || pi < conf->pmin) return;
    s2 = dx * dx + dy * dy;
    if (s2 >= conf->s2max || s2 < conf->s2min) return;
    if (conf->soft_pi) {
      pi_soft = true;
      pi2nd = pi;
      msum = soft_weights_pi(conf, pi, scr_mu, &mlo, &mhi);
    }
    else {
      j0 = (size_t) soft_bin_search(conf->pbin, conf->np, pi);
      off = j0 * (size_t) ns;
    }
  }
  else {
    const real dz2 = dz * dz;
    s2 = dx * dx + dy * dy + dz2;
    if (s2 >= conf->s2max || s2 < conf->s2min) return;
    if (conf->bintype == FCFC_BIN_SMU) {
      const real mu2 = (s2 < REAL_EPS) ? 0 : dz2 / s2;
      if (soft_mu) {
        if (mu2 >= (real) 1) return;
        mu = (real) sqrt((double) mu2);
        msum = soft_weights_mu(conf, mu, scr_mu, &mlo, &mhi);
      }
      else {
        const int pidx = (int) (mu2 * conf->nmu2);
        if (pidx >= conf->nmu2) return;
        j0 = (size_t) conf->mutab[pidx];
        off = j0 * (size_t) ns;
      }
    }
  }

  if (soft_s)
    ssum = soft_weights(conf, (real) sqrt((double) s2), scr_s, &slo, &shi);
  else
    sb = (size_t) hard_s_bin(conf, s2);

  /* Contract the bin cotangents onto each dimension with the weights of
   * the other one (the kernel is a product of the two softmaxes). */
  const double *g = job->g + off;
  if (soft_s) {
    for (int i = slo; i <= shi; i++) {
      double acc = 0;
      if (soft_mu || pi_soft) {
        for (int j = mlo; j <= mhi; j++)
          acc += g[i + (size_t) j * ns] * scr_mu[j];
        acc /= msum;
      }
      else acc = g[i + j0 * ns];
      gs[i] = acc;
    }
  }
  if (soft_mu || pi_soft) {
    for (int j = mlo; j <= mhi; j++) {
      double acc = 0;
      if (soft_s) {
        for (int i = slo; i <= shi; i++)
          acc += g[i + (size_t) j * ns] * scr_s[i];
        acc /= ssum;
      }
      else acc = g[sb + (size_t) j * ns];
      gm[j] = acc;
    }
  }

  /* Normalised-weight contractions (eqs. 4-7 of the file header). */
  const double t = (double) sqrt((double) s2);
  const double *cen = (const double *) conf->scen;
  const double dm = 1.0 / conf->nmu;
  double cbar = 0, xbar = 0, A = 0, Ax = 0, wbar = 0;
  if (soft_s) {
    const double invS = 1.0 / ssum;
    const int want_s = (job->gsig != NULL);
    /* sigma(s) = f * s: logits x_i = (1 - c_i/s)^2 / (2 f^2) and
     * dW_i/ds = W_i (Qbar - Q_i) / (f^2 s^3) with Q_i = (s - c_i) c_i;
     * for a constant sigma this reduces to Q_i = c_i and the familiar
     * (c_i - cbar) / sigma^2 form. */
    const double inv2_p = conf->sigma_log ?
        0.5 / (conf->f * conf->f * t * t) : conf->inv2sig2;
    for (int i = slo; i <= shi; i++) {
      const double W = scr_s[i] * invS;
      cbar += W * (conf->sigma_log ? (t - cen[i]) * cen[i] : cen[i]);
      wbar += gs[i] * W;
      if (want_s) {
        const double d = t - cen[i];
        xbar += W * (d * d * inv2_p);
      }
    }
    for (int i = slo; i <= shi; i++) {
      const double W = scr_s[i] * invS;
      const double Qi = conf->sigma_log ?
          (t - cen[i]) * cen[i] : cen[i];
      A += gs[i] * W * (Qi - cbar);
      if (want_s) {
        const double d = t - cen[i];
        Ax += gs[i] * W * (d * d * inv2_p - xbar);
      }
    }
  }
  else if (soft_mu) {
    const double invM = 1.0 / msum;
    for (int j = mlo; j <= mhi; j++) wbar += gm[j] * scr_mu[j] * invM;
  }
  else wbar = g[sb + j0 * ns];

  const bool soft2 = (soft_mu || pi_soft);
  const bool is_mu = (conf->bintype == FCFC_BIN_SMU);
  const double inv2_2 = is_mu ? conf->inv2sig2_mu : conf->inv2sig2_pi;
  const double sig2 = is_mu ? conf->sigma_mu : conf->sigma_pi;
  const double mu2nd_d = (double) (is_mu ? mu : pi2nd);
  double mbar = 0, ybar = 0, Amu = 0, Ay = 0;
  if (soft2) {
    const double invM = 1.0 / msum;
    const int want_m = is_mu ? (job->gmus != NULL) : (job->gpi != NULL);
    for (int j = mlo; j <= mhi; j++) {
      const double W = scr_mu[j] * invM;
      const double mj = is_mu ? (j + 0.5) * dm : (double) conf->pcen[j];
      mbar += W * mj;
      if (want_m) {
        const double d = mu2nd_d - mj;
        ybar += W * (d * d * inv2_2);
      }
    }
    for (int j = mlo; j <= mhi; j++) {
      const double W = scr_mu[j] * invM;
      const double mj = is_mu ? (j + 0.5) * dm : (double) conf->pcen[j];
      Amu += gm[j] * W * (mj - mbar);
      if (want_m) {
        const double d = mu2nd_d - mj;
        Ay += gm[j] * W * (d * d * inv2_2 - ybar);
      }
    }
  }

  const double wt = (double) w1 * (double) w2;
  const double ws = job->wscale;
  const double Kt = conf->sigma_log ?
      -1.0 / (conf->f * conf->f * t * t * t) : 2.0 * conf->inv2sig2;
  const double tbar = soft_s ? ws * wt * A * Kt : 0.0;
  const double mubar = soft2 ? ws * wt * Amu * 2.0 * inv2_2 : 0.0;

  if (job->gw1) SOFT_ATOMIC_ADD(job->gw1 + i1, ws * wbar * w2);
  if (job->gw2) SOFT_ATOMIC_ADD(job->gw2 + i2, ws * wbar * w1);
  if (job->gsig && soft_s)
    SOFT_ATOMIC_ADD(job->gsig, ws * wt * 2.0 * Ax /
        (conf->sigma_log ? conf->f : conf->sigma));
  if (soft2 && ((is_mu && job->gmus) || (!is_mu && job->gpi)))
    SOFT_ATOMIC_ADD(is_mu ? job->gmus : job->gpi,
        ws * wt * 2.0 * Ay / sig2);

  /* Chain the separation and mu cotangents to the coordinate difference.
   * t is s (iso, smu) or s_perp (spi); mu = |dz| / s (smu).  The periodic
   * wrap / node shift is piecewise constant, hence its derivative is the
   * identity. */
  double ux = 0, uy = 0, uz = 0;
  if (tbar != 0 && s2 >= REAL_EPS) {
    const double inv = tbar / t;
    if (conf->bintype == FCFC_BIN_SPI) {
      ux = dx * inv;
      uy = dy * inv;
    }
    else {
      ux = dx * inv;
      uy = dy * inv;
      uz = dz * inv;
    }
  }
  if (mubar != 0 && s2 >= REAL_EPS && conf->bintype == FCFC_BIN_SMU) {
    /* dmu/d(d) for mu = |dz| / s */
    const double s3 = t * t * t;
    const double adz = REAL_ABS(dz);
    const double sg = (dz >= 0) ? 1.0 : -1.0;
    ux += mubar * (-adz * dx / s3);
    uy += mubar * (-adz * dy / s3);
    uz += mubar * (sg * ((double) dx * dx + (double) dy * dy) / s3);
  }
  if (mubar != 0 && conf->bintype == FCFC_BIN_SPI) {
    /* dpi/d(d) for pi = |dz| */
    uz += mubar * ((dz >= 0) ? 1.0 : -1.0);
  }
  if (ux != 0 || uy != 0 || uz != 0) {
    double *g1 = job->gx1 + 3 * i1;
    double *g2 = job->gx2 + 3 * i2;
    SOFT_ATOMIC_ADD(g1 + 0, ux);
    SOFT_ATOMIC_ADD(g1 + 1, uy);
    SOFT_ATOMIC_ADD(g1 + 2, uz);
    SOFT_ATOMIC_ADD(g2 + 0, -ux);
    SOFT_ATOMIC_ADD(g2 + 1, -uy);
    SOFT_ATOMIC_ADD(g2 + 2, -uz);
  }
  (void) x1; (void) y1; (void) z1;
  (void) x2; (void) y2; (void) z2;
}

/*============================================================================*\
                Functions for counting pairs on leaf nodes
\*============================================================================*/

/* Pairs within a single node (auto pairs, i < j). */
static void soft_count_single(real *const *x, const real *w, const size_t n,
    const SOFT_JOB *job, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  const bool hw = (w != NULL);
  for (size_t i = 0; i + 1 < n; i++) {
    const real wti = hw ? w[i] : (real) 1;
    for (size_t j = i + 1; j < n; j++) {
      const real wt = hw ? wti * w[j] : (real) 1;
      const real dx = x[0][i] - x[0][j];
      const real dy = x[1][i] - x[1][j];
      const real dz = x[2][i] - x[2][j];
      if (job->adjoint)
        soft_bin_pair_adj(conf, dx, dy, dz, x[0][i], x[1][i], x[2][i],
            x[0][j], x[1][j], x[2][j], wti, (hw ? w[j] : (real) 1), i, j,
            job, scratch);
      else
        soft_bin_pair(conf, dx, dy, dz, wt, job->cnt, scratch);
    }
  }
}

/* Pairs between two nodes, with a fixed periodic shift applied to the
 * first node (valid when the node extents are inside half the box). */
static void soft_count_dual(real *const *a, const real *wa, const size_t na,
    real *const *b, const real *wb, const size_t nb, const real shift[3],
    const SOFT_JOB *job, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  const bool ha = (wa != NULL);
  const bool hb = (wb != NULL);
  for (size_t i = 0; i < na; i++) {
    const real ax = a[0][i] + shift[0];
    const real ay = a[1][i] + shift[1];
    const real az = a[2][i] + shift[2];
    const real wti = ha ? wa[i] : (real) 1;
    for (size_t j = 0; j < nb; j++) {
      const real wt = hb ? wti * wb[j] : (real) 1;
      if (job->adjoint)
        soft_bin_pair_adj(conf, ax - b[0][j], ay - b[1][j], az - b[2][j],
            ax, ay, az, b[0][j], b[1][j], b[2][j], wti,
            (hb ? wb[j] : (real) 1), i, j, job, scratch);
      else
        soft_bin_pair(conf, ax - b[0][j], ay - b[1][j], az - b[2][j],
            wt, job->cnt, scratch);
    }
  }
}

/* Pairs between two nodes with per-pair periodic wrapping (minimum image
 * convention). */
static void soft_count_dual_wrap(real *const *a, const real *wa,
    const size_t na, real *const *b, const real *wb, const size_t nb,
    const SOFT_JOB *job, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  const bool ha = (wa != NULL);
  const bool hb = (wb != NULL);
  for (size_t i = 0; i < na; i++) {
    const real wti = ha ? wa[i] : (real) 1;
    for (size_t j = 0; j < nb; j++) {
      const real wt = hb ? wti * wb[j] : (real) 1;
      real dx = a[0][i] - b[0][j];
      if (dx > conf->hsize[0]) dx -= conf->bsize[0];
      else if (dx < -conf->hsize[0]) dx += conf->bsize[0];
      real dy = a[1][i] - b[1][j];
      if (dy > conf->hsize[1]) dy -= conf->bsize[1];
      else if (dy < -conf->hsize[1]) dy += conf->bsize[1];
      real dz = a[2][i] - b[2][j];
      if (dz > conf->hsize[2]) dz -= conf->bsize[2];
      else if (dz < -conf->hsize[2]) dz += conf->bsize[2];
      if (job->adjoint)
        soft_bin_pair_adj(conf, dx, dy, dz, a[0][i], a[1][i], a[2][i],
            b[0][j], b[1][j], b[2][j], wti, (hb ? wb[j] : (real) 1), i, j,
            job, scratch);
      else
        soft_bin_pair(conf, dx, dy, dz, wt, job->cnt, scratch);
    }
  }
}


/*============================================================================*\
             Functions for checking distances between k-D tree nodes
\*============================================================================*/

/* Scalar counterpart of `compare_single_node_kdtree_*` (metric_kdtree.c). */
static inline soft_node_dist_e soft_cmp_single_kdt(const KDT *node,
    const SOFT_CONF *conf) {
  if (conf->bintype == FCFC_BIN_SPI) {
    real wz = 0;
    if (!conf->pmin0) {
      wz = node->wid[2] * 2;
      if (wz < conf->pmin) return SOFT_NODE_DIST_OUT;
    }
    if (!conf->smin0) {
      const real wx = node->wid[0] * 2;
      const real wy = node->wid[1] * 2;
      const real dist = wx * wx + wy * wy;
      if (dist < conf->s2min) return SOFT_NODE_DIST_OUT;
    }
    else if (conf->pmin0) {
      const real wx = node->wid[0] * 2;
      const real wy = node->wid[1] * 2;
      const real dist = wx * wx + wy * wy;
      wz = node->wid[2] * 2;
      if (dist < conf->s2max && wz < conf->pmax) return SOFT_NODE_DIST_IN;
    }
    return SOFT_NODE_DIST_X;
  }

  const real wx = node->wid[0] * 2;
  const real wy = node->wid[1] * 2;
  const real wz = node->wid[2] * 2;
  const real dist = wx * wx + wy * wy + wz * wz;  /* maximum separation */
  if (!conf->smin0) {
    if (dist < conf->s2min) return SOFT_NODE_DIST_OUT;
  }
  else if (dist < conf->s2max) return SOFT_NODE_DIST_IN;
  return SOFT_NODE_DIST_X;
}

/* Scalar counterpart of `compare_dual_node_kdtree_*` (metric_kdtree.c). */
static inline soft_node_dist_e soft_cmp_dual_kdt(const KDT *node1,
    const KDT *node2, const SOFT_CONF *conf, real shift[3]) {
  shift[0] = shift[1] = shift[2] = 0;
  real dx = node1->cen[0] - node2->cen[0];
  if (dx > conf->hsize[0]) shift[0] = -conf->bsize[0];
  else if (dx < -conf->hsize[0]) shift[0] = conf->bsize[0];
  dx = REAL_ABS(dx + shift[0]);

  real dy = node1->cen[1] - node2->cen[1];
  if (dy > conf->hsize[1]) shift[1] = -conf->bsize[1];
  else if (dy < -conf->hsize[1]) shift[1] = conf->bsize[1];
  dy = REAL_ABS(dy + shift[1]);

  real dz = node1->cen[2] - node2->cen[2];
  if (dz > conf->hsize[2]) shift[2] = -conf->bsize[2];
  else if (dz < -conf->hsize[2]) shift[2] = conf->bsize[2];
  dz = REAL_ABS(dz + shift[2]);

  const real wx = node1->wid[0] + node2->wid[0];
  const real wy = node1->wid[1] + node2->wid[1];
  const real wz = node1->wid[2] + node2->wid[2];

  if (conf->bintype == FCFC_BIN_SPI) {
    /* Check the minimum pi. */
    if (dz - wz >= conf->pmax) return SOFT_NODE_DIST_OUT;
    /* Check the maximum pi. */
    const real zmax = dz + wz;
    if (!conf->pmin0 && zmax < conf->pmin) return SOFT_NODE_DIST_OUT;

    /* Check the minimum s_perp. */
    real d, sum = 0;
    d = dx - wx;
    if (d > 0) sum += d * d;
    d = dy - wy;
    if (d > 0) sum += d * d;
    if (sum >= conf->s2max) return SOFT_NODE_DIST_OUT;
    const real sum_min = sum;

    /* Check the maximum s_perp. */
    const real xmax = dx + wx;
    const real ymax = dy + wy;
    sum = xmax * xmax + ymax * ymax;

    const bool pin = conf->pmin0 ?
        (zmax < conf->pmax) :
        (dz - wz >= conf->pmin && zmax < conf->pmax);
    if (conf->smin0) {
      if (sum < conf->s2max && pin) return SOFT_NODE_DIST_IN;
    }
    else {
      if (sum < conf->s2min) return SOFT_NODE_DIST_OUT;
      if (sum_min >= conf->s2min && sum < conf->s2max && pin)
        return SOFT_NODE_DIST_IN;
    }

    if (xmax >= conf->hsize[0] || ymax >= conf->hsize[1] ||
        zmax >= conf->hsize[2]) return SOFT_NODE_DIST_X_WRAP;
    return SOFT_NODE_DIST_X;
  }

  /* Check the minimum separation. */
  real d, sum = 0;
  d = dx - wx;
  if (d > 0) sum += d * d;
  d = dy - wy;
  if (d > 0) sum += d * d;
  d = dz - wz;
  if (d > 0) sum += d * d;
  if (sum >= conf->s2max) return SOFT_NODE_DIST_OUT;
  const real sum_min = sum;

  /* Check the maximum separation. */
  const real xmax = dx + wx;
  const real ymax = dy + wy;
  const real zmax = dz + wz;
  sum = xmax * xmax + ymax * ymax + zmax * zmax;
  if (conf->smin0) {
    if (sum < conf->s2max) return SOFT_NODE_DIST_IN;
  }
  else {
    if (sum < conf->s2min) return SOFT_NODE_DIST_OUT;
    if (sum_min >= conf->s2min && sum < conf->s2max)
      return SOFT_NODE_DIST_IN;
  }

  if (xmax >= conf->hsize[0] || ymax >= conf->hsize[1] ||
      zmax >= conf->hsize[2]) return SOFT_NODE_DIST_X_WRAP;
  return SOFT_NODE_DIST_X;
}


/*============================================================================*\
             Functions for checking distances between ball tree nodes
\*============================================================================*/

/* Scalar counterpart of `compare_single_node_balltree_*`
 * (metric_balltree.c). */
static inline soft_node_dist_e soft_cmp_single_blt(const BLT *node,
    const SOFT_CONF *conf) {
  const real diameter = node->r * 2;
  if (conf->bintype == FCFC_BIN_SPI) {
    if (!conf->pmin0 && diameter < conf->pmin) return SOFT_NODE_DIST_OUT;
    if (!conf->smin0 && diameter < conf->smin) return SOFT_NODE_DIST_OUT;
    if (conf->smin0 && conf->pmin0 &&
        diameter < conf->smax && diameter < conf->pmax)
      return SOFT_NODE_DIST_IN;
  }
  else {
    if (!conf->smin0) {
      if (diameter < conf->smin) return SOFT_NODE_DIST_OUT;
    }
    else if (diameter < conf->smax) return SOFT_NODE_DIST_IN;
  }
  return SOFT_NODE_DIST_X;
}

/* Scalar counterpart of `compare_dual_node_balltree_*`
 * (metric_balltree.c). */
static inline soft_node_dist_e soft_cmp_dual_blt(const BLT *node1,
    const BLT *node2, const SOFT_CONF *conf, real shift[3]) {
  shift[0] = shift[1] = shift[2] = 0;
  real dx = node1->cen[0] - node2->cen[0];
  if (dx > conf->hsize[0]) shift[0] = -conf->bsize[0];
  else if (dx < -conf->hsize[0]) shift[0] = conf->bsize[0];
  dx += shift[0];

  real dy = node1->cen[1] - node2->cen[1];
  if (dy > conf->hsize[1]) shift[1] = -conf->bsize[1];
  else if (dy < -conf->hsize[1]) shift[1] = conf->bsize[1];
  dy += shift[1];

  real dz = node1->cen[2] - node2->cen[2];
  if (dz > conf->hsize[2]) shift[2] = -conf->bsize[2];
  else if (dz < -conf->hsize[2]) shift[2] = conf->bsize[2];

  const real sr = node1->r + node2->r;

  if (conf->bintype == FCFC_BIN_SPI) {
    dz = REAL_ABS(dz + shift[2]);
    if (dz - sr >= conf->pmax) return SOFT_NODE_DIST_OUT;
    const real zmax = dz + sr;
    if (!conf->pmin0 && zmax < conf->pmin) return SOFT_NODE_DIST_OUT;

    const real sum = dx * dx + dy * dy;
    real dcmp = conf->smax + sr;
    if (sum >= dcmp * dcmp) return SOFT_NODE_DIST_OUT;

    const bool pin = conf->pmin0 ?
        (zmax < conf->pmax) :
        (dz - sr >= conf->pmin && zmax < conf->pmax);
    if (conf->smin0) {
      dcmp = conf->smax - sr;
      if (dcmp > 0 && sum < dcmp * dcmp && pin) return SOFT_NODE_DIST_IN;
    }
    else {
      dcmp = conf->smin - sr;
      if (dcmp > 0 && sum < dcmp * dcmp) return SOFT_NODE_DIST_OUT;
      dcmp = conf->smax - sr;
      if (dcmp > 0 && sum < dcmp * dcmp) {
        dcmp = conf->smin + sr;
        if (sum >= dcmp * dcmp && pin) return SOFT_NODE_DIST_IN;
      }
    }

    if (REAL_ABS(dx) + sr >= conf->hsize[0] ||
        REAL_ABS(dy) + sr >= conf->hsize[1] ||
        zmax >= conf->hsize[2]) return SOFT_NODE_DIST_X_WRAP;
    return SOFT_NODE_DIST_X;
  }

  dz += shift[2];
  const real sum = dx * dx + dy * dy + dz * dz;
  real dcmp = conf->smax + sr;
  if (sum >= dcmp * dcmp) return SOFT_NODE_DIST_OUT;

  if (conf->smin0) {
    dcmp = conf->smax - sr;
    if (dcmp > 0 && sum < dcmp * dcmp) return SOFT_NODE_DIST_IN;
  }
  else {
    dcmp = conf->smin - sr;
    if (dcmp > 0 && sum < dcmp * dcmp) return SOFT_NODE_DIST_OUT;
    dcmp = conf->smax - sr;
    if (dcmp > 0 && sum < dcmp * dcmp) {
      dcmp = conf->smin + sr;
      if (sum >= dcmp * dcmp) return SOFT_NODE_DIST_IN;
    }
  }

  if (REAL_ABS(dx) + sr >= conf->hsize[0] ||
      REAL_ABS(dy) + sr >= conf->hsize[1] ||
      REAL_ABS(dz) + sr >= conf->hsize[2]) return SOFT_NODE_DIST_X_WRAP;
  return SOFT_NODE_DIST_X;
}


/*============================================================================*\
                   Recursive dual-tree traversal
\*============================================================================*/

/* k-D tree traversal, replicating `paircnt_kdtree_*` (dual_tree.c). */
static void soft_traverse_kdt(const KDT *node1, const KDT *node2,
    const SOFT_JOB *job, const bool isauto, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  if (isauto && node1 == node2) {         /* the same node */
    const soft_node_dist_e flag = soft_cmp_single_kdt(node1, conf);
    if (conf->strict_single) {
      if (flag == SOFT_NODE_DIST_OUT) return;
      if (node1->left == NULL) {
        soft_count_single(node1->x, node1->w, node1->n, job, scratch);
        return;
      }
    }
    else {
      if (flag == SOFT_NODE_DIST_IN || node1->left == NULL) {
        soft_count_single(node1->x, node1->w, node1->n, job, scratch);
        return;
      }
    }
    if (node1->left->x[0] <= node2->left->x[0])
      soft_traverse_kdt(node1->left, node2->left, job, isauto, scratch);
    if (node1->left->x[0] <= node2->right->x[0])
      soft_traverse_kdt(node1->left, node2->right, job, isauto, scratch);
    if (node1->right->x[0] <= node2->left->x[0])
      soft_traverse_kdt(node1->right, node2->left, job, isauto, scratch);
    if (node1->right->x[0] <= node2->right->x[0])
      soft_traverse_kdt(node1->right, node2->right, job, isauto, scratch);
    return;
  }

  real shift[3] = {0, 0, 0};
  const soft_node_dist_e flag = soft_cmp_dual_kdt(node1, node2, conf, shift);
  if (flag == SOFT_NODE_DIST_OUT) return;
  if (flag == SOFT_NODE_DIST_IN) {
    soft_count_dual(node1->x, node1->w, node1->n, node2->x, node2->w,
        node2->n, shift, job, scratch);
    return;
  }
  if (node1->left == NULL && node2->left == NULL) {
    if (flag != SOFT_NODE_DIST_X_WRAP) {
      soft_count_dual(node1->x, node1->w, node1->n, node2->x, node2->w,
          node2->n, shift, job, scratch);
    }
    else {
      soft_count_dual_wrap(node1->x, node1->w, node1->n, node2->x,
          node2->w, node2->n, job, scratch);
    }
    return;
  }
  if (node1->left && node2->left) {
    if (!isauto || node1->right->x[0] <= node2->right->x[0])
      soft_traverse_kdt(node1->right, node2->right, job, isauto, scratch);
    if (!isauto || node1->right->x[0] <= node2->left->x[0])
      soft_traverse_kdt(node1->right, node2->left, job, isauto, scratch);
    if (!isauto || node1->left->x[0] <= node2->right->x[0])
      soft_traverse_kdt(node1->left, node2->right, job, isauto, scratch);
    if (!isauto || node1->left->x[0] <= node2->left->x[0])
      soft_traverse_kdt(node1->left, node2->left, job, isauto, scratch);
  }
  else if (node1->left) {
    if (!isauto || node1->right->x[0] <= node2->x[0])
      soft_traverse_kdt(node1->right, node2, job, isauto, scratch);
    if (!isauto || node1->left->x[0] <= node2->x[0])
      soft_traverse_kdt(node1->left, node2, job, isauto, scratch);
  }
  else {
    if (!isauto || node1->x[0] <= node2->right->x[0])
      soft_traverse_kdt(node1, node2->right, job, isauto, scratch);
    if (!isauto || node1->x[0] <= node2->left->x[0])
      soft_traverse_kdt(node1, node2->left, job, isauto, scratch);
  }
}

/* Ball tree traversal, replicating `paircnt_balltree_*` (dual_tree.c). */
static void soft_traverse_blt(const BLT *node1, const BLT *node2,
    const SOFT_JOB *job, const bool isauto, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  if (isauto && node1 == node2) {         /* the same node */
    const soft_node_dist_e flag = soft_cmp_single_blt(node1, conf);
    if (conf->strict_single) {
      if (flag == SOFT_NODE_DIST_OUT) return;
      if (node1->left == NULL) {
        soft_count_single(node1->x, node1->w, node1->n, job, scratch);
        return;
      }
    }
    else {
      if (flag == SOFT_NODE_DIST_IN || node1->left == NULL) {
        soft_count_single(node1->x, node1->w, node1->n, job, scratch);
        return;
      }
    }
    if (node1->left->x[0] <= node2->left->x[0])
      soft_traverse_blt(node1->left, node2->left, job, isauto, scratch);
    if (node1->left->x[0] <= node2->right->x[0])
      soft_traverse_blt(node1->left, node2->right, job, isauto, scratch);
    if (node1->right->x[0] <= node2->left->x[0])
      soft_traverse_blt(node1->right, node2->left, job, isauto, scratch);
    if (node1->right->x[0] <= node2->right->x[0])
      soft_traverse_blt(node1->right, node2->right, job, isauto, scratch);
    return;
  }

  real shift[3] = {0, 0, 0};
  const soft_node_dist_e flag = soft_cmp_dual_blt(node1, node2, conf, shift);

  if (flag == SOFT_NODE_DIST_OUT) return;
  if (flag == SOFT_NODE_DIST_IN) {
    soft_count_dual(node1->x, node1->w, node1->n, node2->x, node2->w,
        node2->n, shift, job, scratch);
    return;
  }
  if (node1->left == NULL && node2->left == NULL) {
    if (flag != SOFT_NODE_DIST_X_WRAP) {
      soft_count_dual(node1->x, node1->w, node1->n, node2->x, node2->w,
          node2->n, shift, job, scratch);
    }
    else {
      soft_count_dual_wrap(node1->x, node1->w, node1->n, node2->x,
          node2->w, node2->n, job, scratch);
    }
    return;
  }
  if (node1->left && node2->left) {
    if (!isauto || node1->right->x[0] <= node2->right->x[0])
      soft_traverse_blt(node1->right, node2->right, job, isauto, scratch);
    if (!isauto || node1->right->x[0] <= node2->left->x[0])
      soft_traverse_blt(node1->right, node2->left, job, isauto, scratch);
    if (!isauto || node1->left->x[0] <= node2->right->x[0])
      soft_traverse_blt(node1->left, node2->right, job, isauto, scratch);
    if (!isauto || node1->left->x[0] <= node2->left->x[0])
      soft_traverse_blt(node1->left, node2->left, job, isauto, scratch);
  }
  else if (node1->left) {
    if (!isauto || node1->right->x[0] <= node2->x[0])
      soft_traverse_blt(node1->right, node2, job, isauto, scratch);
    if (!isauto || node1->left->x[0] <= node2->x[0])
      soft_traverse_blt(node1->left, node2, job, isauto, scratch);
  }
  else {
    if (!isauto || node1->x[0] <= node2->right->x[0])
      soft_traverse_blt(node1, node2->right, job, isauto, scratch);
    if (!isauto || node1->x[0] <= node2->left->x[0])
      soft_traverse_blt(node1, node2->left, job, isauto, scratch);
  }
}

/* Dispatch a seed node pair to the traversal for the tree type. */
static inline void soft_traverse(const void *node1, const void *node2,
    const int treetype, const SOFT_JOB *job, const bool isauto,
    double *scratch) {
  if (treetype == FCFC_STRUCT_KDTREE)
    soft_traverse_kdt((const KDT *) node1, (const KDT *) node2, job,
        isauto, scratch);
  else
    soft_traverse_blt((const BLT *) node1, (const BLT *) node2, job,
        isauto, scratch);
}

/* Expand the trees to seed node pairs and run the traversal over the
 * threads; shared by the forward and adjoint drivers. */
static int soft_run_parallel(const void *tree1, const void *tree2,
    const CF *cf, const SOFT_JOB *job, const bool isauto) {
#ifdef OMP
  const int nthread = cf->nthread;
  const int treetype = cf->treetype;
  const size_t sstride = ((2 * ((size_t) cf->ns + (size_t) (cf->nmu > cf->np ?
          cf->nmu : cf->np)) * sizeof(double) + 63) / 64) *
          (64 / sizeof(double));
  double *scratch = NULL;
  if (posix_memalign((void **) &scratch, 64,
      (size_t) nthread * sstride * sizeof(double))) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    return FCFC_ERR_MEMORY;
  }

  size_t nnodes = (size_t) nthread * FCFC_STACK_SIZE_PER_THREAD;
  if (nnodes > (UINT32_MAX >> 1)) nnodes = UINT32_MAX >> 1;

  int status = 0;
  void **nodes1 = NULL, **nodes2 = NULL;
  size_t nnodes2 = nnodes;
  if (treetype == FCFC_STRUCT_KDTREE) {
    nodes1 = (void **) kdtree_get_nodes((const KDT *) tree1,
        (uint32_t) nnodes, &nnodes);
    if (!isauto)
      nodes2 = (void **) kdtree_get_nodes((const KDT *) tree2,
          (uint32_t) nnodes2, &nnodes2);
  }
  else {
    nodes1 = (void **) balltree_get_nodes((const BLT *) tree1,
        (uint32_t) nnodes, &nnodes);
    if (!isauto)
      nodes2 = (void **) balltree_get_nodes((const BLT *) tree2,
          (uint32_t) nnodes2, &nnodes2);
  }
  if (!nodes1 || (!isauto && !nodes2)) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    status = FCFC_ERR_MEMORY;
  }
  else {
#pragma omp parallel num_threads(nthread)
    {
      const int tid = omp_get_thread_num();
      double *scr = scratch + (size_t) tid * sstride;
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < nnodes; i++) {
        if (isauto) {
          for (size_t j = 0; j < nnodes; j++) {
            const void *n1 = nodes1[i];
            const void *n2 = nodes1[j];
            const bool le = (treetype == FCFC_STRUCT_KDTREE) ?
                (((const KDT *) n1)->x[0] <= ((const KDT *) n2)->x[0]) :
                (((const BLT *) n1)->x[0] <= ((const BLT *) n2)->x[0]);
            if (le) soft_traverse(n1, n2, treetype, job, true, scr);
          }
        }
        else {
          for (size_t j = 0; j < nnodes2; j++)
            soft_traverse(nodes1[i], nodes2[j], treetype, job, false, scr);
        }
      }
    }
  }
  if (nodes1) free(nodes1);
  if (nodes2) free(nodes2);
  free(scratch);
  return status;
#else
  /* Serial code: a single scratch buffer. */
  double *scratch = malloc(2 * ((size_t) cf->ns + (size_t) (cf->nmu > cf->np ?
          cf->nmu : cf->np)) * sizeof(double));
  if (!scratch) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    return FCFC_ERR_MEMORY;
  }
  soft_traverse(tree1, tree2, cf->treetype, job, isauto, scratch);
  free(scratch);
  return 0;
#endif
}



/*============================================================================*\
     Index-preserving k-d tree for the adjoint traversal

  FCFC's counting trees store permuted coordinate copies without an index
  map, which is fine for counting but not for accumulating per-object
  cotangents.  The adjoint therefore builds its own k-d tree over index
  arrays (median splits, no coordinate copies), reusing the node distance
  criteria of the forward kernels; the node structures start with the same
  `cen` / `wid` pointer layout as KDT so that the compare functions can be
  shared.
\*============================================================================*/

typedef struct soft_node_struct {
  real *cen;                            /* centre of the box        */
  real *wid;                            /* half-width of the box    */
  size_t n;                             /* number of objects        */
  size_t *idx;                          /* catalogue indices (leaves;
                                         * owned iff owns_idx)      */
  int owns_idx;                         /* idx ownership flag       */
  size_t key;                           /* canonical ordering key:
                                         * first index at build time*/
  real cenbuf[3];
  real widbuf[3];
  struct soft_node_struct *left;
  struct soft_node_struct *right;
} SOFT_NODE;

/* Select the index of the k-th smallest key into position k (Hoare). */
static void soft_qselect(size_t *idx, const real *key, const size_t n,
    const size_t k) {
  size_t lo = 0, hi = n - 1;
  while (hi > lo) {
    size_t i = lo, j = hi;
    const real p = key[idx[(lo + hi) >> 1]];
    while (i <= j) {
      while (key[idx[i]] < p) i++;
      while (key[idx[j]] > p) j--;
      if (i <= j) {
        const size_t t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
        i++;
        if (j == 0) break;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
}

static void soft_node_bounds(real *const *x, const size_t *idx,
    const size_t n, real *cen, real *wid) {
  for (int d = 0; d < 3; d++) {
    real mn = x[d][idx[0]], mx = mn;
    for (size_t i = 1; i < n; i++) {
      const real v = x[d][idx[i]];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    cen[d] = (mn + mx) * 0.5;
    wid[d] = (mx - mn) * 0.5;
  }
}

static void soft_node_destroy(SOFT_NODE *nd);

static SOFT_NODE *soft_node_build(real *const *x, size_t *idx, const size_t n,
    const size_t leaf) {
  SOFT_NODE *nd = calloc(1, sizeof(SOFT_NODE));
  if (!nd) {
    free(idx);                    /* ownership stays with the caller
                                   * until the node exists */
    return NULL;
  }
  nd->cen = nd->cenbuf;
  nd->wid = nd->widbuf;
  nd->n = n;
  nd->idx = idx;
  nd->owns_idx = 1;
  nd->key = idx[0];
  soft_node_bounds(x, idx, n, nd->cenbuf, nd->widbuf);
  if (n > leaf) {
    int dim = 0;
    if (nd->widbuf[1] > nd->widbuf[dim]) dim = 1;
    if (nd->widbuf[2] > nd->widbuf[dim]) dim = 2;
    if (nd->widbuf[dim] > 0) {
      const size_t half = n >> 1;
      soft_qselect(idx, x[dim], n, half);
      size_t *idx2 = malloc(sizeof(size_t) * (n - half));
      if (idx2) {
        memcpy(idx2, idx + half, sizeof(size_t) * (n - half));
        SOFT_NODE *l = soft_node_build(x, idx, half, leaf);
        SOFT_NODE *r = soft_node_build(x, idx2, n - half, leaf);
        if (l && r) {
          nd->left = l;
          nd->right = r;
          nd->idx = l->idx;       /* non-owning: only for completeness;
                                   * internal nodes are keyed by `key' */
          nd->owns_idx = 0;
        }
        else {
          /* Failed split: stay a leaf; release whatever came back. */
          if (l) soft_node_destroy(l);
          else free(idx);
          if (r) soft_node_destroy(r);
          else free(idx2);
          nd->idx = NULL;
          nd->owns_idx = 0;
          /* Rebuild a private index array for the leaf. */
          nd->idx = malloc(sizeof(size_t) * n);
          if (nd->idx) {
            for (size_t i = 0; i < n; i++) nd->idx[i] = i;   /* placeholder */
            nd->owns_idx = 1;
          }
        }
      }
    }
  }
  return nd;
}

static void soft_node_destroy(SOFT_NODE *nd) {
  if (!nd) return;
  soft_node_destroy(nd->left);
  soft_node_destroy(nd->right);
  if (nd->owns_idx && nd->idx) free(nd->idx);
  free(nd);
}

/* Collect all nodes at a fixed depth (leaves early) as parallel work
 * seeds: the collected node sets partition each tree, hence the seed
 * pairs partition the pair space.  A level holds at most 2^depth nodes. */
static size_t soft_node_seeds(SOFT_NODE *nd, const size_t depth,
    SOFT_NODE **list, size_t n) {
  if (!nd) return n;
  if (depth == 0 || (!nd->left && !nd->right)) {
    list[n++] = nd;
    return n;
  }
  n = soft_node_seeds(nd->left, depth - 1, list, n);
  n = soft_node_seeds(nd->right, depth - 1, list, n);
  return n;
}

static SOFT_NODE **soft_tree_seeds(SOFT_NODE *root, const size_t target,
    size_t *nnodes) {
  size_t depth = 0;
  while (((size_t) 1 << depth) < target) depth++;
  const size_t cap = (size_t) 1 << depth;
  SOFT_NODE **list = malloc(sizeof(SOFT_NODE *) * cap);
  if (!list) return NULL;
  *nnodes = soft_node_seeds(root, depth, list, 0);
  return list;
}


/*============================================================================*\
        Leaf pair kernels for the adjoint (index-preserving tree)
\*============================================================================*/

static void soft_adj_single(const SOFT_NODE *nd, real *const *x,
    const real *w, const SOFT_JOB *job, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  const bool hw = (w != NULL);
  for (size_t i = 0; i + 1 < nd->n; i++) {
    const size_t I = nd->idx[i];
    const real wti = hw ? w[I] : (real) 1;
    for (size_t j = i + 1; j < nd->n; j++) {
      const size_t J = nd->idx[j];
      const real wtj = hw ? w[J] : (real) 1;
      const real dx = x[0][I] - x[0][J];
      const real dy = x[1][I] - x[1][J];
      const real dz = x[2][I] - x[2][J];
      soft_bin_pair_adj(conf, dx, dy, dz, x[0][I], x[1][I], x[2][I],
          x[0][J], x[1][J], x[2][J], wti, wtj, I, J, job, scratch);
    }
  }
}

static void soft_adj_dual(const SOFT_NODE *n1, const SOFT_NODE *n2,
    real *const *a, const real *wa, real *const *b, const real *wb,
    const real shift[3], const SOFT_JOB *job, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  const bool ha = (wa != NULL);
  const bool hb = (wb != NULL);
  for (size_t i = 0; i < n1->n; i++) {
    const size_t I = n1->idx[i];
    const real ax = a[0][I] + shift[0];
    const real ay = a[1][I] + shift[1];
    const real az = a[2][I] + shift[2];
    const real wti = ha ? wa[I] : (real) 1;
    for (size_t j = 0; j < n2->n; j++) {
      const size_t J = n2->idx[j];
      const real wtj = hb ? wb[J] : (real) 1;
      soft_bin_pair_adj(conf, ax - b[0][J], ay - b[1][J], az - b[2][J],
          ax, ay, az, b[0][J], b[1][J], b[2][J], wti, wtj, I, J, job,
          scratch);
    }
  }
}

static void soft_adj_dual_wrap(const SOFT_NODE *n1, const SOFT_NODE *n2,
    real *const *a, const real *wa, real *const *b, const real *wb,
    const SOFT_JOB *job, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  const bool ha = (wa != NULL);
  const bool hb = (wb != NULL);
  for (size_t i = 0; i < n1->n; i++) {
    const size_t I = n1->idx[i];
    const real wti = ha ? wa[I] : (real) 1;
    for (size_t j = 0; j < n2->n; j++) {
      const size_t J = n2->idx[j];
      const real wtj = hb ? wb[J] : (real) 1;
      real dx = a[0][I] - b[0][J];
      if (dx > conf->hsize[0]) dx -= conf->bsize[0];
      else if (dx < -conf->hsize[0]) dx += conf->bsize[0];
      real dy = a[1][I] - b[1][J];
      if (dy > conf->hsize[1]) dy -= conf->bsize[1];
      else if (dy < -conf->hsize[1]) dy += conf->bsize[1];
      real dz = a[2][I] - b[2][J];
      if (dz > conf->hsize[2]) dz -= conf->bsize[2];
      else if (dz < -conf->hsize[2]) dz += conf->bsize[2];
      soft_bin_pair_adj(conf, dx, dy, dz, a[0][I], a[1][I], a[2][I],
          b[0][J], b[1][J], b[2][J], wti, wtj, I, J, job, scratch);
    }
  }
}

/* Adjoint traversal on the index-preserving tree.  The node distance
 * criteria are shared with the forward kernels (SOFT_NODE starts with the
 * same cen / wid pointer layout as KDT). */
static void soft_adj_traverse(const SOFT_NODE *n1, const SOFT_NODE *n2,
    real *const *a, const real *wa, real *const *b, const real *wb,
    const SOFT_JOB *job, const bool isauto, double *scratch) {
  const SOFT_CONF *conf = job->conf;
  if (isauto && n1 == n2) {
    const soft_node_dist_e flag =
        soft_cmp_single_kdt((const KDT *) n1, conf);
    if (conf->strict_single) {
      if (flag == SOFT_NODE_DIST_OUT) return;
      if (n1->left == NULL) {
        soft_adj_single(n1, a, wa, job, scratch);
        return;
      }
    }
    else {
      if (flag == SOFT_NODE_DIST_IN || n1->left == NULL) {
        soft_adj_single(n1, a, wa, job, scratch);
        return;
      }
    }
    /* Each unordered child pair is visited exactly once:
     * (L,L), (L,R) and (R,R); the reversed (R,L) is omitted. */
    soft_adj_traverse(n1->left, n2->left, a, wa, b, wb, job, isauto,
          scratch);
    soft_adj_traverse(n1->left, n2->right, a, wa, b, wb, job, isauto,
          scratch);
    soft_adj_traverse(n1->right, n2->right, a, wa, b, wb, job, isauto,
          scratch);
    return;
  }

  real shift[3] = {0, 0, 0};
  const soft_node_dist_e flag =
      soft_cmp_dual_kdt((const KDT *) n1, (const KDT *) n2, conf, shift);
  if (flag == SOFT_NODE_DIST_OUT) return;
  if (flag == SOFT_NODE_DIST_IN) {
    soft_adj_dual(n1, n2, a, wa, b, wb, shift, job, scratch);
    return;
  }
  if (n1->left == NULL && n2->left == NULL) {
    if (flag != SOFT_NODE_DIST_X_WRAP)
      soft_adj_dual(n1, n2, a, wa, b, wb, shift, job, scratch);
    else
      soft_adj_dual_wrap(n1, n2, a, wa, b, wb, job, scratch);
    return;
  }
  if (n1->left && n2->left) {
    soft_adj_traverse(n1->right, n2->right, a, wa, b, wb, job, isauto,
          scratch);
    soft_adj_traverse(n1->right, n2->left, a, wa, b, wb, job, isauto,
          scratch);
    soft_adj_traverse(n1->left, n2->right, a, wa, b, wb, job, isauto,
          scratch);
    soft_adj_traverse(n1->left, n2->left, a, wa, b, wb, job, isauto,
          scratch);
  }
  else if (n1->left) {
    soft_adj_traverse(n1->right, n2, a, wa, b, wb, job, isauto, scratch);
    soft_adj_traverse(n1->left, n2, a, wa, b, wb, job, isauto, scratch);
  }
  else {
    soft_adj_traverse(n1, n2->right, a, wa, b, wb, job, isauto, scratch);
    soft_adj_traverse(n1, n2->left, a, wa, b, wb, job, isauto, scratch);
  }
}
/*============================================================================*\
                        Interface for pair counting
\*============================================================================*/

int soft_count_pairs(const void *tree1, const void *tree2, CF *cf, COUNT *cnt,
    const bool isauto, const bool withwt
#ifdef MPI
    , const PARA *para
#endif
    ) {
#ifdef MPI
  (void) tree1; (void) tree2; (void) cf; (void) cnt;
  (void) isauto; (void) withwt; (void) para;
  P_ERR("soft histogramming (SOFT_HIST) does not support MPI yet\n");
  return FCFC_ERR_ARG;
#else
  /* Soft counts are fractional: they are always accumulated as doubles,
   * independently of the `withwt` flag (cf_setup forces cf->wt in soft
   * mode, so that the results are read back as doubles). */
  (void) withwt;

  SOFT_CONF sconf;
  int e = soft_conf_init(&sconf, cf);
  if (e) return e;

#ifdef OMP
  const int nthread = cf->nthread;
  const size_t ntot = cf->ntot;
  /* Thread-private buffers: 64-byte aligned bases and strides padded to
   * cache line multiples, so that the counting threads never share a
   * cache line (false sharing would serialise the pair counting). */
  const size_t stride = ((ntot * sizeof(double) + 63) / 64) *
      (64 / sizeof(double));
  double *pcnt = NULL;
  if (posix_memalign((void **) &pcnt, 64,
      (size_t) nthread * stride * sizeof(double))) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    soft_conf_destroy(&sconf);
    return FCFC_ERR_MEMORY;
  }
  memset(pcnt, 0, (size_t) nthread * stride * sizeof(double));

  SOFT_JOB job;
  memset(&job, 0, sizeof job);
  job.conf = &sconf;
  job.adjoint = 0;
  job.cnt = NULL;               /* set per thread below */

  /* The per-thread count buffer is selected inside the parallel region,
   * so run the seeding loop by hand here. */
  const int treetype = cf->treetype;
  const size_t sstride = ((2 * ((size_t) cf->ns + (size_t) (cf->nmu > cf->np ?
          cf->nmu : cf->np)) * sizeof(double) + 63) / 64) *
          (64 / sizeof(double));
  double *scratch = NULL;
  if (posix_memalign((void **) &scratch, 64,
      (size_t) nthread * sstride * sizeof(double))) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    free(pcnt);
    soft_conf_destroy(&sconf);
    return FCFC_ERR_MEMORY;
  }

  size_t nnodes = (size_t) nthread * FCFC_STACK_SIZE_PER_THREAD;
  if (nnodes > (UINT32_MAX >> 1)) nnodes = UINT32_MAX >> 1;

  int status = 0;
  void **nodes1 = NULL, **nodes2 = NULL;
  size_t nnodes2 = nnodes;
  if (treetype == FCFC_STRUCT_KDTREE) {
    nodes1 = (void **) kdtree_get_nodes((const KDT *) tree1,
        (uint32_t) nnodes, &nnodes);
    if (!isauto)
      nodes2 = (void **) kdtree_get_nodes((const KDT *) tree2,
          (uint32_t) nnodes2, &nnodes2);
  }
  else {
    nodes1 = (void **) balltree_get_nodes((const BLT *) tree1,
        (uint32_t) nnodes, &nnodes);
    if (!isauto)
      nodes2 = (void **) balltree_get_nodes((const BLT *) tree2,
          (uint32_t) nnodes2, &nnodes2);
  }
  if (!nodes1 || (!isauto && !nodes2)) {
    P_ERR("failed to allocate memory for soft histogramming\n");
    status = FCFC_ERR_MEMORY;
  }
  else {
#pragma omp parallel num_threads(nthread)
    {
      const int tid = omp_get_thread_num();
      SOFT_JOB jjob = job;
      jjob.cnt = pcnt + (size_t) tid * stride;
      double *scr = scratch + (size_t) tid * sstride;
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < nnodes; i++) {
        if (isauto) {
          for (size_t j = 0; j < nnodes; j++) {
            const void *n1 = nodes1[i];
            const void *n2 = nodes1[j];
            const bool le = (treetype == FCFC_STRUCT_KDTREE) ?
                (((const KDT *) n1)->x[0] <= ((const KDT *) n2)->x[0]) :
                (((const BLT *) n1)->x[0] <= ((const BLT *) n2)->x[0]);
            if (le) soft_traverse(n1, n2, treetype, &jjob, true, scr);
          }
        }
        else {
          for (size_t j = 0; j < nnodes2; j++)
            soft_traverse(nodes1[i], nodes2[j], treetype, &jjob, false, scr);
        }
      }
    }
  }
  if (nodes1) free(nodes1);
  if (nodes2) free(nodes2);

  if (!status) {
    /* Gather pair counts from different threads. */
#pragma omp parallel for schedule(static) num_threads(nthread)
    for (size_t k = 0; k < ntot; k++) {
      double s = 0;
      for (int t = 0; t < nthread; t++) s += pcnt[(size_t) t * stride + k];
      cnt[k].d += s;
    }
  }

  free(pcnt);
  free(scratch);
  soft_conf_destroy(&sconf);
  return status;
#else
  /* Serial code. */
  SOFT_JOB job;
  memset(&job, 0, sizeof job);
  job.conf = &sconf;
  job.adjoint = 0;
  job.cnt = (double *) cnt;     /* zero-initialised by cf_setup; COUNT.d
                                 * is at offset 0 */
  const int e2 = soft_run_parallel(tree1, tree2, cf, &job, isauto);
  soft_conf_destroy(&sconf);
  return e2;
#endif
#endif  /* MPI */
}


/*============================================================================*\
                     Interface for the adjoint (VJP)
\*============================================================================*/

/******************************************************************************
Function `soft_hist_vjp`:
  Reverse-mode derivative of `soft_count_pairs`: given the cotangents `g`
  of a scalar loss with respect to the (raw, unnormalised) pair count bins
  of the pair count formed by catalogues `ia` and `ib`, accumulate the
  cotangents of the catalogue coordinates (in internal units; the caller
  rescales them), and optionally of the catalogue weights and of the
  kernel width sigma.  The adjoint builds its own index-preserving k-d
  tree and replays the forward pair selection and kernel windows exactly;
  auto pair counts follow the ordered-pair convention of FCFC (each
  unordered pair counted twice), so the cotangents are consistent with the
  doubled raw auto counts.
Arguments:
  * `cf`:       structure for configurations of correlation functions;
  * `g`:        cotangents of the pair count bins (cf->ntot elements);
  * `gx1`:      coordinate cotangents of catalogue `ia` (3 * n elements,
                internal units, accumulated);
  * `gx2`:      coordinate cotangents of catalogue `ib`;
  * `gw1`:      weight cotangents of catalogue `ia` (or NULL);
  * `gw2`:      weight cotangents of catalogue `ib` (or NULL);
  * `gsigma`:   kernel width cotangent (scalar, or NULL);
  * `ia`, `ib`: indices of the two catalogues of the pair count.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
int soft_hist_vjp(CF *cf, const double *g, double *gx1, double *gx2,
    double *gw1, double *gw2, double *gsigma, double *gmusigma,
    double *gpisigma, const int ia, const int ib
#ifdef MPI
    , const PARA *para
#endif
    ) {
#ifdef MPI
  (void) cf; (void) g; (void) gx1; (void) gx2; (void) gw1; (void) gw2;
  (void) gsigma; (void) gmusigma; (void) gpisigma; (void) ia;
  (void) ib; (void) para;
  P_ERR("soft histogramming (SOFT_HIST) does not support MPI yet\n");
  return FCFC_ERR_ARG;
#else
  if (!g || !gx1 || !gx2 || ia < 0 || ib < 0) return FCFC_ERR_ARG;
  const bool isauto = (ia == ib);

  SOFT_CONF sconf;
  int e = soft_conf_init(&sconf, cf);
  if (e) return e;

  SOFT_JOB job;
  memset(&job, 0, sizeof job);
  job.conf = &sconf;
  job.adjoint = 1;
  job.g = g;
  job.gx1 = gx1;
  job.gx2 = gx2;
  job.gw1 = gw1;
  job.gw2 = gw2;
  job.gsig = gsigma;
  job.gmus = gmusigma;
  job.gpi = gpisigma;
  /* FCFC reports auto pair counts doubled (ordered pairs); the VJP of
   * those counts carries the same factor. */
  job.wscale = isauto ? 2.0 : 1.0;

  DATA *da = cf->data + ia;
  DATA *db = cf->data + ib;
  int status = FCFC_ERR_MEMORY;

  /* The adjoint builds its own trees from cf->data, which still holds the
   * coordinates in input units: replicate the tree-construction
   * preprocessing by rescaling the C-owned copies in place (the caller's
   * arrays are untouched). */
  for (int d = 0; d < 3; d++)
    for (size_t i = 0; i < da->n; i++) da->x[d][i] *= cf->rescale;
  if (!isauto)
    for (int d = 0; d < 3; d++)
      for (size_t i = 0; i < db->n; i++) db->x[d][i] *= cf->rescale;

  size_t *idxa = malloc(sizeof(size_t) * (da->n ? da->n : 1));
  size_t *idxb = isauto ? idxa :
      malloc(sizeof(size_t) * (db->n ? db->n : 1));
  SOFT_NODE *ta = NULL, *tb = NULL;
  if (idxa && idxb) {
    for (size_t i = 0; i < da->n; i++) idxa[i] = i;
    if (!isauto)
      for (size_t i = 0; i < db->n; i++) idxb[i] = i;
    ta = soft_node_build(da->x, idxa, da->n, FCFC_KDTREE_LEAF_SIZE);
    idxa = NULL;                  /* ownership transferred */
    if (isauto) {
      tb = ta;
      idxb = NULL;
    }
    else {
      tb = soft_node_build(db->x, idxb, db->n, FCFC_KDTREE_LEAF_SIZE);
      idxb = NULL;
    }
  }
  if (idxa) free(idxa);
  if (idxb) free(idxb);

  if (ta && tb) {
    status = 0;
#ifdef OMP
    const int nthread = cf->nthread;
    const size_t sstride = ((2 * ((size_t) cf->ns + (size_t) (cf->nmu >
        cf->np ? cf->nmu : cf->np)) * sizeof(double) + 63) / 64) *
        (64 / sizeof(double));
    double *scratch = NULL;
    if (posix_memalign((void **) &scratch, 64,
        (size_t) nthread * sstride * sizeof(double))) {
      soft_node_destroy(ta);
      if (!isauto) soft_node_destroy(tb);
      soft_conf_destroy(&sconf);
      return FCFC_ERR_MEMORY;
    }
    const size_t target = (size_t) nthread * FCFC_STACK_SIZE_PER_THREAD;
    size_t na = 0, nb = 0;
    SOFT_NODE **la = soft_tree_seeds(ta, target, &na);
    SOFT_NODE **lb = isauto ? la : soft_tree_seeds(tb, target, &nb);
    if (la && lb) {
#pragma omp parallel num_threads(nthread)
      {
        const int tid = omp_get_thread_num();
        double *scr = scratch + (size_t) tid * sstride;
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < na; i++) {
          if (isauto) {
            for (size_t j = i; j < na; j++)
              soft_adj_traverse(la[i], la[j], da->x, da->w, da->x, da->w,
                  &job, true, scr);
          }
          else {
            for (size_t j = 0; j < nb; j++)
              soft_adj_traverse(la[i], lb[j], da->x, da->w, db->x, db->w,
                  &job, false, scr);
          }
        }
      }
    }
    else status = FCFC_ERR_MEMORY;
    if (la) free(la);
    if (!isauto && lb) free(lb);
    free(scratch);
#else
    double *scratch = malloc(2 * ((size_t) cf->ns + (size_t) (cf->nmu > cf->np ?
          cf->nmu : cf->np)) * sizeof(double));
    if (!scratch) status = FCFC_ERR_MEMORY;
    else {
      soft_adj_traverse(ta, tb, da->x, da->w, db->x, db->w, &job, isauto,
          scratch);
      free(scratch);
    }
#endif
  }
  soft_node_destroy(ta);
  if (!isauto) soft_node_destroy(tb);
  soft_conf_destroy(&sconf);
  return status;
#endif  /* MPI */
}
