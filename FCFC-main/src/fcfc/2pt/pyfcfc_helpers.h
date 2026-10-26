/*******************************************************************************
* pyfcfc_helpers.h: helper layer between the FCFC C library and the pyfcfc
* Cython bindings.  All accesses to the internals of the `CF' and `DATA'
* structures go through these functions, so that the Cython side never needs
* to duplicate (and risk desynchronising) the structure layouts.

* FCFC: Fast Correlation Function Calculator.
* Github repository:  https://github.com/cheng-zhao/FCFC
* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]
*******************************************************************************/

#ifndef PYFCFC_HELPERS_H_
#define PYFCFC_HELPERS_H_

#include <stddef.h>
#include "eval_cf.h"
#include "fcfc.h"

/* Name of the "program", used as argv[0] by the configuration parser. */
const char *pyfcfc_arg0(void);

/* SIMD instruction set compiled into this build (FCFC_SIMD_* level and
 * a human-readable name), for introspection and benchmarking. */
int pyfcfc_simd_level(void);
const char *pyfcfc_simd_name(void);

/* Input data handling.  Coordinates are copied from interleaved (N, 3)
 * buffers into the structure-of-arrays layout expected by FCFC. */
DATA *pyfcfc_data_alloc(int ncat);
int pyfcfc_data_fill(DATA *dat, int idx, size_t n, const double *xyz,
    const double *w);
int pyfcfc_data_fill_f(DATA *dat, int idx, size_t n, const float *xyz,
    const float *w);
void pyfcfc_data_free(DATA *dat, int ncat);

/* Metadata of the correlation function evaluation. */
int pyfcfc_cf_ncat(const CF *cf);
int pyfcfc_cf_ns(const CF *cf);
int pyfcfc_cf_np(const CF *cf);
int pyfcfc_cf_nmu(const CF *cf);
int pyfcfc_cf_npc(const CF *cf);
int pyfcfc_cf_ncf(const CF *cf);
int pyfcfc_cf_nl(const CF *cf);
int pyfcfc_cf_bintype(const CF *cf);
int pyfcfc_cf_has_mp(const CF *cf);
int pyfcfc_cf_has_wp(const CF *cf);
char pyfcfc_cf_label(const CF *cf, int i);
size_t pyfcfc_cf_data_n(const CF *cf, int i);
double pyfcfc_cf_data_wt(const CF *cf, int i);
int pyfcfc_cf_pc_idx(const CF *cf, int i, int j);
double pyfcfc_cf_norm(const CF *cf, int i);
int pyfcfc_cf_pole(const CF *cf, int i);

/* Bin edges in input (unrescaled) units. */
void pyfcfc_cf_sbin_raw(const CF *cf, double *dst);
void pyfcfc_cf_pbin_raw(const CF *cf, double *dst);

/* Flattened result arrays (the caller allocates `dst'):
 * - pair counts and correlation functions: `cf->ntot' elements, with
 *   indices ordered as  s_idx + ns * (mu_idx | pi_idx);
 * - multipoles: nl * ns elements, ordered as  s_idx + ns * pole_idx;
 * - projected correlation functions: ns elements. */
void pyfcfc_cf_ncnt(const CF *cf, int i, double *dst);
void pyfcfc_cf_cfval(const CF *cf, int i, double *dst);
void pyfcfc_cf_mp(const CF *cf, int i, double *dst);
void pyfcfc_cf_wp(const CF *cf, int i, double *dst);


/* Reverse-mode derivative of the soft histogramming pair counts: `g'
 * holds the cotangents of the raw bins of the pair count `pair'; the
 * coordinate cotangents are written to `gx0' / `gx1' (3 * n arrays,
 * zero-initialised by the caller, input units on return); `gw0' / `gw1'
 * (n arrays, may be NULL) and `gsigma' (scalar, may be NULL) receive the
 * weight and kernel-width cotangents. */
int pyfcfc_soft_vjp(int argc, char *argv[], DATA *dat, int ncat,
    double *sbins, int ns, double *pbins, int np_, int nmu,
    const char *pair, const double *g,
    double *gx0, double *gx1, double *gw0, double *gw1, double *gsigma,
    double *gmusigma, double *gpisigma);

#endif
