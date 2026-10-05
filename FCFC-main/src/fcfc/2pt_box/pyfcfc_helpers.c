/*******************************************************************************
* pyfcfc_helpers.c: helper layer between the FCFC C library and the pyfcfc
* Cython bindings.  See pyfcfc_helpers.h for the API description.

* FCFC: Fast Correlation Function Calculator.
* Github repository:  https://github.com/cheng-zhao/FCFC
* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]
*******************************************************************************/

#include "pyfcfc_helpers.h"
#include "define_para.h"
#include <stdlib.h>
#include <string.h>
#ifdef OMP
#include <omp.h>
#endif

#define PYFCFC_ARG0 "FCFC_BOX"

const char *pyfcfc_arg0(void) {
  return PYFCFC_ARG0;
}

int pyfcfc_simd_level(void) {
  return FCFC_SIMD;
}

const char *pyfcfc_simd_name(void) {
#if   FCFC_SIMD == FCFC_SIMD_NONE
  return "none";
#elif FCFC_SIMD == FCFC_SIMD_AVX
  return "AVX";
#elif FCFC_SIMD == FCFC_SIMD_AVX2
  return "AVX2";
#else
  return "AVX512";
#endif
}

DATA *pyfcfc_data_alloc(int ncat) {
  if (ncat < 1) return NULL;
  DATA *dat = calloc((size_t) ncat, sizeof(DATA));
  if (!dat) return NULL;
  for (int i = 0; i < ncat; i++) {
    dat[i].x[0] = dat[i].x[1] = dat[i].x[2] = NULL;
    dat[i].w = NULL;
    dat[i].n = 0;
    dat[i].wt = dat[i].w2 = 0;
  }
  return dat;
}

/* Internal worker: copy from interleaved single- or double-precision buffers.
 * `is_float' indicates the precision of the inputs.
 * Under SIMD compilations, the counting kernels may read up to
 * FCFC_NUM_REAL elements beyond the end of the coordinate and weight
 * arrays, so the allocations are padded and the padding is zeroed,
 * exactly as the original FCFC input routines do. */
static int data_fill_impl(DATA *dat, int idx, size_t n, const void *xyz,
    const void *w, int is_float) {
  if (!dat || !xyz || !w || n == 0) return -1;
  DATA *d = dat + idx;
  d->n = n;
#ifdef FCFC_NUM_REAL
  size_t npad = n + FCFC_NUM_REAL;
#else
  size_t npad = n;
#endif
  /* Only the first 3 dimensions are provided by the user.  For the
   * survey-like (2pt) component, FCFC_XDIM is 4 and the 4th dimension is
   * allocated internally by the tree creation routines. */
  for (int j = 0; j < 3; j++) {
    if (!(d->x[j] = malloc(npad * sizeof(real)))) return -1;
    if (npad > n) memset(d->x[j] + n, 0, sizeof(real) * (npad - n));
  }
  if (!(d->w = malloc(npad * sizeof(real)))) return -1;
  if (npad > n) memset(d->w + n, 0, sizeof(real) * (npad - n));

#ifdef OMP
#pragma omp parallel default(none) shared(d, xyz, w, n, is_float)
  {
#pragma omp for
#endif
    for (size_t i = 0; i < n; i++) {
      /* The user-supplied coordinates always have 3 columns (stride 3),
       * even for the survey-like component where FCFC_XDIM is 4. */
      if (is_float) {
        const float *p = (const float *) xyz + i * 3;
        d->x[0][i] = (real) p[0];
        d->x[1][i] = (real) p[1];
        d->x[2][i] = (real) p[2];
        d->w[i] = (real) ((const float *) w)[i];
      }
      else {
        const double *p = (const double *) xyz + i * 3;
        d->x[0][i] = (real) p[0];
        d->x[1][i] = (real) p[1];
        d->x[2][i] = (real) p[2];
        d->w[i] = (real) ((const double *) w)[i];
      }
    }
#ifdef OMP
  }
#endif
  return 0;
}

int pyfcfc_data_fill(DATA *dat, int idx, size_t n, const double *xyz,
    const double *w) {
  return data_fill_impl(dat, idx, n, xyz, w, 0);
}

int pyfcfc_data_fill_f(DATA *dat, int idx, size_t n, const float *xyz,
    const float *w) {
  return data_fill_impl(dat, idx, n, xyz, w, 1);
}

void pyfcfc_data_free(DATA *dat, int ncat) {
  if (!dat) return;
  for (int i = 0; i < ncat; i++) {
    for (int j = 0; j < FCFC_XDIM; j++)
      if (dat[i].x[j]) free(dat[i].x[j]);
    if (dat[i].w) free(dat[i].w);
  }
  free(dat);
}

int pyfcfc_cf_ncat(const CF *cf) { return cf->ncat; }
int pyfcfc_cf_ns(const CF *cf) { return cf->ns; }
int pyfcfc_cf_np(const CF *cf) { return cf->np; }
int pyfcfc_cf_nmu(const CF *cf) { return cf->nmu; }
int pyfcfc_cf_npc(const CF *cf) { return cf->npc; }
int pyfcfc_cf_ncf(const CF *cf) { return cf->ncf; }
int pyfcfc_cf_nl(const CF *cf) { return cf->nl; }
int pyfcfc_cf_bintype(const CF *cf) { return cf->bintype; }
int pyfcfc_cf_has_mp(const CF *cf) { return cf->mp != NULL; }
int pyfcfc_cf_has_wp(const CF *cf) { return cf->wp != NULL; }
char pyfcfc_cf_label(const CF *cf, int i) { return cf->label[i]; }
size_t pyfcfc_cf_data_n(const CF *cf, int i) { return cf->data[i].n; }
double pyfcfc_cf_data_wt(const CF *cf, int i) { return cf->data[i].wt; }
int pyfcfc_cf_pc_idx(const CF *cf, int i, int j) { return cf->pc_idx[j][i]; }
double pyfcfc_cf_norm(const CF *cf, int i) { return cf->norm[i]; }
int pyfcfc_cf_pole(const CF *cf, int i) { return cf->poles[i]; }

void pyfcfc_cf_sbin_raw(const CF *cf, double *dst) {
  for (int i = 0; i <= cf->ns; i++) dst[i] = (double) cf->sbin_raw[i];
}

void pyfcfc_cf_pbin_raw(const CF *cf, double *dst) {
  if (!cf->pbin_raw) return;
  for (int i = 0; i <= cf->np; i++) dst[i] = (double) cf->pbin_raw[i];
}

void pyfcfc_cf_ncnt(const CF *cf, int i, double *dst) {
  memcpy(dst, cf->ncnt[i], cf->ntot * sizeof(double));
}

void pyfcfc_cf_cfval(const CF *cf, int i, double *dst) {
  memcpy(dst, cf->cf[i], cf->ntot * sizeof(double));
}

void pyfcfc_cf_mp(const CF *cf, int i, double *dst) {
  memcpy(dst, cf->mp[i], (size_t) cf->nl * cf->ns * sizeof(double));
}

void pyfcfc_cf_wp(const CF *cf, int i, double *dst) {
  memcpy(dst, cf->wp[i], (size_t) cf->ns * sizeof(double));
}
