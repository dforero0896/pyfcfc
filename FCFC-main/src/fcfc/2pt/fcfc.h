/*******************************************************************************
* 2pt/fcfc.h: this file is part of the FCFC program, modified for the
* pyfcfc Python wrapper.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#ifndef _FCFC_H_
#define _FCFC_H_

#include "eval_cf.h"
#include "define_para.h"

/******************************************************************************
Function `compute_cf':
  Evaluate pair counts and correlation functions from in-memory catalogues.
  On success, the returned CF structure owns the input data `dat', and must
  be released with `cf_destroy'.  On failure, NULL is returned and all the
  inputs (including `dat' and its coordinate/weight arrays) are released.
Arguments:
  * `argc':     number of command-line-style configuration arguments;
  * `argv':     command-line-style configuration arguments;
  * `dat':      array of input catalogues (ownership transferred on success);
  * `ncat':     number of input catalogues;
  * `sbins':    edges of the separation (or s_perp) bins, length `ns' + 1;
  * `ns':       number of separation (or s_perp) bins;
  * `pbins':    edges of the pi bins, length `np' + 1 (NULL if unused);
  * `np':       number of pi bins;
  * `nmu':      number of mu bins (used by the (s, mu) binning scheme).
Return:
  Address of the structure for correlation function evaluations on success;
  NULL on error.
******************************************************************************/
CF *compute_cf(int argc, char *argv[], DATA *dat, int ncat,
    real *sbins, int ns, real *pbins, int np, int nmu);


/******************************************************************************
Function `compute_cf_soft_vjp':
  Reverse-mode derivative of the soft histogramming pair counts: given the
  cotangents `g' of a scalar loss with respect to the raw bins of one pair
  count (`pair', e.g. "DD"), evaluate the cotangents of the input catalogue
  coordinates (and optionally of the weights and of the kernel width).
  The configuration is parsed and the trees are built exactly as in
  `compute_cf'; the cotangents are returned in the units of the input
  coordinates.  `gx', `gw' hold two entries (the two catalogues of the
  pair; identical buffers for auto pairs); NULL entries disable the
  respective output (`gw', `gsigma').
Return:
  Zero on success; non-zero on error.
******************************************************************************/
int compute_cf_soft_vjp(int argc, char *argv[], DATA *dat, int ncat,
    real *sbins, int ns, real *pbins, int np, int nmu,
    const char *pair, const double *g,
    double *gx[], double *gw[], double *gsigma, double *gmusigma,
    double *gpisigma);

#endif
