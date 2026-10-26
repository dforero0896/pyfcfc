/*******************************************************************************
* 2pt_box/soft_hist.h: this file is part of the FCFC program.

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

#ifndef __SOFT_HIST_H__
#define __SOFT_HIST_H__

#include "eval_cf.h"

/*============================================================================*\\\
              Interface for soft histogramming pair counts
\*============================================================================*/

/******************************************************************************
Function `soft_count_pairs`:
  Count pairs with soft histogramming: instead of assigning each pair to a
  single separation bin, the pair weight is spread over all separation bins
  with normalised Gaussian-kernel (softmax) weights.  It is a drop-in
  replacement of `count_pairs` used when the SOFT_HIST option is enabled;
  pair counts are always accumulated as doubles (COUNT.d), thus `cf_setup`
  forces the weighted counting path (`cf->wt[i] = true`) in soft mode.
Arguments:
  * `tree1`:    pointer to the root of the first tree;
  * `tree2`:    pointer to the root of the second tree;
  * `cf`:       structure for configurations of correlation functions;
  * `cnt`:      array for storing pair counts (accumulated into COUNT.d);
  * `isauto`:   true for counting auto pairs;
  * `withwt`:   true for enabling weights (ignored: soft counts are always
                accumulated as weighted doubles);
  * `para`:     structure for parallelisms.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
int soft_count_pairs(const void *tree1, const void *tree2, CF *cf, COUNT *cnt,
    const bool isauto, const bool withwt
#ifdef MPI
    , const PARA *para
#endif
    );


/******************************************************************************
Function `soft_hist_vjp`:
  Reverse-mode derivative (VJP) of `soft_count_pairs`: given the cotangents
  of a scalar loss with respect to the raw pair count bins, accumulate the
  cotangents of the catalogue coordinates, and optionally of the catalogue
  weights and of the kernel width sigma.  See `soft_hist.c` for the maths.
******************************************************************************/
int soft_hist_vjp(CF *cf, const double *g, double *gx1, double *gx2,
    double *gw1, double *gw2, double *gsigma, double *gmusigma,
    double *gpisigma, const int ia, const int ib
#ifdef MPI
    , const PARA *para
#endif
    );

#endif
