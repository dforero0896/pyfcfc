/*******************************************************************************
* 2pt_box/fcfc.c: this file is part of the FCFC program, modified for the
* pyfcfc Python wrapper.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#include "fcfc.h"
#include "soft_hist.h"
#include <stdlib.h>

/******************************************************************************
Function `free_input_data':
  Release the memory of the input catalogues passed from Python.
  It is only called on failure paths where CF does not own the data yet.
Arguments:
  * `dat':      array of input catalogues;
  * `ncat':     number of input catalogues.
******************************************************************************/
static void free_input_data(DATA *dat, const int ncat) {
  if (!dat) return;
  for (int i = 0; i < ncat; i++) {
    for (int j = 0; j < FCFC_XDIM; j++)
      if (dat[i].x[j]) free(dat[i].x[j]);
    if (dat[i].w) free(dat[i].w);
  }
  free(dat);
}

/******************************************************************************
Function `compute_cf':
  Evaluate pair counts and correlation functions from in-memory catalogues.
******************************************************************************/
CF *compute_cf(int argc, char *argv[], DATA *dat, int ncat,
    real *sbins, int ns, real *pbins, int np, int nmu) {
  CONF *conf = NULL;
  CF *cf = NULL;

#ifdef WITH_PARA
  /* Initialize parallelisms. */
  PARA para;
  para_init(&para);
#endif

#ifdef MPI
  /* Initialize configurations with the root rank only. */
  if (para.rank == para.root) {
#endif

    if (!(conf = load_conf(argc, argv
#ifdef WITH_PARA
        , &para
#endif
        ))) {
      printf(FMT_FAIL);
      P_EXT("failed to load configuration parameters\n");
      free_input_data(dat, ncat);
      return NULL;
    }

    if (!(cf = cf_setup(conf, dat, sbins, ns, pbins, np, nmu
#ifdef OMP
        , &para
#endif
        ))) {
      printf(FMT_FAIL);
      P_EXT("failed to initialise correlation function evaluations\n");
      conf_destroy(conf);
      /* `cf_setup' attaches the data only on success, so `dat' is intact. */
      free_input_data(dat, ncat);
      return NULL;
    }

#ifdef MPI
  }

  /* Broadcast configurations. */
  cf_setup_worker(&cf, &para);
#endif

  if (eval_cf(conf, cf
#ifdef MPI
      , &para
#endif
      )) {
    printf(FMT_FAIL);
    P_EXT("failed to evaluate correlation functions\n");
    conf_destroy(conf);
    cf_destroy(cf);       /* this also releases the input data */
    return NULL;
  }

  /* The labels have been deep-copied into CF, so `conf' can be released. */
  conf_destroy(conf);

#ifdef MPI
  if (MPI_Finalize()) {
    P_ERR("failed to finalize MPI\n");
    return NULL;
  }
#endif

  return cf;
}

/******************************************************************************
Function `compute_cf_soft_vjp':
  Reverse-mode derivative of the soft histogramming pair counts with
  respect to the input catalogues.  See `fcfc.h' for the interface.
******************************************************************************/
int compute_cf_soft_vjp(int argc, char *argv[], DATA *dat, int ncat,
    real *sbins, int ns, real *pbins, int np, int nmu,
    const char *pair, const double *g,
    double *gx[], double *gw[], double *gsigma) {
#ifdef MPI
  (void) argc; (void) argv; (void) dat; (void) ncat; (void) sbins;
  (void) ns; (void) pbins; (void) np; (void) nmu; (void) pair; (void) g;
  (void) gx; (void) gw; (void) gsigma;
  P_ERR("soft histogramming (SOFT_HIST) does not support MPI yet\n");
  free_input_data(dat, ncat);
  return FCFC_ERR_ARG;
#else
  CONF *conf = NULL;
  CF *cf = NULL;
  int status = FCFC_ERR_UNKNOWN;

#ifdef WITH_PARA
  /* Initialize parallelisms. */
  PARA para;
  para_init(&para);
#endif

  if (!(conf = load_conf(argc, argv
#ifdef WITH_PARA
      , &para
#endif
      ))) {
    P_EXT("failed to load configuration parameters\n");
    free_input_data(dat, ncat);
    return FCFC_ERR_CFG;
  }
  /* The VJP is defined for the soft histogramming method. */
  conf->soft = true;

  if (!(cf = cf_setup(conf, dat, sbins, ns, pbins, np, nmu
#ifdef OMP
      , &para
#endif
      ))) {
    P_EXT("failed to initialise correlation function evaluations\n");
    conf_destroy(conf);
    free_input_data(dat, ncat);
    return FCFC_ERR_CF;
  }

  /* Locate the requested pair count. */
  int ipc = -1;
  if (pair && pair[0] && pair[1] && !pair[2]) {
    for (int i = 0; i < cf->npc; i++) {
      if (cf->label[cf->pc_idx[0][i]] == pair[0] &&
          cf->label[cf->pc_idx[1][i]] == pair[1]) {
        ipc = i;
        break;
      }
    }
  }
  if (ipc < 0) {
    P_ERR("unknown pair count for the soft histogram VJP: %s\n",
        pair ? pair : "(null)");
    conf_destroy(conf);
    cf_destroy(cf);              /* this also releases the input data */
    return FCFC_ERR_ARG;
  }
  const int ia = cf->pc_idx[0][ipc];
  const int ib = cf->pc_idx[1][ipc];
  const bool isauto = (ia == ib);

  /* The adjoint builds its own index-preserving trees internally. */
  status = soft_hist_vjp(cf, g, gx[0], gx[1], gw[0], gw[1], gsigma, ia, ib);

  /* The traversal works in rescaled coordinates (and sigma): convert the
   * cotangents back to the units of the inputs,
   * d/dx_input = rescale * d/dx_internal. */
  if (!status) {
    /* NB: for auto pair counts gx[0] and gx[1] may alias (a single buffer
     * receiving both sides): convert each distinct buffer only once. */
    for (int c = 0; c < 2; c++) {
      if (!gx[c] || (c == 1 && gx[1] == gx[0])) continue;
      const int idx = (c == 0) ? ia : ib;
      for (size_t k = 0; k < (size_t) 3 * cf->data[idx].n; k++)
        gx[c][k] *= (double) cf->rescale;
    }
    if (gsigma) *gsigma *= (double) cf->rescale;
  }

  conf_destroy(conf);
  cf_destroy(cf);                /* this also releases the input data */
  return status;
#endif
}
