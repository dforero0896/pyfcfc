/*******************************************************************************
* 2pt_box/fcfc.c: this file is part of the FCFC program, modified for the
* pyfcfc Python wrapper.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#include "fcfc.h"
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
