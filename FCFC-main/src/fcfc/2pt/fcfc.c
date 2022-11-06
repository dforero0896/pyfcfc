/*******************************************************************************
* 2pt/fcfc.c: this file is part of the FCFC program.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#include "eval_cf.h"
#include "define_para.h"
#ifdef MPI
#include <stdlib.h>
#endif

CF* compute_cf(int argc, char *argv[], DATA* dat) {
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
      return NULL;
    }
  
    if (!(cf = cf_setup(conf
#ifdef OMP
        , &para
#endif
        ))) {
      printf(FMT_FAIL);
      P_EXT("failed to initialise correlation function evaluations\n");
      conf_destroy(conf);
      return NULL;
    }
    
  cf->data = dat; 
  
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
    conf_destroy(conf); cf_destroy(cf);
    return NULL;
  }
  /* Deep copying labels to name results */
  cf->label = malloc(sizeof(char) * cf->ncat);
  memcpy(cf->label, conf->label, cf->ncat);
  conf_destroy(conf);
  
#ifdef MPI
  if (MPI_Finalize()) {
    P_ERR("failed to finalize MPI\n");
    return NULL;
  }
#endif
  return cf;
}
