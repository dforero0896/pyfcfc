/*******************************************************************************
* 2pt/build_tree.c: this file is part of the FCFC program.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2021 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#include "define.h"
#include "build_tree.h"
#include "read_file.h"
#include "kdtree.h"
#include <stdio.h>

/*============================================================================*\
                 Functions for tree creation and deconstruction
\*============================================================================*/

/******************************************************************************
Function `tree_create`:
  Construct the tree from an input catalogue for pair counting.
Arguments:
  * `conf`:     structure for storing configurations;
  * `cf`:       structure for correlation function evaluations;
  * `idx`:      index of the catalogue to be processed;
  * `type`:     type of the tree.
Return:
  Address of the tree on success; NULL on error.
******************************************************************************/
void *tree_create(const CONF *conf, CF *cf, const int idx, const int type) {
  if (!conf) {
    P_ERR("configuration parameters are not loaded\n");
    return NULL;
  }
  if (!cf) {
    P_ERR("correlation function evaluation has not been initialised\n");
    return NULL;
  }
  if (idx < 0 || idx > conf->ninput) {
    P_ERR("unexpected index of the catalog: %d\n", idx);
    return NULL;
  }

  printf("Construct the tree for catalog '%c' ...", cf->label[idx]);
  if (conf->verbose)  printf("\n");
  fflush(stdout);

  /* Read catalogue from file. */
  const size_t skip = (conf->skip) ? conf->skip[idx] : DEFAULT_ASCII_SKIP;
  const char cmt = (conf->comment) ? conf->comment[idx] : DEFAULT_ASCII_COMMENT;
  const char *wt = (conf->has_wt[idx]) ? conf->wt[idx] : NULL;
  const char *sel = (conf->sel) ? conf->sel[idx] : NULL;
  if (read_ascii_data(conf->input[idx], skip, cmt, conf->fmtr[idx],
      conf->pos + idx * 3, wt, sel, cf->data + idx, cf->ndata + idx,
      conf->verbose)) return NULL;

  /* Apply coordinate conversion if necessary. */
  if ((!conf->cnvt && DEFAULT_COORD_CNVT == true) ||
      (conf->cnvt && conf->cnvt[idx])) {
    if (cnvt_coord(conf, cf->data[idx], cf->ndata[idx], cf->coord)) return NULL;
  }

  /* Precompute the squared distance between tracers and the origin,
     and compute the total weights if necessary. */
  if (cf->wt[idx]) {
    double sum = 0;
#ifdef OMP
#pragma omp parallel for reduction(+:sum)
#endif
    for (size_t i = 0; i < cf->ndata[idx]; i++) {
      cf->data[idx][i].s = cf->data[idx][i].x[0] * cf->data[idx][i].x[0] +
          cf->data[idx][i].x[1] * cf->data[idx][i].x[1] +
          cf->data[idx][i].x[2] * cf->data[idx][i].x[2];
      sum += cf->data[idx][i].w;
    }
    cf->wdata[idx] = sum;
  }
  else {
#ifdef OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < cf->ndata[idx]; i++) {
      cf->data[idx][i].s = cf->data[idx][i].x[0] * cf->data[idx][i].x[0] +
          cf->data[idx][i].x[1] * cf->data[idx][i].x[1] +
          cf->data[idx][i].x[2] * cf->data[idx][i].x[2];
    }
    cf->wdata[idx] = (double) cf->ndata[idx];
  }

  /* Construct the tree. */
  DATA tmp;
  void *tree = NULL;
  int err = 0;
  switch (type) {
    case FCFC_TREE_TYPE_KDTREE:
      tree = kdtree_build(cf->data[idx], cf->ndata[idx], &tmp, &err);
      if (err) return NULL;
      if (conf->verbose) printf("  k-D tree constructed for the catalog\n");
      break;
    default:
      P_ERR("unsupported tree type\n");
      return NULL;
  }

  printf(FMT_DONE);
  return tree;
}

/******************************************************************************
Function `tree_destroy`:
  Deconstruct a tree used for pair counting.
Arguments:
  * `tree`:     address of the tree;
  * `type`:     type of the tree.
******************************************************************************/
void tree_destroy(void *tree, const int type) {
  if (!tree) return;
  switch (type) {
    case FCFC_TREE_TYPE_KDTREE:
      kdtree_free((KDT *) tree);
      break;
    default:
      P_WRN("unsupported tree type\n");
  }
}
