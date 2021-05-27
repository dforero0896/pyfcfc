/*******************************************************************************
* 2pt/count_func.c: this file is part of the FCFC program.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2021 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]
 
*******************************************************************************/

#include "define.h"
#include "load_conf.h"
#include "count_func.h"
#include "kdtree.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef OMP
#include <omp.h>
#endif

/*============================================================================*\
                     Data structure for storing dual nodes
\*============================================================================*/

typedef struct {
  const void *a;
  const void *b;
} DUAL_NODE;

typedef struct {
  DUAL_NODE *nodes;
  size_t size;
  size_t capacity;
} STACK_DUAL_NODE;


/*============================================================================*\
                        Functions for stack manipulation
\*============================================================================*/

/******************************************************************************
Function `stack_push`:
  Push an element to the stack for dual nodes.
Arguments:
  * `s`:        pointer to the stack;
  * `a`:        the first node to be pushed to the stack;
  * `b`:        the second node to be pushed to the stack.
******************************************************************************/
static void stack_push(STACK_DUAL_NODE *s, const void *a, const void *b) {
  if (s->size >= s->capacity) {
    /* Enlarge the memory allocated for the stack. */
    if (s->capacity) {
      if ((FCFC_STACK_MAX_SIZE >> 1) <= s->capacity) {
        P_EXT("too many elements to be pushed to the stack of dual nodes\n");
        exit(FCFC_ERR_MEMORY);
      }
      s->capacity <<= 1;
    }
    else {      /* initialise the stack */
      s->nodes = NULL;
      s->capacity = FCFC_STACK_INIT_SIZE;
    }

    if (s->capacity <= s->size) {
      P_EXT("unable to expand the size of the stack of dual nodes\n");
      exit(FCFC_ERR_UNKNOWN);
    }

    DUAL_NODE *tmp = realloc(s->nodes, s->capacity * sizeof *tmp);
    if (!tmp) {
      P_EXT("failed to allocate memory for the stack of dual nodes\n");
      exit(FCFC_ERR_MEMORY);
    }
    s->nodes = tmp;
  }

  s->nodes[s->size].a = a;
  s->nodes[s->size++].b = b;
}

/******************************************************************************
Function `stack_pop`:
  Pop one pair of nodes from the stack for dual nodes.
Arguments:
  * `s`:        pointer to the stack.
Return:
  Address of the dual nodes on the top of the stack on success; NULL on error.
******************************************************************************/
static inline DUAL_NODE *stack_pop(STACK_DUAL_NODE *s) {
  if (!s->size) return NULL;
  s->size -= 1;
  return s->nodes + s->size;
}

/******************************************************************************
Function `stack_destroy`:
  Deconstruct the stack for dual nodes.
Arguments:
  * `s`:        pointer to the stack.
******************************************************************************/
static void stack_destroy(STACK_DUAL_NODE *s) {
  if (!s->capacity) return;
  s->size = s->capacity = 0;
  if (s->nodes) free(s->nodes);
}


/*============================================================================*\
                       Functions for distance evaluations
\*============================================================================*/

/******************************************************************************
Function `squared_distance`:
  Compute the squared Euclidean distance between two points in 3-D space.
Arguments:
  * `a`:        pointer to the first data point;
  * `b`:        pointer to the second data point.
Return:
  The squared Euclidean distance of the two points.
******************************************************************************/
static inline real squared_distance(const DATA *restrict a,
    const DATA *restrict b) {
  register real t = 2 * (a->x[0]*b->x[0] + a->x[1]*b->x[1] + a->x[2]*b->x[2]);
  return a->s + b->s - t;
}

/******************************************************************************
Function `squared_distance_alter`:
  Compute the squared Euclidean distance between two points in 3-D space.
Arguments:
  * `a`:        pointer to the first data point;
  * `b`:        pointer to the second data point.
Return:
  The squared Euclidean distance of the two points.
******************************************************************************/
static inline real squared_distance_alter(const DATA *restrict a,
    const DATA *restrict b) {
  register real dx = a->x[0] - b->x[0];
  register real dy = a->x[1] - b->x[1];
  register real dz = a->x[2] - b->x[2];
  return dx * dx + dy * dy + dz * dz;
}

/******************************************************************************
Function `squared_distance_with_pi`:
  Compute the squared Euclidean distance between two points in 3-D space,
  and report also the squared radial distance.
Arguments:
  * `a`:        pointer to the first data point;
  * `b`:        pointer to the second data point;
  * `pi`:       the squared radial distance.
Return:
  The squared Euclidean distance of the two points.
******************************************************************************/
static inline real squared_distance_with_pi(const DATA *restrict a,
    const DATA *restrict b, real *pi) {
  register real t = 2 * (a->x[0]*b->x[0] + a->x[1]*b->x[1] + a->x[2]*b->x[2]);
  register real s = a->s + b->s;
  register real d = a->s - b->s;
  *pi = d * d / (s + t);
  return s - t;
}

/******************************************************************************
Function `squared_distance_par_with_pi`:
  Compute the squared transverse and radial Euclidean distance between
  two points in 3-D.
Arguments:
  * `a`:        pointer to the first data point;
  * `b`:        pointer to the second data point;
  * `pi`:       the squared radial distance.
Return:
  The squared transverse Euclidean distance of the two points.
******************************************************************************/
static inline real squared_distance_par_with_pi(const DATA *restrict a,
    const DATA *restrict b, real *pi) {
  register real t = 2 * (a->x[0]*b->x[0] + a->x[1]*b->x[1] + a->x[2]*b->x[2]);
  register real s = a->s + b->s;
  register real d = a->s - b->s;
  *pi = d * d / (s + t);
  return s - t - *pi;
}

/******************************************************************************
Function `min_squared_dist_between_box`:
  Compute the minimum squared distance between two boxes.
Arguments:
  * `min1`:     the lower corner of the first box;
  * `max1`:     the upper corner of the first box;
  * `min2`:     the lower corner of the second box;
  * `max2`:     the upper corner of the second box.
Return:
  The minimum squared distance between the two boxes.
******************************************************************************/
static inline real min_squared_dist_between_box(const DATA *restrict min1,
    const DATA *restrict max1, const DATA *restrict min2,
    const DATA *restrict max2) {
  register real sum = 0;
  register real d = (min1->x[0] < min2->x[0]) ?
      min2->x[0] - max1->x[0] : min1->x[0] - max2->x[0];
  if (d > 0) sum += d * d;
  d = (min1->x[1] < min2->x[1]) ?
      min2->x[1] - max1->x[1] : min1->x[1] - max2->x[1];
  if (d > 0) sum += d * d;
  d = (min1->x[2] < min2->x[2]) ?
      min2->x[2] - max1->x[2] : min1->x[2] - max2->x[2];
  if (d > 0) sum += d * d;
  return sum;
}

/******************************************************************************
Function `max_squared_dist_between_box`:
  Compute the maximum squared distance between two boxes.
Arguments:
  * `min1`:     the lower corner of the first box;
  * `max1`:     the upper corner of the first box;
  * `min2`:     the lower corner of the second box;
  * `max2`:     the upper corner of the second box.
Return:
  The maximum squared distance between the two boxes.
******************************************************************************/
static inline real max_squared_dist_between_box(const DATA *restrict min1,
    const DATA *restrict max1, const DATA *restrict min2,
    const DATA *restrict max2) {
  register real sum = 0;
  register real d = (min1->x[0] + max1->x[0] < min2->x[0] + max2->x[0]) ?
      max2->x[0] - min1->x[0] : max1->x[0] - min2->x[0];
  sum += d * d;
  d = (min1->x[1] + max1->x[1] < min2->x[1] + max2->x[1]) ?
      max2->x[1] - min1->x[1] : max1->x[1] - min2->x[1];
  sum += d * d;
  d = (min1->x[2] + max1->x[2] < min2->x[2] + max2->x[2]) ?
      max2->x[2] - min1->x[2] : max1->x[2] - min2->x[2];
  sum += d * d;
  return sum;
}

/******************************************************************************
Function `find_dist_bin`:
  Find the index of a squared distance in the bins, using binary search.
Arguments:
  * `dist`:     the given distance;
  * `rbin`:     the array for distance bins;
  * `n`:        the number of distance bins.
Output:
  Index of the bin on success; SIZE_MAX if the bin is not found.
******************************************************************************/
static inline size_t find_dist_bin(const real dist, const real *restrict dbin,
    const int n) {
  size_t l, u;
  l = 0;
  u = n - 1;
  while (l <= u) {
    size_t i = (l + u) >> 1;
    if (dbin[i + 1] <= dist) l = i + 1;
    else if (dbin[i] > dist) u = i - 1;
    else return i;
  }
  return SIZE_MAX;
}


/*============================================================================*\
                     Pair counting functions from templates
\*============================================================================*/

/* Clean all the relevant macros first */
#ifdef FCFC_TREE_TYPE
  #undef FCFC_TREE_TYPE
#endif

#ifdef FCFC_CNT_TYPE
  #undef FCFC_CNT_TYPE
#endif
#ifdef FCFC_BIN_TYPE
  #undef FCFC_BIN_TYPE
#endif
#ifdef FCFC_BIN_PREC
  #undef FCFC_BIN_PREC
#endif
#ifdef FCFC_BIN_SMIN
  #undef FCFC_BIN_SMIN
#endif
#ifdef FCFC_BIN_PMIN
  #undef FCFC_BIN_PMIN
#endif
#ifdef FCFC_CNT_WT
  #undef FCFC_CNT_WT
#endif

/*******************************************************************************
                                    k-D tree
*******************************************************************************/

#define FCFC_TREE_TYPE  FCFC_TREE_TYPE_KDTREE

/** auto pair counts  **/

/* kdtree_auto_iso_exact */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_exact_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_intbin */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_intbin_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_trunc */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_trunc_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_exact */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_exact_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_intbin */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_intbin_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_trunc */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_trunc_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_smin0_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_smin0_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_smin0_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_exact_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_exact_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_intbin_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_intbin_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_trunc_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_iso_trunc_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_exact_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_exact_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_intbin_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_intbin_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_trunc_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_smu_trunc_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_exact_smin0_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_intbin_smin0_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"

/* kdtree_auto_spi_trunc_smin0_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_AUTO
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#include "dual_tree.c"


/** cross pair counts **/

/* kdtree_cross_iso_exact */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_exact_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_intbin */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_intbin_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_trunc */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_trunc_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_exact */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_exact_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_intbin */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_intbin_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_trunc */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_trunc_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_smin0_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_smin0_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_smin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_smin0_pmin0 */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_NO_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_iso_exact_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_exact_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_intbin_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_intbin_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_trunc_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_iso_trunc_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_ISO
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_exact_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_exact_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_intbin_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_intbin_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_trunc_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_smu_trunc_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SMU
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_exact_smin0_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_EXACT
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_intbin_smin0_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_INTEG
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_smin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_NONZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_NONZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"

/* kdtree_cross_spi_trunc_smin0_pmin0_wt */
#define FCFC_CNT_TYPE   FCFC_PAIR_COUNT_CROSS
#define FCFC_BIN_TYPE   FCFC_BIN_SPI
#define FCFC_BIN_PREC   FCFC_BIN_TRUNC
#define FCFC_BIN_SMIN   FCFC_BIN_MIN_ZERO
#define FCFC_CNT_WT     FCFC_CNT_WITH_WEIGHT
#define FCFC_BIN_PMIN   FCFC_BIN_MIN_ZERO
#include "dual_tree.c"


/*============================================================================*\
                          Interface for pair counting
\*============================================================================*/

/******************************************************************************
Function `count_pairs`:
  Count cross pairs based on the k-D tree data structure.
Arguments:
  * `tree1`:    pointer to the root of the first k-D tree;
  * `tree2`:    pointer to the root of the second k-D tree;
  * `cf`:       structure for congifurations of correlation functions;
  * `cnt`:      array for storing pair counts;
  * `isauto`:   true for counting auto pairs;
  * `usewt`:    apply weights for pair counts.
******************************************************************************/
void count_pairs(const void *tree1, const void *tree2, CF *cf,
    pair_count_t *cnt, bool isauto, bool usewt) {
      
  /* Choose the optimal pair counting function. */
  
  void (*pair_count_func) (STACK_DUAL_NODE *, const CF *, pair_count_t *) =
      NULL;
  
  bool smin0 = (cf->sbin[0] < REAL_TOL && cf->sbin[0] > -REAL_TOL);
  if (isauto) {
    if (usewt) {
      if (cf->bintype == FCFC_BIN_ISO) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_auto_iso_exact_smin0_wt;
          else pair_count_func = kdtree_auto_iso_exact_wt;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_auto_iso_intbin_smin0_wt;
          else pair_count_func = kdtree_auto_iso_intbin_wt;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_auto_iso_trunc_smin0_wt;
          else pair_count_func = kdtree_auto_iso_trunc_wt;
        }
      }
      else if (cf->bintype == FCFC_BIN_SMU) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_auto_smu_exact_smin0_wt;
          else pair_count_func = kdtree_auto_smu_exact_wt;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_auto_smu_intbin_smin0_wt;
          else pair_count_func = kdtree_auto_smu_intbin_wt;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_auto_smu_trunc_smin0_wt;
          else pair_count_func = kdtree_auto_smu_trunc_wt;
        }
      }
      else {                /* FCFC_BIN_SPI */
        bool pmin0 = (cf->pbin[0] < REAL_TOL && cf->pbin[0] > -REAL_TOL);
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_auto_spi_exact_smin0_pmin0_wt;
            else pair_count_func = kdtree_auto_spi_exact_smin0_wt;
          }
          else {
            if (pmin0) pair_count_func = kdtree_auto_spi_exact_pmin0_wt;
            else pair_count_func = kdtree_auto_spi_exact_wt;
          }
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_auto_spi_intbin_smin0_pmin0_wt;
            else pair_count_func = kdtree_auto_spi_intbin_smin0_wt;
          }
          else {
            if (pmin0) pair_count_func = kdtree_auto_spi_intbin_pmin0_wt;
            else pair_count_func = kdtree_auto_spi_intbin_wt;
          }
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_auto_spi_trunc_smin0_pmin0_wt;
            else pair_count_func = kdtree_auto_spi_trunc_smin0_wt;
          }
          else {
            if (pmin0) pair_count_func = kdtree_auto_spi_trunc_pmin0_wt;
            else pair_count_func = kdtree_auto_spi_trunc_wt;
          }
        }
      }
    }
    else {      /* !usewt */
      if (cf->bintype == FCFC_BIN_ISO) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_auto_iso_exact_smin0;
          else pair_count_func = kdtree_auto_iso_exact;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_auto_iso_intbin_smin0;
          else pair_count_func = kdtree_auto_iso_intbin;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_auto_iso_trunc_smin0;
          else pair_count_func = kdtree_auto_iso_trunc;

        }
      }
      else if (cf->bintype == FCFC_BIN_SMU) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_auto_smu_exact_smin0;
          else pair_count_func = kdtree_auto_smu_exact;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_auto_smu_intbin_smin0;
          else pair_count_func = kdtree_auto_smu_intbin;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_auto_smu_trunc_smin0;
          else pair_count_func = kdtree_auto_smu_trunc;
        }
      }
      else {                /* FCFC_BIN_SPI */
        bool pmin0 = (cf->pbin[0] < REAL_TOL && cf->pbin[0] > -REAL_TOL);
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_auto_spi_exact_smin0_pmin0;
            else pair_count_func = kdtree_auto_spi_exact_smin0;
          }
          else {
            if (pmin0) pair_count_func = kdtree_auto_spi_exact_pmin0;
            else pair_count_func = kdtree_auto_spi_exact;
          }
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_auto_spi_intbin_smin0_pmin0;
            else pair_count_func = kdtree_auto_spi_intbin_smin0;
          }
          else {
            if (pmin0) pair_count_func = kdtree_auto_spi_intbin_pmin0;
            else pair_count_func = kdtree_auto_spi_intbin;
          }
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_auto_spi_trunc_smin0_pmin0;
            else pair_count_func = kdtree_auto_spi_trunc_smin0;
          }
          else {
            if (pmin0) pair_count_func = kdtree_auto_spi_trunc_pmin0;
            else pair_count_func = kdtree_auto_spi_trunc;
          }
        }
      }
    }
  }
  else {        /* !isauto */
    if (usewt) {
      if (cf->bintype == FCFC_BIN_ISO) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_cross_iso_exact_smin0_wt;
          else pair_count_func = kdtree_cross_iso_exact_wt;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_cross_iso_intbin_smin0_wt;
          else pair_count_func = kdtree_cross_iso_intbin_wt;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_cross_iso_trunc_smin0_wt;
          else pair_count_func = kdtree_cross_iso_trunc_wt;
        }
      }
      else if (cf->bintype == FCFC_BIN_SMU) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_cross_smu_exact_smin0_wt;
          else pair_count_func = kdtree_cross_smu_exact_wt;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_cross_smu_intbin_smin0_wt;
          else pair_count_func = kdtree_cross_smu_intbin_wt;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_cross_smu_trunc_smin0_wt;
          else pair_count_func = kdtree_cross_smu_trunc_wt;
        }
      }
      else {                /* FCFC_BIN_SPI */
        bool pmin0 = (cf->pbin[0] < REAL_TOL && cf->pbin[0] > -REAL_TOL);
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_cross_spi_exact_smin0_pmin0_wt;
            else pair_count_func = kdtree_cross_spi_exact_smin0_wt;
          }
          else {
            if (pmin0) pair_count_func = kdtree_cross_spi_exact_pmin0_wt;
            else pair_count_func = kdtree_cross_spi_exact_wt;
          }
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_cross_spi_intbin_smin0_pmin0_wt;
            else pair_count_func = kdtree_cross_spi_intbin_smin0_wt;
          }
          else {
            if (pmin0) pair_count_func = kdtree_cross_spi_intbin_pmin0_wt;
            else pair_count_func = kdtree_cross_spi_intbin_wt;
          }
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_cross_spi_trunc_smin0_pmin0_wt;
            else pair_count_func = kdtree_cross_spi_trunc_smin0_wt;
          }
          else {
            if (pmin0) pair_count_func = kdtree_cross_spi_trunc_pmin0_wt;
            else pair_count_func = kdtree_cross_spi_trunc_wt;
          }
        }
      }
    }
    else {      /* !usewt */
      if (cf->bintype == FCFC_BIN_ISO) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_cross_iso_exact_smin0;
          else pair_count_func = kdtree_cross_iso_exact;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_cross_iso_intbin_smin0;
          else pair_count_func = kdtree_cross_iso_intbin;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_cross_iso_trunc_smin0;
          else pair_count_func = kdtree_cross_iso_trunc;
        }
      }
      else if (cf->bintype == FCFC_BIN_SMU) {
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) pair_count_func = kdtree_cross_smu_exact_smin0;
          else pair_count_func = kdtree_cross_smu_exact;
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) pair_count_func = kdtree_cross_smu_intbin_smin0;
          else pair_count_func = kdtree_cross_smu_intbin;
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) pair_count_func = kdtree_cross_smu_trunc_smin0;
          else pair_count_func = kdtree_cross_smu_trunc;
        }
      }
      else {                /* FCFC_BIN_SPI */
        bool pmin0 = (cf->pbin[0] < REAL_TOL && cf->pbin[0] > -REAL_TOL);
        if (cf->prec == REAL_NAN) {         /* FCFC_BIN_EXACT */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_cross_spi_exact_smin0_pmin0;
            else pair_count_func = kdtree_cross_spi_exact_smin0;
          }
          else {
            if (pmin0) pair_count_func = kdtree_cross_spi_exact_pmin0;
            else pair_count_func = kdtree_cross_spi_exact;
          }
        }
        else if (cf->prec == 1) {           /* FCFC_BIN_INTEG */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_cross_spi_intbin_smin0_pmin0;
            else pair_count_func = kdtree_cross_spi_intbin_smin0;
          }
          else {
            if (pmin0) pair_count_func = kdtree_cross_spi_intbin_pmin0;
            else pair_count_func = kdtree_cross_spi_intbin;
          }
        }
        else {                              /* FCFC_BIN_TRUNC */
          if (smin0) {
            if (pmin0) pair_count_func = kdtree_cross_spi_trunc_smin0_pmin0;
            else pair_count_func = kdtree_cross_spi_trunc_smin0;
          }
          else {
            if (pmin0) pair_count_func = kdtree_cross_spi_trunc_pmin0;
            else pair_count_func = kdtree_cross_spi_trunc;
          }
        }
      }
    }
  }

  /* Initialise the stack for dual nodes. */

  STACK_DUAL_NODE stack;
  stack.size = stack.capacity = 0;
  stack_push(&stack, tree1, tree2);
  


#ifdef OMP
  /* Assign tasks to different OpenMP threads. */
  
  size_t size = stack.size;     /* for visiting all nodes at the same level */
  
  while (stack.size != size ||
      stack.size < (size_t) cf->nthread * FCFC_STACK_SIZE_PER_THREAD) {
    pair_count_func(&stack, cf, cnt);
    
    if (!stack.size) return;    /* all pairs have been recorded */
    if (size > 1) {
      size -= 1;
      /* reorder dual nodes, to ensure nodes at the same level are visited */
      DUAL_NODE tmp = stack.nodes[stack.size - 1];
      
      stack.nodes[stack.size - 1] = stack.nodes[size - 1];
      stack.nodes[size - 1] = tmp;
    }
    else size = stack.size;
  }
  
  
  /* Clean the array for storing thread-private pair counts. */
  memset(cf->pcnt, 0, sizeof(pair_count_t) * cf->ntot * cf->nthread);
  
#pragma omp parallel
  {
    STACK_DUAL_NODE ps;         /* thread-private stack */
    ps.size = ps.capacity = 0;
    const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < stack.size; i++) {
    
      stack_push(&ps, stack.nodes[i].a, stack.nodes[i].b);
      while (ps.size) pair_count_func(&ps, cf, cf->pcnt + tid * cf->ntot);
    }
    stack_destroy(&ps);
  }
  

  /* Gather pair counts from threads. */
  if (usewt) {
#pragma omp parallel for
    for (size_t i = 0; i < cf->ntot; i++) {
      for (int j = 0; j < cf->nthread; j++)
        cnt[i].d += cf->pcnt[i + j * cf->ntot].d;
    }
  }
  else {
    for (size_t i = 0; i < cf->ntot; i++) {
      for (int j = 0; j < cf->nthread; j++)
        cnt[i].i += cf->pcnt[i + j * cf->ntot].i;
    }
  }
#else
  while (stack.size) pair_count_func(&stack, cf, cnt);
#endif

  stack_destroy(&stack);
}
