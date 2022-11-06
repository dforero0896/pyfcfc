#ifndef _FCFC_H_
#define _FCFC_H_

#include "eval_cf.h"
#include "define_para.h"
#ifdef MPI
#include <stdlib.h>
#endif


CF* compute_cf(int argc, char *argv[], DATA* dat);

#endif
