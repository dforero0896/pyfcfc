#cython: language_level=3
#cython: boundscheck = True
import cython
from cython.parallel import prange, threadid
cimport openmp
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout, stderr
from libc.limits cimport INT_MAX
from libc.math cimport HUGE_VAL
from libc.stdint cimport SIZE_MAX
from libc.string cimport strcpy, strlen
from _load_conf cimport CONF, conf_destroy
cimport _load_conf
import _load_conf



def set_conf(str label, list has_wt, list cnvt, list pc, list comp_pc,
            double omega_m = 0.31, double omega_l = 0, 
            double omega_k=0, double dew=-1, int bintype=1):
    
    cdef CONF* conf = <CONF *> calloc(1, sizeof(conf[0]))

    cdef int ninput = len(has_wt)
    cdef int npc = len(pc)
    cdef bint* chas_wt = <bint *> malloc(ninput * sizeof(bint))
    cdef bint* ccnvt = <bint *> malloc(ninput * sizeof(bint))
    cdef char** cpc = <char **> malloc(npc * sizeof(char *))
    cdef bint* ccomp_pc = <bint *> malloc(npc * sizeof(bint))
    cdef size_t i

    
    for i in range(<size_t>ninput):
        chas_wt[i] = <bint>int(has_wt[i])
        ccnvt[i] = <bint>int(cnvt[i])
    for i in range(<size_t>npc):
        strcpy(cpc[i], pc[i].encode('utf-8'))
        ccomp_pc[i] = <bint> comp_pc[i]
        
    conf = _load_conf.load_conf_py(label.encode('utf-8'), chas_wt, ccnvt, 
                            omega_m, omega_l, omega_k, dew, cpc, npc, bintype, ccomp_pc)
    
    
    _load_conf.conf_destroy(conf)
    

    
