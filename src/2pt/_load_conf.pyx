#cython: language_level=3, boundscheck=True
from libc.stdlib cimport malloc, free, calloc
from _load_conf cimport CONF
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout, stderr

cdef extern from "load_conf.h":
    CONF *load_conf(const int argc, char *const *argv) nogil   
    void conf_destroy(CONF *conf) nogil      
    CONF *conf_init() nogil

cdef CONF* load_conf_py(char * label, bint *has_wt, bint* cnvt,
                        double omega_m, double omega_l, double omega_k,
                        double dew, char **pc, int npc, int bintype, bint *comp_pc) nogil:
    cdef CONF* conf = <CONF *> calloc(1, sizeof(conf[0]))
    if ( not conf): return NULL
    conf.fconf = conf.label = conf.comment = NULL
    conf.fcnvt = conf.fsbin = conf.fpbin = NULL
    conf.input = conf.fmtr = conf.pos = conf.wt = conf.sel = NULL
    conf.pc = conf.pcout = conf.cf = conf.cfout = NULL
    conf.mpout = conf.wpout = NULL
    conf.ftype = conf.poles = NULL
    conf.has_wt = conf.cnvt = NULL
    conf.comp_pc = NULL
    conf.skip = NULL

    conf.label = label
    conf.has_wt = <bint*> has_wt
    conf.cnvt = cnvt
    conf.omega_m = omega_m
    conf.omega_l = omega_l
    conf.omega_k = omega_k
    conf.dew = dew
    conf.ecnvt = 1e-8

    conf.bintype = bintype
    conf.pc = pc
    conf.comp_pc = comp_pc


    printf("%s\n", conf.label)
    printf("%i\n", conf.has_wt[0])
    printf("%i\n", conf.cnvt[0])
    printf("%s %s %s\n", conf.pc[0], conf.pc[1], conf.pc[2])
    printf("%i %i %i\n", conf.comp_pc[0], conf.comp_pc[1], conf.comp_pc[2])
    


    return conf
    
