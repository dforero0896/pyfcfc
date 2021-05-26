#cython: language_level=3
#cython: boundscheck = True
import cython
from cython.parallel import prange, threadid
cimport openmp
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout, stderr
import numpy as np


DEF OMP = 1

# Define comm

cdef extern from "define_comm.h":
    ctypedef double real
    cdef int REAL_NAN
    cdef int INT_MAX
    #cdef char* FMT_ERR

# count_func

cdef extern from "count_func.h":
    void count_pairs(const void *tree1, const void *tree2, CF *cf,
    pair_count_t *cnt, bint isauto, bint usewt) nogil

# Interface structure for data
cdef extern from "define.h":
    ctypedef struct DATA:
        real x[3]    # x, y, z         
        real w       # weight          
        real s       # x^2 + y^2 + z^2 

    cdef int FCFC_BIN_SMU
    cdef int FCFC_BIN_SPI
    cdef int FCFC_MAX_BIN_NUM
    cdef int KDTREE_LEAF_SIZE
    cdef double REAL_TOL
    cdef int FCFC_ERR_MEMORY    =     (-1)
    cdef int FCFC_ERR_ARG       =    (-2)
    cdef int FCFC_ERR_FILE      =    (-3)
    cdef int FCFC_ERR_CFG       =    (-4)
    cdef int FCFC_ERR_AST       =    (-5)
    cdef int FCFC_ERR_ASCII     =    (-6)
    cdef int FCFC_ERR_CONF      =    (-10)
    cdef int FCFC_ERR_DATA      =    (-11)
    cdef int FCFC_ERR_CNVT      =    (-12)
    cdef int FCFC_ERR_TREE      =    (-13)
    cdef int FCFC_ERR_CF        =    (-14)
    cdef int FCFC_ERR_SAVE      =    (-15)
    cdef int FCFC_ERR_UNKNOWN   =    (-99)

# cnvt_coord

cdef extern from "cnvt_coord.h":
    ctypedef struct COORD_CNVT:
        size_t nsp           # number of sample points, excluding (0,0)      
        double *z            # redshifts                                     
        double *d            # radial comoving distances                     
        double *ypp          # second derivative for spline interpolation    

    COORD_CNVT *cnvt_init() nogil
    void cnvt_destroy(COORD_CNVT *cnvt) nogil
    #int cnvt_coord(const CONF *conf, DATA *data, const size_t ndata,
    #COORD_CNVT *coord) nogil
# CF structure 

cdef extern from "eval_cf.h":

    ctypedef union pair_count_t:
        size_t i
        double d


    ctypedef struct CF:
        real s2min           # minimum squared separation of interest         
        real s2max           # maximum squared separation of interest         
        real p2min           # minimum squared pi of interest                 
        real p2max           # maximum squared pi of interest                 
        real prec            # precision for truncating (squared) distances   

        real *s2bin          # edges of squared separation (or s_perp) bins   
        int ns               # number of separation (or s_perp) bins          
        real *p2bin          # edges of squared pi bins                       
        int np               # number of pi bins                              
        int nmu              # number of mu bins                              
        size_t nmu2          # squared number of mu bins                      
        size_t *stab         # lookup table for separation (or s_perp) bins   
        size_t *ptab         # lookup table for squared pi bins               
        size_t *mutab        # lookup table for mu bins                       

        real sp2min          # minimum squared s_perp of interest             
        real sp2max          # maximum squared s_pere of interest             


        int nthread          # number of threads to be used                   
        int bintype          # binning scheme: iso, smu, or spi               
        size_t ntot          # total number of bins                           
        real *sbin           # edges of separation (or s_perp) bins           
        real *pbin           # edges of pi bins                               

        int ncat             # number of catalogues to be read                
        #const char *label    # labels of the input catalogues                 
        #DATA **data          # structures for saving the input catalogues     
        size_t *ndata        # number of objects in the input catalogues      
        double *wdata        # weighted number of objects in the inputs       
        const bint *wt       # indicate whether weights are available         
        #const bint *cnvt     # indicate whether to run coordinate conversion  
        #COORD_CNVT *coord    # structure for coordinate interpolation         

        int npc              # number of pair counts to be evaluated          
        #int *pc_idx[2]       # pairs to be counted, defined as input indices  
        #const bint *comp_pc  # indicate whether to evaluate the pair counts   
        pair_count_t **cnt   # array for storing evaluated pair counts        
        double *norm         # normalisation factors for pair counts          
        double **ncnt        # array for normalised pair counts               
        pair_count_t *pcnt   # thread-private array for counting in parallel  
        int ncf              # number of correlation functions to be computed 
        #char **cf_exp        # expression for correlation function estimators 
        #ast_t **ast_cf       # abstract syntax trees for 2PCF estimators      
        #double **cf          # array for storing correlation functions        

        int nl               # number of multipoles to be evaluated           
        const int *poles     # orders of Legendre polynomials to be evaluated 
        double **mp          # array for storing 2PCF multipoles              
        bint comp_wp         # indicate whether to compute projected 2PCF     
        double **wp          # array for storing projected 2PCFs              

    #CF *cf_init(const CONF *conf) nogil
    void cf_destroy(CF *cf) nogil
    #int eval_cf(const CONF *conf, CF *cf) nogil

# Tree creation
cdef extern from "kdtree.h":
    cdef struct kdtree_struct:
        size_t n                             # number of objects       
        DATA *data                           # pointer to the data     
        DATA min                             # lower corner of the box 
        DATA max                             # upper corner of the box 
        kdtree_struct *left                  # left child              
        kdtree_struct *right                 # right child    
    ctypedef kdtree_struct KDT

    KDT* kdtree_build(DATA *data, const size_t ndata, DATA *buf, int *err) nogil
    void kdtree_free(KDT* node) nogil
    #KDT* kdtree_init(DATA* data, const size_t ndata) nogil

    
#cdef int kth_compare(const DATA *a, const DATA *b, int k) nogil:
#    fflush(stdout);
#    if (a.x[k] > b.x[k]): return 1;
#    if (a.x[k] < b.x[k]): return -1;

#    k = (k + 1) % 3;
#    if (a.x[k] > b.x[k]):return 1;
#    if (a.x[k] < b.x[k]): return -1;

#    k = (k + 1) % 3;
#    if (a.x[k] > b.x[k]): return 1;
#    if (a.x[k] < b.x[k]): return -1;
#    return 0;

cdef extern from *:
    """
    static inline int kth_compare(const DATA *a, const DATA *b, int k) {
  
        fflush(stdout);
        if (a->x[k] > b->x[k]) return 1;
        if (a->x[k] < b->x[k]) return -1;

        k = (k + 1) % 3;
        if (a->x[k] > b->x[k]) return 1;
        if (a->x[k] < b->x[k]) return -1;

        k = (k + 1) % 3;
        if (a->x[k] > b->x[k]) return 1;
        if (a->x[k] < b->x[k]) return -1;
        return 0;
        }

    #ifdef QSELECT_COMPARE
    #undef QSELECT_COMPARE
    #endif
    #ifdef QSELECT_DTYPE
    #undef QSELECT_DTYPE
    #endif

    #define QSELECT_COMPARE(a,b,k)  kth_compare(a,b,*((int *)(k)))
    #define QSELECT_DTYPE           DATA

    #include "quick_select.c"

    #undef QSELECT_DTYPE
    #undef QSELECT_COMPARE
    """
    
    void qselect(DATA *A, size_t n, size_t len, void *buf,
    void *arg) nogil

cdef CF* cf_init_noconf(bint verbose, 
                        int nthread, 
                        int bin_scheme, # bintype
                        int nmu, 
                        #int ninput, 
                        #char* label, 
                        double [:] sbin_arr, 
                        double [:] pibin_arr, 
                        int dprec,
                        const bint* use_wt,
                        bint compute_wp,
                        int *poles,
                        int npole) nogil:

    printf("Initialising the correlation function calculation ...")


    if verbose: printf("\n")
    fflush(stdout)

    cdef CF* cf = <CF*> calloc(1, sizeof(CF))

    if not cf: 
        fprintf(stderr, "ERROR: failed to allocate memory for the intialisation\n")
        #P_ERR(b"failed to allocate memory for the intialisation\n");
        return NULL

    cf.sbin = cf.s2bin = cf.p2bin = NULL
    cf.stab = cf.mutab = cf.ptab =  NULL
    #cf.data = NULL
    #cf.coord = NULL
    #cf.pc_idx[0] = cf.pc_idx[1] = NULL
    cf.cnt = NULL
    cf.norm = NULL
    cf.wdata = NULL
    cf.pcnt = NULL
    cf.ncnt = NULL
    #cf.cf = NULL
    cf.mp = cf.wp = NULL
    #cf.cf_exp = NULL
    #cf.ast_cf = NULL
    cf.nthread = nthread
    cf.bintype = bin_scheme
    cf.prec = REAL_NAN
    cdef int prec
    if dprec != INT_MAX:
        cf.prec = 1
        prec = dprec
        while prec > 0:
            cf.prec *= 0.1
            prec -= 1
        while prec < 0:
            cf.prec *= 10
            prec += 1
    cf.ns = sbin_arr.shape[0] - 1
    cf.nmu=nmu
    cf.np = pibin_arr.shape[0] - 1

    if cf.bintype == FCFC_BIN_SMU:
        cf.ntot = <size_t> cf.ns * cf.nmu
    elif cf.bintype == FCFC_BIN_SPI:
        cf.ntot = <size_t> cf.ns * cf.np
    else:
        cf.ntot = cf.ns     #cf->bintype == FCFC_BIN_ISO 

    #cf.ncat = ninput
    #cf.label = label
    cf.wt = use_wt
    #cf.cnvt = cnvt
    cf.npc = 1 #npc
    #cf.comp_pc = comp_pc
    cf.ncf = 1
    cf.nl = npole
    cf.poles = poles
    cf.comp_wp = compute_wp
    #cf.coord = cnvt_init()
    #if fcnvt != NULL:
    #    if cf.coord == NULL :
    #        fprintf(stderr, "ERROR: failed to allocate memory for coordinate interpolation")
    #        cf_destroy(cf)
    #        return NULL

    # Define separation bins
    # I will assume a bin array is always passed from hereon
    # Assuming bin array is sorted
    
    if cf.ns > FCFC_MAX_BIN_NUM:
        fprintf(stderr, "ERROR: too many separation bins in `sbin_arr` array.")
        cf_destroy(cf)
        return NULL

    cf.sbin = <real*> malloc(sizeof(real) * (cf.ns +1))
    cf.s2bin = <real*> malloc(sizeof(real) * (cf.ns +1))
    if not cf.sbin or not cf.s2bin:
            fprintf(stderr, "ERROR: failed to allocate memory for pi bins.\n")
            free(cf)
            return NULL
    #cdef Py_ssize_t i
    cdef size_t i
    cf.sbin[0] = sbin_arr[0]
    

    #for i in range(1, cf.ns):
    #    printf("%lf\n",sbin_arr[i])
    #    if sbin_arr[i] != sbin_arr[i-1]:
    #        cf.sbin[i] = 0.5 * (sbin_arr[i] + sbin_arr[i-1])
    #    else: cf.sbin[i] = sbin_arr[i]
    for i in range(cf.ns+1):
        cf.sbin[i] = sbin_arr[i]
        cf.s2bin[i] = cf.sbin[i] * cf.sbin[i]
    cf.s2min = cf.s2bin[0]
    cf.s2max = cf.s2bin[cf.ns]
    if verbose:
        printf("    %d separation bins loaded from bin array. \n", cf.ns)
    
    # Define pi bins
    # Assuming pi bins are also passed
    # Assuming bin array is sorted
    
    if cf.bintype == FCFC_BIN_SPI:

        
        if cf.np > FCFC_MAX_BIN_NUM:
            fprintf(stderr, "ERROR: too many separation bins in `pibin_arr` array.")
            cf_destroy(cf)
            return NULL

    
        cf.pbin = <real*> malloc(sizeof(real) * (cf.np +1))
        cf.p2bin = <real*> malloc(sizeof(real) * (cf.np +1))

        if not cf.pbin or not cf.p2bin:
            fprintf(stderr, "ERROR: failed to allocate memory for pi bins.\n")
            free(cf)
            return NULL

        #cf.pbin[0] = pibin_arr[0]
        #for i in range(1, cf.np):
        #    if pibin_arr[i] != pibin_arr[i-1]:
        #        cf.pbin[i] = 0.5 * (pibin_arr[i] + pibin_arr[i-1])
        #    else:
        #        cf.pbin[i] = pibin_arr[i]
        #    cf.p2bin[i] = cf.pbin[i] * cf.pbin[i]
        for i in range(cf.np + 1):
            cf.pbin[i] = pibin_arr[i]
            cf.p2bin[i] = cf.pbin[i]*cf.pbin[i]
        cf.p2min=cf.p2bin[0]
        cf.p2max=cf.p2bin[cf.np]

        if verbose:
            printf("    %d pi bins loaded from array.\n", cf.np)

    # Setup lookup tables
    cdef real min, max
    cdef size_t offset, ntab
    cdef int j
    if cf.prec != REAL_NAN:
        min = cf.s2bin[0]
        max = cf.s2bin[cf.ns]
        offset = <size_t> (min * cf.prec)
        # Number of elements in lookup table
        ntab = <size_t> (max * cf.prec) - offset

        cf.stab = <size_t*> malloc(sizeof(size_t) * ntab)
        if not cf.stab:
            fprintf(stderr, "ERROR: failed to allocate memory for the lookup table of separations.\n")
            #P_ERR(b"failed to allocate memory for the lookup table of separations\n")
            cf_destroy(cf)
            return NULL

        # Set values for the lookup table of squared separations

        j=1

        for i in range(ntab):
            if <size_t> i+offset < <size_t> (cf.s2bin[j] * cf.prec):
                cf.stab[i] = <size_t> (j-1)
            else:
                j = j+1
                if j > cf.ns:
                    fprintf(stderr, "ERROR: failed to create the lookup table of separations.\n")
                    #P_ERR(b"failed to create the lookup table of separations\n");
                    cf_destroy(cf)
                    return NULL 
                i -= 1

        # Setup table for squared pi bins

        if cf.bintype == FCFC_BIN_SPI:
            cf.sp2min = cf.s2min
            cf.sp2max = cf.s2max
            cf.s2min += cf.p2min
            cf.s2max += cf.p2max        

            min = cf.p2bin[0]
            max = cf.p2bin[cf.np]

            offset = <size_t> (min * cf.prec)
            ntab = <size_t> (max * cf.prec) - offset

            cf.ptab = <size_t*> malloc(sizeof(size_t) * ntab)
            if not cf.ptab:
                fprintf(stderr, "ERROR: failed to allocate memory for the lookup table of pi bins.\n")
                #P_ERR(b"failed to allocate memory for the lookup table of pi bins\n")
                cf_destroy(cf)
                return NULL

            j = 1 

            for i in range(ntab):
                if <size_t> i+offset < <size_t> (cf.p2bin[j] * cf.prec):
                    cf.ptab[i] = <size_t> (j-1)
                else:
                    j = j+1
                    if j > cf.np:
                        fprintf(stderr, "ERROR: failed to allocate memory for the lookup table of pi bins.\n")
                        #P_ERR(b"failed to create the lookup table of pi bins\n")
                        cf_destroy(cf)
                        return NULL 
                    i -= 1

    # Setup lookup table for mu bins

    if cf.bintype == FCFC_BIN_SMU:
        cf.nmu2 = <size_t> cf.nmu * cf.nmu
        cf.mutab = <size_t*> malloc(sizeof(size_t) * cf.nmu2) 
        if not cf.mutab:
            fprintf(stderr, "ERROR: failed to allocate memory for the lookup table of mu bins.\n")
            #P_ERR(b"failed to allocate memory for the lookup table of mu bins\n");
            cf_destroy(cf)
            return NULL
        j = 1 

        for i in range(cf.nmu2):
            if <size_t> i < <size_t> j*j:
                cf.mutab[i] = <size_t> (j-1)
            else:
                j = j+1
                if j > cf.nmu:
                    fprintf(stderr, "ERROR: failed to allocate memory for the lookup table of mu bins.\n")
                    #P_ERR(b"failed to create the lookup table for mu bins\n");
                    cf_destroy(cf)
                    return NULL 
                i -= 1
    if verbose:
        printf("    Separation bins initialised successfully\n")


    # Initialise pair counts

    #cf.pc_idx[0] = <int*> malloc(sizeof(int) * ( cf.npc))
    #cf.pc_idx[1] = <int*> malloc(sizeof(int) * ( cf.npc))

    #if not cf.pc_idx[0] or not cf.pc_idx[1]:
    #    fprintf(stderr, "ERROR: failed to allocate memory for initialising pair counts.\n")
    #    #P_ERR(b"failed to allocate memory for initialising pair counts\n");
    #    cf_destroy(cf)
    #    return NULL
    
    #for i in range(cf.npc):
    #    cf.pc_idx[0][i] = cf.pc_idx[1][i] = -1
    #    if not cf.comp_pc[i]:
    #        continue
    #    for j in range(cf.ncat):
    #        if cf.label[j] == pc[i][0] : cf.pc_idx[0][i] = j
    #        if cf.label[j] == pc[i][1] : cf.pc_idx[1][i] = j
    #    if cf.pc_idx[0][i] == -1 or cf.pc_idx[1][i] == -1:
    #        fprintf(stderr, "ERROR: catalog not found for pair count: %s\n", pc[i]) 
    #        cf_destroy(cf)
    #        return NULL
        
    # Check if any catalog is not used
    #cdef bint found
    
    #for i in range(cf.ncat):
    #    found=False
    #    for j in range(cf.npc):
    #        if cf.comp_pc[j] and (<int> i == cf.pc_idx[0][j] or <int> i == cf.pc_idx[1][j]):
    #            found = True
    #            break
    #    if not found:
    #        fprintf(stderr, "WARNING: catalog <%c> is not required for pair counting.\n", cf.label[i])

    # Allocate memory for the calatogs, pair counts and 2PCFs.

    #cf.data = <DATA **> malloc(sizeof(DATA*) * cf.ncat)

    #if not cf.data:
    #    fprintf(stderr, "ERROR: failed to allocate memory for the input catalogs.\n")
    #    cf_destroy(cf)
    #    return NULL

    #for i in range(cf.ncat):
    #    cf.data[i] = NULL
    
    #cf.ndata = <size_t*> calloc(cf.ncat, sizeof(size_t))
    #if not cf.ndata:
    #    fprintf(stderr, "ERROR: failed to allocate memory for number of inputs.\n")
    #    cf_destroy(cf)
    #    return NULL 

    #cf.wdata = <double*> calloc(cf.ncat, sizeof(double))
    #if not cf.wdata:
    #    fprintf(stderr, "ERROR: failed to allocate memory for weighted number of inputs.\n")
    #    cf_destroy(cf)
    #    return NULL 

    cf.cnt = <pair_count_t**> malloc(sizeof(pair_count_t*) * cf.npc)
    if not cf.cnt:
        fprintf(stderr, "ERROR: failed to allocate memory for pair counts.\n")
        cf_destroy(cf)
        return NULL 
    cf.cnt[0] = NULL # Memory allocated only at the first element

    cf.norm = <double*> calloc(cf.npc, sizeof(double))
    if not cf.norm:
        fprintf(stderr, "ERROR: failed to allocate memory for the normalization of pair counts.\n")
        cf_destroy(cf)
        return NULL 

    cf.ncnt = <double **> malloc(sizeof(double *) * cf.npc)
    if not cf.ncnt:
        fprintf(stderr, "ERROR: failed to allocate memory for normalized pair counts.\n")
        cf_destroy(cf)
        return NULL 
    cf.ncnt[0] = NULL

    # Allocate memory only for the first elements of arrays
    cf.cnt[0] = <pair_count_t *> calloc(cf.ntot * cf.npc, sizeof(pair_count_t))
    cf.ncnt[0] = <double *> malloc(sizeof(double) * cf.ntot * cf.npc)
    if not cf.cnt[0] or not cf.ncnt[0]:
        fprintf(stderr, "ERROR: failed to allocate memory for pair counts.\n")
        cf_destroy(cf)
        return NULL        

    IF OMP == 1   : 
        # Assume OMP always
        printf("    WARNING: Assuming OMP always\n")
        fflush(stdout)
        # Thread-private pair counting pool.
        cf.pcnt = <pair_count_t *> malloc(sizeof(pair_count_t)  *  cf.ntot * cf.nthread)
        if not cf.pcnt:
            fprintf(stderr, "ERROR: failed to allocate memory for thread-private counting array.\n")
            cf_destroy(cf)
            return NULL 
        

    #cf.cf_exp = <char **> malloc(sizeof(char*) * cf.ncf)
    #if not cf.cf_exp:
    #    fprintf(stderr, "ERROR: failed to allocate memory for correlation function estimators.\n")
    #    cf_destroy(cf)
    #    return NULL 
    #for i in range(cf.ncf):
    #    cf.cf_exp[i] = NULL
        

    #cf.ast_cf = <ast_t **> malloc(sizeof(ast_t*) * cf.ncf)
    #if not cf.ast_cf:
    #    fprintf(stderr, "ERROR: failed to allocate memory for correlation function estimators.\n")
    #    cf_destroy(cf)
    #    return NULL 
    #for i in range(cf.ncf):
    #    cf.ast_cf[i] = NULL

        
    #cf.cf = <double **> malloc(sizeof(double*) * cf.ncf)
    #if not cf.cf:
    #    fprintf(stderr, "ERROR: failed to allocate memory for correlation functions.\n")
    #    cf_destroy(cf)
    #    return NULL 
    #cf.cf[0] = NULL
    #cf.cf[0] = <double *> malloc(sizeof(double) * cf.ntot * cf.ncf)
    #if not cf.cf[0]:
    #    fprintf(stderr, "ERROR: failed to allocate memory for correlation functions.\n")
    #    cf_destroy(cf)
    #    return NULL 
    #for i in range(1, cf.ncf):
    #    cf.cf[i] = cf.cf[0] + cf.ntot * i
    cdef size_t ntot
    if cf.nl: # Multipoles are required
        cf.mp = <double **> malloc(sizeof(double *) * cf.ncf)
        if not cf.mp:
            fprintf(stderr, "ERROR: failed to allocate memory for correlation function multipoles.\n")
            cf_destroy(cf)
            return NULL 
            
        cf.mp[0] = NULL
        ntot = <size_t> cf.nl * cf.ns
        cf.mp[0] = <double *> calloc(ntot * cf.ncf, sizeof(double))
        if not cf.mp[0]:
            fprintf(stderr, "ERROR: failed to allocate memory for correlation function multipoles.\n")
            cf_destroy(cf)
            return NULL
        for i in range(1, cf.ncf):
            cf.mp[i] = cf.mp[0] + ntot * i
    elif cf.comp_wp: # Projected CFs are required
        cf.wp = <double **> malloc(sizeof(double *) * cf.ncf)
        if not cf.wp:
            fprintf(stderr, "ERROR: failed to allocate memory for projected correlation function.\n")
            cf_destroy(cf)
            return NULL
        cf.wp[0] = NULL
        cf.wp[0] = <double *> calloc(cf.ns * cf.ncf, sizeof(double))
        if not cf.wp[0]:
            fprintf(stderr, "ERROR: failed to allocate memory for projected correlation function.\n")
            cf_destroy(cf)
            return NULL
        for i in range(1, cf.ncf):
            cf.wp[i] = cf.wp[0] + <size_t> cf.ns * i

    if verbose:
        printf("  Memory allocated for pair counts and correlation functions\n");

    return cf

cdef KDT* kdtree_init_(DATA *data, const size_t ndata) nogil:

    cdef KDT* node = <KDT *> malloc(sizeof(KDT))
    node.n = ndata
    node.data = data
    node.left = NULL
    node.right = NULL
    return node

cdef KDT* kdtree_build_(DATA *data, const size_t ndata, DATA *buf, int *err) nogil:
    if err[0]: return NULL
    if not data or not ndata:
        err[0] = FCFC_ERR_ARG
        return NULL
    
    cdef KDT *node = kdtree_init_(data, ndata)
    if not node:
        err[0] = FCFC_ERR_MEMORY    
        return NULL
    cdef int k
    cdef Py_ssize_t i
    cdef real min, max
    if ndata <= KDTREE_LEAF_SIZE :
        for k in range(3):
            min = data[0].x[k]
            max = data[0].x[k]
            for i in range(ndata):
                if min > data[i].x[k] : min = data[i].x[k]
                if max < data[i].x[k] : max = data[i].x[k]
            node.min.x[k] = min
            node.max.x[k] = max
        return node
    cdef int dir = 0
    cdef real var_max = 0
    cdef real mean, x, var, d
    for k in range(3):
        mean = 0
        min = data[0].x[k]
        max = data[0].x[k]
        for i in range(ndata):
            x = data[i].x[k]
            mean+=x
            if min > x : min = x
            if max < x : max = x
        mean /= <real> ndata
        node.min.x[k] = min
        node.max.x[k] = max
        var = 0
        for i in range(ndata):
            d = data[i].x[k] - mean
            var += d*d
        
        if var > var_max:
            dir = k
            var_max = var
    cdef size_t n = ndata >> 1 

    qselect(data, n, ndata, buf, &dir)
  
    node.left = kdtree_build_(data, n, buf, err);
    node.right = kdtree_build_(data + n, ndata - n, buf, err);
    return node


cdef DATA* npy_to_data(real [:,:] positions, real [:] weights, int nobj, int nthread) nogil:

    cdef Py_ssize_t i

    cdef DATA* data = <DATA *> malloc(nobj * sizeof(DATA))

    for i in range(nobj):
        data[i].x[0] = positions[i,0]
        data[i].x[1] = positions[i,1]
        data[i].x[2] = positions[i,2]
        data[i].s = data[i].x[0]**2 + data[i].x[1]**2 +data[i].x[2]**2 
        data[i].w = weights[i]

    return data


def count_pairs_npy(bint is_auto,
                    real[:,:] data_1, 
                    real[:] weights_1, 
                    double[:] sbin_arr,
                    int n_mu_bin,
                    double[:] pibin_arr, 
                    int bin_scheme,
                    list poles,
                    bint compute_wp,
                    list use_wt,
                    real[:,:] data_2, 
                    real[:] weights_2,
                    int nthreads):

    if is_auto and len(use_wt) > 1:
        raise ValueError("'use_wt' must be one element long for autocorrelations.")
    elif not is_auto and len(use_wt) != 2:
        raise ValueError("'use_wt' must be of length 2 for cross correlations.")
    cdef int n_s_bin, n_tot_bin, n_mp, n_p_bin
    n_s_bin = sbin_arr.shape[0] - 1
    n_p_bin = pibin_arr.shape[0] - 1
    n_mp = len(poles)
    if n_mp>0:
        n_tot_bin = n_s_bin * n_mp
    else:
        n_tot_bin = n_s_bin
    if bin_scheme == 0:
        printf("Computing isotropic 2PCF.\n")
        n_tot_bin = n_s_bin
    elif bin_scheme == 1:
        printf("Computing 2D (s, mu) 2PCF.\n")
        n_tot_bin = n_s_bin * n_mu_bin
    elif bin_scheme == 2:
        printf("Computing 2D (s, pi) 2PCF.\n")
        n_tot_bin = n_s_bin * n_p_bin
        if n_mp > 0:
            raise ValueError("'poles' should be an empty list with 'bin_scheme=2'")
    else:
        raise ValueError(f"'bin_scheme' must be 0, 1 or 2, not {bin_scheme}")
    fflush(stdout)


    cdef DATA buf 
    cdef int err = 0    
    cdef DATA* dat_1 = npy_to_data(data_1, weights_1, data_1.shape[0], nthreads)
    printf("%s", b"Building KDTree for data 1\n")
    fflush(stdout)
    cdef KDT* tree_1 = kdtree_build(dat_1, data_1.shape[0], &buf, &err)    
    printf("%s", b"    Done\n")

    cdef DATA* dat_2
    cdef KDT* tree_2
    if not is_auto:
        dat_2 = npy_to_data(data_2, weights_2, data_2.shape[0], nthreads)
        printf("%s", b"Building KDTree for data 2\n")
        fflush(stdout)
        err = 0
        tree_2 = kdtree_build(dat_2, data_2.shape[0], &buf, &err)    
        printf("%s", b"    Done\n")

    cdef int dprec = -2
    
    
    

    

    
    printf("Using %d threads\n", nthreads)
    fflush(stdout)
    
    cdef int* c_poles = <int*> malloc(len(poles) * sizeof(int))
    for err in range(len(poles)):
        c_poles[err] = poles[err]
    
    cdef bint* c_use_wt = <bint*> malloc(sizeof(bint))
    for err in range(len(use_wt)):
        c_use_wt[err] = use_wt[err]
    cdef size_t* c_ndata = <size_t*> malloc(sizeof(size_t))
    cdef double* c_wdata = <double*> malloc(sizeof(double))
    c_ndata[0] = <size_t> data_1.shape[0]
    c_ndata[1] = <size_t> data_2.shape[0]

    cdef Py_ssize_t j
    for j in range(weights_1.shape[0]):
        c_wdata[0]+=weights_1[j]
    if not is_auto:
        for j in range(weights_2.shape[0]):
            c_wdata[1]+=weights_2[j]
    
        
    cdef CF* cf = cf_init_noconf(verbose=True, 
                                nthread=nthreads, 
                                bin_scheme=bin_scheme, 
                                nmu=n_mu_bin, 
                                sbin_arr = sbin_arr, 
                                pibin_arr = pibin_arr, 
                                dprec=dprec,
                                use_wt = c_use_wt,
                                compute_wp = False,
                                poles = c_poles,
                                npole = len(poles))
    cf.ndata = c_ndata
    cf.wdata = c_wdata
    printf("Built CF struct\n")
    fflush(stdout)

    if use_wt[0] or use_wt[1]:
        out_pair_counts = np.empty(cf.ntot, dtype=np.double)
        
    else:
        out_pair_counts = np.empty(cf.ntot, dtype=np.int64)
    
    out_norm_pair_counts = np.empty(cf.ntot, dtype=np.double)

    # Must add part for cross counts
    cdef Py_ssize_t i 
    if is_auto:
        printf("Counting auto pairs\n")
        fflush(stdout)
        count_pairs(tree_1, tree_1, cf, cf.cnt[0], is_auto, use_wt[0])
        printf("Exit count func\n")
        for i in range(cf.ntot):
            if use_wt[0]:
                cf.cnt[0][i].d *= 2
            else:
                cf.cnt[0][i].i *= 2
    else:
        printf("Counting cross pairs\n")
        fflush(stdout)
        count_pairs(tree_1, tree_2, cf, cf.cnt[0], is_auto, use_wt[0] or use_wt[1])
        printf("Exit count func\n")
    printf("Free tree\n")
    fflush(stdout)
    kdtree_free(tree_1)
    if not is_auto:
        kdtree_free(tree_2)
    # Memory leaks in Data structures
    for i in range(cf.ntot):
        if use_wt:
            out_pair_counts[i] = cf.cnt[0][i].d
            if is_auto:
                cf.norm[0] = <double> (cf.wdata[0] * (cf.wdata[0] - 1 ))
            else:
                cf.norm[0] = <double> (cf.wdata[0] * cf.wdata[1])
        
        else:
            out_pair_counts[i] = cf.cnt[0][i].i
            if is_auto:
                cf.norm[0] = <double> (cf.ndata[0] * (cf.ndata[0] - 1 ))
            else:
                cf.norm[0] = <double> (cf.ndata[0] * cf.ndata[1])
        out_norm_pair_counts[i] = out_pair_counts[i] / (cf.norm[0])
        

    cf_destroy(cf) 
    return out_pair_counts, out_norm_pair_counts

    
    