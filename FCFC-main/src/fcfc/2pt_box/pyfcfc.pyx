#cython: language_level=3
#cython: boundscheck = False
import cython
from cython.parallel import prange, threadid
cimport openmp
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout, stderr
cimport libc.limits
import numpy
from scipy.special import legendre
from cython cimport floating



arg0_bytes = "FCFC_BOX".encode('utf-8') + b'\x00'
cdef char* arg0_str = arg0_bytes
cdef extern from "define_comm.h":
    ctypedef double real
    ctypedef long int int64_t
    ctypedef unsigned long int uint8_t
    ctypedef union COUNT:
        int64_t i
        double d

cdef extern from "libast.h":
    ctypedef struct ast_t:
        ast_dtype_t dtype;    # Data type for the expression.        */
        long nvar;            # Number of unique variables.          */
        void *var;            # The list of unique variables.        */
        long *vidx;           # Unique indices of variables.         */
        char *exp;            # A copy of the expression string.     */
        void *ast;            # The root node of the AST.            */
        void *error;          # Data structure for error handling.   */
    ctypedef enum ast_dtype_t:
  
        AST_DTYPE_BOOL   = 1,
        AST_DTYPE_INT    = 2,
        AST_DTYPE_LONG   = 4,
        AST_DTYPE_FLOAT  = 8,
        AST_DTYPE_DOUBLE = 16,
        AST_DTYPE_STRING = 32


cdef extern from "define.h":
        

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


cdef extern from "eval_cf.h":
    

    ctypedef struct DATA:
        size_t n;             # number of objects         
        real *x[3];   # coordinates               
        real *w;              # weights                   
        double wt;            # weighted number of objects
    IF WITH_SIMD:
        ctypedef struct CF:
            real bsize[3];        # side lengths of the periodic box               */
            int bintype;          # binning scheme: iso, smu, or spi               */
            real *s2bin;          # edges of squared separation (or s_perp) bins   */
            real *pbin;           # edges of pi bins                               */
            void *stab;           # lookup table for separation (or s_perp) bins   */
            void *ptab;           # lookup table for pi bins                       */
            uint8_t *mutab;       # lookup table for mu bins                       */
            int swidth;           # separation (or s_perp) lookup table entry size */
            int pwidth;           # pi lookup table entry size                     */
            int tabtype;          # type of the lookup tables (integer vs. hybrid) */
            int ns;               # number of separation (or s_perp) bins          */
            int np;               # number of pi bins                              */
            int nmu;              # number of mu bins                              */
            #if FCFC_SIMD  <  FCFC_SIMD_AVX512
            COUNT *pcnt;          # thread-private array for counting in parallel  */
            #else
            #void *pcnt;           # vector-private array for counting in parallel  */
            #endif
            int treetype;         # type of the tree structure.                    *
            int ncat;             #/* number of catalogues to be read                */
            const char *label;    #/* labels of the input catalogues                 */
            DATA *data;           #/* structures for the input catalogues            */
            real rescale;         #/* rescaling factor for the input coordinates     */
            real *sbin_raw;       #/* unrescaled separation (or s_perp) bin edges    */
            real *pbin_raw;       #/* unrescaled pi bin edges                        */
            int nthread;          #/* number of OpenMP threads                       */
            size_t ntot;          #/* total number of bins                           */
            real *sbin;           #/* edges of separation (or s_perp) bins           */
            int verbose;          #/* indicate whether to show detailed outputs      */
            int npc;              #/* number of pair counts to be evaluated          */
            int *pc_idx[2];       #/* pairs to be counted, defined as input indices  */
            #ifdef MPI
            #bint *comp_pc;        #/* indicate whether to evaluate the pair counts   */
            #else#
            const bint *comp_pc;  #/* indicate whether to evaluate the pair counts   */
            #endif#
            bint *wt;             #/* indicate whether using weights for pair counts */
            bint *cat_wt;         #/* indicate whether using weights in catalogues   */
            COUNT **cnt;          #/* array for storing evaluated pair counts        */
            double *norm;         #/* normalisation factors for pair counts          */
            double **ncnt;        #/* array for normalised pair counts               */
            double *rr;           #/* array for analytical RR counts                 */
            int ncf;              #/* number of correlation functions to be computed */
            char **cf_exp;        #/* expression for correlation function estimators */
            ast_t **ast_cf;       #/* abstract syntax trees for 2PCF estimators      */
            double **cf;          #/* array for storing correlation functions        */
            int nl;               #/* number of multipoles to be evaluated           */
            const int *poles;     #/* orders of Legendre polynomials to be evaluated */
            double **mp;          #/* array for storing 2PCF multipoles              */
            bint comp_wp;         #/* indicate whether to compute projected 2PCF     */
            double **wp;          #/* array for storing projected 2PCFs              */
    ELSE:
        ctypedef struct CF:
            real bsize[3];        # side lengths of the periodic box               */
            int bintype;          # binning scheme: iso, smu, or spi               */
            real *s2bin;          # edges of squared separation (or s_perp) bins   */
            real *pbin;           # edges of pi bins                               */
            void *stab;           # lookup table for separation (or s_perp) bins   */
            void *ptab;           # lookup table for pi bins                       */
            uint8_t *mutab;       # lookup table for mu bins                       */
            int swidth;           # separation (or s_perp) lookup table entry size */
            int pwidth;           # pi lookup table entry size                     */
            int tabtype;          # type of the lookup tables (integer vs. hybrid) */
            int ns;               # number of separation (or s_perp) bins          */
            int np;               # number of pi bins                              */
            int nmu;              # number of mu bins                              */
            #if FCFC_SIMD  <  FCFC_SIMD_AVX512
            COUNT *pcnt;          # thread-private array for counting in parallel  */
            #else
            #void *pcnt;           # vector-private array for counting in parallel  */
            #endif
            int treetype;         # type of the tree structure.                    *
            int ncat;             #/* number of catalogues to be read                */
            const char *label;    #/* labels of the input catalogues                 */
            DATA *data;           #/* structures for the input catalogues            */
            real rescale;         #/* rescaling factor for the input coordinates     */
            real *sbin_raw;       #/* unrescaled separation (or s_perp) bin edges    */
            real *pbin_raw;       #/* unrescaled pi bin edges                        */
            int nthread;          #/* number of OpenMP threads                       */
            size_t ntot;          #/* total number of bins                           */
            real *sbin;           #/* edges of separation (or s_perp) bins           */
            int verbose;          #/* indicate whether to show detailed outputs      */
            int npc;              #/* number of pair counts to be evaluated          */
            int *pc_idx[2];       #/* pairs to be counted, defined as input indices  */
            #ifdef MPI
            #bint *comp_pc;        #/* indicate whether to evaluate the pair counts   */
            #else#
            const bint *comp_pc;  #/* indicate whether to evaluate the pair counts   */
            #endif#
            bint *wt;             #/* indicate whether using weights for pair counts */
            bint *cat_wt;         #/* indicate whether using weights in catalogues   */
            COUNT **cnt;          #/* array for storing evaluated pair counts        */
            double *norm;         #/* normalisation factors for pair counts          */
            double **ncnt;        #/* array for normalised pair counts               */
            double *rr;           #/* array for analytical RR counts                 */
            int ncf;              #/* number of correlation functions to be computed */
            char **cf_exp;        #/* expression for correlation function estimators */
            ast_t **ast_cf;       #/* abstract syntax trees for 2PCF estimators      */
            double **cf;          #/* array for storing correlation functions        */
            int nl;               #/* number of multipoles to be evaluated           */
            const int *poles;     #/* orders of Legendre polynomials to be evaluated */
            double **mp;          #/* array for storing 2PCF multipoles              */
            bint comp_wp;         #/* indicate whether to compute projected 2PCF     */
            double **wp;          #/* array for storing projected 2PCFs              */
        
    CF * cf_init() nogil
    void cf_destroy(CF *cf) nogil

cdef extern from "define_para.h":

    ctypedef struct PARA:
        pass

cdef extern from "load_conf.h":

    ctypedef struct CONF:
        pass
    void conf_destroy(CONF* conf) nogil
    void conf_template(void *args) nogil
    

cdef extern from *:
    
    """
    static inline void data_init(DATA *data) {
            data->x[0] = data->x[1] = data->x[2] = data->w = NULL;
            }
    """

    void data_init(DATA* data) nogil
cdef extern from "fcfc.h":

    CF* compute_cf(int argc, char *argv[], DATA* dat, real* sbins, int ns, real* pbins, int np, int nmu) nogil


cdef void npy_to_data(DATA* c_data, 
                        double[:,:] npy_pos,
                        double[:] npy_wt,
                        size_t data_id) noexcept nogil:

    cdef size_t i, j
    c_data[data_id].n = <size_t> npy_wt.shape[0]
    for i in range(3):
        c_data[data_id].x[i] = <double *> malloc(c_data[data_id].n * sizeof(real))
    c_data[data_id].w = <double *> malloc(c_data[data_id].n * sizeof(double))

    for j in prange(c_data[data_id].n, nogil=True):
        for i in range(3):
            c_data[data_id].x[i][j] = <real> npy_pos[j,i]
        c_data[data_id].w[j] = <real> npy_wt[j]
cdef void npy_to_data_f(DATA* c_data, 
                        float[:,:] npy_pos,
                        float[:] npy_wt,
                        size_t data_id) noexcept nogil:

    cdef size_t i, j
    c_data[data_id].n = <size_t> npy_wt.shape[0]
    for i in range(3):
        c_data[data_id].x[i] = <double *> malloc(c_data[data_id].n * sizeof(real))
    c_data[data_id].w = <double *> malloc(c_data[data_id].n * sizeof(double))

    for j in prange(c_data[data_id].n, nogil=True):
        for i in range(3):
            c_data[data_id].x[i][j] = <real> npy_pos[j,i]
        c_data[data_id].w[j] = <real> npy_wt[j]


cdef dict retrieve_paircounts(CF* cf):
    # Results of count(s,mu) or xi(s,mu) as a list. 
    result = {}
    if cf.mp is not NULL:
        result['smin'] = numpy.empty((cf.ns, cf.nmu))
        result['smax'] = numpy.copy(result['smin'])
        
        result['mumin'] = numpy.copy(result['smin'])
        result['mumax'] = numpy.copy(result['smin'])
        for j in range(cf.nmu):
            for i in range(cf.ns):
                result['smin'][i,j] = cf.sbin_raw[i]
                result['smax'][i,j] = cf.sbin_raw[i+1]
                result['mumin'][i,j] = j / <double> cf.nmu
                result['mumax'][i,j] = (j + 1) / <double> cf.nmu

        for idx in range(cf.npc):
            pcnt_label = (<bytes> cf.label[cf.pc_idx[0][idx]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][idx]]).decode('utf-8')
            result[pcnt_label] = numpy.copy(result['smin'])
            for j in range(cf.nmu):
                for i in range(cf.ns):
                    result[pcnt_label][i,j] = cf.ncnt[idx][i + j * cf.ns]
    elif cf.wp is not NULL:
        result['smin'] = numpy.empty((cf.ns, cf.np))
        result['smax'] = numpy.copy(result['smin'])
        
        result['pimin'] = numpy.copy(result['smin'])
        result['pimax'] = numpy.copy(result['smin'])
        for j in range(cf.np):
            for i in range(cf.ns):
                result['smin'][i,j] = cf.sbin_raw[i]
                result['smax'][i,j] = cf.sbin_raw[i+1]
                result['pimin'][i,j] = cf.pbin_raw[j]
                result['pimax'][i,j] = cf.pbin_raw[j+1]

        for idx in range(cf.npc):
            pcnt_label = (<bytes> cf.label[cf.pc_idx[0][idx]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][idx]]).decode('utf-8')
            result[pcnt_label] = numpy.copy(result['smin'])
            for j in range(cf.np):
                for i in range(cf.ns):
                    result[pcnt_label][i,j] = cf.ncnt[idx][i + j * cf.ns]
    else:
        result['smin'] = numpy.empty(cf.ns)
        result['smax'] = numpy.copy(result['smin'])
        for i in range(cf.ns):
                result['smin'][i] = cf.sbin_raw[i]
                result['smax'][i] = cf.sbin_raw[i+1]
        for idx in range(cf.npc):
            pcnt_label = (<bytes> cf.label[cf.pc_idx[0][idx]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][idx]]).decode('utf-8')
            result[pcnt_label] = numpy.copy(result['smin'])
            for i in range(cf.ns):
                result[pcnt_label][i] = cf.ncnt[idx][i]
        
    return result


cdef double[:,:,:] retrieve_correlations(CF* cf):
    # Results of count(s,mu) or xi(s,mu) as a list. 
    
    if cf.mp is not NULL:
        result = numpy.empty((cf.ncf, cf.ns, cf.nmu))
        for idx in range(cf.ncf):
            for j in range(cf.nmu):
                for i in range(cf.ns):
                    result[idx,i,j] = cf.cf[idx][i + j * cf.ns]
    elif cf.wp is not NULL:
        result = numpy.empty((cf.ncf, cf.ns, cf.np))
        for idx in range(cf.ncf):
            for j in range(cf.np):
                for i in range(cf.ns):
                    result[idx,i,j] = cf.cf[idx][i + j * cf.ns]
    else:
        result = numpy.empty((1,cf.ncf, cf.ns))
        for idx in range(cf.ncf):
            for i in range(cf.ns):
                result[0,idx,i] = cf.cf[idx][i]
        
    return result



cdef double[:,:,:] retrieve_multipoles(CF* cf):
    results = numpy.empty((cf.ncf, cf.nl, cf.ns))
    for idx in range(cf.ncf):
        for j in range(cf.nl):
            for i in range(cf.ns):
                results[idx, j, i] = cf.mp[idx][i + j * cf.ns]
    return results

cdef double[:,:] retrieve_projected(CF* cf):
    results = numpy.empty((cf.ncf, cf.ns))
    for idx in range(cf.ncf):
        for i in range(cf.ns):
            results[idx, i] = cf.wp[idx][i]
    return results

def py_compute_cf(list data_cats,
                 list data_wts,
                real[:] sedges,
                real[:] pedges,
                int nmu,
                **kwargs):
    assert len(data_cats) == len(data_wts)
    cdef size_t i,j
    cdef CF* cf = cf_init()
    cdef size_t n_catalogs = len(data_cats)
    
    cdef DATA* dat = <DATA*> calloc(<unsigned int> n_catalogs, sizeof(DATA))
    
    for i in range(n_catalogs):
        data_init(dat + i)
        cat_dtype = data_cats[i].dtype
        wt_dtype = data_wts[i].dtype
        if cat_dtype == numpy.float64 and wt_dtype == numpy.float64:
            npy_to_data(dat, data_cats[i], data_wts[i], i)
        elif cat_dtype == numpy.float32 and wt_dtype == numpy.float32:
            npy_to_data_f(dat, data_cats[i], data_wts[i], i)
        else:
            raise TypeError(f"Positions and weights must have the same dtype. Got {cat_dtype} and {wt_dtype}.")
        
    

    cdef int argc = len(kwargs) + 1
    cdef char** argv = process_kwargs_to_args(kwargs)
    argv[0] = arg0_str
    #argv[1] = conf_string

    cdef real* sedges_ptr = &sedges[0]
    cdef int ns = len(sedges) - 1

    cdef real* pedges_ptr = NULL
    cdef int npi = 0

    if pedges is not None:
        pedges_ptr = &pedges[0]
        npi = len(pedges) - 1
    
    
    
    cf = compute_cf(argc, argv, dat, sedges_ptr, ns, pedges_ptr, npi, nmu)
    if cf is NULL: raise ValueError("C-extension failed, see message above.")
    

    results = {}
    results['number'] = {}
    results['weighted_number'] = {}
    results['labels'] = []
    for i in range(cf.ncat):
        label = (<bytes> cf.label[i]).decode('utf-8')
        results['number'][label] = cf.data[i].n
        results['weighted_number'][label] = cf.data[i].n
        results['labels'].append(label)
    results['normalization'] = {}
    for i in range(cf.npc):
        pcnt_label = (<bytes> cf.label[cf.pc_idx[0][i]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][i]]).decode('utf-8')
        results['normalization'][pcnt_label] = cf.norm[i]
    results['pairs'] = retrieve_paircounts(cf)
    if cf.ncf > 0 :
        results['cf'] = numpy.squeeze(retrieve_correlations(cf))
    results['s'] = numpy.empty(cf.ns)
    for i in range(cf.ns):
        results['s'][i] = 0.5 * (cf.sbin_raw[i] + cf.sbin_raw[i+1])
    if cf.mp is not NULL:
        results['multipoles'] = numpy.asarray(retrieve_multipoles(cf))
    if cf.wp is not NULL:
        results['projected'] = retrieve_projected(cf)
    
    
    
    cf_destroy(cf)

    return results

cdef char** process_kwargs_to_args(dict kwargs):
    str_args_list = []
    cdef int argc = len(kwargs) + 1
    cdef char** argv = <char**> malloc(sizeof(char*) * argc)
    argv[0] = arg0_str
    cdef char *c_string
    for i, (key, val) in enumerate(kwargs.items()):
        str_args_list.append(f"--{key.replace('_', '-')}={str(val)}".replace('\'', '').encode('utf-8') + b'\x00')
        c_string = str_args_list[-1]
        argv[i+1] = c_string
    return argv

