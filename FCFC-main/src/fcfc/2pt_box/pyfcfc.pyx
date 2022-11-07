#cython: language_level=3
#cython: boundscheck = False
import cython
from cython.parallel import prange, threadid
cimport openmp
from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout, stderr
cimport libc.limits
import numpy as np
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
        #COUNT *pcnt;          # thread-private array for counting in parallel  */
        #else
        void *pcnt;           # vector-private array for counting in parallel  */
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

    CF* compute_cf(int argc, char *argv[], DATA* dat) nogil


cdef void npy_to_data(DATA* c_data, 
                        double[:,:] npy_pos,
                        double[:] npy_wt,
                        size_t data_id) nogil:

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
        result['smin'] = np.empty((cf.ns, cf.nmu))
        result['smax'] = np.copy(result['smin'])
        
        result['mumin'] = np.copy(result['smin'])
        result['mumax'] = np.copy(result['smin'])
        for j in range(cf.nmu):
            for i in range(cf.ns):
                result['smin'][i,j] = cf.sbin_raw[i]
                result['smax'][i,j] = cf.sbin_raw[i+1]
                result['mumin'][i,j] = j / <double> cf.nmu
                result['mumax'][i,j] = (j + 1) / <double> cf.nmu

        for idx in range(cf.npc):
            pcnt_label = (<bytes> cf.label[cf.pc_idx[0][idx]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][idx]]).decode('utf-8')
            result[pcnt_label] = np.copy(result['smin'])
            for j in range(cf.nmu):
                for i in range(cf.ns):
                    result[pcnt_label][i,j] = cf.ncnt[idx][i + j * cf.ns]
    elif cf.wp is not NULL:
        result['s_perp_min'] = np.empty((cf.ns, cf.np))
        result['s_perp_max'] = np.copy(result['s_perp_min'])
        
        result['pimin'] = np.copy(result['s_perp_min'])
        result['pimax'] = np.copy(result['s_perp_min'])
        for j in range(cf.np):
            for i in range(cf.ns):
                result['s_perp_min'][i,j] = cf.sbin_raw[i]
                result['s_perp_max'][i,j] = cf.sbin_raw[i+1]
                result['pimin'][i,j] = cf.pbin_raw[j]
                result['pimax'][i,j] = cf.pbin_raw[j+1]

        for idx in range(cf.npc):
            pcnt_label = (<bytes> cf.label[cf.pc_idx[0][idx]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][idx]]).decode('utf-8')
            result[pcnt_label] = np.copy(result['s_perp_min'])
            for j in range(cf.np):
                for i in range(cf.ns):
                    result[pcnt_label][i,j] = cf.ncnt[idx][i + j * cf.ns]
    else:
        result['smin'] = np.empty(cf.ns)
        result['smax'] = np.copy(result['smin'])
        for i in range(cf.ns):
                result['smin'][i] = cf.sbin_raw[i]
                result['smax'][i] = cf.sbin_raw[i+1]
        for idx in range(cf.npc):
            pcnt_label = (<bytes> cf.label[cf.pc_idx[0][idx]]).decode('utf-8')+(<bytes> cf.label[cf.pc_idx[1][idx]]).decode('utf-8')
            result[pcnt_label] = np.copy(result['smin'])
            for i in range(cf.ns):
                result[pcnt_label][i] = cf.ncnt[idx][i]
        
    return result


cdef dict retrieve_correlations(CF* cf):
    # Results of count(s,mu) or xi(s,mu) as a list. 
    result = {}
    if cf.mp is not NULL:
        result['smin'] = np.empty((cf.ns, cf.nmu))
        result['smax'] = np.copy(result['smin'])
        
        result['mumin'] = np.copy(result['smin'])
        result['mumax'] = np.copy(result['smin'])
        for j in range(cf.nmu):
            for i in range(cf.ns):
                result['smin'][i,j] = cf.sbin_raw[i]
                result['smax'][i,j] = cf.sbin_raw[i+1]
                result['mumin'][i,j] = j / <double> cf.nmu
                result['mumax'][i,j] = (j + 1) / <double> cf.nmu
        result['cf'] = np.empty((cf.ncf, cf.ns, cf.nmu))
        for idx in range(cf.ncf):
            for j in range(cf.nmu):
                for i in range(cf.ns):
                    result['cf'][idx,i,j] = cf.cf[idx][i + j * cf.ns]
    elif cf.wp is not NULL:
        result['s_perp_min'] = np.empty((cf.ns, cf.np))
        result['s_perp_max'] = np.copy(result['s_perp_min'])
        
        result['pimin'] = np.copy(result['s_perp_min'])
        result['pimax'] = np.copy(result['s_perp_min'])
        for j in range(cf.np):
            for i in range(cf.ns):
                result['s_perp_min'][i,j] = cf.sbin_raw[i]
                result['s_perp_max'][i,j] = cf.sbin_raw[i+1]
                result['pimin'][i,j] = cf.pbin_raw[j]
                result['pimax'][i,j] = cf.pbin_raw[j+1]
        result['cf'] = np.empty((cf.ncf, cf.ns, cf.np))
        for idx in range(cf.ncf):
            for j in range(cf.np):
                for i in range(cf.ns):
                    result['cf'][idx,i,j] = cf.cf[idx][i + j * cf.ns]
    else:
        result['smin'] = np.empty(cf.ns)
        result['smax'] = np.copy(result['smin'])
        for i in range(cf.ns):
                result['smin'][i] = cf.sbin_raw[i]
                result['smax'][i] = cf.sbin_raw[i+1]
        result['cf'] = np.empty((cf.ncf, cf.ns))
        for idx in range(cf.ncf):
            for i in range(cf.ns):
                result['cf'][idx,i] = cf.cf[idx][i]
        
    return result



cdef double[:,:,:] retrieve_multipoles(CF* cf):
    results = np.empty((cf.ncf, cf.nl, cf.ns))
    for idx in range(cf.ncf):
        for j in range(cf.nl):
            for i in range(cf.ns):
                results[idx, j, i] = cf.mp[idx][i + j * cf.ns]
    return results

cdef double[:,:] retrieve_projected(CF* cf):
    results = np.empty((cf.ncf, cf.ns))
    for idx in range(cf.ncf):
        for i in range(cf.ns):
            results[idx, i] = cf.wp[idx][i]
    return results

def py_compute_cf(list data_cats,
                  list data_wts,
                  fcfc_conf_file,
                ) :
    assert len(data_cats) == len(data_wts)
    cdef size_t i,j
    cdef CF* cf = cf_init()
    cdef size_t n_catalogs = len(data_cats)
    
    cdef DATA* dat = <DATA*> calloc(<unsigned int> n_catalogs, sizeof(DATA))
    for i in range(n_catalogs):
        data_init(dat + i)
        npy_to_data(dat, data_cats[i], data_wts[i], i)
    

    # Define name of the configuration file to use
    # TODO: Generate temporary configuration file at fixed location
    #       from options passed to function. See i.e. 
    #       https://github.com/dforero0896/fcfcwrap
    # TODO: (Alternative/harder) override CONF structure
    conf = f"--conf={fcfc_conf_file}"
    conf_bytes = conf.encode('utf-8') + b'\x00'
    cdef char* conf_string = conf_bytes

    

    # Define dummy argc, argv to send to powspec main function
    # This should remain similar once we generate a conf file.
    cdef int argc = 2
    cdef char* argv[3]
    argv[0] = arg0_str
    argv[1] = conf_string

    print(n_catalogs)
    cf = compute_cf(argc, argv, dat)
    if cf is NULL: raise ValueError("Could not compute correlations.")
    cdef int idx = 0

    results = {}
    results['number'] = [cf.data[i].n for i in range(cf.ncat)]
    results['weighted_number'] = [cf.data[i].wt for i in range(cf.ncat)]
    results['normalization'] = [cf.norm[i] for i in range(cf.npc)]



    
    results['pairs'] = retrieve_paircounts(cf)
    results['cf'] = retrieve_correlations(cf)
    results['s'] = np.empty(cf.ns)
    for i in range(cf.ns):
        results['s'][i] = 0.5 * (cf.sbin_raw[i] + cf.sbin_raw[i+1])
    if cf.mp is not NULL:
        results['multipoles'] = retrieve_multipoles(cf)
    if cf.wp is not NULL:
        results['projected'] = retrieve_projected(cf)
    
    
    
    cf_destroy(cf)

    return results

def gen_box_conf(box_size,
                data_struct,
                binning_scheme,
                pair_count,
                pair_count_file,
                cf_estimator,
                cf_output_file,
                multipole,
                multipole_file,
                projected_cf,
                projected_file,
                sep_bin_file,
                sep_bin_min=0,
                sep_bin_max=200,
                sep_bin_size=5,
                mu_bin_num="",
                pi_bin_file="",
                pi_bin_min="",
                pi_bin_max="",
                pi_bin_size="",
                output_format=1,
                overwrite=1,
                verbose=1):
    conf_text=f"""
    WEIGHT          = 
    # Weights for pair counts (unset: 1, i.e. no weight).
    # Column indicator or expression, same dimension as `DATA_CATALOG`.
    BOX_SIZE        = {box_size}
        # Side length of the periodic box for the input catalogs.
        # Double-precision number.
    ################################################################
    #  Configurations for the 2-point correlation function (2PCF)  #
    ################################################################
    DATA_STRUCT     = {data_struct}
        # Data structure for evaluating pair counts, integer (unset: 0).
        # Allowed values are:
        # * 0: k-d tree;
        # * 1: ball tree.
    BINNING_SCHEME  = {binning_scheme}
        # Binning scheme of the 2PCFs, integer (unset: 0).
        # Allowed values are:
        # * 0: isotropic separation bins;
        # * 1: (s, mu) bins (required by 2PCF multipoles);
        # * 2: (s_perp, pi) bins (required by projected 2PCFs);
    PAIR_COUNT      = {pair_count}
        # Identifiers of pairs to be counted or read, string or string array.
        # Pairs are labelled by their source catalogs.
        # E.g., "DD" denotes auto pairs from the catalog 'D',
        # while "DR" denotes cross pairs from catalogs 'D' and 'R'.
    PAIR_COUNT_FILE = {pair_count_file}
        # Name of the files for storing pair counts.
        # String, same dimension as `PAIR_COUNT`.
        # If a specified file exists, then the pair counts are read from this file;
        # otherwise the pair counts are evaluated and saved to the file.
    CF_ESTIMATOR    = {cf_estimator}
        # Estimator of the 2PCFs to be evaluated, string or string array.
        # It must be an expression with pair identifiers.
        # In particular, "@@" denotes the analytical RR pair counts.
    CF_OUTPUT_FILE  = {cf_output_file}
        # Name of the files for saving 2PCFs with the desired binning scheme.
        # String, same dimension as `CF_ESTIMATOR`.
    MULTIPOLE       = {multipole}
        # Orders of Legendre multipoles to be evaluated, integer or integer array.
    MULTIPOLE_FILE  = {multipole_file}
        # Name of the files for saving 2PCF multipoles.
        # String, same dimension as `CF_ESTIMATOR`.
    PROJECTED_CF    = {projected_cf}
        # Boolean option, indicate whether computing the projected 2PCFs (unset: F).
    PROJECTED_FILE  = {projected_file}
        # Name of the files for saving projected 2PCFs.
        # String, same dimension as `CF_ESTIMATOR`.
    #############################
    #  Definitions of the bins  #
    #############################
    SEP_BIN_FILE    = {sep_bin_file}
        # Filename of the table defining edges of separation (or s_perp) bins.
        # It mush be a text file with two columns, for the lower and upper limits
        # of the distance bins, respectively.
        # Lines starting with '#' are omitted.
    SEP_BIN_MIN     = {sep_bin_min}
    SEP_BIN_MAX     = {sep_bin_max}
    SEP_BIN_SIZE    = {sep_bin_size}
        # Lower and upper limits, and width of linear separation (or s_perp) bins.
        # Double-precision numbers. They are only used if `SEP_BIN_FILE` is unset.
    MU_BIN_NUM      = {mu_bin_num}
        # Number of linear mu bins in the range [0,1), integer.
    PI_BIN_FILE     = {pi_bin_file}
        # Filename of the table defining edges of pi (a.k.a. s_para) bins.
        # Lines starting with '#' are omitted.
    PI_BIN_MIN      = {pi_bin_min}
    PI_BIN_MAX      = {pi_bin_max}
    PI_BIN_SIZE     = {pi_bin_size}
        # Lower and upper limits, and width of linear pi bins.
        # Double-precision numbers. They are only used if `PI_BIN_FILE` is unset.
    
    ####################
    #  Other settings  #
    ####################
    OUTPUT_FORMAT   = {output_format}
        # Format of the output `PAIR_COUNT_FILE`, integer (unset: 0).
        # Allowed values are:
        # * 0: FCFC binary format;
        # * 1: ASCII text format.
    OVERWRITE       = {overwrite}
        # Flag indicating whether to overwrite existing files, integer (unset: 0).
        # Allowed values are:
        # * 0: quit the program when an output file exist;
        # * 1: overwrite 2PCF files silently, but keep existing pair count files;
        # * 2 or larger: overwrite all files silently;
        # * negative: notify for decisions, and the maximum allowed number of failed
        #             trials are given by the absolute value of this number.
    VERBOSE         = {verbose}
        # Boolean option, indicate whether to show detailed outputs (unset: T).
    """
    return conf_text
    


    

    


    
    