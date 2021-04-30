#cython: language_level=3, boundscheck=False
cdef extern from "load_conf.h":
    ctypedef struct CONF:
        char *fconf          # Name of the configuration file
        char **input         # CATALOG              
        int ninput           # Number of input catalogues. 
        bint ascii           # Indicate whether there are ASCII catalogues. 
        char *label          # CATALOG_LABEL        
        int *ftype           # CATALOG_TYPE         
        long *skip           # ASCII_SKIP           
        char *comment        # ASCII_COMMENT        
        char **fmtr          # ASCII_FORMATTER      
        char **pos           # POSITION             
        char **wt            # WEIGHT               
        bint *has_wt         # Indicate whether weights are available. 
        char **sel           # SELECTION            
        bint *cnvt           # COORD_CONVERT        

        bint has_cnvt        # Indicate whether coordinate conversion is needed. 
        double omega_m       # OMEGA_M              
        double omega_l       # OMEGA_LAMBDA         
        double omega_k       # 1 - OMEGA_M - OMEGA_LAMBDA 
        double dew           # DE_EOS_W             
        double ecnvt         # CMVDST_ERR           
        char *fcnvt          # Z_CMVDST_CNVT        

        int bintype          # BINNING_SCHEME       
        char **pc            # PAIR_COUNT           
        bint *comp_pc        # Indicate whether to evaluate the pair counts. 
        int npc              # Number of pair counts to be computed. 
        char **pcout         # PAIR_COUNT_FILE      
        char **cf            # CF_ESTIMATOR         
        int ncf              # Number of correlation functions to be computed. 
        char **cfout         # CF_OUTPUT_FILE       
        int *poles           # MULTIPOLE            
        int npole            # Number of multipoles to be evaluated. 
        char **mpout         # MULTIPOLE_FILE       
        bint wp              # PROJECTED_CF         
        char **wpout         # PROJECTED_FILE       

        char *fsbin          # SEP_BIN_FILE         
        double smin          # SEP_BIN_MIN          
        double smax          # SEP_BIN_MAX          
        double ds            # SEP_BIN_SIZE         
        int nsbin            # Number of separation bins 
        int nmu              # MU_BIN_NUM           
        char *fpbin          # PI_BIN_FILE          
        double pmin          # PI_BIN_MIN           
        double pmax          # PI_BIN_MAX           
        double dpi           # PI_BIN_SIZE          
        int npbin            # Number of pi bins    
        int dprec            # SQ_DIST_PREC         

        int ostyle           # OUTPUT_STYLE         
        int ovwrite          # OVERWRITE            
        bint verbose         # VERBOSE     

    CONF *load_conf(const int argc, char *const *argv) nogil   
    void conf_destroy(CONF *conf) nogil      
    CONF *conf_init() nogil

cdef CONF* load_conf_py(char * label, bint *has_wt, bint* cnvt,
                        double omega_m, double omega_l, double omega_k,
                        double dew, char **pc, int npc, int bintype, bint *comp_pc) nogil