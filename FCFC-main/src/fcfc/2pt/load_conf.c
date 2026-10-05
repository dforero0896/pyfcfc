/*******************************************************************************
* 2pt/load_conf.c: this file is part of the FCFC program.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2022 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#include "define.h"
#include "load_conf.h"
#include "libcfg.h"
#include "libast.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <math.h>

/*============================================================================*\
                           Macros for error handling
\*============================================================================*/
/* Check existence of configuration parameters. */
#define CHECK_EXIST_PARAM(name, cfg, var)                       \
  if (!cfg_is_set((cfg), (var))) {                              \
    P_ERR(FMT_KEY(name) " is not set\n");                       \
    return FCFC_ERR_CFG;                                        \
  }
#define CHECK_EXIST_ARRAY(name, cfg, var, num)                  \
  if (!(num = cfg_get_size((cfg), (var)))) {                    \
    P_ERR(FMT_KEY(name) " is not set\n");                       \
    return FCFC_ERR_CFG;                                        \
  }

/* Check length of array. */
#define CHECK_ARRAY_LENGTH(name, cfg, var, fmt, num, nexp)      \
  if (num < (nexp)) {                                           \
    P_ERR("too few elements of " FMT_KEY(name) "\n");           \
    return FCFC_ERR_CFG;                                        \
  }                                                             \
  if (num > (nexp)) {                                           \
    P_WRN("omitting the following " FMT_KEY(name) ":");         \
    for (int i = (nexp); i < num; i++)                          \
      fprintf(stderr, " " fmt, (var)[i]);                       \
    fprintf(stderr, "\n");                                      \
  }

#define CHECK_STR_ARRAY_LENGTH(name, cfg, var, num, nexp)       \
  if (num < (nexp)) {                                           \
    P_ERR("too few elements of " FMT_KEY(name) "\n");           \
    return FCFC_ERR_CFG;                                        \
  }                                                             \
  if (num > (nexp)) {                                           \
    P_WRN("omitting the following " FMT_KEY(name) ":\n");       \
    for (int i = (nexp); i < num; i++)                          \
      fprintf(stderr, "  %s\n", (var)[i]);                      \
  }

/* Release memory for configuration parameters. */
#define FREE_ARRAY(x)           {if(x) free(x);}
#define FREE_STR_ARRAY(x)       {if(x) {if (*(x)) free(*(x)); free(x);}}

/* Print the warning and error messages. */
#define P_CFG_WRN(cfg)  cfg_pwarn(cfg, stderr, FMT_WARN);
#define P_CFG_ERR(cfg)  {                                       \
  cfg_perror(cfg, stderr, FMT_ERR);                             \
  cfg_destroy(cfg);                                             \
  return NULL;                                                  \
}


/*============================================================================*\
                    Functions called via command line flags
\*============================================================================*/

/******************************************************************************
Function `conf_init`:
  Initialise the structure for storing configurations.
Return:
  Address of the structure.
******************************************************************************/
static CONF *conf_init(void) {
  CONF *conf = calloc(1, sizeof *conf);
  if (!conf) return NULL;
  conf->fconf = conf->label = conf->comment = NULL;
  conf->fcnvt =  NULL;
  conf->input = conf->fmtr = conf->pos = conf->wt = conf->sel = NULL;
  conf->pc = conf->pcout = conf->cf = conf->cfout = NULL;
  conf->mpout = conf->wpout = NULL;
  conf->ftype = conf->poles = NULL;
  conf->has_wt = conf->cnvt = NULL;
  conf->comp_pc = NULL;
  conf->skip = NULL;
  return conf;
}

/******************************************************************************
Function `conf_read`:
  Read configurations.
Arguments:
  * `conf`:     structure for storing configurations;
  * `argc`:     number of arguments passed via command line;
  * `argv`:     array of command line arguments.
Return:
  Interface of libcfg.
******************************************************************************/
static cfg_t *conf_read(CONF *conf, const int argc, char *const *argv) {
  if (!conf) {
    P_ERR("the structure for configurations is not initialised\n");
    return NULL;
  }
  cfg_t *cfg = cfg_init();
  if (!cfg) P_CFG_ERR(cfg);

  /* Configuration parameters. */
  const cfg_param_t params[] = {
    {'c', "conf"        , "CONFIG_FILE"    , CFG_DTYPE_STR , &conf->fconf   },
    {'l', "label"       , "CATALOG_LABEL"  , CFG_ARRAY_CHAR, &conf->label   },
    {'w', "weight"      , "WEIGHT"         , CFG_ARRAY_STR , &conf->wt      },
    { 0 , "convert"     , "COORD_CONVERT"  , CFG_ARRAY_BOOL, &conf->cnvt    },
    {'d', "omega-m"     , "OMEGA_M"        , CFG_DTYPE_DBL , &conf->omega_m },
    { 0 , "omega-l"     , "OMEGA_LAMBDA"   , CFG_DTYPE_DBL , &conf->omega_l },
    { 0 , "eos-w"       , "DE_EOS_W"       , CFG_DTYPE_DBL , &conf->dew     },
    { 0 , "cmvdst-err"  , "CMVDST_ERR"     , CFG_DTYPE_DBL , &conf->ecnvt   },
    { 0 , "cmvdst-file" , "Z_CMVDST_FILE"  , CFG_DTYPE_STR , &conf->fcnvt   },
    {'S', "data-struct" , "DATA_STRUCT"    , CFG_DTYPE_INT , &conf->dstruct },
    {'B', "bin"         , "BINNING_SCHEME" , CFG_DTYPE_INT , &conf->bintype },
    {'p', "pair"        , "PAIR_COUNT"     , CFG_ARRAY_STR , &conf->pc      },
    {'P', "pair-output" , "PAIR_COUNT_FILE", CFG_ARRAY_STR , &conf->pcout   },
    {'e', "cf"          , "CF_ESTIMATOR"   , CFG_ARRAY_STR , &conf->cf      },
    {'E', "cf-output"   , "CF_OUTPUT_FILE" , CFG_ARRAY_STR , &conf->cfout   },
    {'m', "multipole"   , "MULTIPOLE"      , CFG_ARRAY_INT , &conf->poles   },
    {'M', "mp-output"   , "MULTIPOLE_FILE" , CFG_ARRAY_STR , &conf->mpout   },
    {'u', "wp"          , "PROJECTED_CF"   , CFG_DTYPE_BOOL, &conf->wp      },
    {'U', "wp-output"   , "PROJECTED_FILE" , CFG_ARRAY_STR , &conf->wpout   },
    {'F', "out-format"  , "OUTPUT_FORMAT"  , CFG_DTYPE_INT , &conf->ofmt    },
    {'O', "overwrite"   , "OVERWRITE"      , CFG_DTYPE_INT , &conf->ovwrite },
    {'v', "verbose"     , "VERBOSE"        , CFG_DTYPE_BOOL, &conf->verbose }
  };

  /* Register parameters.  (The CLI-only help/version/template functions
   * of the original program are not registered: in library mode every
   * option arrives through the validated keyword arguments.) */
  if (cfg_set_params(cfg, params, sizeof(params) / sizeof(params[0])))
      P_CFG_ERR(cfg);
  P_CFG_WRN(cfg);

  /* Read configurations from command line options. */
  int optidx;
  if (cfg_read_opts(cfg, argc, argv, FCFC_PRIOR_CMD, &optidx))
    P_CFG_ERR(cfg);
  P_CFG_WRN(cfg);

  /* Read parameters from configuration file. */
  bool use_default_conf = false;
  if (!cfg_is_set(cfg, &conf->fconf)) {
    conf->fconf = DEFAULT_CONF_FILE;
    use_default_conf = true;
  }
  if (access(conf->fconf, R_OK)) {
    if (!use_default_conf) {
      P_ERR("cannot access the configuration file: `%s'\n", conf->fconf);
      P_CFG_ERR(cfg);
    }
  }
  else if (cfg_read_file(cfg, conf->fconf, FCFC_PRIOR_FILE)) P_CFG_ERR(cfg);
  P_CFG_WRN(cfg);

  return cfg;
}


/*============================================================================*\
                      Functions for parameter verification
\*============================================================================*/

/******************************************************************************
Function `check_input`:
  Check whether an input file can be read.
Arguments:
  * `fname`:    filename of the input file;
  * `key`:      keyword of the input file.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
static inline int check_input(const char *fname, const char *key) {
  if (!fname || *fname == '\0') {
    P_ERR("the input " FMT_KEY(%s) " is not set\n", key);
    return FCFC_ERR_CFG;
  }
  if (access(fname, R_OK)) {
    P_ERR("cannot access " FMT_KEY(%s) ": `%s'\n", key, fname);
    return FCFC_ERR_FILE;
  }
  return 0;
}

/******************************************************************************
Function `check_output`:
  Check whether an output file can be written.
Arguments:
  * `fname`:    filename of the input file;
  * `key`:      keyword of the input file;
  * `ovwrite`:  option for overwriting exisiting files;
  * `force`:    flag for overwriting files without notification.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
static int check_output(char *fname, const char *key, int ovwrite,
    const int force) {
  if (!fname || *fname == '\0') {
    P_ERR("the output file " FMT_KEY(%s) " is not set\n", key);
    return FCFC_ERR_CFG;
  }

  /* Check if the file exists. */
  if (!access(fname, F_OK)) {
    if (ovwrite < 0) {                          /* ask for decision */
      P_WRN("the output file " FMT_KEY(%s) " exists: `%s'\n", key, fname);
      char confirm = 0;
      for (int i = 0; i != ovwrite; i--) {
        fprintf(stderr, "Are you going to overwrite it? (y/n): ");
        if (scanf("%c", &confirm) != 1) continue;
        int c;
        while((c = getchar()) != '\n' && c != EOF) continue;
        if (confirm == 'n' || confirm == 'N') {
          ovwrite = force - 1;
          break;
        }
        else if (confirm == 'y' || confirm == 'Y') {
          ovwrite = force;
          break;
        }
      }
      if (confirm != 'y' && confirm != 'Y' &&
          confirm != 'n' && confirm != 'N') {
        P_ERR("too many failed inputs\n");
        return FCFC_ERR_FILE;
      }
    }

    if (ovwrite <= FCFC_OVERWRITE_NONE) {       /* not overwriting */
      P_ERR("abort to avoid overwriting " FMT_KEY(%s) ": `%s'\n", key, fname);
      return FCFC_ERR_FILE;
    }
    else if (ovwrite >= force) {                /* force overwriting */
      P_WRN(FMT_KEY(%s) " will be overwritten: `%s'\n", key, fname);
    }
    else {                                      /* this is an input file */
      if (access(fname, R_OK)) {
        P_ERR("cannot access " FMT_KEY(%s) ": `%s'\n", key, fname);
        return FCFC_ERR_FILE;
      }
      return FCFC_ERR_SAVE;     /* indicate that the file will be read */
    }

    /* Check file permission for overwriting. */
    if (access(fname, W_OK)) {
      P_ERR("cannot write to file: `%s'\n", fname);
      return FCFC_ERR_FILE;
    }
  }
  /* Check the path permission. */
  else {
    char *end;
    if ((end = strrchr(fname, FCFC_PATH_SEP)) != NULL) {
      *end = '\0';
      if (access(fname, X_OK)) {
        P_ERR("cannot access the directory `%s'\n", fname);
        return FCFC_ERR_FILE;
      }
      *end = FCFC_PATH_SEP;
    }
  }
  return 0;
}

/******************************************************************************
Function `check_cosmo`:
  Verify cosmological parameters for coordinate conversion.
Arguments:
  * `cfg`:      interface of libcfg;
  * `conf`:     structure for storing configurations.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
static int check_cosmo(const cfg_t *cfg, CONF *conf) {
  /* Check OMEGA_M. */
  CHECK_EXIST_PARAM(OMEGA_M, cfg, &conf->omega_m);
  if (conf->omega_m <= 0 || conf->omega_m > 1) {
    P_ERR(FMT_KEY(OMEGA_M) " must be > 0 and <= 1\n");
    return FCFC_ERR_CFG;
  }

  /* Check OMEGA_LAMBDA */
  if (!cfg_is_set(cfg, &conf->omega_l)) {
    conf->omega_l = 1 - conf->omega_m;
    conf->omega_k = 0;
  }
  else if (conf->omega_l < 0) {
    P_ERR(FMT_KEY(OMEGA_LAMBDA) " must be >= 0\n");
    return FCFC_ERR_CFG;
  }
  else conf->omega_k = 1 - conf->omega_m - conf->omega_l;

  /* Check DE_EOS_W. */
  if (!cfg_is_set(cfg, &conf->dew)) conf->dew = DEFAULT_DE_EOS_W;
  else if (conf->dew > -1 / (double) 3) {
    P_ERR(FMT_KEY(DE_EOS_W) " must be <= -1/3\n");
    return FCFC_ERR_CFG;
  }

  /* Finally, make sure that H^2 (z) > 0. */
  double w3 = conf->dew * 3;
  double widx = w3 + 1;
  if (conf->omega_k * pow(conf->omega_l * (-widx), widx / w3) <=
      conf->omega_l * w3 * pow(conf->omega_m, widx / w3)) {
    P_ERR("negative H^2 given the cosmological parameters\n");
    return FCFC_ERR_CFG;
  }
  return 0;
}

/******************************************************************************
Function `conf_verify`:
  Verify configuration parameters.
Arguments:
  * `cfg`:      interface of libcfg;
  * `conf`:     structure for storing configurations.
Return:
  Zero on success; non-zero on error.
******************************************************************************/
static int conf_verify(const cfg_t *cfg, CONF *conf) {
  int e, num;

  /* CATALOG_LABEL */
  num = cfg_get_size(cfg, &conf->label);
  conf->ninput = num;
  if (!num) {
    P_ERR("no " FMT_KEY(CATALOG_LABEL) " is specified\n");
    return FCFC_ERR_CFG;
  }
  else {
    CHECK_ARRAY_LENGTH(CATALOG_LABEL, cfg, conf->label, "%c", num, conf->ninput);
    for (int i = 0; i < conf->ninput; i++) {
      if (conf->label[i] < 'A' || conf->label[i] > 'Z') {
        P_ERR("invalid " FMT_KEY(CATALOG_LABEL) ": %c\n", conf->label[i]);
        return FCFC_ERR_CFG;
      }
    }
    /* Check duplicates. */
    for (int i = 0; i < conf->ninput - 1; i++) {
      for (int j = i + 1; j < conf->ninput; j++) {
        if (conf->label[i] == conf->label[j]) {
          P_ERR("duplicate " FMT_KEY(CATALOG_LABEL) ": %c\n", conf->label[i]);
          return FCFC_ERR_CFG;
        }
      }
    }
  }


  /* WEIGHT */
  if (!(conf->has_wt = malloc(conf->ninput * sizeof(bool)))) {
    P_ERR("failed to allocate memory for " FMT_KEY(WEIGHT) "\n");
    return FCFC_ERR_MEMORY;
  }
  /* The Python wrapper passes WEIGHT = '1' for catalogues whose weights
   * are all exactly 1, which enables the (faster and exact) integer
   * counting path; any other value enables weighted counting. */
  if ((num = cfg_get_size(cfg, &conf->wt))) {
    CHECK_STR_ARRAY_LENGTH(WEIGHT, cfg, conf->wt, num, conf->ninput);
    for (int i = 0; i < conf->ninput; i++) {
      /* Disable weighting if the weight is 1 or empty. */
      if ((conf->wt[i][0] == '1' && conf->wt[i][1] == '\0') ||
          (((conf->wt[i][0] == '\'' && conf->wt[i][1] == '\'') ||
          (conf->wt[i][0] == '"' && conf->wt[i][1] == '"')) &&
          conf->wt[i][2] == '\0')) conf->has_wt[i] = false;
      else conf->has_wt[i] = true;
    }
  }
  else {
    for (int i = 0; i < conf->ninput; i++) conf->has_wt[i] = true;
  }
  /* COORD_CONVERT */
  if ((num = cfg_get_size(cfg, &conf->cnvt))) {
    if (num == 1 && conf->ninput > 1) {
      bool *tmp = realloc(conf->cnvt, conf->ninput * sizeof(bool));
      if (!tmp) {
        P_ERR("failed to allocate memory for " FMT_KEY(COORD_CONVERT) "\n");
        return FCFC_ERR_MEMORY;
      }
      conf->cnvt = tmp;
      for (int i = 1; i < conf->ninput; i++) conf->cnvt[i] = conf->cnvt[0];
    }
    else if (num < conf->ninput) {
      P_ERR("too few elements of " FMT_KEY(COORD_CONVERT) "\n");
      return FCFC_ERR_CFG;
    }
    if (num > conf->ninput) {
      P_WRN("omitting the following " FMT_KEY(COORD_CONVERT) ":");
      for (int i = conf->ninput; i < num; i++)
        fprintf(stderr, " %c", conf->cnvt[i] ? 'T' : 'F');
      fprintf(stderr, "\n");
    }
  }

  /* Check the fiducial cosmology. */
  conf->has_cnvt = false;
  if (!conf->cnvt && DEFAULT_COORD_CNVT == true) conf->has_cnvt = true;
  else if (conf->cnvt) {
    for (int i = 0; i < conf->ninput; i++) {
      if (conf->cnvt[i]) {
        conf->has_cnvt = true;
        break;
      }
    }
  }
  if (conf->has_cnvt) {
    /* Check Z_CMVDST_FILE. */
    if (cfg_is_set(cfg, &conf->fcnvt)) {
      if ((e = check_input(conf->fcnvt, "Z_CMVDST_FILE"))) return e;
    }
    else {
      if ((e = check_cosmo(cfg, conf))) return e;
      /* Check CMVDST_ERR. */
      if (!cfg_is_set(cfg, &conf->ecnvt)) conf->ecnvt = DEFAULT_CNVT_ERR;
      if (conf->ecnvt < DBL_EPSILON) {
        P_ERR(FMT_KEY(CMVDST_ERR) " is smaller than the machine epsilon.\n");
        return FCFC_ERR_CFG;
      }
    }
  }

  /* OVERWRITE */
  if (!cfg_is_set(cfg, &conf->ovwrite)) conf->ovwrite = DEFAULT_OVERWRITE;

  /* DATA_STRUCT */
  if (!cfg_is_set(cfg, &conf->dstruct)) conf->dstruct = DEFAULT_STRUCT;
  switch (conf->dstruct) {
    case FCFC_STRUCT_KDTREE:
    case FCFC_STRUCT_BALLTREE:
      break;
    default:
      P_ERR("invalid " FMT_KEY(DATA_STRUCT) ": %d\n", conf->dstruct);
      return FCFC_ERR_CFG;
  }

  /* BINNING_SCHEME */
  if (!cfg_is_set(cfg, &conf->bintype)) conf->bintype = DEFAULT_BINNING;
  switch (conf->bintype) {
    case FCFC_BIN_ISO:
    case FCFC_BIN_SMU:
    case FCFC_BIN_SPI:
      break;
    default:
      P_ERR("invalid " FMT_KEY(BINNING_SCHEME) ": %d\n", conf->bintype);
      return FCFC_ERR_CFG;
  }

  /* PAIR_COUNT */
  CHECK_EXIST_ARRAY(PAIR_COUNT, cfg, &conf->pc, conf->npc);
  /* Simple validation. */
  for (int i = 0; i < conf->npc; i++) {
    char *s = conf->pc[i];
    if (s[0] < 'A' || s[0] > 'Z' || s[1] < 'A' || s[1] > 'Z' || s[2]) {
      P_ERR("invalid " FMT_KEY(PAIR_COUNT) ": %s\n", s);
      return FCFC_ERR_CFG;
    }
  }
  /* Check duplicates. */
  for (int i = 0; i < conf->npc - 1; i++) {
    for (int j = i + 1; j < conf->npc; j++) {
      if (conf->pc[i][0] == conf->pc[j][0] &&
          conf->pc[i][1] == conf->pc[j][1]) {
        P_ERR("duplicate " FMT_KEY(PAIR_COUNT) ": %s\n", conf->pc[i]);
        return FCFC_ERR_CFG;
      }
    }
  }

  if (!(conf->comp_pc = malloc(conf->npc * sizeof(bool)))) {
    P_ERR("failed to allocate memory for checking pair counts\n");
    return FCFC_ERR_MEMORY;
  }

  /* PAIR_COUNT_FILE */
  if (cfg_is_set(cfg, &conf->pcout)){
    CHECK_EXIST_ARRAY(PAIR_COUNT_FILE, cfg, &conf->pcout, num);
    CHECK_STR_ARRAY_LENGTH(PAIR_COUNT_FILE, cfg, conf->pcout, num, conf->npc);
    for (int i = 0; i < conf->npc; i++) {
      e = check_output(conf->pcout[i], "PAIR_COUNT_FILE", conf->ovwrite,
          FCFC_OVERWRITE_ALL);
      if (!e) conf->comp_pc[i] = true;
      else if (e == FCFC_ERR_SAVE) conf->comp_pc[i] = false;
      else return e;

      /* Check if the labels exist if evaluating pair counts. */
      if (conf->comp_pc[i]) {
        int label_found = 0;
        for (int j = 0; j < conf->ninput; j++) {
          if (conf->pc[i][0] == conf->label[j]) label_found += 1;
          if (conf->pc[i][1] == conf->label[j]) label_found += 1;
        }
        if (label_found != 2) {
          P_ERR("catalog label not found for " FMT_KEY(PAIR_COUNT) ": %s\n",
              conf->pc[i]);
          return FCFC_ERR_CFG;
        }
      }
    }
  }
  else{
    conf->pcout = NULL;
    for (int i = 0; i < conf->npc; i++) {
      conf->comp_pc[i] = true;
      /* Check if the labels exist if evaluating pair counts. */
      if (conf->comp_pc[i]) {
        int label_found = 0;
        for (int j = 0; j < conf->ninput; j++) {
          if (conf->pc[i][0] == conf->label[j]) label_found += 1;
          if (conf->pc[i][1] == conf->label[j]) label_found += 1;
        }
        if (label_found != 2) {
          P_ERR("catalog label not found for " FMT_KEY(PAIR_COUNT) ": %s\n",
              conf->pc[i]);
          return FCFC_ERR_CFG;
        }
      }
    }
  }

  /* CF_ESTIMATOR */
  if ((conf->ncf = cfg_get_size(cfg, &conf->cf))) {
    for (int i = 0; i < conf->ncf; i++) {
      if (!conf->cf[i] || !(*conf->cf[i])) {
        P_ERR("unexpected empty " FMT_KEY(CF_ESTIMATOR) "\n");
        return FCFC_ERR_CFG;
      }
    }
    /* CF_OUTPUT_FILE */
    if (cfg_is_set(cfg, &conf->cfout)){
      CHECK_EXIST_ARRAY(CF_OUTPUT_FILE, cfg, &conf->cfout, num);
      CHECK_STR_ARRAY_LENGTH(CF_OUTPUT_FILE, cfg, conf->cfout, num, conf->ncf);
      for (int i = 0; i < conf->ncf; i++) {
        if ((e = check_output(conf->cfout[i], "CF_OUTPUT_FILE", conf->ovwrite,
            FCFC_OVERWRITE_CFONLY))) return e;
      }
    }
    else conf->cfout = NULL;

    if (conf->bintype == FCFC_BIN_SMU) {
      /* MULTIPOLE */
      if ((conf->npole = cfg_get_size(cfg, &conf->poles))) {
        /* Sort multipoles and remove duplicates. */
        if (conf->npole > 1) {
          /* 5-line insertion sort from https://doi.org/10.1145/3812.315108 */
          for (int i = 1; i < conf->npole; i++) {
            int tmp = conf->poles[i];
            for (num = i; num > 0 && conf->poles[num - 1] > tmp; num--)
              conf->poles[num] = conf->poles[num - 1];
            conf->poles[num] = tmp;
          }
          /* Remove duplicates from the sorted array. */
          num = 0;
          for (int i = 1; i < conf->npole; i++) {
            if (conf->poles[i] != conf->poles[num]) {
              num++;
              conf->poles[num] = conf->poles[i];
            }
          }
          conf->npole = num + 1;
        }
        if (conf->poles[0] < 0 || conf->poles[conf->npole - 1] > FCFC_MAX_ELL) {
          P_ERR(FMT_KEY(MULTIPOLE) " must be between 0 and %d\n", FCFC_MAX_ELL);
          return FCFC_ERR_CFG;
        }

        /* MULTIPOLE_FILE */
        if (cfg_is_set(cfg, &conf->mpout)){
          CHECK_EXIST_ARRAY(MULTIPOLE_FILE, cfg, &conf->mpout, num);
          CHECK_STR_ARRAY_LENGTH(MULTIPOLE_FILE, cfg, conf->mpout,
              num, conf->ncf);
          for (int i = 0; i < conf->ncf; i++) {
            if ((e = check_output(conf->mpout[i], "MULTIPOLE_FILE",
                conf->ovwrite, FCFC_OVERWRITE_CFONLY))) return e;
          }
        }
        else conf->mpout = NULL;
      }
    }
    else if (conf->bintype == FCFC_BIN_SPI) {
      /* PROJECTED_CF */
      if (!cfg_is_set(cfg, &conf->wp)) conf->wp = DEFAULT_PROJECTED_CF;
      if (conf->wp) {
        /* PROJECTED_FILE */
        if (cfg_is_set(cfg, &conf->wpout)){
          CHECK_EXIST_ARRAY(PROJECTED_FILE, cfg, &conf->wpout, num);
          CHECK_STR_ARRAY_LENGTH(PROJECTED_FILE, cfg, conf->wpout,
              num, conf->ncf);
          for (int i = 0; i < conf->ncf; i++) {
            if ((e = check_output(conf->wpout[i], "PROJECTED_FILE",
                conf->ovwrite, FCFC_OVERWRITE_CFONLY))) return e;
          }
        }
        else conf->wpout = NULL;
      }
    }
  }
  
  /* SEP_BIN_FILE */
  
  
  /* OUTPUT_FORMAT */
  if (!cfg_is_set(cfg, &conf->ofmt)) conf->ofmt = DEFAULT_OUTPUT_FORMAT;
  switch (conf->ofmt) {
    case FCFC_OFMT_BIN:
    case FCFC_OFMT_ASCII:
      break;
    default:
      P_ERR("invalid " FMT_KEY(OUTPUT_STYLE) ": %d\n", conf->ofmt);
      return FCFC_ERR_CFG;
  }

  /* VERBOSE */
  if (!cfg_is_set(cfg, &conf->verbose)) conf->verbose = DEFAULT_VERBOSE;

  return 0;
}


/*============================================================================*\
                      Function for printing configurations
\*============================================================================*/

/******************************************************************************
Function `conf_print`:
  Print configuration parameters.
Arguments:
  * `conf`:     structure for storing configurations;
  * `para`:     structure for parallelisms.
******************************************************************************/
static void conf_print(const CONF *conf
#ifdef WITH_PARA
    , const PARA *para
#endif
    ) {
  /* Configuration file */
  printf("\n  CONFIG_FILE     = %s", conf->fconf);
  /* Input catalogs. */

  printf("\n  CATALOG_LABEL   = '%c'", conf->label[0]);
  for (int i = 1; i < conf->ninput; i++) printf(" , '%c'", conf->label[i]);


  if (conf->cnvt) {
    printf("\n  COORD_CONVERT   = %c", conf->cnvt[0] ? 'T' : 'F');
    for (int i = 1; i < conf->ninput; i++)
      printf(" , %c", conf->cnvt[i] ? 'T' : 'F');
  }
  else printf("\n  COORD_CONVERT   = %c", DEFAULT_COORD_CNVT ? 'T' : 'F');

  /* Fiducial cosmology. */
  if (conf->has_cnvt) {
    if (conf->fcnvt) printf("\n  Z_CMVDST_FILE   = %s", conf->fcnvt);
    else {
      printf("\n  OMEGA_M         = " OFMT_DBL, conf->omega_m);
      printf("\n  OMEGA_LAMBDA    = " OFMT_DBL, conf->omega_l);
      if (conf->dew != -1)
        printf("\n  DE_EOS_W        = " OFMT_DBL, conf->dew);
      printf("\n  CMVDST_ERR      = " OFMT_DBL, conf->ecnvt);
    }
  }

  /* 2PCF configurations. */
  const char *tname[] = {"k-d tree", "ball tree"};
  const int ntname = sizeof(tname) / sizeof(tname[0]);
  printf("\n  DATA_STRUCT     = %d (%s)", conf->dstruct,
      conf->dstruct < ntname ? tname[conf->dstruct] : "unknown");

  const char *bname[] = {"s", "s & mu", "s_perp & pi"};
  const int nbname = sizeof(bname) / sizeof(bname[0]);
  printf("\n  BINNING_SCHEME  = %d (%s)", conf->bintype,
      conf->bintype < nbname ? bname[conf->bintype] : "unknown");
  printf("\n  PAIR_COUNT      = %s", conf->pc[0]);
  for (int i = 1; i < conf->npc; i++) printf(" , %s", conf->pc[i]);
  if (conf->pcout){
    printf("\n  PAIR_COUNT_FILE = <%c> %s",
        conf->comp_pc[0] ? 'W' : 'R', conf->pcout[0]);
    for (int i = 1; i < conf->npc; i++) {
      printf("\n                    <%c> %s",
          conf->comp_pc[i] ? 'W' : 'R', conf->pcout[i]);
    }
  }

  if (conf->ncf) {
    printf("\n  CF_ESTIMATOR    = %s", conf->cf[0]);
    for (int i = 1; i < conf->ncf; i++)
      printf("\n                    %s", conf->cf[i]);
    if (conf->cfout){  
      printf("\n  CF_OUTPUT_FILE  = %s", conf->cfout[0]);
      for (int i = 1; i < conf->ncf; i++)
        printf("\n                    %s", conf->cfout[i]);
    }

    if (conf->bintype == FCFC_BIN_SMU && conf->npole) {
      printf("\n  MULTIPOLE       = %d", conf->poles[0]);
      for (int i = 1; i < conf->npole; i++) printf(" , %d", conf->poles[i]);
      if (conf->mpout){  
        printf("\n  MULTIPOLE_FILE  = %s", conf->mpout[0]);
        for (int i = 1; i < conf->ncf; i++)
          printf("\n                    %s", conf->mpout[i]);
      }
    }

    if (conf->bintype == FCFC_BIN_SPI) {
      printf("\n  PROJECTED_CF    = %c", conf->wp ? 'T' : 'F');
      if (conf->wp) {
        if (conf->wpout){  
          printf("\n  PROJECTED_FILE  = %s", conf->wpout[0]);
          for (int i = 1; i < conf->ncf; i++)
            printf("\n                    %s", conf->wpout[i]);
        }
      }
    }
  }

  /* Bin definitions. */
  
  /* Others. */
  const char *sname[] = {"binary", "ASCII"};
  const int nsname = sizeof(sname) / sizeof(sname[0]);
  if (conf->wpout || conf->cfout || conf->pcout || conf->mpout){  
    printf("\n  OUTPUT_FORMAT   = %d (%s)", conf->ofmt,
        conf->ofmt < nsname ? sname[conf->ofmt] : "unknown");
    printf("\n  OVERWRITE       = %d", conf->ovwrite);
  }

#ifdef MPI
  printf("\n  MPI_NUM_TASKS   = %d", para->ntask);
#endif
#ifdef OMP
  printf("\n  OMP_NUM_THREADS = %d", para->nthread);
#endif
  printf("\n");
}


/*============================================================================*\
                      Interface for loading configurations
\*============================================================================*/

/******************************************************************************
Function `load_conf`:
  Read, check, and print configurations.
Arguments:
  * `argc`:     number of arguments passed via command line;
  * `argv`:     array of command line arguments;
  * `para`:     structure for parallelisms.
Return:
  The structure for storing configurations.
******************************************************************************/
CONF *load_conf(const int argc, char *const *argv
#ifdef WITH_PARA
    , const PARA *para
#endif
    ) {
  CONF *conf = conf_init();
  if (!conf) return NULL;

  cfg_t *cfg = conf_read(conf, argc, argv);
  if (!cfg) {
    conf_destroy(conf);
    return NULL;
  }


  if (conf_verify(cfg, conf)) {
    if (cfg_is_set(cfg, &conf->fconf)) free(conf->fconf);
    conf_destroy(conf);
    cfg_destroy(cfg);
    return NULL;
  }
  if (conf->verbose)
    conf_print(conf
#ifdef WITH_PARA
        , para
#endif
        );

  if (cfg_is_set(cfg, &conf->fconf)) free(conf->fconf);
  cfg_destroy(cfg);

#ifdef MPI
  fflush(stdout);
#endif
  return conf;
}

/******************************************************************************
Function `conf_destroy`:
  Release memory allocated for the configurations.
Arguments:
  * `conf`:     the structure for storing configurations.
******************************************************************************/
void conf_destroy(CONF *conf) {
  if (!conf) return;
  FREE_ARRAY(conf->label);
  FREE_STR_ARRAY(conf->wt);
  FREE_ARRAY(conf->has_wt);
  FREE_ARRAY(conf->cnvt);
  FREE_ARRAY(conf->fcnvt);
  FREE_STR_ARRAY(conf->pc);
  FREE_ARRAY(conf->comp_pc);
  FREE_STR_ARRAY(conf->pcout);
  FREE_STR_ARRAY(conf->cf);
  FREE_STR_ARRAY(conf->cfout);
  FREE_ARRAY(conf->poles);
  FREE_STR_ARRAY(conf->mpout);
  FREE_STR_ARRAY(conf->wpout);
  free(conf);
}
