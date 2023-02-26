# Python wrapper to Fast Correlation Function Calculator (pyFCFC)

This repository contains a python wrapper to the FCFC code to compute two-point correlation functions. Due to Cython constraints, some compile-time definitions have been hard coded and thus remove some flexibility from the code. Some of this changes are:
+ MPI capabilities have been left out (may be added in the future)
+ The python wrapper assumes there are always weights, which you may set up to be 1 if not needed.
+ Due to the wrapper introducing in-memory data to FCFC, no extra data input flags are set (nor needed, i.e. no FITS nor HDF5)
+ I have added the possibility to compile with and without SIMD instructions. By default they are deactivated but it is **strongly encouraged to compile with SIMD** it leads to a noticeable speedup (in NERSC Perlmutter). To compile the code with SIMD run `export PYFCFC_WITH_SIMD=1` before `make`.

This wrapper now is now able to work **without configuration files**. When calling the functions, keyword arguments are passed to the C extension. 
You can still use configuration files, however, they do not require input-data information since all direct FCFC I/O is bypassed through python. This implies that there's also no data selection from the configuration file, it should be done through python. Furthermore, any input catalog must correspond to a catalog label in the configuration file. The keyword arguments follow the same names as the command line arguments in the FCFC code except for the fact that the `-` have been replaced by `_`.

## Usage with pycorr
To encourage user adoption, some light benchmarking against [pycorr](https://github.com/cosmodesi/pycorr) has been added. **pyfcfc is ~2x faster than pycorr (when using SIMD)** in NERSC Perlmutter login nodes. `pyfcfc` is also faster without SIMD but not as much. Some extra utilities have also been added so some of the advantages of `pycorr` are available. The `pyfcfc.utils` submodule now includes the `add_pair_counts` function which operates on the dictionary resulting from a pyfcfc call and allows the user to perform split-random two-point correlation measurements. Moreover, `compute_multipoles` and `compute_wp` functions have also been added so correlation functions can be integrated after split-random pair counting. 
Finally the same module contains a `pairs_to_pycorr(results, estimator_name, pair_mapping)` so `pyfcfc` results can be converted to a `pycorr.TwoPointEstimator` state, this allows for pair counting using the (fast) `pyfcfc` functions and analyzing using `pycorr`. So far the resulting state has been tested for multipole integration and rebinning.

## Gotchas

+ Make sure the number of input catalogs is equal to the number of catalog labels in the configuration file. (I did not add checks so any different leads to undefined behavior).
+ It is possible to not pass a CF estimator (either in a conf. file or a kwarg), however this causes the pair counter to default to isotropic binning (`bin_type=0`) so the resulting pair counts can't be integrated into moltipoles or projected. While I may deal with this in a cleaner way later, to ensure the pair counter does the correct binning, it suffices to **always provide a CF estimator, even if you plan on using the external integrator/pycorr afterwards**, for example to minimize computation, you could pass the `cf = ['DD']` kwarg if the `DD` pair counts are computed. 
+ In order to save the results in a `pycorr`-readable format, you must provide a mapping between the `pyfcfc` labels nd `pycorr` ones. This is a dictionary `dict(DD='D1D2', DR = ('D1R2', 'R1D2'), RR = 'R1R2')`. Notice that to make it work the mapping for "reversible" pair counts should contain both `pycorr` labels.

## Interpreting results
The results of using the wrapper are saved in a python `dict`. In all cases there are `normalization`, `number`, `pairs`, `s` and `weighted_number` keys. Other keys depend on which correlation function is computed (if any). Below some examples.

The value of the `cf` key is contains the correlation functions computed by FCFC as defined by the `cf` keyword argument. Mind the **gotchas** related to providing estimators. If a correlation function estimator is provided, the `cf` key contains an array of size `(n_cf, n_s, n_p/mu)`. This is the raw correlation function which may be integrated into either projection.

The `pairs` key contains a dictionary, which in turn contains also the `Xmin/Xmax` arrays: `smin`, `smax` and either `mumin`, `mumax` for multipoles, `pmin`, `pmax` for projected and nothing for isotropic. Moreover, it contains a key-value pair for each of the pair counts that were computed labeled by the labels provided.



See the `test` directory for some example calculations and configuration files. In a nutshell, the relevant function is
```python
py_compute_cf(data_cats, #Data catalogs
              data_wts,  # Data weights
              sedges, # s bin edges
              pedges, # pi bin edges (only used if bin type = 2)
              int nmu, # number of mu bins (only used if bin type = 1)
              **kwargs # keyword arguments that override the conf. file. Following the syntax of command line options for FCFC
                        # for example: pairs = ['DD', 'DR', 'RR']. For command line args with hyphens (-), replace them with underscore (_).
                )
```

### Multipoles of 2PCF

When computing multipoles, the results contain a `multipoles` key which corresponds to an array of size `(n_pc, n_ell, n_s)`

```python

#returns something like
{'cf': array([[7.25353714e-03, 2.54287530e-03],
       [4.27435967e-05, 6.06266121e-05]]),
 'labels': ['A', 'B'],
 'multipoles': <MemoryView of 'ndarray' at 0x14c031968860>,
 'normalization': {'AA': 302499450000.0,
                   'AB': 605000000000.0,
                   'BB': 1209998900000.0},
 'number': {'A': 550000, 'B': 1100000},
 'pairs': {'AA': array([[0.00839842, 0.00835914]]),
           'AB': array([[0.00833746, 0.00833748]]),
           'BB': array([[0.00833781, 0.00833799]]),
           'mumax': array([[0.5, 1. ]]),
           'mumin': array([[0. , 0.5]]),
           'smax': array([[158.48931925, 158.48931925]]),
           'smin': array([[0.001, 0.001]])},
 's': array([79.24515962]),
 'weighted_number': {'A': 550000, 'B': 1100000}}
```

### Projected CF
When computing multipoles, the results contain a `projected` key which corresponds to an array of size `(n_pc, n_s)`
```python
{'cf': array([1.31933576e-02, 3.62523994e-05]),
 'labels': ['A', 'B'],
 'normalization': {'AA': 302499450000.0,
                   'AB': 605000000000.0,
                   'BB': 1209998900000.0},
 'number': {'A': 550000, 'B': 1100000},
 'pairs': {'AA': array([[0.00636608]]),
           'AB': array([[0.00628289]]),
           'BB': array([[0.00628312]]),
           'pimax': array([[100.]]),
           'pimin': array([[0.]]),
           'smax': array([[100.]]),
           'smin': array([[0.]])},
 'projected': <MemoryView of 'ndarray' at 0x14c031973040>,
 's': array([50.]),
 'weighted_number': {'A': 550000, 'B': 1100000}}
 ```
 

### Isotropic 2PCF

For isotropic CF there are no extra keys
```python
{'cf': array([1.92575450e-02, 1.94183401e-05]),
 'labels': ['A', 'B'],
 'normalization': {'AA': 302499450000.0,
                   'AB': 605000000000.0,
                   'BB': 1209998900000.0},
 'number': {'A': 550000, 'B': 1100000},
 'pairs': {'AA': array([0.00426946]),
           'AB': array([0.00418867]),
           'BB': array([0.00418875]),
           'smax': array([100.]),
           'smin': array([0.])},
 's': array([50.]),
 'weighted_number': {'A': 550000, 'B': 1100000}}
 ```

## Examples

### Periodic Boxes
```python
import numpy as np
import proplot as pplt
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.boxes import py_compute_cf

data = 1000. * np.random.random((nobj, 3)).astype(np.double)
w = np.ones(data.shape[0])
rand = 1000. * np.random.random((2 * nobj, 3)).astype(np.double)
wran = np.ones(rand.shape[0])

# Multipoles: Conf sets the CF type. BIN_TYPE = 1
results = py_compute_cf([data, data, rand, rand], [w, w, wran, wran], 
                        10**(np.linspace(-3, 2.2, 51)), # s edges can be non-linear
                        None, # pi edges not used for multipoles
                        100, # 100 mu bins
                        conf = "test/fcfc_box_ell.conf", # conf file can still be used but is not mandatory (if all relevant kwargs are set)
                        label = ['A', 'B', 'C', 'D']) # kwargs override configuration file
                        
## Alternatively, without a configuration file
results = py_compute_cf([data, data, rand, rand], [w, w, wran, wran], 
                        10**(np.linspace(-3, 2.2, 51)), 
                        None, 
                        100, 
                        label = ['A', 'B', 'C', 'D'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA', 'AB', 'AC', 'BD', 'BB', 'CD'], # Desired pair counts
                        box=1000, 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1', '(AB - AC - BD + CD) / CD']) # CF estimator (not necessary if only pair counts are required)

# Projected: Conf sets the CF type. BIN_TYPE = 2
results = py_compute_cf([data, data, rand, rand], [w, w, wran, wran], conf = "test/fcfc_box_wp.conf") 

# Isotropic: Conf sets the CF type. BIN_TYPE = 0
results = py_compute_cf([data, data, rand, rand], [w, w, wran, wran],  conf = "test/fcfc_box_iso.conf") 

```
### Survey-like data
```python
import numpy as np
import pandas as pd
import proplot as pplt
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.sky import py_compute_cf

data = pd.read_csv(dat_fname, usecols = (0,1,3,4), engine='c', delim_whitespace=True, names = ['ra', 'dec', 'zrsd', 'nz']).values.astype(np.float64)
rand = pd.read_csv(ran_fname, usecols = (0,1,3,4), engine='c', delim_whitespace=True, names = ['ra', 'dec', 'zrsd', 'nz']).values.astype(np.float64)

# Do data selection and define weights in python
data = data[(data[:,2] > 0.9) & (data[:,2] < 1)]
rand = rand[(rand[:,2] > 0.9) & (rand[:,2] < 1)]
wdat = 1. / (1 + P0 * data[:,3])
wran = 1. / (1 + P0 * rand[:,3])

# Multipoles: Conf sets the CF type. BIN_TYPE = 1
results = py_compute_cf([data, rand], [wdat, wran], 
                        np.arange(0, 200, 1, dtype=np.double), 
                        None, 
                        100, 
                        conf = "test/fcfc_lc_ell.conf")
## Alternatively w/o configuration file
results = py_compute_cf([data, rand], [wdat, wran], 
                        np.arange(0, 200, 1, dtype=np.double), 
                        None, 
                        100, 
                        label = ['D', 'R'], 
                        omega_m = 0.31, # Cosmo params for coordinate conversion
                        omega_l = 0.69, 
                        eos_w = -1, 
                        bin = 1, 
                        pair = ['DD', 'DR', 'RR'], 
                        cf = ['(DD - 2*DR + RR) / RR'], 
                        multipole = [0,2,4], 
                        convert = 'T') #Coordinate conversion

# Projected: Conf sets the CF type. BIN_TYPE = 2
results = py_compute_cf([data, rand], [wdat, wran],
                        np.arange(0, 200, 1, dtype=np.double), 
                        np.arange(0, 200, 1, dtype=np.double), 
                        1,
                        conf = "test/fcfc_lc_wp.conf")

# Isotropic: Conf sets the CF type. BIN_TYPE = 0
results = py_compute_cf([data, rand], [wdat, wran],  
                        np.arange(0, 200, 1, dtype=np.double), 
                        None,
                        1,
                        conf = "test/fcfc_lc_iso.conf")

```

## Add pair counts
```python
from pyfcfc.utils import add_pair_counts
edges = np.arange(0, 201, 1, dtype=np.double), np.linspace(-1, 1, 201)
n_rand_splits = 20
total_results = {}
for i, (_shifted, _rand, _wshifted, _wrand) in enumerate(zip(*map(lambda x: np.array_split(x, n_rand_splits), (shifted, rand[['RA', 'DEC', 'Z']].values, wshifted, wrand)))):
    s_ = time.time()
    results = py_compute_cf([data, _rand, _shifted], 
                            [wdata, _wrand, _wshifted], 
                            edges[0], None, (edges[1].shape[0] - 1) // 2, # edges is set for pycorr
                            label = ['D', 'R', 'S'],
                            omega_m = (0.02237 + 0.1200) / h**2, # Abacus c000 / DESI fiducial
                            eos_w = -1, 
                            bin = 1, # Bin in s & mu
                            pair = ['DD', 'RR', 'DS', 'SS'] if i == 0 else ['DS', 'SS', 'RR'], # Compute only required pairs
                            cf = ['DS'], # Required to force the pair counter to respect the binning scheme
                            multipole = [0,2,4], 
                            convert = 'T', # Use cosmological quantities to convert coordinates.
                            data_struct = 0, #Use KDTree
                            verbose = 'F')
    
    total_results = add_pair_counts(total_results, results) if i > 0 else results
    print(f"pyfcfc single split {time.time() - s_}s", flush=True)
```
## Save and load in `pycorr` state format
```python
from pyfcfc.utils import pairs_to_pycorr
from pycorr import TwoPointCorrelationFunction
pycorr_states = pairs_to_pycorr(total_results, # pyfcfc results dict
                                'landyszalay', # Defines which estimator is used in pycorr
                                dict(DD = "D1D2", RR = "R1R2", DS = ("D1S2", "S1D2"), SS = "S1S2") # Maps pyfcfc labels to pycorr attribute names
                                )
np.save("test/DD_pycorr_state.pkl.npy", pycorr_states) # Save as pickle, use numpy for simplicity
result = TwoPointCorrelationFunction.load("test/DD_pycorr_state.pkl.npy") #Load same file with pycorr
result2 = result[::2,::4] # Can be rebinned
print('Initially {:d} sep, {:d} mu.'.format(*result.shape))
print('After rebinning {:d} sep, {:d} mu.'.format(*result2.shape))
```
## Using `pyfcfc` integration
```python
from pyfcfc.utils import compute_multipoles, compute_wp
# total_results as in previous examples, pyfcfc results dict
# Compute using the estimator your heart desires
total_results['cf'] = (total_results['pairs']['DD'] - 2 * total_results['pairs']['DS'] + total_results['pairs']['SS']) / total_results['pairs']['RR']
# Integrate to multipoles if binned in s-mu (bin_type 1)
total_results['multipoles'] = compute_multipoles(total_results['cf'], [0,2,4])
# Project to wp if binned in s-mu (bin_type 1)
total_results['projected'] = compute_wp(total_results['cf'], pi_edges) #pi_edges should be the same passed to FCFC
```


## Compilation of the wrapper library

Given that the original code does not contain extra dependencies, it should be enough to type `make` from the containing folder. Please raise an issue if you find any trouble during compilation. As mentioned before it is strongly encouraged to compile with SIMD, to do so run 
```bash
PYFCFC_WITH_SIMD=1 make
```

## Below is the README of the original C implementation
## Table of Contents

-   [Introduction](#introduction)
-   [Compilation](#compilation)
-   [Components and configurations](#components-and-configurations)
-   [Acknowledgements](#acknowledgements)
-   [References](#references)

## Introduction

**F**ast **C**orrelation **F**unction **C**alculator (FCFC) is a C toolkit for computing cosmological correlation functions. It is designed in the hope of being (both time and space) efficient, portable, and user-friendly.

So far the following products are supported:
-   Isotropic 2-point correlation function (2PCF): *&xi;*(*s*);
-   Anisotropic 2PCF: *&xi;*(*s*, *&mu;*);
-   2-D 2PCF: *&xi;*(*s*<sub>perp</sub>, *s*<sub>para</sub>), also known as *&xi;*(*s*<sub>perp</sub>, *&pi;*);
-   2PCF multipoles: *&xi;*<sub>*&ell;*</sub>(*s*);
-   Projected 2PCF: *w*<sub>*p*</sub>(*s*<sub>perp</sub>).

This program is compliant with the ISO C99 and IEEE POSIX.1-2008 standards, and no external library is mandatory. Parallelisation can be enabled with [OpenMP](https://www.openmp.org). Thus it is compatible with most of the modern C compilers and operating systems. It is written by Cheng Zhao (&#36213;&#25104;), and is distributed under the [MIT license](LICENSE.txt).

If you use this program in research that results in publications, please cite the following paper:

> Zhao et al., in preparation.

<sub>[\[TOC\]](#table-of-contents)</sub>

## Compilation

The following command should compile the code for most of the time:

```bash
make
```

To compile only a certain component of the program, the name of the component can be supplied via

```bash
make [COMPONENT_NAME]
```

<sub>[\[TOC\]](#table-of-contents)</sub>

## Components and configurations

FCFC comes along with several components for different tasks. They are served as separate executables, and have to be supplied the corresponding configurations, either via command line options or a text file with configuration parameters.

The list of available command line options can be consulted using the `-h` or `--help` flags, and a template configuration file can be printed via the `-t` or `--template` flags.

An introduction of the components and the corresponding configuration parameters are listed below:

| Component    | Description                                                     | Configuration parameters               |
|:------------:|-----------------------------------------------------------------|:--------------------------------------:|
| FCFC_2PT     | Compute 2PCF for survey-like data                               | [FCFC_2PT.md](doc/FCFC_2PT.md)         |
| FCFC_2PT_BOX | Compute 2PCF for periodic simulation boxes<sup>[*](#tab1)</sup> | [FCFC_2PT_BOX.md](doc/FCFC_2PT_BOX.md) |

<span id="tab1">*: treat the 3<sup>rd</sup> dimension (*z*-direction) as the line of sight</span>

<sub>[\[TOC\]](#table-of-contents)</sub>

## Acknowledgements

This program benefits from the following open-source projects:
-   [Fast Cubic Spline Interpolation](https://doi.org/10.5281/zenodo.3611922) (see also [arXiv:2001.09253](https://arxiv.org/abs/2001.09253))
-   [https://github.com/andralex/MedianOfNinthers](https://github.com/andralex/MedianOfNinthers) (see also [this paper](http://dx.doi.org/10.4230/LIPIcs.SEA.2017.24))
-   [https://github.com/swenson/sort](https://github.com/swenson/sort)

<sub>[\[TOC\]](#table-of-contents)</sub>

## References

TBE

<sub>[\[TOC\]](#table-of-contents)</sub>

## TODO

-   FITS and HDF5 file formats
-   Jackknife covariance estimation
-   3-point correlation functions
-   Approximate correlation function calculators
-   More data structures for potentially faster correlation function evaluation

<sub>[\[TOC\]](#table-of-contents)</sub>
