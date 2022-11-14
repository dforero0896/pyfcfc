# Python wrapper to Fast Correlation Function Calculator (pyFCFC)

This repository contains a python wrapper to the FCFC code to compute two-point correlation functions. Due to Cython constraints, some compile-time definitions have been hard coded and thus remove some flexibility from the code. Some of this changes are:
+ SIMD instructions have been coded to `FCFC_SIMD  =  FCFC_SIMD_AVX512`
+ MPI capabilities have been left out (may be added in the future)
+ The python wrapper assumes there are always weights, which you may set up to be 1 if not needed.
+ Due to the wrapper introducing in-memory data to FCFC, no extra data input flags are set (nor needed, i.e. no FITS nor HDF5)

This wrapper now is now able to work **without configuration files**. When calling the functions, keyword arguments are passed to the C extension. 
You can still use configuration files, however, they do not require input-data information since all direct FCFC I/O is bypassed through python. This implies that there's also no data selection from the configuration file, it should be done through python. Furthermore, any input catalog must correspond to a catalog label in the configuration file.

## Gotchas

+ Make sure the number of input catalogs is equal to the number of catalog labels in the configuration file. (I did not add checks so any different leads to unexpected behavior).

## Interpreting results
The results of using the wrapper are saved in a python `dict`. In all cases there are `normalization`, `number`, `pairs`, `s` and `weighted_number` keys. Other keys depend on which correlation function is computed (if any). Below some examples.

The value of the `pairs` key is also a dict containing the keys `cf`, `smin`, `smax` and either `mumin`, `mumax` for multipoles, `pmin`, `pmax` for projected and nothing for isotropic. 

The `pairs` key also contains a dictionary, which in turn contains also the `Xmin/Xmax` arrays: `smin`, `smax` and either `mumin`, `mumax` for multipoles, `pmin`, `pmax` for projected and nothing for isotropic. Moreover, it contains a key-value pair for each of the pair counts that were computed labeled by the labels provided.

If a correlation function estimator is provided, the `cf` key contains an array of size `(n_cf, n_s, n_p/mu)`. This is the raw correlation function which may be integrated into either projection.

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
{'cf': array([[[18.95727695, 40.82757349],
        [ 8.06542191, 15.86000714]],

       [[ 3.03672644, 42.12724307],
        [ 8.03875268, 15.89928789]]]),
       
 'multipoles': <MemoryView of 'ndarray' at 0x14cb8fa96380>,
 'normalization': [302499450000.0,
                   302500000000.0,
                   605000000000.0,
                   605000000000.0,
                   302499450000.0,
                   1210000000000.0],
 'number': [550000, 550000, 1100000, 1100000],
 'pairs': {'AA': array([[4.17984231e-08, 8.76034651e-08],
       [1.32906027e-07, 2.47180615e-07]]),
           'AB': array([[1.85998017e-06, 8.76033058e-08],
       [1.32905785e-07, 2.47180165e-07]]),
           'AC': array([[2.07603306e-09, 2.13057851e-09],
       [1.49652893e-08, 1.46528926e-08]]),
           'BB': array([[4.17984231e-08, 8.76034651e-08],
       [1.32906027e-07, 2.47180615e-07]]),
           'BD': array([[2.07603306e-09, 2.13057851e-09],
       [1.49652893e-08, 1.46528926e-08]]),
           'CD': array([[9.11181818e-07, 2.02644628e-09],
       [1.46297521e-08, 1.46231405e-08]]),
           'mumax': array([[0.5, 1. ],
       [0.5, 1. ]]),
           'mumin': array([[0. , 0.5],
       [0. , 0.5]]),
           'smax': array([[1., 1.],
       [2., 2.]]),
           'smin': array([[0., 0.],
       [1., 1.]])},
 's': array([0.5, 1.5]),
 'weighted_number': [550000.0, 550000.0, 1100000.0, 1100000.0]}
```

### Projected CF
When computing multipoles, the results contain a `projected` key which corresponds to an array of size `(n_pc, n_s)`
```python
{'cf': array([[[24.80051005, 24.353297  ],
        [ 7.92567097,  7.05509623]],

       [[ 3.14970962, 24.17714584],
        [ 7.89526819,  7.02678261]]]),
        
 'normalization': [302499450000.0,
                   302500000000.0,
                   605000000000.0,
                   605000000000.0,
                   302499450000.0,
                   1210000000000.0],
 'number': [550000, 550000, 1100000, 1100000],
 'pairs': {'AA': array([[1.62109386e-07, 1.59299463e-07],
       [1.68244934e-07, 1.51834987e-07]]),
           'AB': array([[1.98029091e-06, 1.59299174e-07],
       [1.68244628e-07, 1.51834711e-07]]),
           'AC': array([[6.33884298e-09, 6.22975207e-09],
       [1.92099174e-08, 1.86380165e-08]]),
           'BB': array([[1.62109386e-07, 1.59299463e-07],
       [1.68244934e-07, 1.51834987e-07]]),
           'BD': array([[6.33884298e-09, 6.22975207e-09],
       [1.92099174e-08, 1.86380165e-08]]),
           'CD': array([[9.15292562e-07, 6.33553719e-09],
       [1.88280992e-08, 1.90082645e-08]]),
           'pimax': array([[1., 2.],
       [1., 2.]]),
           'pimin': array([[0., 1.],
       [0., 1.]]),
           's_perp_max': array([[1., 1.],
       [2., 2.]]),
           's_perp_min': array([[0., 0.],
       [1., 1.]])},
 'projected': <MemoryView of 'ndarray' at 0x14cb8f5df6c0>,
 's': array([0.5, 1.5]),
 'weighted_number': [550000.0, 550000.0, 1100000.0, 1100000.0]}
 ```
 

### Isotropic 2PCF

For isotropic CF there are no extra keys
```python
{'cf': array([[29.89242522, 11.96271453],
       [ 3.12346988, 11.96813199]]),
 'normalization': [302499450000.0,
                   302500000000.0,
                   605000000000.0,
                   605000000000.0,
                   302499450000.0,
                   1210000000000.0],
 'number': [550000, 550000, 1100000, 1100000],
 'pairs': {'AA': array([1.29401888e-07, 3.80086641e-07]),
           'AB': array([1.94758347e-06, 3.80085950e-07]),
           'AC': array([4.20661157e-09, 2.96181818e-08]),
           'BB': array([1.29401888e-07, 3.80086641e-07]),
           'BD': array([4.20661157e-09, 2.96181818e-08]),
           'CD': array([9.13208264e-07, 2.92528926e-08]),
           'smax': array([1., 2.]),
           'smin': array([0., 1.])},
 's': array([0.5, 1.5]),
 'weighted_number': [550000.0, 550000.0, 1100000.0, 1100000.0]}
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
results = py_compute_cf([np.c_[data, wdat], np.c_[rand, wran]], conf = "test/fcfc_lc_ell.conf")
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
results = py_compute_cf([data, rand], [wdat, wran],  conf = "test/fcfc_lc_wp.conf")

# Isotropic: Conf sets the CF type. BIN_TYPE = 0
results = py_compute_cf([data, rand], [wdat, wran],  conf = "test/fcfc_lc_iso.conf")

```


## Compilation of the wrapper library

Given that the original code does not contain extra dependencies, it should be enough to type `make` from the containing folder. Please raise an issue if you find any trouble during compilation.


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
