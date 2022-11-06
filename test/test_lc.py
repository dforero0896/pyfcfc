import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import numpy as np
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.sky import py_compute_cf


import pandas as pd
P0 = 5000
dat_fname = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_NGC.dat"
ran_fname = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.dat"
data = pd.read_csv(dat_fname, usecols = (0,1,3,4), engine='c', delim_whitespace=True, names = ['ra', 'dec', 'zrsd', 'nz']).values.astype(np.float64)
rand = pd.read_csv(ran_fname, usecols = (0,1,3,4), engine='c', delim_whitespace=True, names = ['ra', 'dec', 'zrsd', 'nz']).values.astype(np.float64)

data = data[(data[:,2] > 0.9) & (data[:,2] < 1)]
rand = rand[(rand[:,2] > 0.9) & (rand[:,2] < 1)]
wdat = 1. / (1 + P0 * data[:,3])
wran = 1. / (1 + P0 * rand[:,3])
assert wdat.shape[0] > 1





fig, ax = pplt.subplots(nrows=2, ncols=2, share=0)

results = py_compute_cf([np.c_[data, wdat], np.c_[rand, wran]], "test/fcfc_lc_ell.conf")
for j in range(results['multipoles'].shape[0]):
    for i in range(3):
        ax[i].plot(results['s'], results['s']**2*results['multipoles'][j,i,:])
        ax[i].format(xlabel = "$s$ [Mpc/$h$]", ylabel = r"$s^2\xi$", title=f"$\ell = {2*i}$", titleloc="ur")
fig.savefig("test/test_lc.png", dpi=300)

results = py_compute_cf([np.c_[data, wdat], np.c_[rand, wran]], "test/fcfc_lc_wp.conf")
for j in range(results['projected'].shape[0]):
    ax[3].plot(results['s'], results['s'] * results['projected'][j,:])
    ax[3].format(xlabel = "$s$ [Mpc/$h$]", ylabel = r"$sw_p$")
fig.savefig("test/test_lc.png", dpi=300)

results = py_compute_cf([np.c_[data, wdat], np.c_[rand, wran]], "test/fcfc_lc_iso.conf")  
for j in range(results['cf']['cf'].shape[0]):
    ax[0].plot(results['s'], results['s']**2 * results['cf']['cf'][j,:])    
fig.savefig("test/test_lc.png", dpi=300)



