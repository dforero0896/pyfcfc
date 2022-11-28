import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import numpy as np
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.sky import py_compute_cf
import pandas as pd
import time


P0 = 5000
dat_fname = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_NGC.dat"
ran_fname = "/global/cfs/projectdirs/desi/mocks/UNIT/HOD_Shadab/multiple_snapshot_lightcone/UNIT_lightcone_multibox_ELG_footprint_nz_1xdata_5.ran_NGC.dat"
data = pd.read_csv(dat_fname, usecols = (0,1,3,4), engine='c', delim_whitespace=True, names = ['ra', 'dec', 'zrsd', 'nz']).values.astype(np.float64)
rand = pd.read_csv(ran_fname, usecols = (0,1,3,4), engine='c', delim_whitespace=True, names = ['ra', 'dec', 'zrsd', 'nz']).values.astype(np.float64)

data = data[(data[:,2] > 0.8) & (data[:,2] < 1)]
rand = rand[(rand[:,2] > 0.8) & (rand[:,2] < 1)]
wdat = 1. / (1 + P0 * data[:,3])
wran = 1. / (1 + P0 * rand[:,3])
print(wdat.sum())
print(wran.sum())
#wdat = np.ones(data.shape[0])
#wran = np.ones(rand.shape[0])
assert wdat.shape[0] > 1




fig, ax = pplt.subplots(nrows=2, ncols=2, share=0)
s = time.time()
#results = py_compute_cf([data, rand], [wdat, wran], np.arange(0, 200, 1, dtype=np.double), None, 100, conf = "test/fcfc_lc_ell.conf")
results = py_compute_cf([data, rand], [wdat, wran], np.arange(0, 200, 1, dtype=np.double), None, 100, label = ['D', 'R'], omega_m = 0.31, omega_l = 0.69, eos_w = -1, bin = 1, pair = ['DD', 'DR', 'RR'], cf = ['(DD - 2*DR + RR) / RR'], multipole = [0,2,4], convert = 'T')
print(f"Par counting takes {time.time() - s}s", flush=True)

for j in range(results['multipoles'].shape[0]):
    for i in range(3):
        ax[i].plot(results['s'], results['s']**2*results['multipoles'][j,i,:])
        ax[i].format(xlabel = "$s$ [Mpc/$h$]", ylabel = r"$s^2\xi$", title=f"$\ell = {2*i}$", titleloc="ur")
fig.savefig("test/test_lc.png", dpi=300)

results = py_compute_cf([data, rand], [wdat, wran], np.logspace(-2, 2.3, 100), np.arange(0, 200, 1, dtype=np.double), 0, conf = "test/fcfc_lc_wp.conf")
for j in range(results['projected'].shape[0]):
    ax[3].plot(results['s'], results['s'] * results['projected'][j,:])
    ax[3].format(xlabel = "$s$ [Mpc/$h$]", ylabel = r"$sw_p$")
fig.savefig("test/test_lc.png", dpi=300)

results = py_compute_cf([data, rand], [wdat, wran], np.arange(0, 200, 1, dtype=np.double), None, 0, conf = "test/fcfc_lc_iso.conf")  
ax[0].plot(results['s'], results['s']**2 * results['cf'])    
fig.savefig("test/test_lc.png", dpi=300)




