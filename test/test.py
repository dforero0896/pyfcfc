import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import numpy as np
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.boxes import py_compute_cf
import pprint

try:
    import pandas as pd
    test_fname = "/global/cfs/projectdirs/desi/cosmosim/UNIT-BAO-RSD-challenge/Stage2Recon/UNITSIM/LRG/LRG-wpmax-v3-snap103-redshift0.74_dens0.dat"
    data = pd.read_csv(test_fname, usecols = (0,1,3), engine='c', delim_whitespace=True, names = ['x', 'y', 'zrsd'], comment = "#").values
except:
    print("WARNING: Read catalog failed, testing with uniform random.", flush=True)
    seed = 42
    np.random.seed(seed)
    nobj = int(1e4)
    data = 1000. * np.random.random((nobj, 3)).astype(np.double)


nobj = data.shape[0]
w = np.ones(nobj)
labels = ['Nat', "LS"]
rand = 1000. * np.random.random((2 * nobj, 3)).astype(np.double)
wran = np.ones(rand.shape[0])



results = py_compute_cf([np.c_[data, w], np.c_[data, w], np.c_[rand, wran], np.c_[rand, wran]], "test/fcfc_box_ell.conf")


fig, ax = pplt.subplots(nrows=2, ncols=2, share=0)
for j in range(results['multipoles'].shape[0]):
    for i in range(3):
        ax[i].plot(results['s'], results['s']**2*results['multipoles'][j,i,:], label = f'ells {labels[j]}')
        ax[i].format(xlabel = "$s$ [Mpc/$h$]", ylabel = r"$s^2\xi$", title=f"$\ell = {2*i}$", titleloc="ur")
fig.savefig("test/test.png", dpi=300)


results = py_compute_cf([np.c_[data, w], np.c_[data, w], np.c_[rand, wran], np.c_[rand, wran]], "test/fcfc_box_wp.conf")
for j in range(results['projected'].shape[0]):
    ax[3].plot(results['s'], results['s'] * results['projected'][j,:], label = f'wp {labels[j]}')
    ax[3].format(xlabel = "$s$ [Mpc/$h$]", ylabel = r"$sw_p$")
fig.savefig("test/test.png", dpi=300)



results = py_compute_cf([np.c_[data, w], np.c_[data, w], np.c_[rand, wran], np.c_[rand, wran]], "test/fcfc_box_iso.conf")
for j in range(results['cf']['cf'].shape[0]):
    ax[0].plot(results['s'], results['s']**2 * results['cf']['cf'][j,:], label = f'Iso {labels[j]}')

ax.legend(loc='top')
fig.savefig("test/test.png", dpi=300)

