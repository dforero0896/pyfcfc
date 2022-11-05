import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import numpy as np
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.boxes import py_compute_cf

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

results = py_compute_cf([np.c_[data, w], np.c_[data, w]], "test/fcfc_box_auto.conf")

fig, ax = pplt.subplots(nrows=1, ncols=3, share=0)
for j in range(results['multipoles'].shape[0]):
    for i in range(3):
        ax[i].plot(results['s'], results['s']**2*results['multipoles'][j,i,:])
fig.savefig("test/test.png", dpi=300)
#py_compute_cf([np.c_[data, w]], "test/fcfc_box_auto.conf")

