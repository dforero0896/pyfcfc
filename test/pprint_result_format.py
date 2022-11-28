
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
rand = 1000. * np.random.random((2 * nobj, 3)).astype(np.double)
wran = np.ones(rand.shape[0])




results = py_compute_cf([data, rand], [w, wran], 10**(np.linspace(-3, 2.2, 2)), None, 2, label = ['A', 'B'], bin=1, pair = ['AA', 'AB', 'BB'], box=1000, multipole = [0, 2, 4], cf = ['AA / @@ - 1', '(AB - 2 * AB + BB) / BB'])
pprint.pprint(results)
results = py_compute_cf([data, rand], [w, wran], np.arange(0, 200, 100, dtype=np.double), np.arange(0, 200, 100, dtype=np.double), 0, label = ['A', 'B'], bin=2, pair = ['AA', 'AB', 'BB'], box=1000, cf = ['AA / @@ - 1', '(AB - 2 * AB + BB) / BB'], wp = 'true')
pprint.pprint(results)
results = py_compute_cf([data, rand], [w, wran], np.arange(0, 200, 100, dtype=np.double), None, 1, label = ['A', 'B'], bin=0, pair = ['AA', 'AB', 'BB'], box=1000, cf = ['AA / @@ - 1', '(AB - 2 * AB + BB) / BB'])
pprint.pprint(results)

    

