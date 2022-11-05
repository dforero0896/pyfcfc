import matplotlib
matplotlib.use("Agg")
import proplot as pplt
import numpy as np
import sys
sys.path.append("/global/homes/d/dforero/codes/pyfcfc/")
from pyfcfc.boxes import py_compute_cf

try:
    import pandas as pd
    test_fname = "/global/project/projectdirs/desi/mocks/UNIT/HOD_Shadab/HOD_boxes/redshift0.9873/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt"
    data = pd.read_csv(test_fname, usecols = (0,1,3), engine='c', delim_whitespace=True, names = ['x', 'y', 'zrsd']).values
except:
    print("WARNING: Read catalog failed, testing with uniform random.", flush=True)
    seed = 42
    np.random.seed(seed)
    nobj = int(1e4)
    data = 1000. * np.random.random((nobj, 3)).astype(np.double)


nobj = data.shape[0]
w = np.ones(nobj)

py_compute_cf([np.c_[data, w]], "test/fcfc_box_auto.conf")

