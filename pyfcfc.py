from pyfcfc import count_pairs_npy

import numpy as np
import os
import time

if __name__=='__main__':
    np.random.seed(42)
    Nelem = int(1e5)
    NBINS=40
    data = np.random.random((Nelem, 3)) * 2500
    weights = np.ones(Nelem)
    sbin_arr = np.linspace(0,200, NBINS+1)
    pibin_arr = np.linspace(0,200, NBINS+1)
    nthreads = int(os.getenv('OMP_NUM_THREADS', 1))
    bin_scheme = 1 # 0 isotropic, 1 s mu, 2 s pi
    use_wt = True
    compute_wp = True
    is_auto = False
    s = time.time()
    out, out_norm = count_pairs_npy(is_auto,
                            data.astype(np.double), 
                            weights, 
                            sbin_arr, 
                            40,
                            pibin_arr, 
                            bin_scheme,
                            [],
                            compute_wp,
                            [use_wt, False],
                            data.astype(np.double), 
                            weights, 
                            nthreads)
    print(f"FCFC took {time.time() - s} s", flush=True)
    print(out)
    print(out.shape)
    print(out_norm)
    print(out_norm.shape)