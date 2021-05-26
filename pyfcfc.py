from pyfcfc.lightcones import count_pairs_npy, pair_counts_to_cf, pair_counts_to_mp, pair_counts_to_wp

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
    is_auto = True
    n_mu_bins = 40
    poles = [0,2,4]
    s = time.time()
    dd_out, dd_out_norm = count_pairs_npy(is_auto,
                            data.astype(np.double), 
                            weights, 
                            sbin_arr, 
                            n_mu_bins,
                            pibin_arr, 
                            bin_scheme,
                            [use_wt],
                            data.astype(np.double), 
                            weights, 
                            nthreads)
    rr_out, rr_out_norm = count_pairs_npy(is_auto,
                            data.astype(np.double), 
                            weights, 
                            sbin_arr, 
                            n_mu_bins,
                            pibin_arr, 
                            bin_scheme,
                            [use_wt],
                            data.astype(np.double), 
                            weights, 
                            nthreads)
    print(f"pyFCFC took {time.time() - s} s", flush=True)
    print(dd_out)
    print(dd_out.shape)
    print(dd_out_norm)
    print(dd_out_norm.shape)

    cf = pair_counts_to_cf(dd=dd_out_norm, rr=rr_out_norm, dr=None)

    multipoles = pair_counts_to_mp(cf, sbin_arr, n_mu_bins, poles)
    print(multipoles.shape)

    wp = pair_counts_to_wp(cf, sbin_arr, pibin_arr)
    print(wp.shape)