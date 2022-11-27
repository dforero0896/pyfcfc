import numpy as np

def add_pair_counts(results_a, results_b, repeat_missing = True):
    new_results = {}
    new_results['labels'] = results_a['labels']
    new_results['number'] = {}
    new_results['weighted_number'] = {}
    for l in results_a['labels']:
        try:
            new_results['number'][l] = results_a['number'][l] + results_b['number'][l]
            new_results['weighted_number'][l] = results_a['weighted_number'][l] + results_b['weighted_number'][l]
        except KeyError:
            if repeat_missing:
                new_results['number'][l] = 2 * results_a['number'][l]
                new_results['weighted_number'][l] = 2 * results_a['weighted_number'][l]

    for key, val in results_a['pairs'].items():
        if len(key) > 2: continue # Only iterate over keys that correspond to pair counts
        try:
            results_a['pairs'][key] = results_a['pairs'][key] * results_a['normalization'][key] + results_b['pairs'][key] * results_b['normalization'][key]
            results_a['pairs'][key] /= results_a['normalization'][key] + results_b['normalization'][key]
            results_a['normalization'][key] += results_b['normalization'][key]
                
        except KeyError:
            print(f"WARNING: {key} pairs not found in `results_b`.", flush=True)
            if repeat_missing:
                print(f"\t Repeating missing pair counts.", flush=True)
                results_a['pairs'][key] = 2 * results_a['pairs'][key] * results_a['normalization'][key]
                results_a['pairs'][key] /= 2 * results_a['normalization'][key]
                results_a['normalization'][key] += results_a['normalization'][key]
                
    results_a['number'] = new_results['number']
    results_a['weighted_number'] = new_results['weighted_number']
    return results_a

def compute_multipoles(correlation_function, poles):
    from scipy.special import eval_legendre
    multipoles = np.empty((len(poles), correlation_function.shape[0]))
    nmu = correlation_function.shape[1]
    mu = (np.arange(0., 1, nmu) + 0.5) / nmu
    leg_cache = np.empty(correlation_function.shape[1])
    for i, ell in enumerate(poles):
        fac = (2 * ell + 1) / nmu
        eval_legendre(ell, mu, leg_cache)
        multipoles[i,:] = (correlation_function * fac * leg_cache[None,:]).sum(axis=1)
    
    return multipoles

def compute_wp(correlation_function, pi_edges):

    dpi = np.diff(pi_edges)
    wp = (correlation_function * dpi[None,:] * 2).sum(axis=1)
    return wp


    
