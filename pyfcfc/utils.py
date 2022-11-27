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
    from scipy.special import eval_legendre, legendre
    multipoles = np.empty((len(poles), correlation_function.shape[0]))
    nmu = correlation_function.shape[1]
    mu_edges = np.linspace(0., 1., nmu+1)
    mu = mu_edges[:-1] + 0.5 * np.diff(mu_edges) 
    

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



def pairs_to_pycorr(results, estimator_name, pair_mapping):
    
    s_edges = results['pairs']['smin'][:,0]
    s_edges = np.append(s_edges, results['pairs']['smax'][-1,0])
    if 'multipoles' in results:
        mode = 'smu'
        mu_edges = results['pairs']['mumin'][0,:]
        mu_edges = np.append(mu_edges, results['pairs']['mumax'][0,-1])
        mu_edges = np.concatenate((-mu_edges[::-1][:-1], mu_edges))
        edges = (s_edges, mu_edges)
    elif 'wp' in results:
        mode = 'rppi'
        pi_edges = results['pairs']['pimin'][0,:]
        pi_edges = np.append(pi_edges, results['pairs']['pimax'][0,-1])
        edges = (s_edges, pi_edges)
    else:
        mode = 's'
        edges = (s_edges,)

    estimator_state = {}
    estimator_state['name'] = estimator_name
    for key, val in results['pairs'].items():
        
        if len(key) > 2: continue
        state = {}
        #['name', 'autocorr', 'is_reversible', 'seps', 'ncounts', 'wcounts', 'wnorm', 'size1', 'size2', 'edges', 'mode', 'bin_type',
        #         'boxsize', 'los_type', 'compute_sepsavg', 'weight_attrs', 'cos_twopoint_weights', 'dtype', 'attrs']
        state['name'] = 'base'
        state['autocorr'] = int(key[0] == key[1])
        state['is_reversible'] = int(key[0] == key[1])
        state['seps'] = [results['pairs']['smin'] + 0.5 * (results['pairs']['smax'] - results['pairs']['smin'])]
        if mode == 'smu':
            second = results['pairs']['mumin'] + 0.5 * (results['pairs']['mumax'] - results['pairs']['mumin'])
            state['seps'].append(np.concatenate((-second[:,::-1], second), axis=1))
            state['seps'][0] = np.concatenate((state['seps'][0], state['seps'][0]), axis=1)
        elif mode =='rppi':
            state['seps'].append(results['pairs']['pimin'] + 0.5 * (results['pairs']['pimax'] - results['pairs']['pimin']))
        state['ncounts'] = results['pairs'][key] * results['normalization'][key]
        state['wcounts'] = results['pairs'][key] * results['normalization'][key]
        if mode == 'smu':
            state['ncounts'] = np.concatenate((state['ncounts'][:,::-1], state['ncounts']), axis=1)
            state['wcounts'] = np.concatenate((state['wcounts'][:,::-1], state['wcounts']), axis=1)
        state['wnorm'] = results['normalization'][key]
        state['size1'] = results['weighted_number'][key[0]]
        state['size2'] = results['weighted_number'][key[1]]
        state['edges'] = edges
        state['mode'] = mode
        state['bin_type'] = 'auto'
        state['boxsize'] = np.nan
        state['los_type'] = 'firstpoint'
        state['compute_sepsavg'] = [False, False] if mode == 'smu' or mode == 'rppi' else [False]
        state['weight_attrs'] = {}
        state['dtype'] = results['pairs'][key].dtype
        if isinstance(pair_mapping[key], str):
            estimator_state[pair_mapping[key]] = state
        else:
            for i in range(2):
                estimator_state[pair_mapping[key][i]] = state
    return estimator_state









    
