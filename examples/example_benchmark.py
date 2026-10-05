"""Benchmark pyfcfc against pycorr (Corrfunc engine).

Measures wall-clock time of complete correlation-function measurements
(pair counting + estimator + integration) on identical periodic-box
catalogues, for the three binning schemes and a range of catalogue
sizes:

- ``smu``  : (s, mu) counts + Landy-Szalay xi(s, mu) + multipoles 0/2/4
- ``rppi`` : (s_perp, pi) counts + Landy-Szalay + projected w_p
- ``s``    : isotropic counts + Landy-Szalay xi(s)

Notes on fairness:

- pycorr's Landy-Szalay evaluation runs four counters (D1D2, D1R2, R1D2,
  R1R2) while pyfcfc evaluates three pair counts (DD, DR, RR); this is
  exactly what each code does in normal use, so it is kept as is.
- ``compute_sepsavg`` is disabled in pycorr (pyfcfc never computes
  pair-weighted bin averages), so both time the same operations.
- Unit weights are used by default, which lets pyfcfc run its exact
  integer counting path; pass ``--weighted`` to benchmark the
  floating-point path instead.

The script reports the SIMD level compiled into pyfcfc
(``pyfcfc.boxes.simd_info``): on an AVX-capable machine, rebuild with
``PYFCFC_WITH_SIMD=1`` before benchmarking to measure the vectorised
kernels.

Outputs
-------
- a table on stdout;
- ``examples/figures/benchmark_results.csv`` with all timings;
- ``examples/figures/benchmark.png`` (timings vs N, and thread scaling
  when ``--threads`` lists several values).

Usage
-----
    python examples/example_benchmark.py                     # quick default run
    python examples/example_benchmark.py --sizes 1e5,5e5,1e6 --repeats 3
    python examples/example_benchmark.py --threads 1,2,4,8   # thread scaling
    python examples/example_benchmark.py --weighted
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
BOX = 1000.0


def make_catalogs(n_data, n_rand, weighted, seed):
    rng = np.random.default_rng(seed)
    data = np.ascontiguousarray(rng.uniform(0, BOX, (n_data, 3)))
    rand = np.ascontiguousarray(rng.uniform(0, BOX, (n_rand, 3)))
    if weighted:
        w_d = rng.uniform(0.8, 1.2, n_data)
        w_r = rng.uniform(0.8, 1.2, n_rand)
    else:
        w_d = np.ones(n_data)
        w_r = np.ones(n_rand)
    return data, rand, w_d, w_r


def timeit(fn, repeats):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_pyfcfc(mode, data, rand, w_d, w_r, s_edges, second_edges, nmu,
                 repeats):
    from pyfcfc.boxes import py_compute_cf

    def run():
        if mode == 'smu':
            return py_compute_cf([data, rand], [w_d, w_r], s_edges, None,
                                 nmu, label=['D', 'R'], bin=1,
                                 pair=['DD', 'DR', 'RR'],
                                 cf=['(DD - 2 * DR + RR) / RR'],
                                 multipole=[0, 2, 4], box=BOX)
        if mode == 'rppi':
            return py_compute_cf([data, rand], [w_d, w_r], s_edges,
                                 second_edges, 0, label=['D', 'R'], bin=2,
                                 pair=['DD', 'DR', 'RR'],
                                 cf=['(DD - 2 * DR + RR) / RR'], wp=True,
                                 box=BOX)
        return py_compute_cf([data, rand], [w_d, w_r], s_edges, None, 0,
                             label=['D', 'R'], bin=0, pair=['DD', 'DR', 'RR'],
                             cf=['(DD - 2 * DR + RR) / RR'], box=BOX)

    return timeit(run, repeats)


def bench_pycorr(mode, data, rand, w_d, w_r, s_edges, second_edges_full,
                 repeats, nthreads):
    from pycorr import TwoPointCorrelationFunction

    kwargs = dict(position_type='xyz', boxsize=BOX, los='z',
                  estimator='landyszalay', engine='corrfunc',
                  nthreads=nthreads, compute_sepsavg=False,
                  randoms_positions1=rand.T, randoms_weights1=w_r)

    def run():
        if mode == 'smu':
            corr = TwoPointCorrelationFunction(
                'smu', (s_edges, second_edges_full),
                data_positions1=data.T, data_weights1=w_d, **kwargs)
            return [corr(ell=ell) for ell in (0, 2, 4)]
        if mode == 'rppi':
            corr = TwoPointCorrelationFunction(
                'rppi', (s_edges, second_edges_full),
                data_positions1=data.T, data_weights1=w_d, **kwargs)
            return corr(mode='wp')
        corr = TwoPointCorrelationFunction(
            's', (s_edges,), data_positions1=data.T, data_weights1=w_d,
            **kwargs)
        return corr()

    return timeit(run, repeats)


def run_benchmark(sizes, nmu, npi, repeats, weighted, nthreads):
    from pyfcfc.boxes import simd_info

    simd = simd_info()
    rows = []
    s_edges = np.arange(0, 151, 5, dtype=np.float64)
    mu_full = np.linspace(-1, 1, 2 * nmu + 1)
    pi_half = np.linspace(0, 80., npi + 1)
    pi_full = np.linspace(-80., 80., 2 * npi + 1)

    print(f"pyfcfc SIMD build: {simd[1]}   |   threads: {nthreads}   |   "
          f"weighted: {weighted}")
    header = (f"{'mode':5s} {'N_data':>8s} {'pyfcfc [s]':>11s} "
              f"{'pycorr [s]':>11s} {'speedup':>8s}")
    print(header)
    print('-' * len(header))
    for n_data in sizes:
        n_data = int(n_data)
        n_rand = 2 * n_data
        data, rand, w_d, w_r = make_catalogs(n_data, n_rand, weighted,
                                             seed=1234)
        for mode, second_full, second_half in (
                ('smu', mu_full, None), ('rppi', pi_full, pi_half),
                ('s', None, None)):
            t_fc = bench_pyfcfc(mode, data, rand, w_d, w_r, s_edges,
                                second_half, nmu, repeats)
            t_pc = bench_pycorr(mode, data, rand, w_d, w_r, s_edges,
                                second_full, repeats, nthreads)
            print(f"{mode:5s} {n_data:8d} {t_fc:11.3f} {t_pc:11.3f} "
                  f"{t_pc / t_fc:7.2f}x")
            rows.append(dict(mode=mode, n_data=n_data, n_rand=n_rand,
                             threads=nthreads, simd=simd[1],
                             weighted=bool(weighted),
                             t_pyfcfc=t_fc, t_pycorr=t_pc))
    return rows, simd


def worker_main(args):
    """Run the benchmark in-process and print the rows as JSON."""
    rows, simd = run_benchmark(args.sizes, args.nmu, args.npi, args.repeats,
                               args.weighted,
                               int(os.environ.get('OMP_NUM_THREADS', '1')))
    print("@@JSON@@" + json.dumps(dict(rows=rows, simd=simd[1],
                                       threads=int(os.environ.get(
                                           'OMP_NUM_THREADS', '1')))))


def plot(results, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thread_values = sorted({r['threads'] for r in results})
    simd = results[0]['simd']
    modes = ['smu', 'rppi', 's']

    if len(thread_values) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                                 layout='constrained')
        ax, axr = axes
        threads = thread_values[0]
        for i, mode in enumerate(modes):
            sub = [r for r in results
                   if r['mode'] == mode and r['threads'] == threads]
            ns = [r['n_data'] for r in sub]
            ax.plot(ns, [r['t_pyfcfc'] for r in sub], '--o',
                    color=f'C{3 + i}', lw=1.3, ms=4,
                    label=f'pyfcfc [{mode}]')
            ax.plot(ns, [r['t_pycorr'] for r in sub], '-o',
                    color=f'C{i}', lw=1.3, ms=4, label=f'pycorr [{mode}]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$N_{\rm data}$  ($N_{\rm rand} = 2 N_{\rm data}$)')
        ax.set_ylabel('wall-clock time [s]')
        ax.set_title(f'pyfcfc (SIMD={simd}, {threads} threads) vs pycorr')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, ncol=2)
        for i, mode in enumerate(modes):
            sub = [r for r in results
                   if r['mode'] == mode and r['threads'] == threads]
            ns = [r['n_data'] for r in sub]
            axr.plot(ns, [r['t_pycorr'] / r['t_pyfcfc'] for r in sub],
                     '-o', color=f'C{i}', lw=1.3, ms=4, label=mode)
        axr.axhline(1, color='k', lw=0.8)
        axr.set_xscale('log')
        axr.set_xlabel(r'$N_{\rm data}$')
        axr.set_ylabel('pycorr time / pyfcfc time')
        axr.grid(alpha=0.3)
        axr.legend(fontsize=8)
        fig.savefig(fname, dpi=160)
        plt.close(fig)
        return

    # thread-scaling figure
    n_ref = max(r['n_data'] for r in results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    ax, axr = axes
    for i, mode in enumerate(modes):
        sub = [r for r in results
               if r['mode'] == mode and r['n_data'] == n_ref]
        ts = [r['threads'] for r in sub]
        ax.plot(ts, [r['t_pyfcfc'] for r in sub], '--o', color=f'C{3 + i}',
                lw=1.3, ms=4, label=f'pyfcfc [{mode}]')
        ax.plot(ts, [r['t_pycorr'] for r in sub], '-o', color=f'C{i}',
                lw=1.3, ms=4, label=f'pycorr [{mode}]')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('OpenMP threads')
    ax.set_ylabel(f'wall-clock time [s]  (N_data = {n_ref:.0e})')
    ax.set_title(f'thread scaling (SIMD={simd})')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, ncol=2)
    for i, mode in enumerate(modes):
        sub = [r for r in results
               if r['mode'] == mode and r['n_data'] == n_ref]
        ts = [r['threads'] for r in sub]
        base_fc = sub[0]['t_pyfcfc']
        base_pc = sub[0]['t_pycorr']
        axr.plot(ts, [base_fc / r['t_pyfcfc'] for r in sub], '--o',
                 color=f'C{3 + i}', lw=1.3, ms=4, label=f'pyfcfc [{mode}]')
        axr.plot(ts, [base_pc / r['t_pycorr'] for r in sub], '-o',
                 color=f'C{i}', lw=1.3, ms=4, label=f'pycorr [{mode}]')
    axr.set_xscale('log', base=2)
    axr.set_xlabel('OpenMP threads')
    axr.set_ylabel('speedup vs 1 thread')
    axr.grid(alpha=0.3)
    axr.legend(fontsize=7, ncol=2)
    fig.savefig(fname, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--sizes', default='5e4,2e5',
                   help='comma-separated data catalogue sizes '
                        '(randoms = 2x); default: %(default)s')
    p.add_argument('--nmu', type=int, default=40)
    p.add_argument('--npi', type=int, default=40)
    p.add_argument('--repeats', type=int, default=3,
                   help='timing repetitions (best kept)')
    p.add_argument('--weighted', action='store_true',
                   help='use non-trivial weights (float counting path)')
    p.add_argument('--threads', default=None,
                   help='comma-separated thread counts to scan; each value '
                        'is run in a subprocess with OMP_NUM_THREADS set. '
                        'Default: the current OMP_NUM_THREADS.')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    args.sizes = [float(s) for s in args.sizes.split(',')]

    if args.worker:
        worker_main(args)
        return

    os.makedirs(FIGDIR, exist_ok=True)
    threads = ([int(t) for t in args.threads.split(',')]
               if args.threads else
               [int(os.environ.get('OMP_NUM_THREADS', '1'))])

    results = []
    if len(threads) == 1:
        os.environ.setdefault('OMP_NUM_THREADS', str(threads[0]))
        rows, _ = run_benchmark(args.sizes, args.nmu, args.npi,
                                args.repeats, args.weighted, threads[0])
        results.extend(rows)
    else:
        for t in threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(t))
            cmd = [sys.executable, os.path.abspath(__file__), '--worker',
                   '--sizes', ','.join(str(s) for s in args.sizes),
                   '--nmu', str(args.nmu), '--npi', str(args.npi),
                   '--repeats', str(args.repeats)]
            if args.weighted:
                cmd.append('--weighted')
            out = subprocess.run(cmd, env=env, capture_output=True,
                                 text=True)
            if out.returncode != 0:
                print(out.stdout[-3000:])
                print(out.stderr[-3000:])
                raise SystemExit(f"worker run with {t} threads failed")
            payload = out.stdout.split('@@JSON@@')[-1]
            print(out.stdout.split('@@JSON@@')[0])
            decoded = json.loads(payload)
            results.extend(decoded['rows'])

    # save the raw numbers
    import csv
    csv_name = os.path.join(FIGDIR, 'benchmark_results.csv')
    with open(csv_name, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    fname = os.path.join(FIGDIR, 'benchmark.png')
    plot(results, fname)
    print(f"\nresults saved to:\n  {csv_name}\n  {fname}")


if __name__ == '__main__':
    main()
