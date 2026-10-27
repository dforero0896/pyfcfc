#!/usr/bin/env python3
"""JIT caching benchmark: first (compiling) run vs all subsequent runs.

Demonstrates that the ``pyfcfc.jax`` soft-histogramming primitive does
*not* recompile across repeated calls.  ``soft_pair_counts`` caches one
``jax.custom_vjp`` primitive per static configuration (``_PRIM_CACHE``),
and ``jax.jit`` caches one compiled executable per input signature
(shapes/dtypes).  Consequently:

* the **first** jitted call with a given signature pays the tracing +
  XLA compilation cost,
* **every subsequent** call with the same signature reuses the cached
  executable and only pays dispatch + the C pair counting itself,
* a **new** signature (e.g. a different number of points) triggers
  exactly one additional compilation.

The script measures all three regimes, for the forward pass and for
``jax.grad`` through it, and verifies the compilation caches via
``fn._cache_size()`` where the installed jax exposes it.

Run from anywhere (adjust the two sys.path lines below if you move it):

    python examples/benchmark_jit_cache.py [--n-small 400] [--n-large 4000]
                                           [--reps 5] [--json FILE]

Timings depend on the machine, the jax/XLA version and the OpenMP
thread count; the *ratios* (first vs subsequent) are the informative
part.  With ``--json FILE`` the raw numbers are also written as JSON.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import pyfcfc  # noqa: E402
import pyfcfc.jax as pjax  # noqa: E402
from mocks import make_clustered_box  # noqa: E402

BOX = 400.0


def _cache_size(fn):
    """Compilation-cache size of a jitted fn (None if not introspectable)."""
    sz = getattr(fn, "_cache_size", None)
    if not callable(sz):
        return None
    try:
        return sz()
    except Exception:
        return None


def _time_one(fn, *args):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - t0


def measure(fn, *args, reps):
    """First-call and subsequent-call timings (seconds)."""
    first = _time_one(fn, *args)
    warm = [_time_one(fn, *args) for _ in range(reps)]
    return dict(first_s=first, subsequent_min_s=min(warm),
                subsequent_mean_s=float(np.mean(warm)),
                subsequent_all_s=warm, speedup=first / float(np.mean(warm)))


def _fmt_ms(s):
    return f"{s * 1e3:9.1f}"


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'task':<38}{'first [ms]':>11}{'subseq. min':>13}"
          f"{'subseq. mean':>14}{'speedup':>9}")
    for name, m in rows:
        if m is None:
            print(f"  {name:<38}{'-':>11}{'-':>13}{'-':>14}{'-':>9}")
            continue
        print(f"  {name:<38}{_fmt_ms(m['first_s'])}{_fmt_ms(m['subsequent_min_s'])}"
              f"{_fmt_ms(m['subsequent_mean_s'])}"
              f"{m['speedup']:8.1f}x")


def env_info():
    info = dict(
        platform=platform.platform(),
        python=sys.version.split()[0],
        jax=jax.__version__,
        jaxlib=getattr(jax.lib, "__version__", "unknown"),
        numpy=np.__version__,
        pyfcfc=pyfcfc.__version__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
        x64=bool(jax.config.jax_enable_x64),
        persistent_cache=str(
            getattr(jax.config, "jax_enable_persistent_cache", "n/a")),
    )
    print("environment")
    for k, v in info.items():
        print(f"  {k:<20}{v}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-small", type=int, default=400,
                    help="catalogue size isolating the compilation cost")
    ap.add_argument("--n-large", type=int, default=4000,
                    help="catalogue size for realistic per-call costs")
    ap.add_argument("--reps", type=int, default=5,
                    help="repetitions per measurement (after the first)")
    ap.add_argument("--json", default=None,
                    help="optional path to dump the raw numbers as JSON")
    args = ap.parse_args()

    results = dict(env=env_info(), configs={})

    # log-spaced s bins (20 bins, 0.5 - 40 Mpc/h) with the scale-free
    # kernel sigma(s) = f * s (soft_sigma_log): the recommended setup for
    # projected clustering; one auto-catalogue 'DD' count per call
    s_edges = np.geomspace(0.5, 40.0, 21)
    kw = dict(module="boxes", s_edges=s_edges, pairs=("DD",), labels=["D"],
              soft_hist=True, soft_sigma_log=True, bin=0, box=BOX)

    for tag, n in (("small", args.n_small), ("large", args.n_large)):
        pos, w = make_clustered_box(n, BOX, seed=7)
        X, W = jnp.asarray(pos), jnp.asarray(w)
        rng = np.random.default_rng(7)
        g = jnp.asarray(rng.normal(size=len(s_edges) - 1))

        counts = lambda p, u: pjax.soft_pair_counts(  # noqa: E731
            [p], [u], **kw)["DD"]

        def loss(p, u):
            return jnp.sum(counts(p, u) * g)

        print(f"\n=== configuration '{tag}': N = {n} points, "
              f"{len(s_edges) - 1} log bins 0.5-40 Mpc/h ===")

        cfg = dict(n=n)

        # (0) eager reference (no jit at all)
        eager = measure(lambda p, u: counts(p, u), X, W, reps=args.reps)
        cfg["eager_forward"] = eager

        # (1) jitted forward: first call = trace + compile + callback
        jf = jax.jit(counts)
        fwd = measure(jf, X, W, reps=args.reps)
        cfg["jitted_forward"] = fwd

        # (2) jitted gradient (forward + adjoint callbacks)
        jg = jax.jit(jax.grad(loss, argnums=(0, 1)))
        grad = measure(jg, X, W, reps=args.reps)
        cfg["jitted_grad"] = grad

        # (3) a genuinely new signature: one recompilation, then cached.
        # (A distinct lambda wrapper on purpose: jax.jit deduplicates
        # wrapper creation for identical function objects, so reusing
        # `counts` here would return `jf` itself and just add the new
        # shape to its cache.)
        pos2, w2 = make_clustered_box(n + 1, BOX, seed=8)
        X2, W2 = jnp.asarray(pos2), jnp.asarray(w2)
        jf2 = jax.jit(lambda p, u: counts(p, u))   # fresh wrapper, new shape
        newshape = measure(jf2, X2, W2, reps=max(args.reps // 2, 2))
        cfg["jitted_forward_new_shape"] = newshape

        # cache verification: one compilation per (wrapper, signature)
        cfg["cache_sizes"] = dict(jf=_cache_size(jf), jg=_cache_size(jg),
                                  jf2=_cache_size(jf2),
                                  prim_cache=len(pjax._PRIM_CACHE))

        print_table(f"N = {n}: first run vs subsequent runs", [
            ("forward, eager (no jit)", eager),
            ("forward, jitted", fwd),
            ("grad, jitted", grad),
            ("forward, jitted, NEW shape (recompiles)", newshape),
        ])
        series = [round(t * 1e3, 1)
                  for t in [fwd["first_s"]] + fwd["subsequent_all_s"]]
        print(f"  per-call times of the jitted forward (ms): {series}")
        print(f"  compilation-cache sizes (each wrapper saw a single "
              f"signature; 1 expected): {cfg['cache_sizes']}")

        results["configs"][tag] = cfg

    print("\ninterpretation: only the FIRST jitted call per input signature")
    print("is slow (tracing + XLA compilation); all subsequent calls reuse")
    print("the cached executable.  A new shape costs one extra compilation.")
    print("The C pair counting itself runs in every call (it dominates the")
    print("warm times at large N).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nraw numbers written to {args.json}")


if __name__ == "__main__":
    main()
