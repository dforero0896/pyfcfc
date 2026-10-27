"""Realistic log-binned projected correlation function w_p, differentiable
end to end with pyfcfc.jax.

w_p is normally measured on small scales with logarithmic r_p bins; this
example uses ``geomspace(0.5, 40, 200)`` on a hierarchically clustered
mock (clump scales from 0.05 to 8 Mpc/h, see ``mocks.make_wp_mock``),
with the scale-free soft kernel
``sigma(s) = f * s`` (``soft_sigma_log=True``) so that the smoothing is
uniform in log r_p instead of washing out the small bins.

The whole post-processing chain - Landy-Szalay xi(r_p, pi), the pi
integration to w_p, and the model (a transverse stretch of the galaxy
slab) - is built with JAX ops on top of the ``pyfcfc.jax.soft_pair_counts``
custom-VJP primitive, so ``jax.grad`` differentiates w_p with respect to
the model parameter through the C adjoint:

  1. gradient check of a w_p functional against finite differences;
  2. Adam recovery of the transverse stretch from w_p alone;
  3. panels: the 2D xi(r_p, pi) map, w_p on semilog-x axes with a linear
     y axis (target / start / fitted), the loss curve and the parameter
     trajectory;
  4. a second figure comparing the hard histogram with several kernel
     widths, as in example_soft_hist.py, on an intermediate setup (a
     denser mock than the fit, moderately coarse display bins).  All
     kernel widths are absolute: in r_p the scale-free kernel
     sigma_s(r_p) = f * r_p [Mpc/h] is specified through the
     dimensionless fraction f (soft_sigma_log), and in pi through a
     fixed sigma_pi [Mpc/h].  The right-hand panels show soft - hard
     differences in Mpc/h (a soft / hard ratio would diverge wherever
     w_p crosses zero).

It saves ``examples/figures/soft_hist_wp.png`` and
``examples/figures/soft_hist_wp_sigmas.png``.  Requires ``jax``
(``pip install jax``).  Run from the root of the repository:

    python examples/example_soft_hist_wp.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from mocks import make_wp_mock  # noqa: E402

try:
    import jax
    import jax.numpy as jnp
except ImportError:                                   # pragma: no cover
    print("this example requires jax:  pip install jax")
    sys.exit(1)

from pyfcfc.boxes import py_compute_cf  # noqa: E402
from pyfcfc.jax import (soft_pair_counts, pair_normalizations,  # noqa: E402
                         xi_landy_szalay, projected_wp)

jax.config.update("jax_enable_x64", True)

BOX = 100.0
RP_EDGES = np.geomspace(0.5, 40, 200)           # realistic log w_p bins
PI_EDGES = np.arange(0, 41, 10, dtype=np.float64)
SIGMA_PI = 3.0
A_TRUE = 0.03
CENTRE = np.array([43.5, 43.5, 50.0])           # centre of the mock slab
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

KW = dict(module='boxes', s_edges=RP_EDGES, pedges=PI_EDGES, nmu=1,
          bin=2, pairs=('GG', 'GR', 'RR'), labels=['G', 'R'],
          soft_sigma=None,                # auto: half the mean log bin
          soft_sigma_log=True, soft_pi=True, soft_pi_sigma=SIGMA_PI,
          box=BOX)


def stretch(pos, a):
    """Transverse stretch about the slab centre (LOS untouched).

    The model parameter enters through the separation definition (both
    catalogues stretched, AP-style), so w_p shifts monotonically in
    log r_p and the fit is well posed.
    """
    return CENTRE + (pos - CENTRE) * jnp.array([1 + a, 1 + a, 1.0])


def counts_at(a):
    return soft_pair_counts([stretch(GPOS, a), stretch(RAND, a)], [WG, WR],
                            **KW)


def wp_at(a):
    c = counts_at(a)
    n = NORMS
    u = c['GG'] / n['GG']
    v = c['GR'] / n['GR']
    w = c['RR'] / n['RR']
    # fixed support mask (see main); the mask also guards the division so
    # that reverse mode never sees 0 * inf in unsupported cells
    xi = jnp.where(SUPPORT, (u - 2 * v + w) / jnp.where(SUPPORT, w, 1.0),
                   0.0)
    return projected_wp(xi, PI_EDGES), xi


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    global RAND, WG, WR, NORMS, SUPPORT
    rng = np.random.default_rng(2024)

    g = make_wp_mock(2500, seed=11)
    # uniform randoms in the same slab: a strong, smooth xi signal
    RAND = jnp.asarray(np.array([30., 30., 0.]) +
                       rng.uniform(0, 1, (2500, 3)) *
                       np.array([27., 27., 100.]))
    WG = jnp.ones(len(g))
    WR = jnp.ones(len(RAND))
    NORMS = pair_normalizations([WG, WR], ('GG', 'GR', 'RR'),
                                labels=['G', 'R'])
    # fixed support mask: (r_p, pi) cells with random-pair support
    global GPOS
    GPOS = jnp.asarray(g)
    c0 = counts_at(jnp.float64(0.0))
    SUPPORT = (c0['RR'] / NORMS['RR']) > 1e-5

    g_j = GPOS
    G = jnp.asarray(rng.normal(size=len(RP_EDGES) - 1))
    # analysis cut for the *loss*: the smallest log bins carry only a
    # handful of random pairs (shot noise dominates); the plots still
    # show every bin.  Fixed mask => smooth loss.
    LOSS_MASK = jnp.asarray(c0['RR'].sum(axis=1) > 30)
    print(f"loss uses {int(LOSS_MASK.sum())}/{LOSS_MASK.size} r_p bins "
          f"(RR support cut)")

    # ------------------------------------------------------------------
    # 1. gradient check of a w_p functional
    # ------------------------------------------------------------------
    print("checking jax.grad of a w_p functional vs FD "
          f"({len(RP_EDGES) - 1} log bins) ...")

    def lin(a):
        wp, _ = wp_at(a)
        return jnp.sum(G * wp * LOSS_MASK)

    a0 = jnp.float64(0.01)
    grad = float(jax.grad(lin)(a0))
    eps = 1e-6
    fd = (float(lin(a0 + eps)) - float(lin(a0 - eps))) / (2 * eps)
    rel = abs(grad - fd) / abs(fd)
    print(f"  adjoint {grad:.10e}  FD {fd:.10e}  rel {rel:.2e}")

    # ------------------------------------------------------------------
    # 2. recover the transverse stretch from w_p
    # ------------------------------------------------------------------
    print(f"recovering transverse stretch a (true {A_TRUE}) from w_p ...")

    wp_t, xi_t = wp_at(jnp.float64(A_TRUE))
    wp_t = jnp.where(LOSS_MASK, wp_t, 0.0)   # unsupported bins: plot/loss cut

    def loss(a):
        wp, _ = wp_at(a)
        d = (wp - wp_t) * LOSS_MASK
        return 0.5 * jnp.sum(d * d)

    a = jnp.float64(0.0)
    m = v = 0.0
    n_iter = 50
    hist = []
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        val, gr = jax.value_and_grad(loss)(a)
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        mh = m / (1 - 0.9 ** step)
        vh = v / (1 - 0.999 ** step)
        lr = 4e-3 * (1 - step / n_iter) ** 2 + 1e-5
        a = a - lr * mh / (np.sqrt(vh) + 1e-8)
        hist.append((float(val), float(a)))
        if step % 20 == 0 or step == 1:
            print(f"  step {step:3d}: loss {float(val):12.5f}  "
                  f"a = {float(a):.5f}")
    dt = time.perf_counter() - t0
    hist = np.array(hist)
    print(f"  fit a = {hist[-1, 1]:.5f} (true {A_TRUE})  [{dt:.0f} s]")

    wp_0, xi_0 = wp_at(jnp.float64(0.0))
    wp_f, _ = wp_at(jnp.float64(hist[-1, 1]))
    lmask = np.asarray(LOSS_MASK).astype(bool)
    wp_t = np.where(lmask, np.asarray(wp_t), np.nan)
    wp_0 = np.where(lmask, np.asarray(wp_0), np.nan)
    wp_f = np.where(lmask, np.asarray(wp_f), np.nan)
    rp = np.sqrt(RP_EDGES[:-1] * RP_EDGES[1:])
    pic = 0.5 * (PI_EDGES[:-1] + PI_EDGES[1:])

    # xi(r_p, pi) map on coarser display bins (same catalogue, fitted
    # stretch): the 199-bin map is shot-noise dominated per cell
    rn = np.asarray(RAND)
    wn = np.asarray(WG)
    wrn = np.asarray(WR)
    RP_M = np.geomspace(0.5, 40, 40)
    PI_M = np.arange(0, 41, 5, dtype=np.float64)
    cd = py_compute_cf([np.asarray(stretch(g_j, jnp.float64(hist[-1, 1]))),
                        rn], [wn, wrn], RP_M, PI_M, 1,
                       label=['G', 'R'], bin=2, pair=['GG', 'GR', 'RR'],
                       soft_hist=True, soft_sigma_log=True, soft_pi=True,
                       soft_pi_sigma=SIGMA_PI, box=BOX)
    sup_d = cd['pairs']['RR'] > 1e-5
    xi_d = np.where(sup_d, (cd['pairs']['GG'] - 2 * cd['pairs']['GR'] +
                            cd['pairs']['RR']) /
                    np.where(sup_d, cd['pairs']['RR'], 1.0), np.nan)
    rpm = np.sqrt(RP_M[:-1] * RP_M[1:])
    picm = 0.5 * (PI_M[:-1] + PI_M[1:])

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    vir = plt.get_cmap("viridis")

    ax = axes[0, 0]
    im = ax.pcolormesh(rpm, picm, xi_d.T, norm=SymLogNorm(1e-2),
                       cmap="viridis", shading='auto')
    ax.set_xscale("log")
    ax.set_xlabel("$r_p$ [Mpc/h]")
    ax.set_ylabel("$\\pi$ [Mpc/h]")
    ax.set_title(r"soft $\xi(r_p, \pi)$ at the fitted stretch")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 1]
    ax.plot(rp, np.asarray(wp_t), "k-", lw=1.6, label="target $w_p$")
    ax.plot(rp, np.asarray(wp_0), "--", color="crimson", lw=1.1,
            label="start ($a=0$)")
    ax.plot(rp, np.asarray(wp_f), "-", color=vir(0.7), lw=1.1,
            label=f"fitted ($a={hist[-1, 1]:.4f}$)")
    ax.set_xscale("log")          # semilogx: log r_p, linear w_p
    ax.set_xlabel("$r_p$ [Mpc/h]")
    ax.set_ylabel("$w_p(r_p)$ [Mpc/h]")
    ax.set_title("Projected correlation function on log bins")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.semilogy(hist[:, 0], lw=1.5, color=vir(0.25))
    ax.set_xlabel("Adam step")
    ax.set_ylabel("$\\frac{1}{2}|w_p(a)-w_p(a_*)|^2$")
    ax.set_title("Differentiable $w_p$ fitting")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 1]
    ax.plot(hist[:, 1], lw=1.5, color=vir(0.25), label="gradient descent")
    ax.axhline(A_TRUE, ls=":", color="k", label="truth")
    ax.axhline(0.0, ls=":", color="crimson", label="start")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("$\\alpha_\\perp - 1$")
    ax.set_title(f"Recovered stretch (grad check rel err {rel:.1e})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "soft_hist_wp.png")
    fig.savefig(out_png, dpi=140)
    print(f"figure saved to {out_png}")

    # ------------------------------------------------------------------
    # second figure: how the kernel widths shape w_p.  Intermediate
    # setup: a denser mock than the fit (N=8000 vs 2500) on moderately
    # coarse display bins (50 log r_p bins instead of 199): per-bin shot
    # noise stays small without the cost of a very dense mock.  Kernel widths are
    # absolute: sigma_s(r_p) = f * r_p [Mpc/h] via the dimensionless
    # fraction f (soft_sigma_log), sigma_pi [Mpc/h] fixed for all bins.
    # ------------------------------------------------------------------
    gD = make_wp_mock(8000, seed=13)
    rD = np.array([30., 30., 0.]) + rng.uniform(0, 1, (8000, 3)) * \
        np.array([27., 27., 100.])
    wD = np.ones(len(gD))
    wDr = np.ones(len(rD))
    RP_D = np.geomspace(0.5, 40, 51)
    PI_D = np.arange(0, 41, 5, dtype=np.float64)
    rpd = np.sqrt(RP_D[:-1] * RP_D[1:])
    # f_auto = half the mean log bin width: sigma_s = f_auto * r_p then
    # equals half the *local* r_p bin width at every scale
    f_auto = 0.5 * np.log(RP_D[-1] / RP_D[0]) / (len(RP_D) - 1)

    def wp_numpy(f=None, spi=None):
        kw = dict(label=['G', 'R'], bin=2, pair=['GG', 'GR', 'RR'], box=BOX)
        if f is not None:
            kw.update(soft_hist=True, soft_sigma_log=True, soft_sigma=f,
                      soft_pi=True, soft_pi_sigma=spi)
        res = py_compute_cf([gD, rD], [wD, wDr], RP_D, PI_D, 1, **kw)
        u, v, w = (res['pairs'][p] for p in ('GG', 'GR', 'RR'))
        sup = w > 1e-5
        xi = np.where(sup, (u - 2 * v + w) / np.where(sup, w, 1.0), 0.0)
        wp = 2.0 * np.sum(xi * np.diff(PI_D)[None, :], axis=1)
        # py_compute_cf returns *normalized* counts: rescale to raw RR
        rr_raw = w.sum(axis=1) * res['normalization']['RR']
        return wp, rr_raw                 # w_p and projected raw RR counts

    print("computing the kernel-width comparison (N=8000, 50 log bins) ...")
    print(f"  f_auto = {f_auto:.4f}  =>  sigma_s = f_auto * r_p: "
          f"{f_auto * 1:.3f} Mpc/h at r_p = 1, {f_auto * 40:.2f} Mpc/h "
          f"at r_p = 40 (half the local bin width)")
    wp_hard, rr_hard = wp_numpy()
    keep = rr_hard > 30                   # display cut on random support
    if not keep.any():                    # defensive: never mask all bins
        keep = rr_hard > 0.0
    print(f"  display uses {int(keep.sum())}/{keep.size} r_p bins "
          f"(RR support cut)")
    fs = [0.5 * f_auto, f_auto, 2.0 * f_auto]
    wp_fs = [wp_numpy(f=f, spi=SIGMA_PI)[0] for f in fs]
    spis = [0.5 * SIGMA_PI, 2.0 * SIGMA_PI]
    wp_ps = [wp_numpy(f=f_auto, spi=q)[0] for q in spis]
    wp_hard = np.where(keep, wp_hard, np.nan)
    wp_fs = [np.where(keep, c, np.nan) for c in wp_fs]
    wp_ps = [np.where(keep, c, np.nan) for c in wp_ps]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax = axes[0, 0]
    ax.plot(rpd, wp_hard, "k-", lw=1.4, label="hard histogram")
    for f, wpc, c in zip(fs, wp_fs, [vir(0.2), vir(0.5), vir(0.85)]):
        ax.plot(rpd, wpc, lw=1.1, color=c,
                label=rf"soft, $f={f / f_auto:g}\,f_{{\rm auto}}"
                      rf"={f:.3f}$")
    ax.set_xscale("log")
    ax.set_xlabel("$r_p$ [Mpc/h]")
    ax.set_ylabel("$w_p(r_p)$ [Mpc/h]")
    ax.set_title(r"Kernel width in $r_p$: $\sigma_s(r_p) = f\,r_p$")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.03, 0.06,
            "$\\sigma_s$ = absolute width [Mpc/h], grows with scale;\n"
            "$f$ = dimensionless fraction, $f_{\\rm auto}=\\Delta\\ln r_p/2$\n"
            "$\\Rightarrow\\sigma_s$ = half the local bin width everywhere",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            bbox=dict(fc="w", alpha=0.75, ec="0.7"))
    ax.grid(alpha=0.25, which="both")

    ax = axes[0, 1]
    for f, wpc, c in zip(fs, wp_fs, [vir(0.2), vir(0.5), vir(0.85)]):
        ax.plot(rpd, wpc - wp_hard, lw=1.1, color=c,
                label=rf"$f={f / f_auto:g}\,f_{{\rm auto}}={f:.3f}$")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("$r_p$ [Mpc/h]")
    ax.set_ylabel("soft $-$ hard $w_p$ [Mpc/h]")
    ax.set_title("Difference to the hard histogram")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(rpd, wp_hard, "k-", lw=1.4, label="hard histogram")
    for q, wpc, c in zip(spis, wp_ps, [vir(0.2), vir(0.85)]):
        ax.plot(rpd, wpc, lw=1.1, color=c,
                label=rf"soft, $\sigma_\pi={q:g}$ Mpc/h")
    ax.plot(rpd, wp_fs[1], lw=1.1, ls="--", color=vir(0.5),
            label=rf"$\sigma_\pi={SIGMA_PI:g}$ Mpc/h (reference)")
    ax.set_xscale("log")
    ax.set_xlabel("$r_p$ [Mpc/h]")
    ax.set_ylabel("$w_p(r_p)$ [Mpc/h]")
    ax.set_title(r"Kernel width in $\pi$: fixed $\sigma_\pi$ [Mpc/h]")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.03, 0.06,
            "$\\sigma_\\pi$ = absolute width [Mpc/h], the same for all\n"
            "bins (here $\\Delta\\pi=5$ Mpc/h; no log scaling for $\\pi$)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            bbox=dict(fc="w", alpha=0.75, ec="0.7"))
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 1]
    for q, wpc, c in zip(spis, wp_ps, [vir(0.2), vir(0.85)]):
        ax.plot(rpd, wpc - wp_hard, lw=1.1, color=c,
                label=rf"$\sigma_\pi={q:g}$ Mpc/h")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("$r_p$ [Mpc/h]")
    ax.set_ylabel("soft $-$ hard $w_p$ [Mpc/h]")
    ax.set_title("Difference to the hard histogram")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "soft_hist_wp_sigmas.png")
    fig.savefig(out_png, dpi=140)
    print(f"figure saved to {out_png}")


if __name__ == "__main__":
    main()
