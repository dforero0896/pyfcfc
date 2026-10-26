"""Differentiable systematics: fitting a catalogue displacement from the
soft histogram, with the pyfcfc adjoint (no autodiff framework needed).

A realistic reverse-mode use case: the galaxy catalogue is mis-positioned
by an unknown rigid shift theta = (tx, ty, tz) (one parameter per axis,
e.g. an astrometric / calibration offset with respect to a fixed random
catalogue).  The soft-histogram cross counts DR(theta) depend smoothly on
theta, so the shift can be recovered by gradient descent on

    L(theta) = 1/2 |DR(theta) - DR(theta_*)|^2 ,

whose gradient follows from the pyfcfc adjoint by the chain rule,

    dL/dtheta_a = sum_i dL/dx_{i,a} ,     dx_i/dtheta = I ,

i.e. simply the sum of the per-object coordinate cotangents returned by
``soft_hist_vjp`` for the cross pair.  (A rigid translation of *all*
catalogues leaves every separation invariant - Euclidean or minimum
image - so it is an exact null direction of any histogram loss; this
example checks that the adjoint reproduces it to round-off.)

The example

  1. validates dL/dtheta at a displaced position against central finite
     differences of the forward pair counts,
  2. runs Adam on theta to recover the true shift from the histograms
     alone,
  3. plots the gradient check, the loss curve, the parameter trajectory
     and the DR histograms along the fit.

It saves ``examples/figures/soft_hist_fit.png`` and does not require jax
(or pycorr).  Run from the root of the repository:

    python examples/example_soft_hist_fit.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from mocks import make_clustered_box  # noqa: E402
from pyfcfc.boxes import py_compute_cf, soft_hist_vjp  # noqa: E402

BOX = 400.0
S_EDGES = np.arange(0, 191, 5, dtype=np.float64)   # smax < box/2
SIGMA = 3.0
NS = len(S_EDGES) - 1
THETA_TRUE = np.array([4.0, -3.0, 3.0])
THETA_0 = np.zeros(3)
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def raw_dr(gpos, rand, wg, wr):
    """Raw soft-histogram cross counts DR (the differentiable model)."""
    res = py_compute_cf([gpos, rand], [wg, wr], S_EDGES, None, 1,
                        label=['G', 'R'], bin=0, pair=['GR'],
                        soft_hist=True, soft_sigma=SIGMA, box=BOX)
    return res['pairs']['GR'] * res['normalization']['GR']


def loss_and_grad(theta, g, rand, wg, wr, target):
    """L(theta) and dL/dtheta via the pyfcfc adjoint."""
    d = raw_dr(g + theta, rand, wg, wr) - target
    loss = 0.5 * float(d @ d)
    out = soft_hist_vjp([g + theta, rand], [wg, wr], S_EDGES, None, 1,
                        pair='GR', cotangent=d, soft_sigma=SIGMA, bin=0,
                        box=BOX, label=['G', 'R'])
    # dx_i/dtheta = I for a rigid shift: the parameter cotangent is the
    # sum of the per-object coordinate cotangents
    return loss, out['grads']['G'].sum(axis=0)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(2024)

    # Clustered galaxies and randoms, squeezed into an interior cube so
    # that every cross separation stays inside [0, smax) along the whole
    # trajectory (the histogram loss is then smooth in theta).
    g, _ = make_clustered_box(700, BOX, seed=5, n_centers=12)
    g = 150.0 + g / BOX * 100.0
    rand, _ = make_clustered_box(700, BOX, seed=9, n_centers=12)
    rand = 150.0 + rand / BOX * 100.0
    wg = rng.uniform(0.8, 1.2, len(g))
    wr = rng.uniform(0.8, 1.2, len(rand))

    target = raw_dr(g + THETA_TRUE, rand, wg, wr)

    # ------------------------------------------------------------------
    # 1. gradient check at a displaced position
    # ------------------------------------------------------------------
    theta_chk = np.array([2.0, -1.5, 1.0])
    loss0, grad_chk = loss_and_grad(theta_chk, g, rand, wg, wr, target)
    grad = grad_chk
    fd = np.zeros(3)
    eps = 1e-5
    for a in range(3):
        h = np.zeros(3)
        h[a] = eps
        lp, _ = loss_and_grad(theta_chk + h, g, rand, wg, wr, target)
        lm, _ = loss_and_grad(theta_chk - h, g, rand, wg, wr, target)
        fd[a] = (lp - lm) / (2 * eps)
    rel = np.abs(grad_chk - fd) / np.abs(fd)
    print(f"gradient check at theta={theta_chk}: adjoint {grad}")
    print(f"{'':24s}finite-diff {fd}  max rel err {rel.max():.2e}")

    # null direction: translating *both* catalogues changes no separation
    out = soft_hist_vjp([g + theta_chk, rand + theta_chk], [wg, wr],
                        S_EDGES, None, 1, pair='GR',
                        cotangent=raw_dr(g + theta_chk, rand + theta_chk,
                                         wg, wr) - target,
                        soft_sigma=SIGMA, bin=0, box=BOX,
                        label=['G', 'R'])
    null = (out['grads']['G'].sum(axis=0) + out['grads']['R'].sum(axis=0))
    scale = np.abs(out['grads']['G']).sum() + np.abs(out['grads']['R']).sum()
    print(f"rigid-translation null direction: |dL/dtheta| = "
          f"{np.linalg.norm(null):.3e} (scale {scale:.3e})")

    # ------------------------------------------------------------------
    # 2. Adam on theta, starting from zero shift
    # ------------------------------------------------------------------
    theta = THETA_0.copy()
    m = v = np.zeros(3)
    n_iter = 220
    hist = []
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        loss, grad = loss_and_grad(theta, g, rand, wg, wr, target)
        hist.append((loss, *theta))
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mh = m / (1 - 0.9 ** step)
        vh = v / (1 - 0.999 ** step)
        lr = 0.2 * (1 - step / n_iter) ** 2 + 5e-4
        theta = theta - lr * mh / (np.sqrt(vh) + 1e-8)
        if step % 55 == 0 or step == 1:
            print(f"  step {step:3d}: loss {loss:12.4f}   theta = "
                  f"({theta[0]:6.3f}, {theta[1]:6.3f}, {theta[2]:6.3f})")
    dt = time.perf_counter() - t0
    print(f"recovered theta = ({theta[0]:.4f}, {theta[1]:.4f}, "
          f"{theta[2]:.4f}) vs true {THETA_TRUE}  [{dt:.0f} s, "
          f"{n_iter} Adam steps]")

    hist = np.array(hist)
    cnt_0 = raw_dr(g + THETA_0, rand, wg, wr)
    cnt_f = raw_dr(g + theta, rand, wg, wr)

    # ------------------------------------------------------------------
    # 3. figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    comps = ["$t_x$", "$t_y$", "$t_z$"]
    cols = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, 2))

    ax = axes[0, 0]
    wdt = 0.38
    idx = np.arange(3)
    ax.bar(idx - wdt / 2, fd, wdt, color=cols[0], label="finite differences")
    ax.bar(idx + wdt / 2, grad_chk, wdt, color=cols[1],
           label="adjoint (soft_hist_vjp)")
    ax.set_xticks(idx)
    ax.set_xticklabels(comps)
    ax.set_ylabel("$dL/d\\theta_a$")
    ax.set_title(f"Gradient of the histogram-matching loss "
                 f"(max rel err {rel.max():.1e})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    ax.semilogy(hist[:, 0], lw=1.5, color=cols[0])
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss  $\\frac{1}{2}|DR(\\theta)-DR(\\theta_*)|^2$")
    ax.set_title("Displacement recovery from cross histograms")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    for a in range(3):
        ax.plot(hist[:, 1 + a], lw=1.4, color=plt.get_cmap("viridis")(
            0.2 + 0.6 * a / 2), label=comps[a])
        ax.axhline(THETA_TRUE[a], ls=":", color="0.4")
    ax.set_xlabel("Adam step")
    ax.ylabel = None
    ax.set_ylabel("shift parameter [Mpc/h]")
    ax.set_title("Trajectory of the shift parameters (dotted: truth)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    cen = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    ax.step(np.repeat(S_EDGES, 2)[1:-1], np.repeat(target, 2), where="mid",
            color="k", lw=1.5, label="target  $\\theta_*$")
    ax.plot(cen, cnt_0, "--", color="crimson", lw=1.3, label="start "
            r"$\theta=0$")
    ax.plot(cen, cnt_f, "-o", ms=3, lw=1.2, color=cols[1], label="fitted")
    ax.set_xlabel("$s$ [Mpc/h]")
    ax.set_ylabel("soft DR pair counts")
    ax.set_title("Cross histograms along the fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "soft_hist_fit.png")
    fig.savefig(out_png, dpi=140)
    print(f"figure saved to {out_png}")


if __name__ == "__main__":
    main()
