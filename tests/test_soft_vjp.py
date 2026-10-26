"""Validate the adjoint (VJP) of the soft histogramming method.

``soft_hist_vjp`` propagates cotangents from the soft pair count bins back
to the catalogue coordinates (and optionally the weights and the kernel
width).  Since the soft histogram is a smooth function of the coordinates,
the adjoint can be validated against central finite differences of the
forward pass; the configurations below are chosen free of range kinks (all
pair separations strictly inside the binning range, single hard mu / pi
bin) so that the forward is differentiable everywhere the FD step moves.

Also checked: the conservation-law invariant (a constant cotangent yields
zero coordinate gradient), linearity of the VJP, the ordered-pair factor
of auto counts, and the ``normalized`` convenience flag.
"""

import numpy as np
import pytest

from pyfcfc import sky
from pyfcfc.boxes import py_compute_cf, soft_hist_vjp

SIGMA = 6.0
S_EDGES = np.arange(0, 181, 20, dtype=np.float64)
PI_EDGES = np.array([0., 180.])


def _make_box_catalogs(rng, n1=110, n2=90):
    """Points in a 100-side cube: all separations < 174 < smax = 180."""
    d = rng.uniform(150, 250, (n1, 3))
    r = rng.uniform(150, 250, (n2, 3))
    return d, r


def _make_sky_catalog(rng, n=110):
    """Points in a thin radial shell: separations stay well inside smax."""
    xyz = rng.normal(size=(n, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    xyz *= rng.uniform(500, 560, (n, 1))
    return xyz


def _cotangent(rng, bin_type, ns, nmu=1, npi=1):
    if bin_type == 0:
        return rng.normal(size=(ns,))
    return rng.normal(size=(ns, nmu if bin_type == 1 else npi))


def _fd_directional(loss, x, v, eps=1e-6):
    return (loss(x + eps * v) - loss(x - eps * v)) / (2 * eps)


def _check(mod, cats, wts, bin_type, pair, sigma, pedges, nmu, grads_of,
           rtol=1e-4, extra=None):
    """Directional finite-difference check of the coordinate VJP."""
    rng = np.random.default_rng(12345 + bin_type)
    ns = len(S_EDGES) - 1
    g = _cotangent(rng, bin_type, ns, nmu, len(pedges) - 1 if pedges is not None else 1)
    labels = [chr(65 + i) for i in range(len(cats))]
    fwd = dict(label=labels, bin=bin_type, pair=[pair], soft_hist=True,
               soft_sigma=sigma, **(extra or {}))

    def loss(cs):
        res = mod.py_compute_cf(cs, wts, S_EDGES, pedges, nmu, **fwd)
        raw = res['pairs'][pair] * res['normalization'][pair]
        return float(np.sum(g * raw))

    which = grads_of
    v = rng.normal(size=cats[which].shape)
    fd = _fd_directional(lambda x: loss([x if k == which else c
                                         for k, c in enumerate(cats)]),
                         cats[which], v)
    out = mod.soft_hist_vjp(cats, wts, S_EDGES, pedges, nmu, pair=pair,
                            cotangent=g, soft_sigma=sigma, bin=bin_type,
                            **(extra or {}))
    an = float(np.sum(out['grads'][labels[which]] * v))
    assert np.isclose(an, fd, rtol=rtol, atol=1e-7 * max(1.0, abs(fd))), \
        f"bin={bin_type} pair={pair}: VJP {an} vs FD {fd}"


@pytest.mark.parametrize("bin_type,nmu", [(0, 1), (1, 1), (2, 1)])
def test_vjp_box_cross(bin_type, nmu):
    rng = np.random.default_rng(1)
    d, r = _make_box_catalogs(rng)
    w1 = rng.uniform(0.5, 1.5, len(d))
    w2 = rng.uniform(0.5, 1.5, len(r))
    pedges = PI_EDGES if bin_type == 2 else None
    import pyfcfc.boxes as boxes
    _check(boxes, [d, r], [w1, w2], bin_type, 'AB', SIGMA, pedges, nmu, 1,
             extra=dict(box=400.))
    _check(boxes, [d, r], [w1, w2], bin_type, 'AB', SIGMA, pedges, nmu, 0,
             extra=dict(box=400.))


@pytest.mark.parametrize("bin_type", [0, 1, 2])
def test_vjp_box_auto(bin_type):
    rng = np.random.default_rng(2)
    d, _ = _make_box_catalogs(rng)
    w = rng.uniform(0.5, 1.5, len(d))
    pedges = PI_EDGES if bin_type == 2 else None
    import pyfcfc.boxes as boxes
    _check(boxes, [d], [w], bin_type, 'AA', SIGMA, pedges, 1, 0,
             extra=dict(box=400.))


@pytest.mark.parametrize("bin_type", [0, 1, 2])
def test_vjp_sky_auto(bin_type):
    rng = np.random.default_rng(3)
    xyz = _make_sky_catalog(rng)
    w = rng.uniform(0.5, 1.5, len(xyz))
    dmax = np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1))
    global S_EDGES
    saved = S_EDGES
    try:
        S_EDGES = np.linspace(0, dmax * 1.05, 9)
        pedges = np.array([0., dmax * 1.05]) if bin_type == 2 else None
        _check(sky, [xyz], [w], bin_type, 'AA', 15.0, pedges, 1, 0)
    finally:
        S_EDGES = saved


def test_vjp_box_balltree_forward():
    """The VJP is tree-independent; the forward may use the ball tree."""
    rng = np.random.default_rng(4)
    d, r = _make_box_catalogs(rng)
    w1 = rng.uniform(0.5, 1.5, len(d))
    w2 = rng.uniform(0.5, 1.5, len(r))
    import pyfcfc.boxes as boxes
    _check(boxes, [d, r], [w1, w2], 0, 'AB', SIGMA, None, 1, 1,
             extra=dict(box=400.))
    g = rng.normal(size=(len(S_EDGES) - 1,))
    out = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1, pair='AB',
                              cotangent=g, soft_sigma=SIGMA, bin=0, box=400.,
                              data_struct=1)
    assert np.all(np.isfinite(out['grads']['A']))


def test_vjp_weights_and_sigma():
    rng = np.random.default_rng(5)
    d, r = _make_box_catalogs(rng)
    w1 = rng.uniform(0.5, 1.5, len(d))
    w2 = rng.uniform(0.5, 1.5, len(r))
    g = rng.normal(size=(len(S_EDGES) - 1,))
    import pyfcfc.boxes as boxes
    fwd = dict(label=['A', 'B'], bin=0, pair=['AB'], soft_hist=True,
               soft_sigma=SIGMA, box=400.)

    def loss(ws, sg):
        res = boxes.py_compute_cf([d, r], ws, S_EDGES, None, 1,
                                  **{**fwd, 'soft_sigma': sg})
        return float(np.sum(g * res['pairs']['AB'] *
                            res['normalization']['AB']))

    eps = 1e-6
    vw = rng.normal(size=w2.shape)
    fd_w = _fd_directional(lambda x: loss([w1, x], SIGMA), w2, vw)
    fd_s = (loss([w1, w2], SIGMA + eps) - loss([w1, w2], SIGMA - eps)) / \
        (2 * eps)
    out = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1, pair='AB',
                              cotangent=g, soft_sigma=SIGMA, bin=0,
                              grad_wts=True, grad_sigma=True, box=400.)
    an_w = float(np.sum(out['grad_wts']['B'] * vw))
    an_s = out['grad_sigma']
    assert np.isclose(an_w, fd_w, rtol=1e-4, atol=1e-7 * abs(fd_w))
    assert np.isclose(an_s, fd_s, rtol=1e-4, atol=1e-7 * abs(fd_s))


def test_vjp_invariants():
    """Conservation => constant cotangents give zero coordinate gradient;
    the VJP is linear in the cotangent."""
    rng = np.random.default_rng(6)
    d, r = _make_box_catalogs(rng)
    w1 = rng.uniform(0.5, 1.5, len(d))
    w2 = rng.uniform(0.5, 1.5, len(r))
    import pyfcfc.boxes as boxes
    ns = len(S_EDGES) - 1
    common = dict(pair='AB', soft_sigma=SIGMA, bin=0, box=400.)

    out = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1,
                              cotangent=np.ones(ns), **common)
    assert np.allclose(out['grads']['A'], 0, atol=1e-10)
    assert np.allclose(out['grads']['B'], 0, atol=1e-10)

    g1, g2 = rng.normal(size=(ns,)), rng.normal(size=(ns,))
    o1 = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1,
                             cotangent=g1, **common)
    o2 = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1,
                             cotangent=g2, **common)
    o3 = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1,
                             cotangent=2 * g1 - 3 * g2, **common)
    for lab in ('A', 'B'):
        assert np.allclose(o3['grads'][lab],
                           2 * o1['grads'][lab] - 3 * o2['grads'][lab],
                           rtol=1e-10, atol=1e-10)


def test_vjp_normalized_flag():
    rng = np.random.default_rng(7)
    d, r = _make_box_catalogs(rng)
    w1 = rng.uniform(0.5, 1.5, len(d))
    w2 = rng.uniform(0.5, 1.5, len(r))
    import pyfcfc.boxes as boxes
    ns = len(S_EDGES) - 1
    g = rng.normal(size=(ns,))
    res = boxes.py_compute_cf([d, r], [w1, w2], S_EDGES, None, 1,
                              label=['A', 'B'], bin=0, pair=['AB'],
                              soft_hist=True, soft_sigma=SIGMA, box=400.)
    norm = res['normalization']['AB']
    o_raw = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1,
                                pair='AB', cotangent=g / norm,
                                soft_sigma=SIGMA, bin=0, box=400.)
    o_nrm = boxes.soft_hist_vjp([d, r], [w1, w2], S_EDGES, None, 1,
                                pair='AB', cotangent=g, normalized=True,
                                soft_sigma=SIGMA, bin=0, box=400.)
    for lab in ('A', 'B'):
        assert np.allclose(o_raw['grads'][lab], o_nrm['grads'][lab],
                           rtol=1e-12, atol=1e-12)


def test_vjp_api_errors():
    rng = np.random.default_rng(8)
    d, _ = _make_box_catalogs(rng, 40, 40)
    w = np.ones(len(d))
    import pyfcfc.boxes as boxes
    ns = len(S_EDGES) - 1
    with pytest.raises(ValueError, match="cotangent"):
        boxes.soft_hist_vjp([d], [w], S_EDGES, None, 1, pair='AA',
                            cotangent=None, bin=0, box=400.)
    with pytest.raises(ValueError, match="shape"):
        boxes.soft_hist_vjp([d], [w], S_EDGES, None, 1, pair='AA',
                            cotangent=np.ones((ns, 3)), soft_sigma=SIGMA,
                            bin=0, box=400.)
    with pytest.raises(ValueError, match="invalid pair"):
        boxes.soft_hist_vjp([d], [w], S_EDGES, None, 1, pair='ZZ',
                            cotangent=np.ones(ns), soft_sigma=SIGMA,
                            bin=0, box=400.)
    with pytest.raises(ValueError, match="soft_sigma"):
        boxes.soft_hist_vjp([d], [w], S_EDGES, None, 1, pair='AA',
                            cotangent=np.ones(ns), soft_sigma=-1.0,
                            bin=0, box=400.)
    with pytest.raises(ValueError, match="normalized"):
        boxes.soft_hist_vjp([d], [w], S_EDGES, None, 1, pair='AA',
                            cotangent=np.ones(ns), normalized=True,
                            grad_wts=True, soft_sigma=SIGMA, bin=0,
                            box=400.)


# ============================================================================
#  Realistic scenario: a displacement model on the catalogue positions
#
#  The galaxy catalogue is translated by theta = (tx, ty, tz) against a
#  fixed random catalogue, and the loss matches the cross (DR) soft
#  histogram to a target:  L(theta) = 1/2 |DR(theta) - DR(theta*)|^2.
#  Since dx_i/dtheta = I, the parameter gradient is the sum of the
#  per-object coordinate cotangents returned by the adjoint.  (A rigid
#  translation of *all* catalogues leaves every separation invariant,
#  which provides the null-direction check below.)
# ============================================================================

DISP_EDGES = np.arange(0, 191, 10, dtype=np.float64)
DISP_SIGMA = 4.0


def _displacement_catalogs(rng, ng=140, nr=140):
    """Galaxies and randoms in an interior cube: every cross separation
    stays below smax - max|theta| along the whole trajectory (no kinks)."""
    g = rng.uniform(150, 250, (ng, 3))
    r = rng.uniform(150, 250, (nr, 3))
    return g, r


def _disp_loss_data(rng, theta_star=None, weighted=True):
    from pyfcfc.boxes import py_compute_cf
    g, r = _displacement_catalogs(rng)
    if weighted:
        wg = rng.uniform(0.5, 1.5, len(g))
        wr = rng.uniform(0.5, 1.5, len(r))
    else:
        wg = np.ones(len(g))
        wr = np.ones(len(r))
    if theta_star is None:
        theta_star = np.zeros(3)

    def raw_dr(theta):
        res = py_compute_cf([g + np.asarray(theta), r], [wg, wr],
                            DISP_EDGES, None, 1, label=['G', 'R'], bin=0,
                            pair=['GR'], soft_hist=True,
                            soft_sigma=DISP_SIGMA, box=400.)
        return res['pairs']['GR'] * res['normalization']['GR']

    target = raw_dr(theta_star)
    return g, r, wg, wr, raw_dr, target


def _disp_grad_theta(g, r, wg, wr, theta, cotangent):
    """dL/dtheta from the C adjoint: sum of the coordinate cotangents."""
    from pyfcfc.boxes import soft_hist_vjp
    out = soft_hist_vjp([g + np.asarray(theta), r], [wg, wr], DISP_EDGES,
                        None, 1, pair='GR', cotangent=cotangent,
                        soft_sigma=DISP_SIGMA, bin=0, box=400.,
                        label=['G', 'R'])
    return out['grads']['G'].sum(axis=0)


@pytest.mark.parametrize("weighted", [False, True])
def test_vjp_displacement_loss_gradient(weighted):
    """Adjoint parameter gradient vs central finite differences."""
    rng = np.random.default_rng(21)
    g, r, wg, wr, raw_dr, target = _disp_loss_data(rng, weighted=weighted)
    theta0 = np.array([6.0, -5.0, 4.0])

    def loss(theta):
        d = raw_dr(theta) - target
        return 0.5 * float(np.dot(d, d))

    cot = raw_dr(theta0) - target
    an = _disp_grad_theta(g, r, wg, wr, theta0, cot)
    fd = np.zeros(3)
    eps = 1e-5
    for a in range(3):
        h = np.zeros(3)
        h[a] = eps
        fd[a] = (loss(theta0 + h) - loss(theta0 - h)) / (2 * eps)
    assert np.allclose(an, fd, rtol=1e-4, atol=1e-7 * np.abs(fd).max()), \
        f"adjoint {an} vs FD {fd}"


def test_vjp_displacement_recovery():
    """Gradient descent on the displacement recovers the true shift."""
    rng = np.random.default_rng(22)
    theta_star = np.array([3.0, -2.0, 2.5])
    g, r, wg, wr, raw_dr, target = _disp_loss_data(rng, theta_star,
                                                   weighted=False)
    theta = np.zeros(3)
    m = v = np.zeros(3)
    n_iter = 120
    losses = []
    for step in range(1, n_iter + 1):
        d = raw_dr(theta) - target
        losses.append(0.5 * float(np.dot(d, d)))
        gr = _disp_grad_theta(g, r, wg, wr, theta, d)
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        mh = m / (1 - 0.9 ** step)
        vh = v / (1 - 0.999 ** step)
        lr = 0.12 * (1 - step / n_iter) ** 2 + 1e-3   # decaying step size
        theta = theta - lr * mh / (np.sqrt(vh) + 1e-8)
    assert losses[-1] < 1e-3 * losses[0]
    assert np.allclose(theta, theta_star, atol=0.1), \
        f"recovered {theta} vs true {theta_star}"


def test_vjp_translation_null_direction():
    """Translating every catalogue leaves all separations (and hence any
    histogram loss) invariant: the adjoint gradient along that direction
    must vanish."""
    rng = np.random.default_rng(23)
    g, _ = _displacement_catalogs(rng)
    w = rng.uniform(0.5, 1.5, len(g))
    ns = len(S_EDGES) - 1
    gg = rng.normal(size=(ns,))
    import pyfcfc.boxes as boxes
    theta = np.array([4.0, -3.0, 2.0])
    out = boxes.soft_hist_vjp([g + theta], [w], S_EDGES, None, 1, pair='AA',
                              cotangent=gg, soft_sigma=SIGMA, bin=0,
                              box=400.)
    gx = out['grads']['A']
    # dL/dtheta = sum_i dL/dx_i for a rigid translation of all points
    dtheta = gx.sum(axis=0)
    scale = np.abs(gx).sum()
    assert np.allclose(dtheta, 0.0, atol=1e-9 * scale), \
        f"null direction violated: {dtheta} (scale {scale})"
