"""
Wideband dielectric model for FDTD: the Djordjevic-Sarkar (wideband Debye) permittivity that matches
a laminate's er / tan d at one frequency, fitted by a few Debye poles plus a conductivity, which
openEMS can run directly (CSPropDebyeMaterial). Without it, the loss is a conductivity fitted at the
band centre: tan d then falls as 1/f, too lossy below the centre and too little above.

    eps(f) = eps_inf + d_eps / (m2 - m1) * log10((w2 + j w) / (w1 + j w)),   w1,2 = 2 pi 10^(m1,2)

with e^(+j w t) (eps = eps' - j eps''), m1 = 3 (1 kHz) and m2 = 12 (1 THz).
"""
import numpy as np
from kicad2ems.api.solver import EPS0

M1, M2 = 3.0, 12.0


def _k(f):
    w = 2 * np.pi * np.asarray(f, float)
    return np.log10((2 * np.pi * 10 ** M2 + 1j * w) / (2 * np.pi * 10 ** M1 + 1j * w)) / (M2 - M1)


def djordjevic_sarkar(er, tand, fref, f):
    """Complex relative permittivity (eps' - j eps'') at frequencies f (Hz), equal to er (1 - j tand)
    at fref."""
    kr = _k(fref)
    d_eps = -er * tand / kr.imag
    eps_inf = er - d_eps * kr.real
    return eps_inf + d_eps * _k(f)


def _nnls(A, b, iters=200):
    """Non-negative least squares, min |A x - b| with x >= 0 (Lawson-Hanson active set; small problems)."""
    n = A.shape[1]
    x, P = np.zeros(n), np.zeros(n, bool)
    for _ in range(iters):
        w = A.T @ (b - A @ x)
        if P.all() or w[~P].max(initial=-np.inf) <= 1e-12 * max(1.0, np.abs(w).max()):
            break
        P[np.argmax(np.where(P, -np.inf, w))] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            q = P & (z <= 0)
            alpha = np.min(x[q] / (x[q] - z[q]))
            x = x + alpha * (z - x)
            P &= x > 1e-15
    return x


def fit_debye(er, tand, fref, fmin, fmax, npoles=None, tau_min=0.0):
    """Debye poles + conductivity reproducing the Djordjevic-Sarkar model over [fmin, fmax] (Hz):
        eps(f) = eps_inf + sum d_eps_i / (1 + j w tau_i) - j kappa / (w eps0)
    Pole frequencies are log-spaced over [fmin / 2, 2 fmax] (1.5 per decade + 1, at least 3); poles with
    tau < tau_min (openEMS skips relaxation times below 2 time steps) are left out. Returns
    (eps_inf, [(d_eps, tau), ...], kappa S/m) with only non-zero poles."""
    fa, fb = max(fmin, fmax / 1e3) / 2, 2 * fmax
    if npoles is None:
        npoles = max(3, int(np.ceil(1.5 * np.log10(fb / fa))) + 1)     # each pole costs FDTD time: tand < 1 %
    fp = np.logspace(np.log10(fa), np.log10(fb), npoles)
    tau = 1 / (2 * np.pi * fp)
    tau = tau[tau >= tau_min]
    f = np.logspace(np.log10(max(fmin, fa)), np.log10(fmax), 60)
    w = 2 * np.pi * f
    target = djordjevic_sarkar(er, tand, fref, f)
    # columns: eps_inf, poles, kappa; rows: eps' / er and eps'' / (er tand) (loss weighted like the real part)
    cols = [np.ones_like(w, complex)] + [1 / (1 + 1j * w * t) for t in tau] + [-1j / (w * EPS0)]
    C = np.array(cols).T
    sr, si = 1 / er, 1 / max(er * tand, 1e-9)
    A = np.vstack([C.real * sr, -C.imag * si])
    b = np.concatenate([target.real * sr, -target.imag * si])
    scale = np.abs(A).max(0)                                        # kappa's column is ~1e9 larger
    x = _nnls(A / scale, b) / scale
    poles = [(float(d), float(t)) for d, t in zip(x[1:-1], tau) if d > 1e-9 * er]
    return float(x[0]), poles, float(x[-1])


def describe(layers, fref, fmin, fmax):
    """Lines: er / tan d of each dielectric at fmin and fmax under the wideband model."""
    out = [f"wideband dielectric (Djordjevic-Sarkar, matched at {fref / 1e9:g} GHz):"]
    for l in layers:
        e = djordjevic_sarkar(l['er'], l['tand'], fref, np.array([max(fmin, fmax / 1e3), fmax]))
        td = -e.imag / e.real
        out.append(f"  {l['name']:<14s} er {e[0].real:.3f} -> {e[1].real:.3f}  tand {td[0]:.4f} -> {td[1]:.4f}  "
                   f"({max(fmin, fmax / 1e3) / 1e9:g} -> {fmax / 1e9:g} GHz)")
    return out


def debye_eps(eps_inf, poles, kappa, f):
    """Complex relative permittivity of a fit_debye() result at f (Hz)."""
    w = 2 * np.pi * np.asarray(f, float)
    return eps_inf + sum(d / (1 + 1j * w * t) for d, t in poles) - 1j * kappa / (w * EPS0)
