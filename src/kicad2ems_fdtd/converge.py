"""
Mesh convergence check: the same model on several mesh levels. Level f uses res x f cells per
wavelength, fine / f at the copper edges (the fine cell of level 1 as the base) and ceil(nz x f) cells
through each dielectric; everything else unchanged. The finest level is the reference; each coarser
level is compared with it.
"""
import numpy as np
from .mesh import auto_mesh


def mesh_levels(m, fmin, fmax, factors, res=20, fine=None, nz_min=4, log=print, **mesh_kw):
    """[(factor, mesh), ...] coarse to fine."""
    quiet = lambda *a: None
    fine0 = fine if fine else auto_mesh(m, fmin, fmax, res=res, fine=None, nz_min=nz_min, log=quiet,
                                        **mesh_kw)['info']['fine_mm']
    out = []
    for f in factors:
        mesh = auto_mesh(m, fmin, fmax, res=res * f, fine=fine0 / f, nz_min=int(np.ceil(nz_min * f - 1e-9)),
                         log=log if f == factors[-1] else quiet, **mesh_kw)
        out.append((f, mesh))
    return out


def describe_levels(levels):
    """Cells, time step and relative run time (cells / time step, level 1 = first) per level."""
    lines = ["mesh convergence levels:"]
    work0 = None
    for f, mesh in levels:
        inf = mesh['info']
        work = inf['cells'] / inf['dt_ps']
        work0 = work0 or work
        lines.append(f"  x{f:<5g} fine {inf['fine_mm']:.4f} mm  {inf['nx']} x {inf['ny']} x {inf['nz']} = "
                     f"{inf['cells'] / 1e6:6.2f} M cells  dt {inf['dt_ps']:.3f} ps  ~{inf['mem_gb']:.1f} GB  "
                     f"run time x{work / work0:.1f}")
    return lines


def _pairs(n):
    return [(i, j) for j in range(n) for i in range(j, n)]


def compare_levels(S_levels, names, floor_db=-40.0):
    """Each level against the last (finest). S_levels: list of (nf, n, n) arrays. Returns one dict per
    coarser level: {'params': {'S(a,b)': (max |d dB|, max |d phase| deg)}, 'max_db', 'max_deg'}.
    Points where the reference is below floor_db are left out (noise floor, undefined phase)."""
    ref = S_levels[-1]
    out = []
    for S in S_levels[:-1]:
        params = {}
        for i, j in _pairs(ref.shape[1]):
            r, s = ref[:, i, j], S[:, i, j]
            ok = np.isfinite(r) & np.isfinite(s) & (20 * np.log10(np.abs(r) + 1e-15) > floor_db)
            if not ok.any():
                continue
            ddb = np.abs(20 * np.log10(np.abs(s[ok]) + 1e-15) - 20 * np.log10(np.abs(r[ok]) + 1e-15))
            dph = np.abs(np.degrees(np.angle(s[ok] / r[ok])))
            params[f"S({names[i]},{names[j]})"] = (float(ddb.max()), float(dph.max()))
        out.append(dict(params=params, max_db=max((v[0] for v in params.values()), default=0.0),
                        max_deg=max((v[1] for v in params.values()), default=0.0)))
    return out


def verdict(factors, cmp, tol_db=0.1, tol_deg=2.0):
    """Text table and a verdict: converged when the second-finest level is within tolerance."""
    lines = [f"convergence (vs the finest level x{factors[-1]:g}, |S| > -40 dB):"]
    for f, c in zip(factors, cmp):
        worst = max(c['params'].items(), key=lambda kv: kv[1][0], default=(None, (0, 0)))
        lines.append(f"  x{f:<5g} max {c['max_db']:.3f} dB / {c['max_deg']:.2f} deg" +
                     (f"  (worst {worst[0]})" if worst[0] else ''))
    last = cmp[-1]
    ok = last['max_db'] <= tol_db and last['max_deg'] <= tol_deg
    lines.append(f"  -> {'converged' if ok else 'NOT converged'}: x{factors[-2]:g} is within "
                 f"{last['max_db']:.3f} dB / {last['max_deg']:.2f} deg of x{factors[-1]:g} "
                 f"(tolerance {tol_db:g} dB / {tol_deg:g} deg)" +
                 ("" if ok else " - refine further (higher factors) before trusting the result"))
    return lines, ok
