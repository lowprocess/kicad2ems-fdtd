"""
The FDTD model: the board and its mesh as CSX geometry (dielectrics, copper sheets, via cylinders, parts),
lumped ports, the Gaussian excitation and the boundaries, built through openEMS's Python interface
(ContinuousStructure, openEMS) of whichever engine runs it (engines.py).
"""
import math
import numpy as np
from kicad2ems.api.solver import EPS0
from kicad2ems.api.solver import split_holes


def _f(v):
    """Plain Python float: older CSXCAD / openEMS releases reject numpy scalars (type checks)."""
    return float(v)



def _fl(v):
    """List of plain floats (coordinates)."""
    return [float(x) for x in v]



def build_fdtd(m, mesh, fmin, fmax, excite, bc=('MUR',) * 6, end_db=-40, sigma=None, max_steps=1_000_000,
               dumps=(), diel='fixed', diel_fref=1e9, exc_fmax=None, api=None):
    """diel: 'fixed' (loss as a conductivity fitted at the band centre) or 'wideband' (Djordjevic-Sarkar
    model matched at diel_fref Hz, as Debye poles; see dielectric.py). exc_fmax: upper edge of the
    Gaussian excitation (default fmax; antenna runs extend it, see antenna.py). api: (ContinuousStructure,
    openEMS) classes to build with (engines.engine_api; None: the default engine; readback.py records the
    calls)."""
    if api is None:
        from .engines import engine_api
        api = engine_api(None)
    ContinuousStructure, openEMS = api
    FDTD = openEMS(NrTS=int(max_steps), EndCriteria=_f(10 ** (end_db / 10)))
    fe = fmax if exc_fmax is None else exc_fmax
    f0, fc = _f(0.5 * (fmin + fe)), _f(0.5 * (fe - fmin) * 1.05)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond([str(b) for b in bc])
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)
    grid.SetLines('x', np.asarray(mesh['x'], float)); grid.SetLines('y', np.asarray(mesh['y'], float))
    grid.SetLines('z', np.asarray(mesh['z'], float))
    lines = [np.asarray(mesh[a], float) for a in 'xyz']

    def on_mesh(s, e, tol=1e-5):
        """A box's flat sides (start = stop on an axis) moved exactly onto the nearest mesh line: openEMS
        drops a zero-thickness port or element that misses its line. Two differential legs whose port planes
        differed by 1e-12 mm got one merged line, and the second leg's excitation found no edges."""
        s, e = list(s), list(e)
        for a in range(3):
            if abs(s[a] - e[a]) < 1e-9 and len(lines[a]):
                near = lines[a][np.argmin(np.abs(lines[a] - s[a]))]
                if abs(near - s[a]) < tol:
                    s[a] = e[a] = float(near)
        return s, e
    ox, oy = m.outline.exterior.xy
    pts = [np.asarray(ox[:-1], float), np.asarray(oy[:-1], float)]
    for l in m.layers:
        name = l['name'].replace(' ', '_')
        if diel == 'wideband' and l['tand'] > 0:
            mat = _debye_material(CSX, name, l, diel_fref, fmin, fmax, mesh['info']['dt_ps'] * 1e-12)
        else:                                                       # loss at the centre frequency
            kappa = _f(2 * np.pi * f0 * EPS0 * l['er'] * l['tand'])
            mat = CSX.AddMaterial(name, epsilon=_f(l['er']), kappa=kappa)
        mat.AddLinPoly(pts, 'z', _f(l['z0']), _f(l['z1'] - l['z0']), priority=1)
    for lay, ps in m.copper.items():
        if sigma:
            metal = CSX.AddConductingSheet(f"cu_{lay}", conductivity=_f(sigma), thickness=_f(m.cu_t[lay] * 1e-3))
        else:
            metal = CSX.AddMetal(f"cu_{lay}")
        for p in ps:
            for q in split_holes(p):
                x, y = q.exterior.xy
                metal.AddPolygon([np.asarray(x[:-1], float), np.asarray(y[:-1], float)], 'z', _f(m.cu_z[lay]),
                                 priority=10)
    vias = mesh.get('vias', m.vias)                                 # as placed by the mesher (snapped)
    if vias:
        vm = CSX.AddMetal('vias')
        for (x, y, r, z0, z1) in vias:
            vm.AddCylinder(_fl([x, y, z0]), _fl([x, y, z1]), _f(r), priority=11)
    walls = mesh.get('walls') or []                                # --edge-walls: PEC sheets where the planes end
    if walls:
        wm = CSX.AddMetal('edge_walls')
        for axis, c, t0, t1, z0, z1 in walls:
            s, e = ([c, t0, z0], [c, t1, z1]) if axis == 'x' else ([t0, c, z0], [t1, c, z1])
            wm.AddBox(_fl(s), _fl(e), priority=11)
    for pt in getattr(m, 'parts', []):                              # lumped R / L / C between two pads
        s, e = pt.box()
        if pt.short:
            CSX.AddMetal(f"part_{pt.ref}").AddBox(_fl(s), _fl(e), priority=12)
            continue
        for suffix, vals, series, ps, pe in pt.pieces():
            ps, pe = on_mesh(ps, pe)
            kw = {k: _f(v) for k, v in vals.items()}
            if series:
                kw['LEtype'] = 1
            try:
                el = CSX.AddLumpedElement(f"part_{pt.ref}{suffix}", ny=int(pt.axis), caps=True, **kw)
            except Exception as ex:                                 # older CSXCAD: no L / LEtype
                raise RuntimeError(f"{pt.ref}: this CSXCAD cannot make a lumped element with {sorted(kw)} ({ex}) - "
                                   f"update openEMS, or skip the part (--parts none / Sim.Enable = 0)") from ex
            el.AddBox(_fl(ps), _fl(pe), priority=12)
    ports = []
    for i, p in enumerate(m.ports):
        s, e = on_mesh(*p.box())
        ports.append(FDTD.AddLumpedPort(i + 1, _f(p.z0), _fl(s), _fl(e), getattr(p, 'direction', 'z'),
                                        excite=1.0 if i == excite else 0.0, priority=12))
    if dumps:
        from kicad2ems.api.solver import add_dumps
        add_dumps(CSX, dumps)
    return FDTD, CSX, ports



def _debye_material(CSX, name, layer, fref, fmin, fmax, dt):
    """Debye-pole material for one dielectric (openEMS skips relaxation times below 2 time steps, so
    the fit only uses poles above 2.5 dt). Every engine's structure has AddDebyeMaterial (openems/engine.py adds it to
    a CSXCAD without it)."""
    from .dielectric import fit_debye
    e_inf, poles, kappa = fit_debye(layer['er'], layer['tand'], fref, fmin, fmax, tau_min=2.5 * dt)
    if not poles:
        return CSX.AddMaterial(name, epsilon=_f(e_inf), kappa=_f(kappa))
    mat = CSX.AddDebyeMaterial(name, order=len(poles), epsilon=_f(e_inf), kappa=_f(kappa))
    for i, (d, tau) in enumerate(poles):
        mat.SetDispersiveMaterialProperty(order=i, eps_delta=_f(d), eps_relax=_f(tau))
    return mat



def port_index(m, names):
    """Indices of single-ended ports to excite: names may be legs (P1+) or whole ports (P1)."""
    if not names:
        return list(range(len(m.ports)))
    want = [s.strip() for s in (names.split(',') if isinstance(names, str) else names)]
    idx = [i for i, p in enumerate(m.ports) if p.name in want or p.pair in want]
    if not idx:
        raise ValueError(f"no port matches {want}; ports: {[p.name for p in m.ports]}")
    return idx



def pulse_steps(fmin, fmax, dt_ps, exc_fmax=None):
    """Length of openEMS's Gaussian excitation in timesteps (2 * 9 / (2 pi fc) / dt, fc as set in build_fdtd),
    with the mesher's time step estimate (openEMS's own step is often ~10 % smaller, so the true count is a
    little higher). The fields must ring down after it: allow --max-steps well above this."""
    fe = fmax if exc_fmax is None else exc_fmax
    fc = 0.5 * (fe - fmin) * 1.05
    return int(np.ceil(2 * 9 / (2 * np.pi * fc) / (dt_ps * 1e-12)))



def pulse_time(fmin, fmax):
    """Length (s) of openEMS's Gaussian excitation for the band (as set in build_fdtd)."""
    fc = 0.5 * (fmax - fmin) * 1.05
    return 2 * 9 / (2 * math.pi * fc)
