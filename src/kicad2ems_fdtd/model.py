"""
The FDTD model: the board and its mesh as CSX geometry (dielectrics, copper sheets, via cylinders, parts),
lumped ports, the Gaussian excitation and the boundaries, built through openEMS's Python interface
(ContinuousStructure, openEMS) of whichever engine runs it (engines.py). With the mesh's copper thickness plan
(--thickness on, thick.py) the copper is solid and the stack moved to make room for it.
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
               dumps=(), diel='fixed', diel_fref=1e9, exc_fmax=None, api=None, rough=None, copper_kw=None):
    """sigma: copper conductivity (conducting sheets; None: PEC). rough: um RMS, the sheets' own roughness (an
    engine with it: gfdtd 1.2; None: smooth). diel: 'fixed' (loss as a conductivity fitted at the band centre) or 'wideband' (Djordjevic-Sarkar
    model matched at diel_fref Hz, as Debye poles; see dielectric.py). exc_fmax: upper edge of the
    Gaussian excitation (default fmax; antenna runs extend it, see antenna.py). api: (ContinuousStructure,
    openEMS) classes to build with (engines.engine_api; None: the default engine; readback.py records the
    calls). copper_kw: dict(faces=, finish=) from the plugin's copper_kw(): faces marks thick copper's sheets as
    faces (gfdtd 1.3); finish puts the surface finish on the outer layers - with thick copper on their outer face's
    own sheet, else on their sheets weighted by the strip's share; as the sheets' own (finish['wideband'], gfdtd
    1.3) or as the conductivity of the same loss at the band centre."""
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
    th = mesh.get('thick')                                          # copper thickness (thick.py), None: sheets
    if th:
        from . import thick
        zm = lambda z: thick.z_map(m, th, z)
    else:
        zm = lambda z: z
    for l in m.layers:
        name = l['name'].replace(' ', '_')
        if diel == 'wideband' and l['tand'] > 0:
            mat = _debye_material(CSX, name, l, diel_fref, fmin, fmax, mesh['info']['dt_ps'] * 1e-12)
        else:                                                       # loss at the centre frequency
            kappa = _f(2 * np.pi * f0 * EPS0 * l['er'] * l['tand'])
            mat = CSX.AddMaterial(name, epsilon=_f(l['er']), kappa=kappa)
        z0, z1 = zm(l['z0']), zm(l['z1'])
        mat.AddLinPoly(pts, 'z', _f(z0), _f(z1 - z0), priority=1)
    ckw = copper_kw or {}
    fin = ckw.get('finish') if sigma else None
    for lay, ps in m.copper.items():
        t = m.cu_t.get(lay, 0.0)
        slab = thick.faces(m, th, lay) if th and t > 0 else None
        body = outer = None
        rkw = dict(roughness=_f(rough * 1e-6)) if rough else {}
        share = fin['share'].get(lay, 0.0) if fin else 0.0
        if slab:                                                    # solid copper: PEC; with sigma its faces lossy
            body = CSX.AddMetal(f"cu_{lay}_body") if sigma else CSX.AddMetal(f"cu_{lay}")
            if sigma:                                               # one side of a conductor t / 2 thick each
                fkw = dict(face=True) if ckw.get('faces') else {}
                metal = CSX.AddConductingSheet(f"cu_{lay}", conductivity=_f(sigma / 4), thickness=_f(2 * t * 1e-3),
                                               **rkw, **fkw)
                if fin and lay in fin['share']:                     # the outer face: finished, a sheet of its own
                    z_out = slab[1] if lay == m.cu_order[-1] else slab[0]
                    outer = (z_out, _finished_sheet(CSX, f"cu_{lay}_finish", sigma, t, fin, 1.0, rkw, face=True))
        elif sigma:
            metal = (_finished_sheet(CSX, f"cu_{lay}", sigma, t, fin, share, rkw, face=False) if share > 0 else
                     CSX.AddConductingSheet(f"cu_{lay}", conductivity=_f(sigma), thickness=_f(t * 1e-3), **rkw))
        else:
            metal = CSX.AddMetal(f"cu_{lay}")
        for p in ps:
            for q in split_holes(p):
                x, y = q.exterior.xy
                xy = [np.asarray(x[:-1], float), np.asarray(y[:-1], float)]
                if not slab:
                    metal.AddPolygon(xy, 'z', _f(zm(m.cu_z[lay])), priority=10)
                    continue
                body.AddLinPoly(xy, 'z', _f(slab[0]), _f(slab[1] - slab[0]), priority=9 if sigma else 10)
                if sigma:
                    for zf in slab:
                        sheet = outer[1] if outer and abs(zf - outer[0]) < 1e-12 else metal
                        sheet.AddPolygon(xy, 'z', _f(zf), priority=10)
    over = (lambda z0, z1: thick.span(m, th, z0, z1)) if th else (lambda z0, z1: tuple(sorted((z0, z1))))
    across = (lambda z: thick.slab(m, th, z)) if th else (lambda z: (z, z))
    vias = mesh.get('vias', m.vias)                                 # as placed by the mesher (snapped)
    if vias:
        vm = CSX.AddMetal('vias')
        for (x, y, r, z0, z1) in vias:
            z0, z1 = over(z0, z1)
            vm.AddCylinder(_fl([x, y, z0]), _fl([x, y, z1]), _f(r), priority=11)
    walls = mesh.get('walls') or []                                # --edge-walls: PEC sheets where the planes end
    if walls:
        wm = CSX.AddMetal('edge_walls')
        for axis, c, t0, t1, z0, z1 in walls:
            z0, z1 = over(z0, z1)
            s, e = ([c, t0, z0], [c, t1, z1]) if axis == 'x' else ([t0, c, z0], [t1, c, z1])
            wm.AddBox(_fl(s), _fl(e), priority=11)

    def flat(s, e):
        """A flat part / gap port on a copper level: across that layer's slab (thick copper)."""
        s, e = np.array(s, float), np.array(e, float)
        if th and abs(s[2] - e[2]) < 1e-9:
            s[2], e[2] = across(s[2])
        return s, e
    for pt in getattr(m, 'parts', []):                              # lumped R / L / C between two pads
        s, e = flat(*pt.box())
        if pt.short:
            CSX.AddMetal(f"part_{pt.ref}").AddBox(_fl(s), _fl(e), priority=12)
            continue
        for suffix, vals, series, ps, pe in pt.pieces():
            ps, pe = on_mesh(*flat(ps, pe))
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
        s, e = p.box()
        if th:
            s, e = np.array(s, float), np.array(e, float)
            if abs(s[2] - e[2]) < 1e-9:                             # a gap port: across the slab
                s, e = flat(s, e)
            else:                                                   # from face to face, toward each other
                s[2], e[2] = thick.toward(m, th, s[2], e[2]), thick.toward(m, th, e[2], s[2])
        s, e = on_mesh(s, e)
        ports.append(FDTD.AddLumpedPort(i + 1, _f(p.z0), _fl(s), _fl(e), getattr(p, 'direction', 'z'),
                                        excite=1.0 if i == excite else 0.0, priority=12))
    if dumps:
        from kicad2ems.api.solver import add_dumps
        add_dumps(CSX, dumps)
    return FDTD, CSX, ports



def _finished_sheet(CSX, name, sigma, t_mm, fin, share, rkw, face):
    """A conducting sheet of copper (sigma, t_mm) with the surface finish fin on `share` of its loss. face: the
    sheet is the outer face of thick copper (sigma / 4, twice the thickness: one face of the copper's half), all of
    it finished. As the sheet's own finish (fin['wideband']: gfdtd 1.3), else as the conductivity whose loss at the
    band centre is the finished sheet's (the face's impedance of half the copper, or the strip's weighted)."""
    s, th = (sigma / 4, 2 * t_mm * 1e-3) if face else (sigma, t_mm * 1e-3)
    if fin['wideband']:
        kw = dict(finish=[(_f(a), _f(b)) for a, b in fin['layers']], finish_share=_f(share), face=face)
        return CSX.AddConductingSheet(name, conductivity=_f(s), thickness=_f(th), **rkw, **kw)
    from kicad2ems.api.solver import finish_band_sigma
    lay = [(None, a, b) for a, b in fin['layers']]
    t_face = (t_mm / 2 if face else t_mm) * 1e-3
    eff = finish_band_sigma(sigma, lay, share, *fin['band'], t=t_face)
    return CSX.AddConductingSheet(name, conductivity=_f(eff / 4 if face else eff), thickness=_f(th), **rkw)


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
