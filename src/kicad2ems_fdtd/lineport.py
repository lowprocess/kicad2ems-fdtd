"""
Line ports for the FDTD model (R-E1, `--port-kind line | auto`): voltage and current probes on a port's straight
feed, behind its lumped port, which stays the source and the termination. The probes are openEMS's (and gfdtd's)
microstrip-port arrangement, not excited: three voltage lines from the reference copper to the strip and two
current loops round it, near a measurement plane `d` along the feed, which give the line's waves there and the
line's own Zc and gamma.

S comes from the waves of every run: a = (V + z0 I) / 2 and b = (V - z0 I) / 2 at each probed port's plane (the
lumped ports' own a and b where a port has no probes), S = B A^-1 in z0 (50 ohm) at those planes, exact whatever
the ports' terminations reflect. Each probed port's reference is then moved back to the lumped port's plane along
the FDTD's own line (its Zc and gamma read at the plane in its own run, fitted over the band): the result is the
board from the same planes as the lumped S, without the lumped port's feed (its inductance and the trace behind
it). The same runs give the lumped S as well.

A port takes probes where its trace runs straight at one width from the port to the probes' last plane and
END_CLEAR more (the next change of the line, its near field), the measurement plane d = max(3 h, 2 w, four mesh cells,
D_MIN) along it, with no other signal copper within 3 h of its edges there; vertical ports of single-ended lines on sheet
copper only. Every port must be driven (B A^-1 needs each port's run).

    plan, notes = plan_ports(m, mesh, 'auto')        # {port index: dict(d, layer, ...)}, why the others are not
    S = solve(m, freq, res, plan)                    # res: run_excitations' RunResult with res.lines
"""
import numpy as np

KINDS = ('lumped', 'line', 'auto')
NEIGHBOUR = 3.0             # x the line's height: other signal copper nearer the trace's edge than this: no probes
END_CLEAR = (2.0, 1.0)      # the probes' last plane this far (x h, x w: the larger) from where the feed changes
D_MIN = 3.0                 # mm: the plane at least this far from the port. Nearer, the port's own field is in what
                            # the probes read: rf_stripline's eps_eff read 4.87 at 0.6 mm (3 h), 4.52 at 1.5, 4.42 at
                            # 3 and 4.400 at 4.5 (the stack's 4.40), its S 0.067, 0.030 and 0.011 from 4.5 mm's (a
                            # port to one of a stripline's two planes also drives the wave between them); a microstrip
                            # read the same at 2.9 and 5.7 mm
WIDTH_TOL = 0.02            # the feed's width may change this much (relative) along the window
FIT_DROP = 0.15             # the lowest share of the band left out of the line's fit (its Zc reads poorly there)
C0 = 299792458.0


def _union(m, layer):
    from shapely.ops import unary_union
    return unary_union(m.copper.get(layer, []))


def _width(g, pt, n, half):
    """The width across n of the copper piece of g through pt (0 where pt is not on copper)."""
    from shapely.geometry import LineString, Point
    seg = LineString([tuple(pt - n * half), tuple(pt + n * half)])
    cut = g.intersection(seg)
    pieces = getattr(cut, 'geoms', [cut])
    here = Point(*pt)
    return next((q.length for q in pieces if not q.is_empty and q.distance(here) < 1e-9), 0.0)


def feed_length(m, p, upto):
    """How far (mm) the port's trace runs straight from the port along its direction at its width (within
    WIDTH_TOL), looked at up to `upto` mm."""
    g = _union(m, p.layer)
    d = np.zeros(2)
    d[p.axis] = p.sgn
    n = np.array([-d[1], d[0]])
    c = np.asarray(p.c, float)[:2]
    step = float(np.clip(p.w / 10, 0.005, 0.05))
    half = 2.0 * p.w + 1.0
    w0 = _width(g, c + d * step / 2, n, half)
    if w0 <= 0:
        return 0.0
    s = step / 2
    while s < upto:
        if abs(_width(g, c + d * (s + step), n, half) - w0) > WIDTH_TOL * w0:
            break
        s += step
    return s + step / 2


def _neighbours(m, p, d, s0, s1):
    """Whether other signal copper (not the port's net, not a ground net) runs within NEIGHBOUR heights of the
    feed's edges between s0 and s1 (None where the board kept no nets: a board.json)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    items = getattr(getattr(m, 'prims', None), 'items', None)
    if not items:
        return None
    grounds = set(getattr(m, 'ground_nets', None) or ())
    other = [g for lay, net, g in items if lay == p.layer and net and net != p.net and net not in grounds]
    if not other:
        return False
    c = np.asarray(p.c, float)[:2]
    seg = LineString([tuple(c + d * s0), tuple(c + d * s1)])
    reach = p.w / 2 + NEIGHBOUR * abs(p.z_sig - p.z_ref)
    return unary_union(other).distance(seg) < reach


def window(m, p, mesh):
    """(dict(d, clear, feed), None) where port p can have probes, else (None, why)."""
    if p.kind != 'z':
        return None, 'a gap port'
    if getattr(p, 'internal', False):
        return None, "a part's port"
    if p.pair or p.leg:
        return None, "a differential pair's leg (its modes: not yet)"
    if mesh.get('thick'):
        return None, 'thick copper (--copper-thickness on)'
    if getattr(p, 'oblique', False):
        return None, 'its trace is at an angle to x and y (the probes stand on the mesh)'
    h, w = abs(p.z_sig - p.z_ref), float(p.w)
    lines = np.asarray(mesh['xy'[p.axis]], float)
    c = float(p.c[p.axis])
    ahead = np.sort(p.sgn * (lines - c))
    ahead = ahead[ahead > 1e-9]
    if len(ahead) < 8:
        return None, 'fewer than 8 mesh lines along its feed'
    d = max(3 * h, 2 * w, ahead[3], D_MIN)
    b = int(np.argmin(np.abs(ahead - d)))                           # the planes: the line nearest d and either side
    if b + 1 >= len(ahead):
        return None, 'too few mesh lines along its feed'
    stop = float(ahead[b + 1])
    need = stop + max(END_CLEAR[0] * h, END_CLEAR[1] * w)
    feed = feed_length(m, p, need + ahead[3] / 4)
    if feed < need:
        return None, f"its straight feed is {feed:.2f} mm, probes need {need:.2f} mm"
    dvec = np.zeros(2)
    dvec[p.axis] = p.sgn
    if _neighbours(m, p, dvec, 0.0, stop):
        return None, f"other signal copper within {NEIGHBOUR:g} heights of its feed"
    return dict(d=float(d), stop=stop, feed=float(feed)), None


def plan_ports(m, mesh, kind):
    """{port index: probe plan} and notes [str] for --port-kind kind: 'lumped' none; 'auto' every port that can
    have probes, a note for the others; 'line' every single-ended vertical port, an error where one cannot (a
    part's internal ports stay lumped)."""
    if kind not in KINDS:
        raise ValueError(f"--port-kind: {kind!r}, not one of {', '.join(KINDS)}")
    if kind == 'lumped':
        return {}, []
    plan, notes, bad = {}, [], []
    for i, p in enumerate(m.ports):
        win, why = window(m, p, mesh)
        if win is not None:
            plan[i] = dict(win, layer=p.layer, c=[float(v) for v in p.c], axis=int(p.axis), sgn=int(p.sgn),
                           w=float(p.w), z_sig=float(p.z_sig), z_ref=float(p.z_ref))
        elif not getattr(p, 'internal', False):
            (bad if kind == 'line' else notes).append(f"{p.name}: lumped ({why})")
    if bad:
        raise ValueError("--port-kind line: " + '; '.join(bad) + " (--port-kind auto keeps those lumped)")
    return plan, notes


def probe_box(pr):
    """The probes' box (start, stop): from the port plane to their last plane along the feed, across the trace,
    from the strip (start's z) to the reference copper (stop's)."""
    a, b = pr['axis'], 1 - pr['axis']
    s, e = np.zeros(3), np.zeros(3)
    s[a], e[a] = pr['c'][a], pr['c'][a] + pr['sgn'] * pr['stop']
    s[b], e[b] = pr['c'][b] - pr['w'] / 2, pr['c'][b] + pr['w'] / 2
    s[2], e[2] = pr['z_sig'], pr['z_ref']
    return s, e


def add(FDTD, cu_props, plan):
    """The probes of every planned port, on its layer's own copper property (cu_props {layer: property}): the
    port objects {index: port} (numbers past the lumped ports', names 'line_port_...')."""
    out = {}
    for i, pr in sorted(plan.items()):
        s, e = probe_box(pr)
        out[i] = FDTD.AddMSLPort(1000 + i + 1, cu_props[pr['layer']], [float(v) for v in s], [float(v) for v in e],
                                 'xy'[pr['axis']], 'z', excite=0, MeasPlaneShift=float(pr['d']),
                                 PortNamePrefix='line_', priority=10)
    return out


def read(probes, path, freq):
    """What each probe read in a run: {index: dict(u, i, z, beta, d)} (V and I at its plane, the line's Zc and
    beta, the plane's distance from the port in mm)."""
    out = {}
    for i, q in probes.items():
        q.ReadUIData(path, np.asarray(freq, float))
        out[i] = dict(u=np.asarray(q.uf_tot, complex), i=np.asarray(q.if_tot, complex),
                      z=np.asarray(q.Z_ref, complex), beta=np.asarray(q.beta, complex), d=float(q.measplane_shift))
    return out


def waves(m, res, plan, z0=None):
    """A, B (nf, n, n): a and b of port i (row) in the run that drives port k (column)."""
    names = [p.name for p in m.ports]
    n = len(names)
    nf = next(iter(res.lines.values()))['lumped'][0].shape[1]
    A, B = np.zeros((nf, n, n), complex), np.zeros((nf, n, n), complex)
    for k, name in enumerate(names):
        got = res.lines[name]
        ainc, bref = got['lumped']
        for i in range(n):
            if i in plan:
                zr = (z0 or m.ports[i].z0)
                u, cur = got['probes'][i]['u'], got['probes'][i]['i']
                A[:, i, k], B[:, i, k] = 0.5 * (u + zr * cur), 0.5 * (u - zr * cur)
            else:
                A[:, i, k], B[:, i, k] = ainc[i], bref[i]
    return A, B


def fit_line(freq, z, beta):
    """The line's Zc and beta over the band, smoothed: Zc linear and the effective index beta / k0 quadratic in f,
    fitted above the band's lowest FIT_DROP (where the probes read poorly)."""
    f = np.asarray(freq, float)
    keep = f >= f[0] + FIT_DROP * (f[-1] - f[0])
    if keep.sum() < 4:
        keep = np.ones_like(f, bool)
    k0 = 2 * np.pi * f / C0
    pz = np.polyfit(f[keep], np.asarray(z)[keep], 1)
    pn = np.polyfit(f[keep], (np.asarray(beta) / k0)[keep], 2)
    return np.polyval(pz, f), np.polyval(pn, f) * k0


def _line_s(zc, beta, length_m, z0):
    """S (nf, 2, 2) in z0 of a line of Zc and beta (arrays over f; gamma = j beta), length_m long."""
    gl = 1j * np.asarray(beta) * length_m
    ch, sh = np.cosh(gl), np.sinh(gl)
    a, b, c, d = ch, zc * sh, sh / zc, ch
    den = a + b / z0 + c * z0 + d
    return np.stack([np.stack([(a + b / z0 - c * z0 - d) / den, 2 * (a * d - b * c) / den], -1),
                     np.stack([2 / den, (-a + b / z0 - c * z0 + d) / den], -1)], -2)


def solve(m, freq, res, plan):
    """S (nf, n, n) over the single-ended ports, each port's z0, from the probes and the lumped ports of every run
    (B A^-1), each probed port's reference moved back to its lumped port's plane; and the lines read
    {port name: (Zc at mid-band, eps_eff at mid-band, plane mm)} for the log."""
    from kicad2ems.deembed import connect
    names = [p.name for p in m.ports]
    missing = [nm for nm in names if nm not in res.lines]
    if missing:
        raise ValueError(f"line ports need every port driven: {', '.join(missing)} not excited (--excite)")
    z0s = {m.ports[i].z0 for i in range(len(names))}
    if len(z0s) != 1:
        raise ValueError("line ports: the ports' impedances differ (one reference throughout)")
    z0 = float(next(iter(z0s)))
    A, B = waves(m, res, plan, z0)
    S = B @ np.linalg.inv(A)
    lines = {}
    for i in sorted(plan):
        own = res.lines[names[i]]['probes'][i]
        zc, beta = fit_line(freq, own['z'], own['beta'])
        S = connect(S, i, _line_s(zc, beta, own['d'] * 1e-3, z0))
        mid = len(freq) // 2
        lines[names[i]] = (float(np.real(zc[mid])), float(np.real(beta[mid] * C0 / (2 * np.pi * freq[mid])) ** 2),
                           own['d'])
    return S, lines
