"""
Automatic rectilinear mesh for openEMS.

  x / y : thirds rule at Manhattan copper edges (1/3 inside, 2/3 outside the metal), end points of
          oblique edges, via centre +- radius, port planes and the board outline kept exactly;
          lines closer than fine/2 merged by priority; graded smoothing (<= ratio growth) up to
          lambda/res in the dielectric and in air
  z     : every interface, >= nz cells per dielectric, copper sheets exactly on lines, air margin
          (none below a PEC bottom plane), PML cells added only on PML sides
"""
import numpy as np
from kicad2ems.api.solver import C0
from .boundaries import SIDES, bottom_is_plane


def _merge(lines, tol, keep=(), prio=None):
    """Cluster lines closer than tol and keep ONE line per cluster: lines in `keep` (exact), else the
    member with the highest priority (ties: the first). Clusters do not chain (every member is within
    tol of the cluster's first line) and lines are never averaged, so the thirds-rule line just inside
    a narrow trace survives instead of drifting outside it."""
    lines = np.asarray(lines, float)
    prio = np.zeros(len(lines)) if prio is None else np.asarray(prio, float)
    if len(keep):
        lines = np.concatenate([lines, keep]); prio = np.concatenate([prio, np.full(len(keep), 1e9)])
    o = np.argsort(lines, kind='stable')
    lines, prio = lines[o], prio[o]
    out, op, cl = [], [], [0]
    for i in range(1, len(lines) + 1):
        if i < len(lines) and lines[i] - lines[cl[0]] <= tol:
            cl.append(i); continue
        c = np.array(cl)
        j = c[np.argmax(prio[c])]
        x, pr = lines[j], prio[j]
        # the winners of two neighbouring clusters can still be closer than tol: keep the stronger one
        while out and x - out[-1] <= tol:
            if pr > op[-1]:
                out.pop(); op.pop()
            else:
                break
        if not out or x - out[-1] > tol:
            out.append(x); op.append(pr)
        cl = [i]
    return np.array(out)


def smooth_lines(fixed, maxres, ratio=1.4, fine=None):
    """Insert lines between fixed lines so that every cell is <= maxres(x) and neighbouring cells
    grow by at most `ratio` away from the fixed (feature) lines. maxres: float or callable(x)."""
    M = maxres if callable(maxres) else (lambda x, m=maxres: m)
    fx = np.unique(np.asarray(fixed, float))
    if len(fx) < 2:
        return fx
    gaps = np.diff(fx)
    out = [fx[0]]
    for i in range(len(fx) - 1):
        a, b = fx[i], fx[i + 1]
        m = M(0.5 * (a + b))
        ha = gaps[i - 1] if i > 0 else gaps[i]                      # neighbouring cells set the start size
        hb = gaps[i + 1] if i + 1 < len(gaps) else gaps[i]
        if fine:
            ha, hb = min(ha, fine * 3), min(hb, fine * 3)
        ha, hb = min(ha, m), min(hb, m)
        L = b - a
        if L <= m and L <= ratio * min(ha, hb) * 1.0001:
            out.append(b)
            continue
        left, right = [a], [b]                                 # graded from both ends towards the middle
        h, g = ha, hb
        while True:
            hn, gn = min(h * ratio, m), min(g * ratio, m)
            if right[-1] - left[-1] <= hn + gn:
                break
            if h <= g:
                left.append(left[-1] + hn); h = hn
            else:
                right.append(right[-1] - gn); g = gn
        if len(left) > 1:                                      # give the last graded cell of each side back
            left.pop()
        if len(right) > 1:
            right.pop()
        lo, hi = left[-1], right[-1]                           # ... and split [lo, hi] evenly: no sliver
        hl = left[-1] - left[-2] if len(left) > 1 else ha
        gr = right[-2] - right[-1] if len(right) > 1 else hb
        k = max(1, int(np.ceil((hi - lo) / min(m, ratio * min(hl, gr)) - 1e-9)))
        out += left[1:] + list(np.linspace(lo, hi, k + 1)[1:-1]) + right[1:][::-1] + [b]
    return np.array(out)


def manhattan(d, L, fine):
    """Edges (vectors d, lengths L) treated as Manhattan: at least `fine` long and drifting at most
    fine/10 across their axis. Near-Manhattan edges come from simplify() merging a tiny step (a 1.5 um
    jog where a track meets a slightly wider rectangle); they still get the thirds rule."""
    tol = fine / 10
    return ((np.abs(d[..., 0]) <= tol) | (np.abs(d[..., 1]) <= tol)) & (L >= fine)


def curve_lines(xy, fine, mode='extremes', step=None, boxes=None, inside=None):
    """Mesh-line candidates of the curved parts of a ring (rounded pad corners, round track ends,
    arcs, circular antipads, oblique edges). Edges that are Manhattan and at least `fine` long are
    handled by the thirds rule instead. mode 'extremes': for each run of consecutive curved edges only
    its two end points and its local x / y extremes (a round antipad gives 2 lines per axis instead of
    one per vertex), and with `step` also a line wherever the run has gone `step` mm in x or y since the
    last one: a thin curved track (the copper's width as the step) stays connected on the mesh. With the
    extremes alone a rat-race's 0.8 mm ring sat on cells up to 1.8 mm, broke apart, and S21 came out
    -35.8 dB instead of -3.4. 'vertices': every vertex (dense, closest to the drawn geometry).
    boxes: a list that gets (x0, y0, x1, y1) of each curved run longer than 4 fine and twice `step` (the
    copper's width): arcs, rings and fans, not the round end of a track or a pad's corner.
    inside: a test of (x, y): only the lines of points it passes (the ground copper within the signal zone)."""
    if inside is not None:
        allx, ally = _curve_points(xy, fine, mode, step, boxes)
        return [p[0] for p in allx if inside(*p)], [p[1] for p in ally if inside(*p)]
    allx, ally = _curve_points(xy, fine, mode, step, boxes)
    return [p[0] for p in allx], [p[1] for p in ally]


def _curve_points(xy, fine, mode='extremes', step=None, boxes=None):
    """curve_lines' candidates as the points they come from: ([(x, y)] for x lines, [(x, y)] for y lines)."""
    xy = np.asarray(xy)[:-1]                                        # closed ring -> unique vertices
    n = len(xy)
    if n < 3:
        return [], []
    d = np.roll(xy, -1, axis=0) - xy                                # edge i: vertex i -> i+1
    L = np.hypot(d[:, 0], d[:, 1])
    manh = manhattan(d, L, fine)
    if mode == 'vertices':
        if boxes is not None:
            curve_lines(np.vstack([xy, xy[:1]]), fine, 'extremes', step=step, boxes=boxes)
        idx = sorted({i for i in range(n) if not manh[i]} | {(i + 1) % n for i in range(n) if not manh[i]})
        return [tuple(xy[i]) for i in idx], [tuple(xy[i]) for i in idx]
    if not (~manh).any():
        return [], []
    xs, ys = [], []
    start = int(np.argmax(manh)) if manh.any() else 0               # walk runs starting after a Manhattan edge
    i, seen = (start + 1) % n if manh.any() else 0, 0
    while seen < n:
        if manh[i]:
            i, seen = (i + 1) % n, seen + 1
            continue
        run = [i]                                                   # consecutive curved edges
        j = (i + 1) % n
        while not manh[j] and len(run) < n:
            run.append(j); j = (j + 1) % n
        verts = [run[0]] + [(e + 1) % n for e in run]               # vertex indices along the run
        pts = xy[verts]
        ext = pts.max(0) - pts.min(0)
        if boxes is not None and ext.max() >= max(4 * fine, 2 * (step or 0)):    # not a round end or corner
            boxes.append((*pts.min(0), *pts.max(0)))
        if ext.max() >= fine / 2 and len(run) == n:                 # closed curve (circle): its extremes only
            xs += [tuple(pts[np.argmin(pts[:, 0])]), tuple(pts[np.argmax(pts[:, 0])])]
            ys += [tuple(pts[np.argmin(pts[:, 1])]), tuple(pts[np.argmax(pts[:, 1])])]
        elif ext.max() >= fine / 2:                                 # open run; tiny chamfers are ignored
            xs += [tuple(pts[0]), tuple(pts[-1])]; ys += [tuple(pts[0]), tuple(pts[-1])]
            lx, ly = pts[0]
            for k in range(1, len(pts) - 1):                        # local extremes inside the run
                if (pts[k, 0] - pts[k - 1, 0]) * (pts[k + 1, 0] - pts[k, 0]) < 0 or \
                        (step and abs(pts[k, 0] - lx) >= step):
                    xs.append(tuple(pts[k])); lx = pts[k, 0]
                if (pts[k, 1] - pts[k - 1, 1]) * (pts[k + 1, 1] - pts[k, 1]) < 0 or \
                        (step and abs(pts[k, 1] - ly) >= step):
                    ys.append(tuple(pts[k])); ly = pts[k, 1]
        seen += len(run)
        i = j
    return xs, ys


THIRDS_CELL = 2 / 3     # the cell across a copper edge, as a fraction of `fine` (see _edge_lines)
ZONE_DIST = 1.0         # zone: the ground copper's edges are resolved within this many mm of the signal copper,
ZONE_H = 3.0            # or this many times the thickest dielectric, whichever is more (signal_zone)
ZONE_SMALL = 2.0        # mm: a ground ring no larger than this (a pad, an antipad, a via) that touches the zone keeps
                        # all its lines (rf_diffvia: half of a return via's lines had moved its common mode by 0.14)


def signal_zone(m, dist=None):
    """Where the mesh resolves the ground copper's edges (model mm): within `dist` of the signal copper (copper of
    any net but the ground nets, on any layer) and of the signal vias. dist None: ZONE_DIST or ZONE_H x the
    thickest dielectric, whichever is more. None when the board has no signal copper apart from the ground."""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    gnd = set(getattr(m, 'ground_nets', ()) or ())
    nets = getattr(m, 'copper_nets', {})
    sig = [p for lay, ps in m.copper.items() for p, n in zip(ps, nets.get(lay, [''] * len(ps))) if n and n not in gnd]
    sig += [Point(v[0], v[1]).buffer(v[2]) for v, s_ in zip(m.vias, getattr(m, 'via_signal', [])) if s_]
    if not sig:
        return None
    if dist is None:
        h = max([l['z1'] - l['z0'] for l in m.layers if l['kind'] == 'diel'] + [0.0])
        dist = max(ZONE_DIST, ZONE_H * h)
    return unary_union(sig).buffer(dist, quad_segs=4)


def _edge_lines(m, fine, curves, boxes=None, zone=None):
    """Candidate x / y lines from the copper: the board outline, thirds-rule pairs at Manhattan edges
    (third: 2/3 outside the metal, third_in: 1/3 inside) and the curve lines (see curve_lines; boxes: the
    curved runs' boxes). zone (signal_zone): the ground copper's edges count only inside it (signal copper's
    always)."""
    from shapely.geometry import LineString, Point
    from shapely.prepared import prep
    zp = prep(zone) if zone is not None else None
    gnd = set(getattr(m, 'ground_nets', ()) or ())
    nets = getattr(m, 'copper_nets', {})
    ob = m.outline.bounds
    fixed = {0: [ob[0], ob[2]], 1: [ob[1], ob[3]]}
    third, third_in = {0: [], 1: []}, {0: [], 1: []}
    # The pair straddling an edge spans c = THIRDS_CELL * fine, not fine: with pairs a full `fine` apart,
    # edges closer than ~1.5 fine (a trace end, the via pad around it, the antipad) gave candidate lines
    # within the merge distance (fine / 2) of each other and the merge kept too few of them. On the
    # transition demo that cost 4 ps of delay (138.8 vs 134.7 ps converged); with 2/3 fine the default
    # mesh gives 135.1 ps with fewer cells (docs/meshing.md, 'Validation').
    c = THIRDS_CELL * fine
    for lay, ps in m.copper.items():
        for p, net in zip(ps, nets.get(lay, [''] * len(ps))):
            ground = zp is not None and (not net or net in gnd)
            for ring in [p.exterior] + list(p.interiors):
                xy = np.asarray(ring.coords)
                cut = ground                                        # filter this ring's lines by the zone
                if ground:
                    b = ring.bounds
                    if max(b[2] - b[0], b[3] - b[1]) <= ZONE_SMALL:   # small: all of it, or none
                        if not zp.intersects(ring):
                            continue
                        cut = False
                for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                    dx, dy = x1 - x0, y1 - y0
                    if not manhattan(np.array([dx, dy]), np.hypot(dx, dy), fine):
                        continue
                    if cut and not zp.intersects(LineString([(x0, y0), (x1, y1)])):
                        continue
                    # orient(): exterior CCW, holes CW -> metal is on the left of every edge
                    if abs(dx) <= abs(dy):                          # edge along y at x ~ x0
                        s, xm = -np.sign(dy), 0.5 * (x0 + x1)       # metal on +x if going down
                        third_in[0].append(xm + s * c / 3); third[0].append(xm - s * 2 * c / 3)
                    else:
                        s, ym = np.sign(dx), 0.5 * (y0 + y1)
                        third_in[1].append(ym + s * c / 3); third[1].append(ym - s * 2 * c / 3)
                cx, cy = curve_lines(xy, fine, curves, step=max(fine, 2 * p.area / p.length), boxes=boxes,  # its width
                                     inside=(lambda x, y: zp.contains(Point(x, y))) if cut else None)
                fixed[0] += cx; fixed[1] += cy
    return fixed, third, third_in


def _via_modes(m, vias):
    """Mesh treatment per via: 'exact' / 'centre' / 'snap' ('auto': exact on signal nets, snap elsewhere)."""
    sig = getattr(m, 'via_signal', [False] * len(m.vias))
    return ['exact' if (vias == 'auto' and s_) else ('snap' if vias == 'auto' else vias) for s_ in sig]


def _kept_lines(m):
    """Lines kept exactly: the board outline, the port planes (the port width snaps to the edge lines)
    and both ends of every lumped-part piece."""
    ob = m.outline.bounds
    keep = {0: [ob[0], ob[2]], 1: [ob[1], ob[3]]}
    for p in m.ports:
        keep[p.axis] += p.planes()
    for pt in getattr(m, 'parts', []):
        for _, _, _, s, e in pt.pieces():
            keep[pt.axis] += [s[pt.axis], e[pt.axis]]
    return keep


def _snap_vias(m, mode, mesh_xy, via_snap):
    """Via positions after snapping ('snap' vias move onto the nearest line crossing within via_snap mm)
    and the indices of those too far from any line."""
    placed, far = [], []
    for i, ((x, y, r, z0, z1), md) in enumerate(zip(m.vias, mode)):
        if md != 'snap':
            placed.append((x, y, r, z0, z1)); continue
        jx, jy = np.argmin(np.abs(mesh_xy[0] - x)), np.argmin(np.abs(mesh_xy[1] - y))
        sx, sy = mesh_xy[0][jx], mesh_xy[1][jy]
        if abs(sx - x) <= via_snap and abs(sy - y) <= via_snap:
            placed.append((float(sx), float(sy), r, z0, z1))
        else:
            placed.append((x, y, r, z0, z1)); far.append(i)
    return placed, far


def signal_sheets(m, plane=0.85):
    """Heights of the copper sheets with signal edges on them: the ports' layers and every layer that is not
    a plane (copper over less than `plane` of the board)."""
    area = m.outline.area
    out = {m.cu_z[p.layer] for p in m.ports if p.layer in m.cu_z}
    for lay, ps in m.copper.items():
        if lay in m.cu_z and sum(p.area for p in ps) < plane * area:
            out.add(m.cu_z[lay])
    return sorted(out)


def _z_lines(m, ext, M_die, M_air, nz_min, ratio, dz_sheet=None):
    """z lines: every interface, >= nz_min cells per dielectric, copper sheets exactly on lines, graded
    air (ext[4] below - none on a PEC bottom plane - and ext[5] above). dz_sheet (mm): the first cell either
    side of each signal sheet (signal_sheets), graded by `ratio` into the dielectric."""
    zs = sorted({0.0, m.thickness} | {l['z0'] for l in m.layers} | {l['z1'] for l in m.layers} | set(m.cu_z.values()))
    sheets = signal_sheets(m) if dz_sheet else []
    zl, extra = [], []
    for z0, z1 in zip(zs[:-1], zs[1:]):
        lay = next((l for l in m.layers if l['z0'] <= z0 + 1e-9 and l['z1'] >= z1 - 1e-9), None)
        Mz = M_die if lay else M_air
        n = max(nz_min, int(np.ceil((z1 - z0) / Mz)))
        h = (z1 - z0) / n
        near = [zc for zc in (z0, z1) if lay and any(abs(zc - sz) < 1e-9 for sz in sheets)]
        if not near or dz_sheet >= h:
            zl += list(np.linspace(z0, z1, n + 1))
            continue
        # graded from dz_sheet at the sheet(s) up to the uniform cell h
        fixed = [z0, z1] + [z0 + dz_sheet if zc == z0 else z1 - dz_sheet for zc in near]
        extra += list(smooth_lines(fixed, h, ratio))
    zl = _merge(np.concatenate([np.round(zl, 9), np.round(extra, 9), zs]), 1e-7)
    for lay, zc in m.cu_z.items():                                  # snap: sheets must lie exactly on a line
        i = np.argmin(np.abs(zl - zc))
        assert abs(zl[i] - zc) < 1e-6, f"{lay} at z={zc} is not on a z mesh line"
        zl[i] = zc
    zfix = np.concatenate([[zl[0] - ext[4]] if ext[4] > 0 else [], zl, [zl[-1] + ext[5]]])
    return smooth_lines(zfix, lambda z: M_die if 0 <= z <= m.thickness else M_air, ratio)


def _check(m, mesh, info, log):
    """Port traces resolved across their width, lumped-part ends on lines."""
    for p in m.ports:
        b = 1 - p.axis
        lines = mesh[b]
        n_in = int(np.sum((lines > p.c[b] - p.w / 2 + 1e-9) & (lines < p.c[b] + p.w / 2 - 1e-9)))
        info.setdefault('lines_across_ports', {})[p.name] = n_in
        if n_in < 2:
            log(f"WARNING: only {n_in} mesh line(s) across the {p.w:.3f} mm trace at {p.name} - use a smaller --fine")
    for pt in getattr(m, 'parts', []):
        lines = mesh[pt.axis]
        ends = [np.min(np.abs(lines - v)) < 1e-9 for _, _, _, s, e in pt.pieces() for v in (s[pt.axis], e[pt.axis])]
        if not all(ends):
            log(f"WARNING: the {pt.gap:.3f} mm gap of {pt.ref} is narrower than half the fine cell - "
                f"use a smaller --fine")


def signal_gaps(m, near=2.0):
    """The gaps (mm) between the copper the ports sit on and other copper of the same layer, closer than `near`:
    coupled lines, a grounded CPW's slots. Copper pieces are the separate polygons of a layer."""
    from shapely.geometry import Point
    out = []
    for p in m.ports:
        polys = m.copper.get(p.layer, [])
        mine = [g for g in polys if g.distance(Point(*p.c)) < 1e-3]
        for g in mine:
            for h in polys:
                if h is not g:
                    d = g.distance(h)
                    if 0 < d < near:
                        out.append(float(d))
    return out


def air_text(air):
    """The air margin as the log says it: '11.5 mm', or per side when they differ."""
    if not isinstance(air, (tuple, list)):
        return f"{air:.1f} mm"
    return " ".join(f"{n}={a:g}" for n, a in zip(SIDES, air)) + " mm"


def auto_mesh(m, fmin, fmax, res=20, fine=None, air=5.0, ratio=1.4, bc=('MUR',) * 6, nz_min=4,
              curves='extremes', vias='auto', via_snap=0.2, refine=(), z_sheet=1.0, curve_res=6.0, zone=False,
              log=print):
    """Mesh lines (mm) for model m. Returns dict(x, y, z, vias, info).

    curves: 'extremes' (default) or 'vertices' - see curve_lines().
    vias:   'exact'  lines at the centre and +-radius of every via (round barrel resolved)
            'centre' one line per axis through the centre (barrel modelled on that line)
            'snap'   no lines of their own: each via moves onto the nearest existing line crossing if it
                     is within via_snap mm (else it gets centre lines)
            'auto'   (default) 'exact' for vias on the port (signal) nets, 'snap' for all others
                     (ground stitching / fence vias)
    refine: [(x0, y0, x1, y1, step)] boxes (model coordinates, mm) with x and y lines every `step`
    z_sheet: the first z cell either side of a copper sheet with signal edges, x fine, graded into the
             dielectric (0: uniform cells, nz_min per dielectric). On 0.8 mm FR-4 the default nz of 4 gave a
             10 dB coupler -8.6 dB (converged about -10.6) and a grounded CPW's S11 at 5 GHz -19 dB (about -33);
             grading from the fine cell gave -10.3 and -31.6 dB for 1.4x the run time. The gap and edge cells in
             x / y hardly mattered (docs/meshing.md#z-lines)
    curve_res: over curved copper (curve_lines' boxes) the largest cell is the dielectric's / curve_res: a radial
             stub's fan had cells up to 1.8 mm and notched at 2.03 GHz instead of 2.13
    zone:   True: the ground copper's edges get mesh lines only near the signal copper (signal_zone); far from it
             the planes' pours, antipads and pads are left to the graded mesh. A number: the zone's reach (mm)"""
    er_max = max([l['er'] for l in m.layers] + [1.0])
    lam0 = C0 / fmax * 1e3                                          # mm
    M_air, M_die = lam0 / res, lam0 / np.sqrt(er_max) / res
    widths = [p.w for p in m.ports]
    h_min = min([l['z1'] - l['z0'] for l in m.layers if l['kind'] == 'diel'] + [1.0])
    if fine is None:
        # a quarter of the narrowest of: the port traces, twice the thinnest dielectric, and the gaps next to the
        # signal copper (the coupler's 0.14 mm gap between 1.5 mm lines had had a cell of 0.38 mm)
        fine = float(np.clip(min(widths + [h_min * 2] + signal_gaps(m)) / 4, 0.01, M_die / 4))
    air = 5.0 if air is None else air
    sides = list(air) if isinstance(air, (tuple, list)) else [air] * 6
    npml = [int(b.split('_')[1]) if b.startswith('PML_') else 0 for b in bc]
    ext = [sides[i] + npml[i] * M_air for i in range(6)]            # per side: air margin (+ PML cells)
    bottom_pec = bc[4] == 'PEC' and bottom_is_plane(m)
    if bottom_pec:
        ext[4] = 0.0                                                # PEC wall = the bottom plane itself
    ob = m.outline.bounds
    boxes = []
    zg = signal_zone(m, None if zone is True else float(zone)) if zone is not False and zone is not None else None
    fixed, third, third_in = _edge_lines(m, fine, curves, boxes, zg)
    mode = _via_modes(m, vias)
    for (x, y, r, z0, z1), md in zip(m.vias, mode):
        if md == 'exact':
            fixed[0] += [x - r, x, x + r]; fixed[1] += [y - r, y, y + r]
        elif md == 'centre':
            fixed[0].append(x); fixed[1].append(y)
    keep = _kept_lines(m)
    for x0, y0, x1, y1, step in refine:                             # local refinement boxes
        fixed[0] += list(np.arange(min(x0, x1), max(x0, x1) + step / 2, step))
        fixed[1] += list(np.arange(min(y0, y1), max(y0, y1) + step / 2, step))

    def xy_lines(extra):
        out = {}
        for k in (0, 1):
            pts = _merge(np.concatenate([fixed[k], extra[k], third[k], third_in[k]]), fine / 2, keep=keep[k],
                         prio=np.r_[np.zeros(len(fixed[k]) + len(extra[k])), np.ones(len(third[k])),
                                    np.full(len(third_in[k]), 2.0)])
            lo, hi = ob[k], ob[2 + k]
            pts = np.concatenate([[lo - ext[2 * k]], pts, [hi + ext[2 * k + 1]]])
            spans = [(b[k], b[2 + k]) for b in boxes] if curve_res > 1 else []
            Mf = (lambda x, lo=lo, hi=hi, sp=spans: M_air if not lo - 1e-9 <= x <= hi + 1e-9 else
                  (M_die / curve_res if any(a <= x <= b for a, b in sp) else M_die))
            out[k] = smooth_lines(pts, Mf, ratio, fine)
        return out

    mesh = xy_lines({0: [], 1: []})
    placed, far = _snap_vias(m, mode, mesh, via_snap)
    if far:                                                         # too far from any line: give them centre lines
        mesh = xy_lines({0: [m.vias[i][0] for i in far], 1: [m.vias[i][1] for i in far]})
        placed, far = _snap_vias(m, mode, mesh, via_snap)
        if far:                                                     # their lines were merged away: snap them anyway
            placed, _ = _snap_vias(m, mode, mesh, np.inf)           # (left off the grid, openEMS would drop them)
    moved = [np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(placed, m.vias)]
    nsnap = sum(1 for md in mode if md == 'snap')
    mesh[2] = _z_lines(m, ext, M_die, M_air, nz_min, ratio, dz_sheet=z_sheet * fine if z_sheet else None)
    nx, ny, nz = (len(mesh[k]) for k in range(3))
    dmin = [np.diff(mesh[k]).min() * 1e-3 for k in range(3)]
    dt = 1 / (C0 * np.sqrt(sum(1 / d ** 2 for d in dmin)))
    info = dict(nx=nx, ny=ny, nz=nz, cells=nx * ny * nz, fine_mm=fine, maxres_die_mm=M_die, maxres_air_mm=M_air,
                air_mm=min(sides), air_sides_mm=sides, air_text=air_text(air), bc=list(bc), curves=curves, vias=vias, vias_snapped=nsnap,
                via_max_move_mm=float(max(moved, default=0.0)), dt_ps=dt * 1e12, min_cell_mm=[d * 1e3 for d in dmin],
                mem_gb=nx * ny * nz * 150e-9)
    log(f"mesh: {nx} x {ny} x {nz} = {nx * ny * nz / 1e6:.2f} M cells (fine {fine:.3f} mm, max {M_die:.2f} mm in "
        f"dielectric / {M_air:.2f} mm in air, air margin {air_text(air)}), boundaries "
        f"{' '.join(f'{n}={b}' for n, b in zip(SIDES, bc))}" + (" (PEC on the bottom plane)" if bottom_pec else "") +
        f", time step {dt * 1e12:.3f} ps, ~{info['mem_gb']:.1f} GB")
    _check(m, mesh, info, log)
    if nsnap:
        log(f"vias: {nsnap} snapped onto existing lines (moved <= {info['via_max_move_mm']:.3f} mm), "
            f"{len(m.vias) - nsnap} with their own lines" +
            (f"; {len(far)} moved further than --via-snap {via_snap:g} mm (no line nearer)" if far else ""))
    return dict(x=mesh[0], y=mesh[1], z=mesh[2], vias=placed, info=info)
