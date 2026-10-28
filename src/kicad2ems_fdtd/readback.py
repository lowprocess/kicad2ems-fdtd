"""
What openEMS gets, read back: build_fdtd run on stand-ins for CSXCAD and openEMS that record every primitive
(no openEMS needed), summarised for the parity checks (parity.py).

    rb = readback(board, mesh, bc)    # dict(vias, walls, copper, parts, ports)

  vias     [x, y, r, z0, z1] of every cylinder of the 'vias' metal (mm, model coordinates)
  walls    [axis, c, t0, t1, z0, z1] of every 'edge_walls' sheet
  copper   {layer: area (mm2)} of the polygons of each copper property (cu_<layer>)
  parts    {ref: elements} lumped elements (or the metal strap of a 0 ohm part) per part
  ports    the lumped ports, in order: [P1, P2, ...] by the board's names
"""
import numpy as np
from shapely.geometry import Polygon


class _Prim:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def AddPolygon(self, pts, nd, elev, priority=0):
        self.log.append((self.name, 'polygon', [np.asarray(pts[0], float), np.asarray(pts[1], float)], float(elev)))

    def AddLinPoly(self, pts, nd, elev, length, priority=0):
        pass                                                        # dielectric slabs: not compared

    def AddBox(self, a, b, priority=0):
        self.log.append((self.name, 'box', [float(v) for v in a], [float(v) for v in b]))

    def AddCylinder(self, a, b, r, priority=0):
        self.log.append((self.name, 'cyl', [float(v) for v in a], [float(v) for v in b], float(r)))


class _Grid:
    def SetDeltaUnit(self, u):
        pass

    def SetLines(self, d, v):
        pass


class _CSX:
    def __init__(self):
        self.log = []

    def GetGrid(self):
        return _Grid()

    def _prim(self, name, *a, **kw):
        return _Prim(name, self.log)

    AddMaterial = AddConductingSheet = AddMetal = _prim

    def AddLumpedElement(self, name, ny, caps=True, **kw):
        return _Prim(name, self.log)


class _FDTD:
    def __init__(self, NrTS=0, EndCriteria=0):
        self.ports = []

    def SetGaussExcite(self, f0, fc):
        pass

    def SetBoundaryCond(self, bc):
        pass

    def SetCSX(self, csx):
        pass

    def AddLumpedPort(self, nr, R, start, stop, d, excite=0, priority=0):
        self.ports.append(nr)
        return object()


def readback(m, mesh, bc, sigma=None):
    """The board as build_fdtd hands it to CSXCAD (see the module)."""
    from .model import build_fdtd
    FDTD, CSX, _ = build_fdtd(m, mesh, 1e9, 2e9, 0, bc=bc, sigma=sigma, api=(_CSX, _FDTD))
    out = dict(vias=[], walls=[], copper={}, parts={}, ports=[m.ports[n - 1].name for n in FDTD.ports])
    for rec in CSX.log:
        name, kind = rec[0], rec[1]
        if kind == 'cyl' and name == 'vias':
            (x, y, z0), (_, _, z1), r = rec[2], rec[3], rec[4]
            out['vias'].append([x, y, r, z0, z1])
        elif kind == 'box' and name == 'edge_walls':
            s, e = rec[2], rec[3]
            axis = 'x' if abs(s[0] - e[0]) < 1e-12 else 'y'
            c, t0, t1 = (s[0], s[1], e[1]) if axis == 'x' else (s[1], s[0], e[0])
            out['walls'].append([axis, c, t0, t1, s[2], e[2]])
        elif kind == 'polygon' and name.startswith('cu_'):
            lay = name[3:]
            out['copper'][lay] = out['copper'].get(lay, 0.0) + Polygon(np.column_stack(rec[2])).area
        elif name.startswith('part_'):
            ref = next((p.ref for p in m.parts if name in (f"part_{p.ref}", f"part_{p.ref}_R", f"part_{p.ref}_X")), name[5:])
            out['parts'][ref] = out['parts'].get(ref, 0) + 1
    return out
