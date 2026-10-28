"""Geometry, ports, automatic mesh, boundaries and post-processing (no openEMS install needed)."""
import os
import numpy as np
import pytest
from shapely.geometry import box, Point

from conftest import K2E as ROOT                                    # kicad2ems's checkout: its example boards
from kicad2ems.geometry import split_holes
from kicad2ems_fdtd.mesh import smooth_lines, auto_mesh
from kicad2ems.board import Board
from kicad2ems.sparams import mixed_mode
from kicad2ems.touchstone import read_touchstone, write_touchstone
from kicad2ems.importers.kicad import KiCadBoard


@pytest.fixture(scope='session')
def demo(demo_files):
    return demo_files[0]


def test_split_holes_preserves_area():
    p = box(0, 0, 10, 6).difference(Point(3, 3).buffer(1)).difference(Point(7, 3).buffer(1.2))
    parts = split_holes(p)
    assert all(not q.interiors for q in parts)
    assert sum(q.area for q in parts) == pytest.approx(p.area, rel=1e-9)


def test_smooth_lines_grading_and_limits():
    fixed = [0, 0.1, 0.2, 5, 5.05, 20, 20.3]
    x = smooth_lines(fixed, 1.0, ratio=1.4)
    d = np.diff(x)
    assert set(np.round(fixed, 9)) <= set(np.round(x, 9))          # fixed lines kept
    assert d.max() <= 1.0 + 1e-9
    assert d.min() >= 0.05 - 1e-12                                  # nothing smaller than the features
    r = d[1:] / d[:-1]
    assert np.max(np.maximum(r, 1 / r)) <= 1.4 * 1.25              # graded (middle cells split evenly)


def test_model_ports_and_mesh(demo):
    m = Board(KiCadBoard(demo + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=demo + '-top.pos', ground='B.Cu')
    p1, p2 = m.ports
    assert (p1.layer, p1.ref_layer, p2.layer, p2.ref_layer) == ('F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu')
    assert p1.w == pytest.approx(0.35, rel=1e-3) and p2.w == pytest.approx(0.15, rel=1e-3)
    assert len(m.vias) == 23
    mesh = auto_mesh(m, 1e9, 8e9, log=lambda *a: None)
    zset = set(mesh['z'].tolist())
    assert all(z in zset for z in m.cu_z.values())                  # every sheet exactly on a z line
    for k in ('x', 'y'):
        d = np.diff(mesh[k])
        assert d.min() > 0.3 * mesh['info']['fine_mm']                 # no slivers next to edge lines
        assert d.max() <= mesh['info']['maxres_air_mm'] + 1e-9
    for p in m.ports:                                               # port plane is a mesh line
        assert np.min(np.abs(mesh['xy'[p.axis]] - p.c[p.axis])) < 1e-9
    # trace edges: thirds rule -> a line 1/3 of the edge cell inside each edge of the 0.35 mm trace
    from kicad2ems_fdtd.mesh import THIRDS_CELL
    yc, w, f = p1.c[1], p1.w, mesh['info']['fine_mm'] * THIRDS_CELL
    for e, s in ((yc - w / 2, 1), (yc + w / 2, -1)):
        assert np.min(np.abs(mesh['y'] - (e + s * f / 3))) < 1e-6


def test_mixed_mode_is_unitary_for_lossless():
    rng = np.random.default_rng(3)
    H = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = H + H.T                                                     # symmetric -> reciprocal
    lam, V = np.linalg.eigh((H + H.conj().T) / 2)
    S = (V @ np.diag(np.exp(1j * lam)) @ V.conj().T)[None]           # exp(j Hermitian) is unitary
    class P:  # minimal stand-in for Model ports
        def __init__(s, n): s.name, s.z0 = n, 50.0
    ps = [P('P1+'), P('P1-'), P('P2+'), P('P2-')]
    m = type('M', (), {'ports': ps, 'user_ports': [('P1', ps[:2]), ('P2', ps[2:])]})()
    Smm, names, zref = mixed_mode(S, m)
    assert names == ['P1d', 'P1c', 'P2d', 'P2c'] and zref == [100, 25, 100, 25]
    assert np.allclose(Smm[0] @ Smm[0].conj().T, np.eye(4), atol=1e-12)


def test_touchstone_roundtrip(tmp_path):
    f = np.linspace(1e9, 5e9, 7)
    S = np.random.default_rng(0).standard_normal((7, 2, 2)) * (1 + 0.5j)
    p = str(tmp_path / 'x.s2p')
    write_touchstone(p, f, S, [50, 50])
    f2, S2 = read_touchstone(p)
    assert np.allclose(f2, f) and np.allclose(S2, S, atol=1e-8)
    p = str(tmp_path / 'y.s4p')
    S4 = np.random.default_rng(1).standard_normal((7, 4, 4)) + 0j
    write_touchstone(p, f, S4, [100, 25, 100, 25])
    assert np.allclose(read_touchstone(p)[1], S4, atol=1e-8)


def test_boundaries_and_air_margin(demo):
    from kicad2ems_fdtd.boundaries import resolve_bc, cavity_check
    m = Board(KiCadBoard(demo + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=demo + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    assert bc == ['MUR'] * 4 + ['PEC', 'MUR']                       # solid ground -> PEC bottom
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, log=lambda *a: None)
    ob = m.outline.bounds
    assert mesh['z'][0] == 0.0                                      # no air below the PEC plane
    assert mesh['z'][-1] == pytest.approx(m.thickness + 5.0)        # 5 mm air above
    assert mesh['x'][0] == pytest.approx(ob[0] - 5.0) and mesh['y'][-1] == pytest.approx(ob[3] + 5.0)
    pml = auto_mesh(m, 1e9, 8e9, bc=['PML_8'] * 6, log=lambda *a: None)
    assert pml['x'][0] < mesh['x'][0] - 8 * 0.9 * pml['info']['maxres_air_mm'] / 2   # PML cells added outside
    front = auto_mesh(m, 1e9, 8e9, bc=bc, air=(5, 5, 5, 40, 0, 5), log=lambda *a: None)   # per side: 40 mm at ymax
    assert front['y'][-1] == pytest.approx(ob[3] + 40.0) and front['y'][0] == pytest.approx(mesh['y'][0])
    assert front['info']['air_mm'] == 0 and front['info']['air_sides_mm'][3] == 40
    from kicad2ems_fdtd.plugin import _air
    assert _air('7') == 7.0 and _air('1,2,3,4,5,6') == (1, 2, 3, 4, 5, 6) and _air(None) is None
    with pytest.raises(ValueError):
        _air('1,2')
    assert resolve_bc('PEC', m) == ['PEC'] * 6
    msgs = []
    f = cavity_check(m, mesh, ['PEC'] * 6, 20e9, log=msgs.append)
    assert f is not None and f < 20e9 and msgs and 'cavity' in msgs[0]
    m2 = Board(KiCadBoard(demo + '.kicad_pcb'), nets=['RF'], pos_file=demo + '-top.pos', ground='B.Cu')
    m2.copper.pop('B.Cu')
    assert resolve_bc('auto', m2)[4] == 'MUR'                       # no solid bottom plane -> absorbing


def test_mixed_mode_keeps_known_entries_when_some_are_unknown():
    class P:
        def __init__(s, n): s.name, s.z0 = n, 50.0
    ps = [P('P1'), P('P2+'), P('P2-')]
    m = type('M', (), {'ports': ps, 'user_ports': [('P1', ps[:1]), ('P2', ps[1:])]})()
    S = np.full((1, 3, 3), 0.1 + 0j)
    S[0, 0, 0], S[0, 1, 0] = 0.2, 0.7
    S[0, 1:, 1:] = np.nan                                          # P2 legs not excited
    Smm, names, _ = mixed_mode(S, m)
    assert names == ['P1', 'P2d', 'P2c']
    assert Smm[0, 0, 0] == pytest.approx(0.2)                       # S11 survives
    assert np.isfinite(Smm[0, 1:, 0]).all()                         # P1 -> P2d / P2c survive
    assert np.isnan(Smm[0, 1:, 1:]).all()                           # only the unknown block is NaN


def test_merge_keeps_inside_line_of_narrow_trace():
    """Regression: chained clusters + averaging pushed the line inside a 0.15 mm trace outside it."""
    from kicad2ems_fdtd.mesh import _merge
    fine = 0.0375
    edge = -5.075                                                   # lower edge of a trace, metal above
    lines = [edge - 2 * fine / 3, -5.081, edge + fine / 3]          # outside third, a curve vertex, inside third
    out = _merge(lines, fine / 2, prio=[1, 0, 2])
    assert np.any(np.abs(out - (edge + fine / 3)) < 1e-12)          # the inside line survives exactly
    assert np.all(np.diff(out) > fine / 2)                          # and nothing closer than tol remains


def test_curves_extremes_and_via_snapping(demo):
    from kicad2ems_fdtd.mesh import curve_lines
    from kicad2ems_fdtd.boundaries import resolve_bc
    circle = np.array(Point(1, 2).buffer(0.5, quad_segs=8).exterior.coords)
    xs, ys = curve_lines(circle, 0.05)
    assert set(np.round(xs, 6)) == {0.5, 1.5} and set(np.round(ys, 6)) == {1.5, 2.5}   # 2 lines per axis, not 32
    assert len(curve_lines(circle, 0.05, 'vertices')[0]) == 32
    m = Board(KiCadBoard(demo + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=demo + '-top.pos', ground='B.Cu')
    assert m.signal_nets == {'RF'} and sum(m.via_signal) == 1
    bc = resolve_bc('auto', m)
    exact = auto_mesh(m, 1e9, 8e9, bc=bc, curves='vertices', vias='exact', log=lambda *a: None)
    simple = auto_mesh(m, 1e9, 8e9, bc=bc, log=lambda *a: None)
    assert simple['info']['cells'] < 0.6 * exact['info']['cells']
    assert simple['info']['via_max_move_mm'] <= 0.2
    sig = [v for v, s in zip(simple['vias'], m.via_signal) if s][0]
    assert sig[:2] == m.vias[m.via_signal.index(True)][:2]          # the signal via never moves
    for mesh in (exact, simple):                                     # both ports resolved across their width
        assert min(mesh['info']['lines_across_ports'].values()) >= 3


def test_near_manhattan_edges_keep_thirds_lines():
    """rf_interdig.kicad_pcb: the 0.35 mm feeds meet 0.353 mm resonator rectangles; simplify() turns the
    1.5 um step into slightly oblique 6.8 mm edges, which must still get the thirds rule."""
    m = Board(KiCadBoard(os.path.join(ROOT, 'examples', 'rf_interdig.kicad_pcb')), nets=['RF', 'GND'],
                log=lambda *a: None)
    mesh = auto_mesh(m, 1e9, 8e9, fine=0.05, log=lambda *a: None)
    assert all(n >= 3 for n in mesh['info']['lines_across_ports'].values()), mesh['info']['lines_across_ports']


def test_region_leaves_out_ports_outside():
    b = KiCadBoard(os.path.join(ROOT, 'examples', 'rf_interdig.kicad_pcb'))
    pos = os.path.join(ROOT, 'examples', 'rf_interdig-all.pos')
    m = Board(b, nets=['RF', 'GND'], pos_file=pos, region=(0, 0, 12, 20), log=lambda *a: None)
    assert [p.name for p in m.ports] == ['P1'] and any('P2 (TP2) is outside the region' in n for n in m.notes)
    assert m.outline.bounds == pytest.approx((0, -20, 12, 0))
    with pytest.raises(ValueError, match='no port footprints inside the region'):
        Board(b, nets=['RF', 'GND'], pos_file=pos, region=(5, 0, 12, 20), log=lambda *a: None)
    with pytest.raises(ValueError, match='does not overlap'):
        Board(b, nets=['RF', 'GND'], pos_file=pos, region=(30, 30, 40, 40), log=lambda *a: None)


def test_curve_lines_follow_a_thin_curved_track():
    """Along a curved run, 'extremes' adds a line every `step` mm (the copper's width), so a thin curved
    track stays connected on the mesh; without a step only the run's ends and extremes."""
    import numpy as np
    from kicad2ems_fdtd.mesh import curve_lines
    a = np.linspace(0, np.pi, 60)
    arc = np.column_stack([10 * np.cos(a), 10 * np.sin(a)])
    ring = np.vstack([arc, arc[::-1] * 0.92, arc[:1]])            # a half annulus 0.8 mm wide, closed
    xs0, ys0 = curve_lines(ring, 0.2)
    xs1, ys1 = curve_lines(ring, 0.2, step=0.8)
    assert len(xs1) > 3 * len(xs0) and len(ys1) > len(ys0)
    gaps = np.diff(np.unique(np.round(xs1, 6)))
    assert gaps.max() < 0.8 + 0.6                                  # never much more than the step apart


def test_mesh_zone_keeps_the_signal_detail():
    """--mesh-zone: the ground copper's edges get lines only near the signal copper. On sma_launch the planes'
    cut-outs far from the line no longer set the y mesh; every line near the signal trace stays."""
    from kicad2ems_fdtd.mesh import signal_zone
    ex = os.path.join(ROOT, 'examples')
    m = Board(KiCadBoard(os.path.join(ex, 'sma_launch.kicad_pcb')), nets=['REF_SMA_N', 'GND'],
              pos_file=os.path.join(ex, 'sma_launch-all.pos'), log=lambda *a: None)
    off = auto_mesh(m, 1e9, 6e9, log=lambda *a: None)
    on = auto_mesh(m, 1e9, 6e9, zone=True, log=lambda *a: None)
    assert on['info']['cells'] < 0.8 * off['info']['cells']
    z = signal_zone(m)
    sig = [p for lay, ps in m.copper.items() for p, n in zip(ps, m.copper_nets[lay]) if n not in m.ground_nets]
    assert z.contains(sig[0]) and z.area < 0.7 * m.outline.area
    y0, y1 = sig[0].bounds[1], sig[0].bounds[3]                    # across the trace: the same lines
    near = lambda g: [round(v, 6) for v in g['y'] if y0 - 0.3 <= v <= y1 + 0.3]
    assert near(on) == near(off)
    assert auto_mesh(m, 1e9, 6e9, zone=0.0, log=lambda *a: None)['info']['cells'] <= on['info']['cells']


@pytest.mark.parametrize('name, plane', [('rf_dgs', False), ('rf_slotline', False), ('rf_gcpw', True), ('rf_siw', True)])
def test_bottom_pec_only_on_an_unbroken_plane(name, plane):
    """bc auto puts a PEC wall on the bottom plane only where nothing is cut out of it: a PEC under the defected
    ground's dumbbell or the slotline's slot shorted them (the DGS came out a plain line)."""
    from kicad2ems_fdtd.boundaries import bottom_is_plane
    m = Board(KiCadBoard(os.path.join(ROOT, 'examples', name + '.kicad_pcb')), nets=['RF', 'GND'], log=lambda *a: None)
    assert bottom_is_plane(m) is plane
