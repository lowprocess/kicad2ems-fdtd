"""--thickness on (thick.py): the stack moved to make room for the copper, faces on z lines, ports from face to face,
and the model built from it on gfdtd (PEC, and lossy faces with --sigma)."""
import contextlib
import io

import numpy as np
import pytest


@pytest.fixture(scope='module')
def board(demo_files):
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    base, _ = demo_files
    return Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')


def test_plan_moves_the_stack_by_the_copper(board):
    """B.Cu grows down, the rest up; each dielectric keeps its thickness between the copper faces."""
    from kicad2ems_fdtd import thick
    m = board
    th = thick.plan(m)
    order = m.cu_order
    assert th['up'] == {lay: lay != order[0] for lay in order}
    f = thick.all_faces(m, th)
    for lay in order:
        assert f[lay][1] - f[lay][0] == pytest.approx(m.cu_t[lay])
    assert f[order[0]][1] == 0.0
    for a, b in zip(order, order[1:]):                              # upper face of one to lower face of the next
        assert f[b][0] - f[a][1] == pytest.approx(m.cu_z[b] - m.cu_z[a])
    for l in m.layers:                                              # dielectrics keep their thickness
        lo = thick.layer_at(m, l['z0'])
        z0, z1 = thick.z_map(m, th, l['z0']), thick.z_map(m, th, l['z1'])
        extra = th['t'][lo] if lo and th['up'][lo] else 0.0          # it holds the copper below it, if that grows up
        assert z1 - z0 == pytest.approx(l['z1'] - l['z0'] + extra)


def test_mesh_has_both_faces_and_one_cell_through_the_copper(board):
    from kicad2ems_fdtd import thick
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.mesh import auto_mesh
    m = board
    bc = resolve_bc('auto', m)
    flat = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, thickness=True, log=lambda *a: None)
    assert flat['thick'] is None and flat['info']['thickness'] == 'off' and mesh['info']['thickness'] == 'on'
    z = np.asarray(mesh['z'])
    for lo, up in thick.all_faces(m, mesh['thick']).values():
        i, j = np.searchsorted(z, lo), np.searchsorted(z, up)
        assert z[i] == lo and z[j] == up and j == i + 1             # exactly on lines, one cell between
    assert mesh['info']['cells'] < 1.5 * flat['info']['cells']


@pytest.mark.parametrize('sigma', [None, 5.8e7])
def test_thick_model_on_gfdtd(board, sigma):
    """Ports from the reference copper's face to the trace's; solid copper (PEC), with --sigma its faces as
    conducting sheets of sigma / 4 and 2 t (one side of a conductor t / 2 thick each)."""
    pytest.importorskip('gfdtd')
    from gfdtd.operator import build
    from kicad2ems_fdtd import thick
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.engines import engine_api
    from kicad2ems_fdtd.mesh import auto_mesh
    from kicad2ems_fdtd.model import build_fdtd
    m = board
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, thickness=True, log=lambda *a: None)
    th = mesh['thick']
    with contextlib.redirect_stdout(io.StringIO()):
        FDTD, CSX, ports = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, sigma=sigma, api=engine_api('gfdtd:numpy'))
        op = build(CSX, FDTD.settings())
    for p, mp in zip(ports, m.ports):
        s, e = mp.box()
        lo, hi = sorted((s[2], e[2]))
        zs = sorted((p.start[2], p.stop[2]))
        assert zs == pytest.approx([thick.faces(m, th, thick.layer_at(m, lo))[1],
                                    thick.faces(m, th, thick.layer_at(m, hi))[0]])
        assert zs[1] - zs[0] == pytest.approx(hi - lo)              # the dielectric between them
    assert len(op.exc[0]) > 0 and op.n_pec[2] > 0                   # the copper's own z edges, the vias'
    if sigma:
        sheets = [p for p in CSX.GetAllProperties() if type(p).__name__ == 'ConductingSheet']
        assert sheets and all(p.conductivity == pytest.approx(sigma / 4) for p in sheets)
        assert {round(p.thickness, 12) for p in sheets} == {round(2 * m.cu_t[lay] * 1e-3, 12) for lay in m.copper}
        assert op.n_sheet > 0
    else:
        assert op.n_sheet == 0
