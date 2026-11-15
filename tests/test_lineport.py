"""Line ports (R-E1, lineport.py): which ports can have probes, S = B A^-1 and the reference moved back to the port
along the line read, and a straight 50 ohm line on gfdtd against its lumped ports."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kicad2ems_fdtd import lineport as LP

K2E = os.environ.get('KICAD2EMS_SRC') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'kicad2ems')


def _line_board(path, length):
    sys.path.insert(0, os.path.join(K2E, 'examples'))
    import make_rf_examples as M
    bd = M.Board()
    yc = 8.0
    bd.track([(0, yc), (length, yc)], M.W50)
    bd.port(1, 0.5, yc, 0)
    bd.port(2, length - 0.5, yc, 180)
    bd.write(str(path), length, 16.0)
    from kicad2ems.board import Board
    from kicad2ems.importers.kicad import KiCadBoard
    return Board(KiCadBoard(str(path)), nets=['RF', 'GND'], ground='B.Cu', log=lambda *a: None)


@pytest.fixture(scope='module')
def boards(tmp_path_factory):
    d = tmp_path_factory.mktemp('lines')
    return dict(long=_line_board(d / 'msline.kicad_pcb', 30.0), short=_line_board(d / 'short.kicad_pcb', 4.0))


def _mesh(m):
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.mesh import auto_mesh
    return auto_mesh(m, 0.5e9, 10e9, bc=resolve_bc('auto', m))


def test_which_ports_have_probes(boards):
    m = boards['long']
    mesh = _mesh(m)
    plan, notes = LP.plan_ports(m, mesh, 'auto')
    h = abs(m.ports[0].z_sig - m.ports[0].z_ref)
    assert sorted(plan) == [0, 1] and not notes
    assert all(pr['d'] >= 3 * h - 1e-9 and pr['stop'] > pr['d'] and pr['feed'] >= pr['stop'] + 2 * h
               for pr in plan.values())
    assert LP.plan_ports(m, mesh, 'lumped') == ({}, [])
    with pytest.raises(ValueError, match='not one of'):
        LP.plan_ports(m, mesh, 'wave')
    s = boards['short']                                             # 4 mm: too short for d, its planes, 2 h
    plan, notes = LP.plan_ports(s, _mesh(s), 'auto')
    assert plan == {} and len(notes) == 2 and 'straight feed is' in notes[0]
    with pytest.raises(ValueError, match='--port-kind line: P1: lumped'):
        LP.plan_ports(s, _mesh(s), 'line')
    gap = SimpleNamespace(kind='gap', internal=False)
    assert LP.window(m, gap, mesh) == (None, 'a gap port')


def _abcd_line(zc, beta, length):
    gl = 1j * beta * length
    return np.stack([np.stack([np.cosh(gl), zc * np.sinh(gl)], -1), np.stack([np.sinh(gl) / zc, np.cosh(gl)], -1)], -2)


def test_the_waves_give_s_and_the_reference_moves_back_to_the_port():
    """Synthetic runs of a known 2-port seen through 3 mm of a 45 ohm line at port 1 (probed) and directly at
    port 2 (lumped): solve() returns the 2-port at the ports, whatever the terminations reflected."""
    from kicad2ems.deembed import abcd_to_s, s_to_abcd
    f = np.linspace(1e9, 10e9, 41)
    w = 2 * np.pi * f
    zc, beta = 45.0 + 0j * f, w * np.sqrt(3.4) / LP.C0 * (1 - 0.01j)    # a lossy line, its index flat
    d = 3.0
    core = np.zeros((len(f), 2, 2), complex)                        # the board at the ports: a series RC
    zser = 20 + 1 / (1j * w * 2e-12)
    core[:, 0, 0] = core[:, 1, 1] = zser / (zser + 100)
    core[:, 0, 1] = core[:, 1, 0] = 100 / (zser + 100)
    T = s_to_abcd(core, 50.0)                                       # what the probes see: the line removed
    Tm = np.linalg.inv(_abcd_line(zc, beta, d * 1e-3)) @ T
    Sm = abcd_to_s(Tm, 50.0)
    rng = np.random.default_rng(3)
    A = np.eye(2)[None] + 0.3 * (rng.normal(size=(len(f), 2, 2)) + 1j * rng.normal(size=(len(f), 2, 2)))
    B = Sm @ A                                                      # each run's waves (terminations reflect)
    ports = [SimpleNamespace(name='P1', z0=50.0), SimpleNamespace(name='P2', z0=50.0)]
    lines = {}
    for k, p in enumerate(ports):
        u, i = A[:, 0, k] + B[:, 0, k], (A[:, 0, k] - B[:, 0, k]) / 50.0
        lines[p.name] = dict(probes={0: dict(u=u, i=i, z=zc, beta=beta, d=d)},
                             lumped=(np.array([A[:, 0, k] * 0, A[:, 1, k]]), np.array([B[:, 0, k] * 0, B[:, 1, k]])))
    S, read = LP.solve(SimpleNamespace(ports=ports), f, SimpleNamespace(lines=lines), {0: dict(d=d)})
    assert np.abs(S - core).max() < 1e-9
    assert read['P1'][0] == pytest.approx(45.0) and read['P1'][1] == pytest.approx(3.4, rel=1e-3) and read['P1'][2] == d
    with pytest.raises(ValueError, match='every port driven'):
        LP.solve(SimpleNamespace(ports=ports), f, SimpleNamespace(lines={'P1': lines['P1']}), {0: dict(d=d)})


def test_a_straight_line_on_gfdtd(boards, tmp_path):
    """The 30 mm, 50 ohm line through gfdtd: its line ports read its own line (Zc near 50 ohm) and match it
    (|S11| below -27 dB from 2 to 9 GHz), where the lumped ports carry their feed and the trace behind them
    (-21 dB at 8 GHz); the same runs give both."""
    pytest.importorskip('gfdtd')
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.run import run_excitations
    m = boards['long']
    mesh = _mesh(m)
    f = np.linspace(0.5e9, 10e9, 96)
    plan, _ = LP.plan_ports(m, mesh, 'line')
    res = run_excitations(m, mesh, f, [0, 1], resolve_bc('auto', m), str(tmp_path), line_plan=plan, engine='gfdtd',
                          log=lambda *a: None)
    S, read = LP.solve(m, f, res, plan)
    band = (f >= 2e9) & (f <= 9e9)
    db = lambda x: 20 * np.log10(np.abs(x))
    assert db(S[band, 0, 0]).max() < -27 and db(res.S[band, 0, 0]).max() > -24
    assert all(48 < zc < 52 and 3.2 < ee < 3.6 for zc, ee, _ in read.values())
    assert np.abs(S[:, 0, 1] - S[:, 1, 0]).max() < 0.01                # reciprocal


def test_a_port_at_an_angle_builds_and_stays_lumped(tmp_path):
    """A vertical port on a trace at 30 deg (R-E1): the FDTD model takes it (its sheet across x, as wide as the trace's
    cut there), and --port-kind auto keeps it lumped (probes stand on the mesh's axes)."""
    sys.path.insert(0, os.path.join(K2E, 'examples'))
    import make_rf_examples as M
    from kicad2ems.board import Board
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.readback import readback
    bd, a, L = M.Board(), np.radians(30.0), 16.0
    u = np.array([np.cos(a), -np.sin(a)])
    p0, p1 = np.array([12.0, 12.0]) - u * L / 2, np.array([12.0, 12.0]) + u * L / 2
    bd.track([tuple(p0), tuple(p1)], M.W50)
    bd.port(1, *(p0 + 0.5 * u), 30.0)
    bd.port(2, *(p1 - 0.5 * u), 210.0)
    bd.write(str(tmp_path / 'b.kicad_pcb'), 24.0, 24.0)
    m = Board(KiCadBoard(str(tmp_path / 'b.kicad_pcb')), nets=['RF', 'GND'], ground='B.Cu', log=lambda *a: None)
    mesh = _mesh(m)
    assert readback(m, mesh, resolve_bc('auto', m))['ports'] == ['P1', 'P2']
    plan, notes = LP.plan_ports(m, mesh, 'auto')
    assert plan == {} and all('at an angle' in n for n in notes)
