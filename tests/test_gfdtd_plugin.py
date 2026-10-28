"""gfdtd's connection (kicad2ems_fdtd.gfdtd): registered, its engine API, and the FDTD model built through
gfdtd's API. Needs a kicad2ems checkout (conftest.py)."""
import os

import pytest

gfdtd = pytest.importorskip('gfdtd')                                # the [gfdtd] extra
from gfdtd.operator import build                                    # noqa: E402


def test_plugin_registered():
    from kicad2ems.solvers import registry, get
    sv = registry()['gfdtd']
    assert get('gpu-fdtd') is sv and sv.in_auto and not sv.in_all
    assert sv.check()[0]                                              # numpy at worst
    assert sv.capabilities['antenna'].startswith('yes')
    assert sv.options[-1].key == 'gfdtd_backend'                    # its own, after the shared FDTD settings
    shared = {o.key: o for o in registry()['openems'].options}
    assert all(shared[o.key] is o for o in sv.options[:-1])


def test_engine_api_and_run_dir():
    from kicad2ems_fdtd.engines import engine_api, engine_name
    from kicad2ems_fdtd.run import run_dir
    assert engine_name(None) == ('openems', 'openEMS') and engine_name('gfdtd:cupy') == ('gfdtd', 'gfdtd')
    with pytest.raises(ValueError, match='unknown FDTD engine'):
        engine_api('hfss')
    cs, fd = engine_api('gfdtd:numpy')
    assert cs is gfdtd.ContinuousStructure and fd().backend == 'numpy'

    class P:
        name = 'P1+'

    class M:
        ports = [P()]
    assert run_dir(M, 0, '/o').endswith(os.path.join('openems', 'exc_P1p'))
    assert run_dir(M, 0, '/o', 'gfdtd:cupy').endswith(os.path.join('gfdtd', 'exc_P1p'))


def test_build_fdtd_builds_the_board_on_gfdtd(demo_files):
    """kicad2ems's openEMS model, built through gfdtd's API, lowers to an operator: ports, copper, vias."""
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.model import build_fdtd
    from kicad2ems_fdtd.engines import engine_api
    from kicad2ems_fdtd.mesh import auto_mesh
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    FDTD, CSX, ports = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, api=engine_api('gfdtd:numpy'))
    op = build(CSX, FDTD.settings())
    assert len(ports) == len(m.ports) == 2 and len(op.exc[0]) > 0
    assert op.n_pec[2] > 0                                            # the vias' z edges
    assert len(op.probes) == 4 and op.bc == bc


def test_engine_api_version(monkeypatch):
    """The connection checks gfdtd's API_VERSION: the same major and at least the minor it was written for; else
    the solver is not usable (kicad2ems --check says which package to update) and the engine refuses to build."""
    from kicad2ems_fdtd import gfdtd as conn
    from kicad2ems_fdtd.engines import engine_api
    assert conn.engine_compatible() is None and conn.GFDTD().check()[0]
    monkeypatch.setattr(gfdtd, 'API_VERSION', (conn.ENGINE_API[0], conn.ENGINE_API[1] + 1), raising=False)
    assert conn.engine_compatible() is None                         # a newer minor: only additions
    monkeypatch.setattr(gfdtd, 'API_VERSION', (conn.ENGINE_API[0] + 1, 0))
    ok, why = conn.GFDTD().check()
    assert not ok and 'update kicad2ems-fdtd' in why
    with pytest.raises(RuntimeError, match='update kicad2ems-fdtd'):
        engine_api('gfdtd:numpy')
    monkeypatch.setattr(gfdtd, 'API_VERSION', (conn.ENGINE_API[0], conn.ENGINE_API[1] - 1) if conn.ENGINE_API[1]
                        else (conn.ENGINE_API[0] - 1, 9))
    assert 'update gfdtd' in conn.engine_compatible()
    monkeypatch.delattr(gfdtd, 'API_VERSION')
    assert 'no API version' in conn.engine_compatible() and 'update gfdtd' in conn.engine_compatible()
