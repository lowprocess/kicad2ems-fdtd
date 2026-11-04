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


def test_sigma_follows_the_engine_api(monkeypatch, demo_files):
    """--sigma: conducting sheets on gfdtd from engine API 1.1 (lossy there), and auto keeps gfdtd for such runs;
    on an older gfdtd its copper stays PEC and auto gives --sigma runs to openEMS."""
    from types import SimpleNamespace
    from kicad2ems.api.solver import AUTO
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    from kicad2ems_fdtd import gfdtd as conn
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.model import build_fdtd
    from kicad2ems_fdtd.engines import engine_api
    from kicad2ems_fdtd.mesh import auto_mesh
    run = SimpleNamespace(plan=[AUTO], s=SimpleNamespace(sigma=5.8e7, converge=None))
    monkeypatch.setattr(gfdtd, 'API_VERSION', conn.LOSS_API)
    sv = conn.GFDTD()
    assert sv.conductor_loss and sv.assess(run) == [] and sv.capabilities['conductor_loss'].startswith('yes')
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    FDTD, CSX, _ = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, sigma=5.8e7 if sv.conductor_loss else None,
                              api=engine_api('gfdtd:numpy'))
    op = build(CSX, FDTD.settings())
    assert op.n_sheet > 0 and any('conducting sheet cu_' in n for n in op.notes)
    monkeypatch.setattr(gfdtd, 'API_VERSION', (1, 0))
    assert not sv.conductor_loss and '--sigma' in sv.assess(run)[0]


def test_rough_copper_by_engine(monkeypatch, demo_files):
    """--rough: gfdtd from engine API 1.2 gets it in its sheets (Hammerstad over the band); openEMS, and an older
    gfdtd, the conductivity at the band centre (band_sigma)."""
    from types import SimpleNamespace
    from kicad2ems.api.solver import band_sigma
    from kicad2ems.solvers import registry
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    from kicad2ems_fdtd import gfdtd as conn
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.model import build_fdtd
    from kicad2ems_fdtd.engines import engine_api
    from kicad2ems_fdtd.mesh import auto_mesh
    s = SimpleNamespace(sigma=5.8e7, rough=1.0)
    run = SimpleNamespace(s=s, fmin=1e9, fmax=9e9)
    centre = band_sigma(s, 1e9, 9e9)
    assert centre < 5.8e7 / 1.5                                       # K(5 GHz, 1 um) ~ 1.65
    assert registry()['openems'].copper(run) == (centre, None)
    monkeypatch.setattr(gfdtd, 'API_VERSION', conn.ROUGH_API)
    sv = conn.GFDTD()
    assert sv.copper(run) == (5.8e7, 1.0) and sv.capabilities['roughness'].startswith('yes')
    monkeypatch.setattr(gfdtd, 'API_VERSION', conn.LOSS_API)
    assert sv.copper(run) == (centre, None)
    assert sv.copper(SimpleNamespace(s=SimpleNamespace(sigma=None, rough=1.0), fmin=1e9, fmax=9e9)) == (None, None)
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    _, CSX, _ = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, sigma=5.8e7, rough=1.0, api=engine_api('gfdtd:numpy'))
    sheets = [p for p in CSX.GetAllProperties() if type(p).__name__ == 'ConductingSheet']
    assert sheets and all(p.roughness == pytest.approx(1e-6) for p in sheets)


def test_finished_copper_by_engine(monkeypatch, demo_files):
    """--finish: gfdtd from engine API 1.3 gets it in its sheets (thick copper: the outer face's own sheet, all
    finished; sheets: weighted by the strip's share) and marks thick copper's sheets as faces; openEMS the
    conductivity of the same loss at the band centre."""
    from types import SimpleNamespace
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    from kicad2ems.solvers import registry
    from kicad2ems_fdtd import gfdtd as conn
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.model import build_fdtd
    from kicad2ems_fdtd.engines import engine_api
    from kicad2ems_fdtd.mesh import auto_mesh
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    s = SimpleNamespace(sigma=5.8e7, rough=None, finish='enig', mask=False)
    run = SimpleNamespace(s=s, m=m, fmin=1e9, fmax=8e9)
    monkeypatch.setattr(gfdtd, 'API_VERSION', conn.FINISH_API)
    ckw = conn.GFDTD().copper_kw(run)
    top = m.cu_order[-1]
    assert ckw['faces'] and ckw['finish']['wideband'] and ckw['finish']['share'][top] > 0.3
    bc = resolve_bc('auto', m)
    sheets = lambda CSX: {p.name: p for p in CSX.GetAllProperties() if type(p).__name__ == 'ConductingSheet'}
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, thickness=True, log=lambda *a: None)
    _, CSX, _ = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, sigma=5.8e7, copper_kw=ckw, api=engine_api('gfdtd:numpy'))
    sh = sheets(CSX)
    assert sh[f'cu_{top}_finish'].finish and sh[f'cu_{top}_finish'].finish_share == 1.0 and sh[f'cu_{top}'].face
    assert not sh[f'cu_{top}'].finish
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    _, CSX, _ = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, sigma=5.8e7, copper_kw=ckw, api=engine_api('gfdtd:numpy'))
    sh = sheets(CSX)
    assert sh[f'cu_{top}'].finish_share == pytest.approx(ckw['finish']['share'][top])
    oc = registry()['openems'].copper_kw(run)
    assert not oc['faces'] and not oc['finish']['wideband']
    _, CSX, _ = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, sigma=5.8e7, copper_kw=oc, api=engine_api('gfdtd:numpy'))
    assert sheets(CSX)[f'cu_{top}'].conductivity < 5.8e7 / 1.5 and not sheets(CSX)[f'cu_{top}'].finish
