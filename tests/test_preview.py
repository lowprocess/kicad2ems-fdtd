"""--preview on the real FDTD plugin: the mesh and its log, the report and summary, the walls, the 3D-view grid, the
antennas and the shared FDTD settings (moved from kicad2ems's tests, which run on stand-ins)."""
import json, os
import numpy as np
import pytest
from kicad2ems import solvers, meshes
from kicad2ems.importers.kicad import KiCadBoard
from kicad2ems.board import Board
from kicad2ems.edges import edge_walls
from test_plugins import _EP, _Fake                                 # kicad2ems's tests (conftest.py)

K2E = os.environ.get('KICAD2EMS_SRC') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'kicad2ems')
FILTER = os.path.join(K2E, 'examples', 'rf_interdig.toml')


@pytest.fixture(scope='module')
def launch(boards):
    base = boards['launch']
    return Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', log=lambda *a: None)


def test_preview_report_and_summary(demo_files, run_cli, tmp_path):
    base, _ = demo_files
    out = tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                            '--fmin', '1', '--fmax', '8', '--out', out, '--preview', '--events', tmp_path / 'ev.jsonl')
    assert code == 0, err
    assert 'mesh:' in so and 'zmin=PEC' in so and 'transmission line P1 -> P2' in so
    assert (out / 'demo_mesh.png').stat().st_size > 10000 and (out / 'demo_tlines.png').exists()
    page = (out / 'demo_report.html').read_text()
    for part in ("id='summary'", "id='model'", "id='tlines'", "id='mesh'", 'data:image/png;base64,', 'P1',
                 'dielectric 1', 'microstrip', 'stripline'):
        assert part in page, part
    assert "id='openems'" not in page and "id='mom25d'" not in page   # no full-wave solve in a preview (auto: the line model)
    sm = json.loads((out / 'demo_run.json').read_text())
    assert sm['mode'] == 'preview' and sm['lines']['paths'][0]['port'] == 'P1'
    from kicad2ems.report import build                         # the report rebuilds from the JSON alone
    assert "id='tlines'" in build(sm, str(out))
    kinds = [json.loads(l)['kind'] for l in (tmp_path / 'ev.jsonl').read_text().splitlines()]
    assert 'stage' in kinds and 'log' in kinds and 'file' in kinds and kinds[-1] == 'done'
    code, _, _ = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                         '--out', tmp_path / 'o2', '--preview', '--no-report')
    assert code == 0 and not (tmp_path / 'o2' / 'demo_report.html').exists()


def test_build_fdtd_adds_the_walls(launch, fake_openems):
    from kicad2ems_fdtd.model import build_fdtd
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.mesh import auto_mesh
    m = launch
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 6e9, bc=bc, log=lambda *a: None)
    mesh['walls'] = edge_walls(m, ['xmin', 'ymax'])
    build_fdtd(m, mesh, 1e9, 6e9, 0, bc=bc)
    assert fake_openems.count('box') >= 2


def test_preview_logs_and_reports_walls(boards, run_cli, tmp_path):
    base = boards['launch']
    out = tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--fmin', '1',
                            '--fmax', '6', '--edge-walls', 'all', '--out', out, '--preview')
    assert code == 0, err
    assert 'edge walls: xmin 1 piece' in so
    assert 'edge walls (PEC)' in (out / 'launch_report.html').read_text()


def test_walls_and_vias_sit_on_mesh_lines(launch, boards):
    """openEMS drops a zero-thickness sheet or a thin cylinder that misses every mesh line (an 'unused
    primitive'): walls are snapped onto lines, and vias whose own lines were merged away are snapped
    even beyond --via-snap."""
    from kicad2ems_fdtd.boundaries import resolve_bc
    from kicad2ems_fdtd.mesh import auto_mesh
    from kicad2ems.edges import snap
    for m in (launch, Board(KiCadBoard(boards['demo'] + '.kicad_pcb'), nets=['RF', 'GND'],
                            pos_file=boards['demo'] + '-top.pos', ground='B.Cu', log=lambda *a: None)):
        for fine in (None, 0.05):
            g = auto_mesh(m, 1e9, 6e9, fine=fine, bc=resolve_bc('auto', m), log=lambda *a: None)
            X, Y = np.asarray(g['x']), np.asarray(g['y'])
            assert all(np.min(np.abs(X - v[0])) < 1e-9 and np.min(np.abs(Y - v[1])) < 1e-9 for v in g['vias'])
            for axis, c, *_ in snap(edge_walls(m, ['xmin', 'xmax', 'ymin', 'ymax']), g):
                assert np.min(np.abs((X if axis == 'x' else Y) - c)) < 1e-12


def test_preview_writes_the_meshes(run_cli, tmp_path):
    out = tmp_path / 'out'
    code, so, se = run_cli(FILTER, '--preview', '--solvers', 'openems', '--points', '21', '--no-report', '--out', out)
    assert code == 0, so + se
    found = meshes.find(str(out))
    assert 'openems' in found
    grid = meshes.segments(found['openems'])
    assert len(grid) and len(grid) % 2 == 0 and grid.shape[1] == 3
    assert 'lines' in meshes.counts(found['openems'])


@pytest.mark.parametrize('kind', ['patch', 'ifa', 'dipole', 'slot', 'vivaldi', 'vivaldi_antipodal'])
def test_antenna_preview(boards, kind, tmp_path, run_cli):
    code, out, err = run_cli(boards[kind] + '.kicad_pcb', '--mode', 'antenna', '--fmin', '2', '--fmax', '3',
                             '--out', tmp_path, '--preview', '--no-report')
    assert code == 0, err
    assert 'antenna mode: boundaries PML_8' in out and 'zmin=PML_8' in out
    assert ('gap port' in out) == (kind == 'dipole')


def test_shared_fdtd_settings(monkeypatch):
    """FDTD solvers share the mesh and run settings (the same Option objects): one setting each, in either solver's
    table; another plugin may list them too, but not define its own of the same name."""
    from kicad2ems.config import all_options
    from kicad2ems_fdtd.plugin import FDTD, MESH, RUN

    class Engine(FDTD):
        api = (2, 0)
        name, aliases, label = 'fdtd2', (), 'another FDTD engine'
        options = MESH + RUN

    class Clash(_Fake):
        name, aliases = 'clash', ()
        options = [solvers.Option('res', 10.0)]                     # the shared setting, redefined: refused
    monkeypatch.setattr(solvers, '_registry', None)
    monkeypatch.setattr(solvers, 'broken', [])
    installed = solvers._entry_points()
    monkeypatch.setattr(solvers, '_entry_points', lambda: installed + [_EP('fdtd2', Engine), _EP('clash', Clash)])
    reg = solvers.registry()
    assert 'fdtd2' in reg and [b[0].split(' =')[0] for b in solvers.broken] == ['clash']
    owner = [(o.key, n) for n, o in all_options(reg)]
    assert len(owner) == len(dict(owner)) and dict(owner)['res'] == 'openems'      # once, under the first
    assert reg['gfdtd'].options[0] is reg['openems'].options[0]
    monkeypatch.setattr(solvers, '_registry', None)
