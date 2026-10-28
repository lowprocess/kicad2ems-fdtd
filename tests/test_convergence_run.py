"""--converge end to end without openEMS: the runs replaced by a canned thru, so every level gives the same answer
and the check says converged; the plot, the summary and the report's section are written (from kicad2ems's
pipeline tests)."""
import json
from kicad2ems_fdtd import run
from kicad2ems_fdtd.openems import engine
from conftest import canned_runs


def test_converge_run(demo_files, run_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'check', lambda: (True, []))
    monkeypatch.setattr(run, 'run_excitations', canned_runs)
    base, _ = demo_files
    out = tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                            '--fmin', '1', '--fmax', '8', '--points', '51', '--out', out, '--converge', '1,1.2',
                            '--solvers', 'openems')
    assert code == 0, so + err
    assert (out / 'demo_converge.png').exists()
    sm = json.loads((out / 'demo_run.json').read_text())
    assert sm['convergence']['converged'] and [lv['factor'] for lv in sm['convergence']['levels']] == [1, 1.2]
    assert "id='convergence'" in (out / 'demo_report.html').read_text()
