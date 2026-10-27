"""--converge end to end without openEMS: the runs replaced by a canned thru, so every level gives the same answer
and the check says converged; the plot, the summary and the report's section are written (from kicad2ems's
pipeline tests)."""
import json
import numpy as np
import pytest
from kicad2ems.fdtd import run
from kicad2ems.fdtd.run import RunResult
from kicad2ems_openems import engine


def _canned(m, mesh, freq, excite, bc, out_dir, **kw):
    """A plausible thru: |S21| ~ -0.3 dB with 130 ps delay, S11 ~ -25 dB."""
    n = len(m.ports)
    r = RunResult(np.full((len(freq), n, n), np.nan + 0j))
    ph = np.exp(-2j * np.pi * freq * 130e-12)
    for k in excite:
        r.S[:, k, k] = 0.056 * ph
        for i in range(n):
            if i != k:
                r.S[:, i, k] = 0.966 * ph
        r.seconds[m.ports[k].name], r.steps[m.ports[k].name] = 12.0, 30000
    return r


def test_converge_run(demo_files, run_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'check', lambda: (True, []))
    monkeypatch.setattr(run, 'run_excitations', _canned)
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
