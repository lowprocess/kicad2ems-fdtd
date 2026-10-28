"""The whole run without openEMS: the solver is replaced by canned results, so the post-processing,
plots, line-model comparison, antenna outputs, JSON summary and report are all exercised (openEMS's --converge:
test_convergence_run.py)."""
import json
import numpy as np
import pytest
from kicad2ems_fdtd import run, antenna
from kicad2ems_fdtd.run import RunResult
from kicad2ems_fdtd.openems import engine


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


@pytest.fixture
def no_openems(monkeypatch):
    monkeypatch.setattr(engine, 'check', lambda: (True, []))
    monkeypatch.setattr(run, 'run_excitations', _canned)


def test_sparams_run_with_fast_model(demo_files, run_cli, tmp_path, no_openems):
    pytest.importorskip('skrf')
    base, _ = demo_files
    out = tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                            '--fmin', '1', '--fmax', '8', '--points', '51', '--out', out, '--fast', 'skrf',
                            '--solvers', 'openems')
    assert code == 0, err
    for f in ('demo_openems.s2p', 'demo_scikit.s2p', 'demo_openems.png', 'demo_tlines.png', 'demo_report.html',
              'demo_run.json'):
        assert (out / f).exists(), f
    sm = json.loads((out / 'demo_run.json').read_text())
    assert sm['mode'] == 'S-parameters' and sm['ports'] == ['P1', 'P2'] and sm['excited'] == ['P1', 'P2']
    cmp = {r['param']: r for r in sm['solved']['scikit']['compare']}
    assert abs(cmp['S(P2,P1)']['max_ddb']) < 1.5 and cmp['S(P2,P1)']['ddelay_ps'] is not None
    page = (out / 'demo_report.html').read_text()
    for part in ("id='results'", "id='scikit'", "id='tlines'", 'line model', 'Δ delay'):
        assert part in page, part


def test_antenna_run(boards, run_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'check', lambda: (True, []))

    def fake_antenna(m, mesh, freq, k, bc, out_dir, ff_extra=(), ff_pattern=None, step=2.0, log=print, **kw):
        r = _canned(m, mesh, freq, [k], bc, out_dir)
        f0 = 2.45e9
        s11 = 0.9 - 0.85 * np.exp(-((freq - f0) / 0.05e9) ** 2) + 0j
        r.S[:, k, k] = s11
        rows = antenna.gain_table(antenna.record_freqs(freq[0], freq[-1]), [3.0] * 21, [0.9e-3] * 21, [1e-3] * 21,
                                  [1.1e-3] * 21)
        ang = list(np.arange(-180, 181, 30.0))
        cut = dict(angle=ang, gain_dbi=[5.0] * len(ang))
        th, ph = np.arange(0, 181, 30.0), np.arange(0, 361, 30.0)
        r.extra[m.ports[k].name] = dict(match=antenna.match_summary(freq, s11, 50.0), rows=rows, f_pattern=f0,
                                        cuts={'xz (phi = 0)': cut, 'yz (phi = 90)': cut, 'xy (theta = 90)': cut},
                                        sphere=dict(theta=th, phi=ph, gain=np.full((len(th), len(ph)), 2.0)),
                                        port=m.ports[k].name)
        return r
    monkeypatch.setattr(antenna, 'run_antenna', fake_antenna)
    out = tmp_path / 'ant'
    code, so, err = run_cli(boards['patch'] + '.kicad_pcb', '--mode', 'antenna', '--fmin', '2', '--fmax', '3',
                            '--points', '41', '--out', out, '--solvers', 'openems')
    assert code == 0, err
    for f in ('patch_antenna.png', 'patch_pattern.png', 'patch_antenna.json', 'patch_report.html'):
        assert (out / f).exists(), f
    sm = json.loads((out / 'patch_run.json').read_text())
    assert sm['antenna']['f_pattern'] == 2.45e9 and 'sphere' not in sm['antenna']
    assert "id='antenna'" in (out / 'patch_report.html').read_text()


def test_touchstone_says_what_was_not_computed(demo_files, run_cli, tmp_path, no_openems):
    """With only P1 driven, S22 is unknown: the file still needs a number there (0), and says so."""
    from kicad2ems.pipeline import unknown_note
    S = np.zeros((3, 2, 2), complex)
    S[:, 1, 1] = np.nan
    assert unknown_note(S, ['P1', 'P2']) == '; not computed (written as 0): S(P2,P2)'
    assert unknown_note(np.zeros((3, 2, 2)), ['P1', 'P2']) == ''
    base, _ = demo_files
    out = tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                            '--fmin', '1', '--fmax', '8', '--points', '21', '--out', out, '--excite', 'P1', '--solvers', 'openems')
    assert code == 0, err
    head = (out / 'demo_openems.s2p').read_text().splitlines()[0]
    assert head.endswith('not computed (written as 0): S(P2,P2)'), head
