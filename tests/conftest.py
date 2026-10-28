"""The plugin's tests use kicad2ems's fixtures and example boards, from a kicad2ems checkout: KICAD2EMS_SRC,
default ../kicad2ems next to this repository."""
import os, sys

K2E = os.environ.get('KICAD2EMS_SRC') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'kicad2ems')
if not os.path.exists(os.path.join(K2E, 'tests', 'k2e_fixtures.py')):
    raise RuntimeError(f"no kicad2ems checkout at {K2E}: set KICAD2EMS_SRC")
sys.path.insert(0, os.path.join(K2E, 'tests'))
from k2e_fixtures import *                                          # noqa: E402,F401,F403


def canned_runs(m, mesh, freq, excite, bc, out_dir, **kw):
    """In place of run.run_excitations: a plausible thru (|S21| ~ -0.3 dB with 130 ps delay, S11 ~ -25 dB)."""
    import numpy as np
    from kicad2ems_fdtd.run import RunResult
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
