"""A real FDTD engine through the whole plugin (R-C6): the demo transition run by gfdtd on its compiled CPU core,
from the board to the S-parameters, against openEMS's answer stored in kicad2ems's examples/reference (the same
model, settings and mesh as that run: examples/demo_transition.toml). The other tests use stand-ins for openEMS.
About 15 s on eight cores; skipped without gfdtd's core (numpy alone takes minutes)."""
import os

import numpy as np
import pytest

from conftest import K2E


def _core():
    try:
        from gfdtd.engine import core_info
        return core_info()[0]
    except ImportError:
        return False


@pytest.mark.skipif(not _core(), reason="gfdtd's compiled CPU core is not built")
def test_gfdtd_on_the_demo_transition_matches_openems_stored(demo_files, run_cli, tmp_path):
    skrf = pytest.importorskip('skrf')
    base, _ = demo_files
    out = tmp_path / 'out'
    code, so, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                            '--fmin', '1', '--fmax', '8', '--points', '51', '--air', '5', '--nz', '6',
                            '--excite', 'P1', '--solvers', 'gfdtd', '--gfdtd-backend', 'cpu', '--out', out)
    assert code == 0, err
    got = skrf.Network(str(out / 'demo_gfdtd.s2p'))
    ref = skrf.Network(os.path.join(K2E, 'examples', 'reference', 'demo_transition_openems.s2p')).interpolate(got.frequency)
    d = np.abs(got.s - ref.s)
    assert d[:, 0, 0].max() < 0.01 and d[:, 1, 0].max() < 0.02          # here: 0.001 and 0.008
    assert 0.94 < abs(got.s[np.argmin(abs(got.f - 5e9)), 1, 0]) < 0.98
