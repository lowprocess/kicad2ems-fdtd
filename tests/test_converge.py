"""Mesh convergence: refinement levels, comparison against the finest level, --preview --converge."""
import numpy as np
import pytest
from kicad2ems.importers.kicad import KiCadBoard
from kicad2ems.board import Board
from kicad2ems_openems.converge import mesh_levels, compare_levels, verdict



def test_levels_refine_monotonically(demo_files):
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu',
              log=lambda *a: None)
    levels = mesh_levels(m, 1e9, 8e9, [1, 1.5, 2], log=lambda *a: None)
    cells = [lm['info']['cells'] for _, lm in levels]
    nz = [lm['info']['nz'] for _, lm in levels]
    fine = [lm['info']['fine_mm'] for _, lm in levels]
    assert cells == sorted(cells) and len(set(cells)) == 3 and nz == sorted(nz)
    assert fine[0] == pytest.approx(2 * fine[2])
    for _, lm in levels:
        assert min(lm['info']['lines_across_ports'].values()) >= 2


def test_compare_levels_and_verdict():
    f = 11
    ref = np.zeros((f, 2, 2), complex)
    ref[:, 0, 0] = 0.1; ref[:, 1, 0] = ref[:, 0, 1] = 0.9; ref[:, 1, 1] = 1e-4      # S22 below -40 dB: ignored
    mid = ref.copy(); mid[:, 1, 0] *= 10 ** (0.05 / 20) * np.exp(1j * np.radians(1.0))
    coarse = ref.copy(); coarse[:, 0, 0] *= 10 ** (1.0 / 20)
    coarse[:, 1, 1] = 0.5                                                          # would be huge if not ignored
    cmp = compare_levels([coarse, mid, ref], ['P1', 'P2'])
    assert cmp[0]['max_db'] == pytest.approx(1.0) and 'S(P2,P2)' not in cmp[0]['params']
    assert cmp[1]['max_db'] == pytest.approx(0.05) and cmp[1]['max_deg'] == pytest.approx(1.0)
    lines, ok = verdict([1, 1.5, 2], cmp, 0.1, 2)
    assert ok and 'converged' in lines[-1]
    assert not verdict([1, 1.5, 2], cmp, 0.01, 2)[1]


def test_preview_converge_end_to_end(demo_files, tmp_path, run_cli):
    base, _ = demo_files
    code, out, err = run_cli(base + '.kicad_pcb', '--pos', base + '-top.pos', '--nets', 'RF,GND', '--ground', 'B.Cu',
                             '--fmin', '1', '--fmax', '8', '--out', tmp_path, '--converge', '1,2', '--preview')
    assert code == 0, err
    assert 'mesh convergence levels:' in out and 'x2 ' in out
