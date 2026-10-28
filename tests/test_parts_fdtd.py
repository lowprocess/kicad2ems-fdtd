"""Lumped R / L / C parts on the FDTD model: mesh lines at the element ends and the CSXCAD calls (fake openEMS: the
fake_openems fixture; moved from kicad2ems's test_parts.py)."""
import numpy as np
import pytest
from kicad2ems.importers.kicad import KiCadBoard
from kicad2ems.board import Board
from kicad2ems.parts import Part
from kicad2ems_fdtd.mesh import auto_mesh


@pytest.fixture(scope='module')
def rlc(boards):
    return boards['rlc']


def test_part_ends_on_mesh_lines(rlc):
    m = Board(KiCadBoard(rlc + '.kicad_pcb'), nets=['RF1', 'RF2', 'GND'], pos_file=rlc + '-top.pos', log=lambda *a: None)
    mesh = auto_mesh(m, 0.5e9, 6e9, log=lambda *a: None)
    c1 = m.parts[0]
    for v in (c1.s[0], c1.e[0]):
        assert np.min(np.abs(mesh['x'] - v)) < 1e-9
    assert np.sum((mesh['x'] > c1.s[0]) & (mesh['x'] < c1.e[0])) >= 1            # cells across the gap


def test_build_fdtd_adds_lumped_elements(rlc, fake_openems):
    from kicad2ems_fdtd.model import build_fdtd
    from kicad2ems_fdtd.boundaries import resolve_bc
    m = Board(KiCadBoard(rlc + '.kicad_pcb'), nets=['RF1', 'RF2', 'GND'], pos_file=rlc + '-top.pos', log=lambda *a: None)
    m.parts.append(Part('L7', 0.2, np.float64(2e-9), None, 'F.Cu', m.cu_z['F.Cu'], 1, [3, 3, 0.2], [3.2, 3.5, 0.2]))
    m.parts.append(Part('R5', 0.0, None, None, 'F.Cu', m.cu_z['F.Cu'], 0, [4, 3, 0.2], [4.4, 3.5, 0.2]))
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 0.5e9, 6e9, bc=bc, log=lambda *a: None)
    build_fdtd(m, mesh, 0.5e9, 6e9, 0, bc=bc)
    lumped = [x for x in fake_openems if isinstance(x, tuple) and x[0] == 'lumped']
    assert lumped[0] == ('lumped', 'part_C1', {'C': 1.5e-12})
    # series R + L: a resistive piece and an inductor piece (openEMS's series R-L update is unstable)
    assert lumped[1] == ('lumped', 'part_L7_R', {'R': 0.2})
    assert lumped[2] == ('lumped', 'part_L7_X', {'L': 2e-9})                     # L without C: parallel type
    assert len(lumped) == 3 and fake_openems.count('box') == 4                    # C1, L7 (2) + the 0 ohm strap
