"""A board read back from its board.json gives the same FDTD mesh (the rest of the round trip: kicad2ems's
test_board.py)."""
import numpy as np
import pytest
from kicad2ems.board import Board
from test_board import board_of, quiet                              # kicad2ems's tests (conftest.py puts them on the path)
from kicad2ems_fdtd.mesh import auto_mesh


@pytest.mark.parametrize('case', ['launch', 'demo_lowpass', 'diff'])
def test_mesh_round_trip(case, boards, tmp_path):
    b = board_of(case, boards)
    b2 = Board.load(b.save(str(tmp_path / 'b.json')), log=quiet)
    m1, m2 = (auto_mesh(x, 1e9, 6e9, log=quiet) for x in (b, b2))
    assert all(np.array_equal(m1[k], m2[k]) for k in ('x', 'y', 'z')) and m1['vias'] == m2['vias']
