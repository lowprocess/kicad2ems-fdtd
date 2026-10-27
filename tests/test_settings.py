"""openEMS's own settings (--converge, --converge-tol), read from a config and the command line (from kicad2ems's
tests, where the solvers are stand-ins without them)."""
import pytest
from kicad2ems.cli import settings_from_argv
from kicad2ems.config import build_settings, ConfigError


def test_converge_from_a_config(tmp_path):
    p = tmp_path / 'c.toml'
    p.write_text('board = "b.kicad_pcb"\nconverge = "1,1.5"\nconverge_tol = [0.2, 3]\n')
    s, _ = settings_from_argv([str(p)])
    assert s.converge == [1.0, 1.5] and s.converge_tol == (0.2, 3.0)


def test_empty_converge_means_off(tmp_path):
    p = tmp_path / 'c.toml'
    p.write_text('board = "b.kicad_pcb"\nconverge = []\n')
    assert build_settings({}, str(p)).converge is None


def test_one_level_is_explained():
    with pytest.raises(ConfigError, match='converge'):
        build_settings(dict(board='b.kicad_pcb', converge='1'))
