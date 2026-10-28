"""--max-steps: excitation length estimate, steps taken read back from a run folder, the step-limit
warning and the CLI / TOML option (no openEMS needed)."""
import pytest
from kicad2ems.cli import settings_from_argv
from kicad2ems_fdtd.model import pulse_steps
from kicad2ems_fdtd.run import steps_taken, step_limit_warning


def test_pulse_steps():
    # openEMS: Gaussian length 2 * 9 / (2 pi fc), fc = 0.5 (fmax - fmin) 1.05; filter example: 1-8 GHz, dt 57.2 fs
    # -> 1e6 steps = 73.4 pulse lengths (from the openEMS log)
    assert 1e6 / pulse_steps(1e9, 8e9, 0.0572) == pytest.approx(73.4, rel=0.01)
    assert pulse_steps(0.05e9, 0.5e9, 0.0572) > 9 * pulse_steps(1e9, 8e9, 0.0572)   # low frequencies: longer


def _run_folder(path, dt, steps, every=60):
    path.mkdir()
    (path / 'et').write_text("".join(f"{i * dt:.6e}\t0\n" for i in range(50)))
    (path / 'port_ut_1').write_text("% probe\n" + "".join(f"{i * dt:.9e}\t0\n" for i in range(0, steps, every)))


def test_steps_taken_and_warning(tmp_path):
    _run_folder(tmp_path / 'a', 4.8e-13, 12000)
    n = steps_taken(str(tmp_path / 'a'), dt_ps=0.53)                 # 'et' wins over the mesher's estimate
    assert 11900 <= n <= 12000
    assert step_limit_warning('P1', n, 1_000_000, -40) is None
    w = step_limit_warning('P1', n, 12000, -40)
    assert w and 'step limit' in w and '--max-steps 12000' in w
    assert steps_taken(str(tmp_path)) is None                         # no probe file


def test_cli_and_toml(tmp_path):
    assert settings_from_argv(['b.kicad_pcb'])[0].max_steps == 1_000_000
    assert settings_from_argv(['b.kicad_pcb', '--max-steps', '5000000'])[0].max_steps == 5_000_000
    cfg = tmp_path / 'c.toml'
    cfg.write_text('[solver.openems]\nmax-steps = 3000000\n')
    s, _ = settings_from_argv(['b.kicad_pcb', '--config', str(cfg)])
    assert s.notes == [] and s.max_steps == 3_000_000
