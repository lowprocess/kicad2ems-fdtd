"""build_fdtd against strict stand-ins for CSXCAD / openEMS: every scalar handed over must be a plain
Python number (older CSXCAD releases test type(v) in (int, float) and reject numpy scalars)."""
import numpy as np
import pytest
from kicad2ems.importers.kicad import KiCadBoard
from kicad2ems.board import Board
from kicad2ems_fdtd.boundaries import resolve_bc
from kicad2ems_fdtd.mesh import auto_mesh


@pytest.mark.parametrize('sigma', [None, 5.8e7])
def test_build_fdtd_passes_plain_scalars(demo_files, fake_openems, sigma):
    from kicad2ems_fdtd.model import build_fdtd
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, log=lambda *a: None)
    freq = np.linspace(1e9, 8e9, 11)                                # numpy scalars, as in a real run
    build_fdtd(m, mesh, freq[0], freq[-1], 0, bc=bc, sigma=np.float64(sigma) if sigma else None)
    assert fake_openems.count('linpoly') == len(m.layers) and fake_openems.count('port') == 2
    assert fake_openems.count('cyl') == len(m.vias)


def test_port_planes_land_on_mesh_lines(demo_files, fake_openems):
    """A port plane a hair off its mesh line (two differential legs differing by 1e-12 mm share one merged
    line) is put exactly on the line: openEMS drops a zero-thickness port that misses it, unexcited."""
    from kicad2ems_fdtd.model import build_fdtd
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, log=lambda *a: None)
    p = m.ports[0]
    p.c = p.c + np.eye(2)[p.axis] * 1e-9                          # off its line by far less than any cell
    build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc)
    boxes = [x for x in fake_openems if isinstance(x, tuple) and x[0] == 'port_box']
    s, e = boxes[0][1], boxes[0][2]
    assert s[p.axis] == e[p.axis] and s[p.axis] in set(np.asarray(mesh['xy'[p.axis]], float).tolist())


def test_build_fdtd_wideband_dielectric(demo_files, fake_openems):
    from kicad2ems_fdtd.model import build_fdtd
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, log=lambda *a: None)
    build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, diel='wideband', diel_fref=1e9)
    debye = [x for x in fake_openems if isinstance(x, tuple) and x[0] == 'debye']
    assert len(debye) == len(m.layers) and all(order >= 3 for _, _, order in debye)
    assert fake_openems.count('linpoly') == len(m.layers)


def test_port_watch_stops_a_decayed_run(tmp_path):
    """PortWatch writes ABORT once every port signal is end_db below its peak after the pulse, and records
    why; a signal still ringing keeps the run going."""
    import json, time, numpy as np
    from kicad2ems_fdtd.run import PortWatch, port_decay, pulse_time
    t = np.arange(0, 20e-9, 10e-12)
    ring = np.exp(-((t - 1e-9) / 0.3e-9) ** 2) + 1e-3 * np.sin(2 * np.pi * 3e9 * t)     # -60 dB ripple: still on
    for name, sig in (('port_ut_1', ring), ('port_it_1', ring / 50), ('port_ut_2', 0.5 * ring), ('port_it_2', ring / 100)):
        np.savetxt(tmp_path / name, np.column_stack([t, sig]), header='t u', comments='% ')
    level = port_decay(str(tmp_path), 2e-9)
    assert level[0] == pytest.approx(t[-1]) and -62 < level[1] < -55
    # the field energy still falling (a ringing filter): the port level alone does not stop the run
    log = "".join(f"[@        {k}s] Timestep:         {1000 * k} || Speed:  100.0 MC/s (1e-3 s/TS) || Energy: ~1e-15 (-{10 + 2 * k:.2f}dB)\n"
                  for k in range(1, 13))
    (tmp_path / 'openems.log').write_text(log)
    w = PortWatch(str(tmp_path), end_db=-40, t_min=2e-9, window=2e-9, interval=0.05).start()
    time.sleep(0.4)
    w.stop()
    assert not (tmp_path / 'ABORT').exists()
    # stalled energy (the SMA launch: a plateau far above end_db): the ports decide
    log = "".join(f"[@        {k}s] Timestep:         {1000 * k} || Speed:  100.0 MC/s (1e-3 s/TS) || Energy: ~1e-15 (-{36 + 0.05 * k:.2f}dB)\n"
                  for k in range(1, 13))
    (tmp_path / 'openems.log').write_text(log)
    w = PortWatch(str(tmp_path), end_db=-40, t_min=2e-9, window=2e-9, interval=0.05).start()
    time.sleep(0.4)
    w.stop()
    assert (tmp_path / 'ABORT').exists()
    assert json.loads((tmp_path / 'run_end.json').read_text())['reason'] == 'ports'
    (tmp_path / 'ABORT').unlink()
    w = PortWatch(str(tmp_path), end_db=-70, t_min=2e-9, window=2e-9, interval=0.05).start()
    time.sleep(0.3)
    w.stop()
    assert not (tmp_path / 'ABORT').exists()                        # -60 dB is not -70 dB
    assert pulse_time(1e9, 6e9) == pytest.approx(2 * 9 / (2 * np.pi * 2.625e9))


def test_stop_request_starts_no_further_openems_run(tmp_path, monkeypatch):
    """After the GUI's Stop (kicad2ems.stop) no further excitation or convergence level starts."""
    from kicad2ems import stop
    from kicad2ems_fdtd.run import execute
    monkeypatch.delenv(stop.ENV, raising=False)
    stop.arm(tmp_path)
    ran = []

    class FDTD:
        def Run(self, path, **kw):
            ran.append(path)
    execute(FDTD(), str(tmp_path / 'exc_1'))
    stop.request(tmp_path)
    with pytest.raises(stop.Stopped):
        execute(FDTD(), str(tmp_path / 'exc_2'))
    assert ran == [str(tmp_path / 'exc_1')]
    stop.disarm()
    assert not stop.requested() and not (tmp_path / stop.NAME).exists()
