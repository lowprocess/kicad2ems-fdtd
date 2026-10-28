"""Progress / ETA: openEMS status-line parsing, the run tracker, the log monitor and output redirection
(no openEMS needed)."""
import os
import pytest
from kicad2ems_fdtd.progress import parse_line, fmt_time, Tracker, Monitor, redirect_output

LINE = "[@     1m32s] Timestep:       265424 || Speed:  377.2 MC/s (3.435e-04 s/TS) || Energy: ~1.56e-19 (-36.27dB)"


def test_parse_line():
    p = parse_line(LINE)
    assert p == dict(t=92.0, step=265424, speed=377.2, db=-36.27)
    assert parse_line("[@        4s] Timestep:   5828 || Speed:  383.4 MC/s (6.871e-04 s/TS) || Energy: ~5.3e-19 (- 0.00dB)")['db'] == 0
    assert parse_line("[@ 1h02m03s] Timestep: 10 || Speed: 1.0 MC/s (1 s/TS) || Energy: ~   inf (-  nandB)")['t'] == 3723
    assert parse_line("Max. number of timesteps: 1000000") is None
    assert fmt_time(5) == '5s' and fmt_time(125) == '2m05s' and fmt_time(3725) == '1h02m' and fmt_time(None) == '?'


def _feed(tr, pts):
    for t, step, db in pts:
        tr.feed(dict(t=t, step=step, speed=100.0, db=db))


def test_eta_from_decay_and_step_limit():
    tr = Tracker('P1', -40, 1_000_000)
    _feed(tr, [(4, 1000, 0.0), (8, 2000, -10.0), (12, 3000, -14.0), (16, 4000, -18.0), (20, 5000, -22.0)])
    eta, by = tr.eta()                                                 # -1 dB/s, 18 dB to go
    assert by == 'decay' and eta == pytest.approx(18, abs=0.5)
    assert tr.fraction() == pytest.approx(22 / 40)
    flat = Tracker('P1', -40, 100_000)                                 # stuck at -24 dB: only the step limit ends it
    _feed(flat, [(4 * i, 1000 * i, -24.0) for i in range(1, 11)])
    eta, by = flat.eta()
    assert by == 'limit' and eta == pytest.approx((100_000 - 10_000) / 250)
    assert flat.flat() and 'energy flat' in flat.status() and '(step limit)' in flat.status()


def test_monitor_bar_and_raw(tmp_path):
    log = tmp_path / 'openems.log'
    log.write_text(LINE + "\n")
    r, w = os.pipe()
    tr = Tracker('P1', -40, 1_000_000)
    mon = Monitor([('P1', str(log), tr)], 'bar', out_fd=w, interval=0.01).start()
    mon.stop()
    os.close(w)
    out = os.read(r, 10000).decode()
    assert 'P1 finished: step 265424, -36.3 dB' in out
    r, w = os.pipe()
    mon = Monitor([('P1', str(log), Tracker('P1')), ('P2', str(log), Tracker('P2'))], 'raw', out_fd=w, interval=0.01)
    mon.start().stop()
    os.close(w)
    out = os.read(r, 10000).decode()
    assert out.count('Timestep:') == 2 and '[P1] ' in out and '[P2] ' in out


def test_redirect_output(tmp_path, capfd):
    log = tmp_path / 'run.log'
    with redirect_output(str(log)) as terminal:
        os.write(1, b"from the solver\n")                             # what C / C++ code writes to stdout
        os.write(2, b"solver warning\n")
        os.write(terminal, b"to the terminal\n")
    assert log.read_text() == "from the solver\nsolver warning\n"
    assert 'to the terminal' in capfd.readouterr().out


def test_status_events_reach_the_events_file(tmp_path):
    """A Monitor feeding Job.status writes JSON 'status' events (openEMS's clock as run_time)."""
    import json
    from kicad2ems.config import Settings
    from kicad2ems.pipeline import Job
    log = tmp_path / 'openems.log'
    log.write_text(LINE + "\n")
    from kicad2ems.events import EventSink
    job = Job(Settings(board='b.kicad_pcb', out=str(tmp_path)), echo=lambda *a: None)
    job.events = EventSink(str(tmp_path / 'ev.jsonl'))                  # what Job.run() opens with --events
    r, w = os.pipe()
    Monitor([('P1', str(log), Tracker('P1'))], 'bar', out_fd=w, interval=0.01, on_status=job.status).start().stop()
    os.close(w)
    job.events.close()
    ev = [json.loads(l) for l in (tmp_path / 'ev.jsonl').read_text().splitlines()]
    st = [e for e in ev if e['kind'] == 'status']
    assert st and st[0]['run'] == 'P1' and st[0]['step'] == 265424 and st[0]['run_time'] == 92.0 and 't' in st[0]
