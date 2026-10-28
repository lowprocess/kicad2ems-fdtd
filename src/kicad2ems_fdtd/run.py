"""
FDTD runs: one run per excited port (in parallel with --jobs), each in its own folder
<out>/<engine>/exc_<port>; the end criteria (PortWatch), progress, and the S-parameter columns from the
port probes. Any engine with openEMS's interface (engines.py) runs the same way.
"""
import glob, json, math, os, threading, time
import numpy as np
from .engines import engine_api, engine_name, engine_module
from .model import build_fdtd, pulse_time


def _times(fn, n=None):
    """First column of an openEMS text probe file (comment lines start with %); the first n values, or all."""
    out = []
    with open(fn) as fh:
        for line in fh:
            if line.strip() and not line.startswith('%'):
                out.append(float(line.split()[0]))
                if n and len(out) >= n:
                    break
    return out



def steps_taken(path, dt_ps=None):
    """Timesteps a finished run took: the last sample time of its port voltage probe divided by openEMS's
    time step, read from the excitation signal file 'et' (one sample per step; the mesher's estimate dt_ps,
    used only without 'et', can be ~10 % off). The probes are subsampled, so this may undercount by a few
    dozen steps. None if there is no probe file."""
    et = os.path.join(path, 'et')
    dt = None
    if os.path.exists(et):
        t = _times(et, 3)
        if len(t) >= 3 and t[2] > t[1]:
            dt = t[2] - t[1]
    if dt is None and dt_ps:
        dt = dt_ps * 1e-12
    for fn in sorted(glob.glob(os.path.join(path, 'port_ut*'))):
        t = _times(fn)
        if t and dt:
            return int(round(t[-1] / dt))
    return None



def step_limit_warning(name, steps, max_steps, end_db):
    """Warning text when a run stopped at --max-steps instead of decaying to --end-db (None otherwise)."""
    if steps is None or steps < 0.97 * max_steps:
        return None
    return (f"WARNING: {name} ran to the step limit (~{steps} of --max-steps {max_steps}) before the energy fell by "
            f"{-end_db:g} dB - the fields may still be ringing; raise --max-steps (or accept a higher --end-db)")



DEAD_DB = -120.0         # PortWatch: port signals this far below their peak end a run at once



class PortWatch:
    """Thread that ends a run whose field energy has stalled once every port signal has died away.
    openEMS's own end criterion is the total field energy, which can level off above --end-db when some
    energy sits where no port sees it (a static field, a mode the ports do not couple to) while the
    S-parameters are long settled. Every `interval` s it reads the port voltage / current probes
    (port_ut* / port_it*, which openEMS flushes as it runs) and the energy from openEMS's status lines
    (openems.log). It writes ABORT into the run folder (openEMS stops and finishes normally) and records
    why in run_end.json when all of these hold:
      - the time is past t_min (the excitation pulse) and the largest u^2 + (z i)^2 of any port over the
        last `window` s is end_db below its peak;
      - the energy has stalled: over the last quarter of the steps so far (and at least 8 status lines)
        it fell by less than `stall_db`.
    The second condition matters: in a high-Q circuit (a filter) the stored energy leaks out through
    the ports slowly, the port signals sit far below their peak while it does, and stopping then cuts
    the ring-down off (the response comes out smoothed and shifted). Port signals below DEAD_DB end it
    without the stall: nothing reaches the ports any more (a radial stub held the pulse's DC charge at
    -33 dB of energy, its ports at -222 dB, for 176k timesteps where 20k were enough)."""
    def __init__(self, path, end_db, t_min, window, z=50.0, interval=2.0, stall_db=1.5):
        self.path, self.end_db, self.t_min, self.window, self.z, self.interval = path, end_db, t_min, window, z, interval
        self.stall_db, self.energy, self._pos = stall_db, [], 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.fired = None

    def _read_energy(self):
        from .progress import parse_line
        try:
            with open(os.path.join(self.path, 'openems.log'), 'rb') as fh:
                fh.seek(self._pos)
                chunk = fh.read()
        except OSError:
            return
        cut = chunk.rfind(b'\n') + 1
        self._pos += cut
        for line in chunk[:cut].decode('utf-8', 'replace').splitlines():
            p = parse_line(line)
            if p and p['db'] == p['db']:
                self.energy.append((p['step'], p['db']))

    def stalled(self):
        """The energy fell by less than stall_db over the last quarter of the steps (>= 8 status lines)."""
        if len(self.energy) < 8:
            return False
        last = self.energy[-1][0]
        recent = [db for st, db in self.energy if st >= 0.75 * last]
        if len(recent) < 4:
            recent = [db for _, db in self.energy[-4:]]
        return max(recent) - min(recent) < self.stall_db

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def level(self):
        """(time of the last sample s, level of the last window below the peak in dB) or None."""
        return port_decay(self.path, self.window, self.z)

    def _loop(self):
        from kicad2ems.api.solver import stop
        while not self._stop.wait(self.interval):
            if stop.requested():                                # the GUI's Stop (stop.py)
                self.fired = dict(reason='stop')
                open(os.path.join(self.path, 'ABORT'), 'w').close()
                return
            try:
                lv = self.level()
            except Exception:                                   # a half-written file: try again later
                continue
            self._read_energy()
            if lv and lv[0] > self.t_min + self.window and lv[1] <= self.end_db and (self.stalled() or lv[1] <= DEAD_DB):
                self.fired = dict(reason='ports', t_ns=lv[0] * 1e9, level_db=lv[1],
                                  energy_db=self.energy[-1][1] if self.energy else None)
                with open(os.path.join(self.path, 'run_end.json'), 'w') as fh:
                    json.dump(self.fired, fh)
                open(os.path.join(self.path, 'ABORT'), 'w').close()
                return



def watch_window(m, fmin):
    """PortWatch's window (s): two periods of the slowest ringing the ports can see. That is fmin's, except on a
    board without lumped parts on its signal nets: nothing there rings below its lowest cavity mode, c0 / (2 D sqrt(er)) over the
    outline's diagonal D, so a band reaching far below it (fmin 0.1 GHz: a 20 ns window, longer than a whole
    run on a 6-layer board) waits no longer than that mode's two periods. At least 1 ns."""
    from kicad2ems.api.solver import C0
    w = 2.0 / fmin
    gnd = set(getattr(m, 'ground_nets', ()) or ())
    signal_parts = [pt for pt in getattr(m, 'parts', []) or [] if any(n and n not in gnd for n in _part_nets(m, pt))]
    if not signal_parts:                                        # (decoupling between two ground nets does not count)
        x0, y0, x1, y1 = m.outline.bounds
        er = max([l['er'] for l in m.layers] + [1.0])
        f_low = C0 / (2 * np.hypot(x1 - x0, y1 - y0) * 1e-3 * np.sqrt(er))
        w = min(w, 2.0 / f_low)
    return max(w, 1e-9)



def _part_nets(m, pt, reach=0.1):
    """The nets of the copper at a lumped part's two ends (its pads, within `reach` mm of the ends)."""
    from shapely.geometry import Point
    ps, nets = m.copper.get(pt.layer, []), getattr(m, 'copper_nets', {}).get(pt.layer, [])
    out = []
    for c in (pt.s, pt.e):
        q = Point(float(c[0]), float(c[1]))
        hit = [(g.distance(q), n) for g, n in zip(ps, nets) if g.distance(q) <= reach]
        out.append(min(hit)[1] if hit else None)
    return out



def port_decay(path, window, z=50.0):
    """(last probe time s, max over ports of the signal power in the last `window` s relative to its peak,
    dB) from the port probe files of a run folder, or None when there are none yet."""
    peak, last, tmax = 0.0, 0.0, None
    for fu in sorted(glob.glob(os.path.join(path, 'port_ut*'))):
        fi = fu.replace('port_ut', 'port_it')
        u = np.loadtxt(fu, comments='%', ndmin=2)
        i = np.loadtxt(fi, comments='%', ndmin=2) if os.path.exists(fi) else None
        n = len(u) if i is None else min(len(u), len(i))
        if n < 4:
            return None
        t = u[:n, 0]
        pw = u[:n, 1] ** 2 + ((z * i[:n, 1]) ** 2 if i is not None else 0.0)
        peak = max(peak, float(pw.max()))
        tmax = t[-1] if tmax is None else min(tmax, t[-1])
        last = max(last, float(pw[t >= t[-1] - window].max()))
    if tmax is None or peak <= 0:
        return None
    return float(tmax), 10 * math.log10(max(last, 1e-300) / peak)



def execute(FDTD, path, threads=0, verbose=1, progress=None, watch=None):
    """FDTD.Run in `path`, restoring the working directory (openEMS changes it). Returns seconds taken.
    progress: None (openEMS writes to the terminal), or dict(mode, label, end_db, max_steps[, on_status]):
    openEMS's output goes to <path>/openems.log and, for mode 'bar' / 'raw', a progress line / the raw
    lines to the terminal ('file': the log only - parallel workers, monitored by the main process);
    on_status(label, status dict) is called for every openEMS status line."""
    from .progress import Monitor, Tracker, redirect_output
    from kicad2ems.api.solver import stop
    stop.check()                                                # stopped: no further excitation or level
    os.makedirs(path, exist_ok=True)
    for f in ('ABORT', 'run_end.json'):                         # leftovers of an earlier run in this folder
        if os.path.exists(os.path.join(path, f)):
            os.remove(os.path.join(path, f))
    cwd, t1 = os.getcwd(), time.time()
    pw = PortWatch(path, **watch).start() if watch else None
    try:
        if progress is None:
            FDTD.Run(path, cleanup=True, numThreads=threads, verbose=verbose)
        elif getattr(FDTD, 'writes_log', False):                 # gfdtd: its output to the log itself (runs in
            log = log_path(path)                                    # threads share the process's stdout)
            open(log, 'w').close()
            mon = None
            if progress['mode'] in ('bar', 'raw'):
                tr = Tracker(progress['label'], progress['end_db'], progress['max_steps'])
                mon = Monitor([(progress['label'], log, tr)], progress['mode'],
                              on_status=progress.get('on_status')).start()
            try:
                FDTD.Run(path, cleanup=True, numThreads=threads, verbose=max(verbose, 1), log=log)
            finally:
                if mon:
                    mon.stop()
        else:
            log = log_path(path)
            open(log, 'w').close()
            with redirect_output(log) as terminal:
                mon = None
                if progress['mode'] in ('bar', 'raw'):
                    tr = Tracker(progress['label'], progress['end_db'], progress['max_steps'])
                    mon = Monitor([(progress['label'], log, tr)], progress['mode'], out_fd=terminal,
                                  on_status=progress.get('on_status')).start()
                try:
                    FDTD.Run(path, cleanup=True, numThreads=threads, verbose=max(verbose, 1))
                finally:
                    if mon:
                        mon.stop()
    finally:
        os.chdir(cwd)
        if pw:
            pw.stop()
    if os.path.exists(os.path.join(path, 'ABORT')):
        os.remove(os.path.join(path, 'ABORT'))
    return time.time() - t1



def log_path(path):
    """openEMS's output of a run (written when the run has a progress setting)."""
    return os.path.join(path, 'openems.log')



class RunResult:
    """Result of run_excitations: S (nf, n, n) over the single-ended ports (NaN in the columns of ports not
    excited), dump_files {(port, dump name): path}, dump_norm {port: {freq: incident voltage}} (field maps
    scaled to a 1 V incident wave), seconds / steps {port: value} per run, extra {port: what the run's
    `analyse` hook returned} (antenna mode: the far-field results)."""
    def __init__(self, S):
        self.S, self.dump_files, self.dump_norm, self.seconds, self.steps, self.extra = S, {}, {}, {}, {}, {}
        self.ended = {}                             # {port: dict(reason='ports', t_ns, level_db)} when PortWatch ended it



def run_dir(m, k, out_dir, engine=None):
    """Folder of the run that excites port k: <out>/openems/exc_<port> (P1+ / P1- -> exc_P1p / exc_P1n); another
    engine's under <out>/<engine>/."""
    tag = m.ports[k].name.replace('+', 'p').replace('-', 'n')
    return os.path.abspath(os.path.join(out_dir, engine_name(engine)[0], f"exc_{tag}"))



def _run_port(k, m, mesh, freq, bc, out_dir, end_db=-40, sigma=None, threads=0, xml_path=None, dumps=(),
              verbose=1, diel='fixed', diel_fref=1e9, max_steps=1_000_000, progress=None, exc_fmax=None,
              setup=None, analyse=None, end_on='ports', engine=None):
    """One openEMS run with port k excited (top level, so it can run in a worker process); engine: another FDTD
    engine with openEMS's interface (engine_api: 'gfdtd'), run the same way. setup(FDTD, CSX)
    may add to the model before the run (antenna mode: the NF2FF box) and its return value is handed to
    analyse(path, ports, token, k, col) after the run. Returns (k, S column (nf, n), {(port, dump): path},
    {freq: incident voltage} or None, seconds, steps taken, what analyse returned or None)."""
    fmin, fmax = freq[0], freq[-1]
    path = run_dir(m, k, out_dir, engine)
    FDTD, CSX, ports = build_fdtd(m, mesh, fmin, fmax, k, bc=bc, end_db=end_db, sigma=sigma, dumps=dumps,
                                  diel=diel, diel_fref=diel_fref, max_steps=max_steps, exc_fmax=exc_fmax,
                                  api=engine_api(engine))
    token = setup(FDTD, CSX) if setup else None
    os.makedirs(path, exist_ok=True)
    if dumps:
        from kicad2ems.api.solver import write_info
        write_info(path, dumps, m, mesh['info'].get('dt_ps'))
    else:                                                       # an earlier run's dumps are gone (openEMS cleans up)
        for f in ('dump_info.json', 'dump_norm.json'):
            if os.path.exists(os.path.join(path, f)):
                os.remove(os.path.join(path, f))
    if xml_path:
        CSX.Write2XML(os.path.abspath(xml_path))
    if progress is not None:
        progress = dict(progress, label=m.ports[k].name, end_db=end_db, max_steps=max_steps)
    watch = None
    if end_on == 'ports':
        fe = fmax if exc_fmax is None else exc_fmax
        watch = dict(end_db=end_db, t_min=pulse_time(fmin, fe), window=watch_window(m, fmin), z=m.ports[k].z0)
    seconds = execute(FDTD, path, threads, verbose, progress, watch=watch)
    files, norm = {}, None
    for d in dumps:
        files[(m.ports[k].name, d.name)] = (os.path.join(path, d.name + '.h5') if d.freqs
                                            else os.path.join(path, d.name + '_*.vtr'))
    fd = sorted({f for d in dumps for f in (d.freqs or [])})
    if fd:                                                      # incident wave at the dump frequencies
        ports[k].CalcPort(path, np.array(fd))
        norm = dict(zip(fd, np.atleast_1d(ports[k].uf_inc)))
        from kicad2ems.api.solver import write_norm
        write_norm(path, norm)
    for p in ports:
        p.CalcPort(path, freq)
    if not np.any(np.abs(ports[k].uf_inc) > 0):                 # nothing went in: no S column to compute
        raise RuntimeError(f"port {m.ports[k].name} was not excited (no incident wave: its box catches no mesh "
                           f"edges; see the run's openems.log, 'Voltage excitations')")
    col = np.array([p.uf_ref / ports[k].uf_inc for p in ports]).T
    extra = analyse(path=path, ports=ports, token=token, k=k, col=col) if analyse else None
    return k, col, files, norm, seconds, steps_taken(path, mesh['info']['dt_ps']), extra



def run_excitations(m, mesh, freq, excite, bc, out_dir, end_db=-40, sigma=None, threads=0, xml_path=None,
                    dumps=(), dump_runs=None, jobs=1, diel='fixed', diel_fref=1e9, max_steps=1_000_000, progress=None,
                    exc_fmax=None, setup=None, analyse=None, log=print, end_on='ports', engine=None):
    """One openEMS run per excited port, `jobs` of them at a time (separate processes; without `threads`,
    half the logical CPUs are shared between them; gfdtd: threads of this process, on the GPU a stream each). Field dumps are added to the runs in dump_runs (port
    indices; default: the first excited port). progress: None or dict(mode='bar' | 'raw'[, on_status]) -
    see execute(). setup / analyse: hooks of every run (see _run_port; picklable for jobs > 1). engine: see
    _run_port. Returns a RunResult."""
    folder, who = engine_name(engine)
    dump_runs = [excite[0]] if dump_runs is None else dump_runs
    n = len(m.ports)
    res = RunResult(np.full((len(freq), n, n), np.nan + 0j))
    jobs = max(1, min(int(jobs or 1), len(excite)))
    if jobs > 1 and not threads:
        # FDTD is memory-bound: half the logical CPUs (no hyper-thread pairs) shared by the runs; giving each
        # run cpu_count / jobs threads oversubscribes the cores and made every run ~4x slower
        threads = max(1, (os.cpu_count() or 2) // 2 // jobs)
    tasks = [dict(k=k, dumps=dumps if k in dump_runs else (), xml_path=xml_path if k == excite[0] else None)
             for k in excite]

    def collect(out):
        k, col, files, norm, seconds, steps, extra = out
        name = m.ports[k].name
        res.S[:, :, k] = col
        res.dump_files.update(files)
        res.seconds[name], res.steps[name] = seconds, steps
        if norm is not None:
            res.dump_norm[name] = norm
        if extra is not None:
            res.extra[name] = extra
        for (pn, dn), path in files.items():
            if not glob.glob(os.path.join(os.path.dirname(path), dn + '*')):
                log(f"  WARNING: no output found for field dump {dn}")
        log(f"  {name} done in {seconds:.0f} s" + (f", ~{steps} timesteps" if steps else ''))
        end = os.path.join(run_dir(m, k, out_dir, engine), 'run_end.json')
        if os.path.exists(end):
            with open(end) as fh:
                res.ended[name] = json.load(fh)
            log(f"  {name}: stopped when the port signals had decayed by {-res.ended[name]['level_db']:.1f} dB "
                f"(at {res.ended[name]['t_ns']:.2f} ns; the field energy had not reached --end-db)")
        w = step_limit_warning(name, steps, max_steps, end_db)
        if w:
            log("  " + w)

    common = dict(m=m, mesh=mesh, freq=freq, bc=bc, out_dir=out_dir, end_db=end_db, sigma=sigma, threads=threads,
                  diel=diel, diel_fref=diel_fref, max_steps=max_steps, exc_fmax=exc_fmax, setup=setup, analyse=analyse,
                  end_on=end_on, engine=engine)
    if jobs == 1:
        for t in tasks:
            log(f"{who}: exciting {m.ports[t['k']].name} -> {run_dir(m, t['k'], out_dir, engine)}")
            collect(_run_port(**t, **common, progress=progress))
        return res
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from .progress import Monitor, Tracker
    in_threads = getattr(engine_module(engine), 'IN_THREADS', False)   # one process: the runs share it (gfdtd: the GPU)
    log(f"{who}: {len(tasks)} excitation(s), {jobs} at a time" + ('' if in_threads else f", {threads} thread(s) each") +
        f" ({who} output in openems.log of each run under {os.path.join(out_dir, folder)})")
    mon = None
    if progress is not None:
        for t in tasks:                                             # fresh logs, so the monitor never reads old ones
            os.makedirs(run_dir(m, t['k'], out_dir, engine), exist_ok=True)
            open(log_path(run_dir(m, t['k'], out_dir, engine)), 'w').close()
        mon = Monitor([(m.ports[t['k']].name, log_path(run_dir(m, t['k'], out_dir, engine)),
                        Tracker(m.ports[t['k']].name, end_db, max_steps)) for t in tasks], progress['mode'],
                      on_status=progress.get('on_status')).start()
    try:
        pool = ThreadPoolExecutor(jobs) if in_threads else \
            ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn'))
        with pool as ex:
            futs = [ex.submit(_run_port, **t, **common, verbose=1, progress=dict(mode='file')) for t in tasks]
            outs = [f.result() for f in as_completed(futs)]
    finally:
        if mon:
            mon.stop()
    for out in outs:                                                # after the progress lines: clean output
        collect(out)
    return res
