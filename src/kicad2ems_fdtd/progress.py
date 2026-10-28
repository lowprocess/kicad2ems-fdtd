"""
Progress and ETA for openEMS runs.

openEMS (C++) writes its status to the process's stdout / stderr. During a run, redirect_output() sends
both file descriptors to <run>/openems.log, and a Monitor thread tails the log(s) and writes to the real
terminal either a progress line per run or (--openems-output) the raw openEMS lines. With --jobs the
workers only redirect; the main process monitors all their logs.

openEMS status lines look like
    [@     1m32s] Timestep:   265424 || Speed:  377.2 MC/s (3.435e-04 s/TS) || Energy: ~1.56e-19 (-36.27dB)
A run ends when the energy has fallen by --end-db or after --max-steps timesteps: progress is the
further of the two, and the ETA the sooner (the energy decay rate over the last status lines, the step
rate towards the limit).
"""
import contextlib, os, re, sys, threading, time

_LINE = re.compile(r'\[@\s*([0-9hms]+)\]\s*Timestep:\s*(\d+).*?Speed:\s*([\d.]+)\s*MC/s.*?'
                   r'Energy:\s*~\s*(\S+)\s*\(\s*(-?)\s*([\d.]+|nan|inf)\s*dB\)', re.I)


def _seconds(text):
    """'1h02m03s' / '1m32s' / '4s' -> seconds."""
    total = 0.0
    for v, u in re.findall(r'(\d+)([hms])', text):
        total += int(v) * {'h': 3600, 'm': 60, 's': 1}[u]
    return total


def parse_line(line):
    """dict(t, step, speed, db) of an openEMS status line (t: wall seconds, db: energy in dB, <= 0), or None."""
    mt = _LINE.search(line)
    if not mt:
        return None
    t, step, speed, _, _, db = mt.groups()
    db = -abs(float(db)) if db not in ('nan', 'inf') else float('nan')   # openEMS prints the decay: (-36.27dB)
    return dict(t=_seconds(t), step=int(step), speed=float(speed), db=db)


def fmt_time(s):
    if s is None or s != s or s == float('inf'):
        return '?'
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else (f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


class Tracker:
    """Progress of one run from its status lines."""
    def __init__(self, name, end_db=-40.0, max_steps=1_000_000):
        self.name, self.end_db, self.max_steps = name, end_db, max_steps
        self.points, self.done = [], False
        self._decay = None                                           # (wall time, ETA) of the last decay estimate

    def feed(self, p):
        if p:
            self.points.append(p)

    @property
    def last(self):
        return self.points[-1] if self.points else None

    def fraction(self):
        p = self.last
        if not p:
            return 0.0
        f_db = min(max(p['db'] / self.end_db, 0.0), 1.0) if p['db'] == p['db'] else 0.0
        return max(f_db, min(p['step'] / self.max_steps, 1.0))

    def eta(self):
        """(seconds, by): by 'decay' - the energy trend reaches --end-db (a straight-line fit of dB over time
        to the later half of the status lines since the energy first fell below -3 dB: the slow tail, not
        the first plunge, with the swings of resonant circuits averaged out; 0 when the trend is already
        past --end-db and only those swings keep the run going); by 'limit' - the step rate reaches
        --max-steps, an upper bound used when it is sooner or the energy is not (yet) falling."""
        if len(self.points) < 2:
            return None, ''
        a, b = self.points[max(0, len(self.points) - 6)], self.points[-1]
        dt = b['t'] - a['t']
        rate = (b['step'] - a['step']) / dt if dt > 0 else 0          # steps per second
        limit = (self.max_steps - b['step']) / rate if rate > 0 else None
        tail = [p for p in self.points if p['db'] == p['db']]
        first = next((i for i, p in enumerate(tail) if p['db'] < -3), None)
        decay = None
        if first is not None and len(tail) - first >= 3:
            recent = tail[first:][-max(4, (len(tail) - first) // 2):]
            t = [p['t'] for p in recent]
            db = [p['db'] for p in recent]
            mt, md = sum(t) / len(t), sum(db) / len(db)
            var = sum((x - mt) ** 2 for x in t)
            slope = sum((x - mt) * (y - md) for x, y in zip(t, db)) / var if var > 0 else 0.0
            if slope < -1e-6:
                now = md + slope * (t[-1] - mt)                       # the trend line at the last status
                decay = max((self.end_db - now) / slope, 0.0)
                self._decay = (b['t'], decay)
            elif self._decay:                                         # a swing upwards: keep counting down
                decay = max(self._decay[1] - (b['t'] - self._decay[0]), 0.0)
        if decay is not None and (limit is None or decay < limit):
            return decay, 'decay'
        return limit, 'limit'

    def status(self, width=20):
        p = self.last
        if self.done:
            return (f"{self.name} finished: step {p['step']}, {p['db']:.1f} dB after {fmt_time(p['t'])}" if p
                    else f"{self.name} finished")
        if not p:
            return f"{self.name} starting"
        f = self.fraction()
        bar = '#' * int(round(f * width)) + '-' * (width - int(round(f * width)))
        eta, by = self.eta()
        db = f"{p['db']:.1f}" if p['db'] == p['db'] else 'nan'
        return (f"{self.name} [{bar}] {f * 100:3.0f}%  step {p['step']}  {db}/{self.end_db:g} dB  "
                f"{p['speed']:.0f} MC/s  ETA " + (f"<={fmt_time(eta)} (step limit)" if by == 'limit' and eta
                                                  else 'soon' if by == 'decay' and eta < 1 else fmt_time(eta))
                + ('  energy flat: a static field? (see --end-db)' if self.flat() else ''))

    def flat(self, n=8, tol=0.2):
        """The energy has stayed within tol dB over the last n status lines, below -3 dB and above --end-db: a
        static field left by the excitation's DC part (sweeps starting near 0 Hz), which never decays."""
        last = [p['db'] for p in self.points[-n:]]
        return (len(last) == n and all(v == v for v in last) and max(last) - min(last) < tol
                and -3 > last[-1] > self.end_db)


class Monitor:
    """Thread tailing openEMS logs: runs = [(label, log path, Tracker)]. mode 'bar' (progress lines) or
    'raw' (the openEMS lines, prefixed with the label when there are several runs). out_fd: where to write
    (the terminal, also while stdout is redirected). on_status(label, status): called for every status
    line with the parsed values plus the run's progress fraction and ETA (seconds or None)."""
    def __init__(self, runs, mode='bar', out_fd=None, interval=0.25, plain_every=30.0, on_status=None):
        self.runs, self.mode, self.interval, self.on_status = runs, mode, interval, on_status
        if out_fd is None:
            try:
                out_fd = sys.stdout.fileno()
            except (AttributeError, OSError, ValueError):            # a stdout without a descriptor (captured)
                out_fd = None
        self.out_fd = out_fd
        self.tty = out_fd is not None and os.isatty(out_fd)
        self.plain_every, self._last_plain = plain_every, 0.0
        self._pos = [0] * len(runs)                                  # read position per run
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._width = 0

    def write(self, text):
        try:
            if self.out_fd is None:
                sys.stdout.write(text)
                sys.stdout.flush()
            else:
                os.write(self.out_fd, text.encode('utf-8', 'replace'))
        except (OSError, ValueError):
            pass

    def _read(self):
        for i, (label, path, tr) in enumerate(self.runs):
            try:
                with open(path, 'rb') as fh:
                    fh.seek(self._pos[i])
                    chunk = fh.read()
            except OSError:
                continue
            cut = chunk.rfind(b'\n') + 1                            # an incomplete last line waits for the next read
            self._pos[i] += cut
            lines = chunk[:cut].decode('utf-8', 'replace').splitlines()
            for line in lines:
                p = parse_line(line)
                tr.feed(p)
                if p and self.on_status:
                    try:
                        self.on_status(label, dict(p, fraction=tr.fraction(), eta=tr.eta()[0]))
                    except Exception:                                # a listener must never stop the monitor
                        self.on_status = None
                if self.mode == 'raw':
                    self.write((f"[{label}] " if len(self.runs) > 1 else '') + line + "\n")

    def render(self, final=False):
        if self.mode != 'bar':
            return
        line = " | ".join(tr.status() for _, _, tr in self.runs)
        if self.tty:
            pad = max(self._width - len(line), 0)
            self._width = len(line)
            self.write("\r" + line + " " * pad + ("\n" if final else ""))
        elif final or time.time() - self._last_plain >= self.plain_every:
            self._last_plain = time.time()
            self.write(line + "\n")

    def _loop(self):
        while not self._stop.is_set():
            self._read()
            self.render()
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._read()
        for _, _, tr in self.runs:
            tr.done = True
        self.render(final=True)


@contextlib.contextmanager
def redirect_output(path):
    """Send this process's stdout and stderr file descriptors (C / C++ output included) to `path`
    (appended). Yields a duplicate of the original stdout descriptor for writing to the terminal."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    fh = open(path, 'ab', buffering=0)
    os.dup2(fh.fileno(), 1)
    os.dup2(fh.fileno(), 2)
    try:
        yield saved_out
    finally:
        try:
            import ctypes
            ctypes.CDLL(None).fflush(None)                           # C stdio buffers (std::cout) first
        except (OSError, AttributeError, TypeError):
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        fh.close()
