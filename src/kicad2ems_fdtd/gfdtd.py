"""
gfdtd as a kicad2ems solver plugin and FDTD engine (engines.py): the FDTD model (plugin.py) solved by gfdtd's
engine (the GPU via CuPy or Vulkan, its compiled CPU core, or numpy) through its CSXCAD / openEMS interface. The board, the mesh (--res, --fine, --nz, ...), the
ports, the parts and the end criteria are the FDTD settings shared with openEMS; only the engine differs. Runs go
to <out>/gfdtd/exc_<port>, the CSX file to <name>_gfdtd.xml (also runnable with `python -m gfdtd`), the antenna
results to <name>_gfdtd_antenna.json.

--sigma: openEMS's conducting-sheet model on gfdtd's engine API 1.1 or later (LOSS_API); before that its
copper is PEC, and --solvers auto gives --sigma runs to openEMS. --rough: the sheets' own (Hammerstad, a wideband
fit) from engine API 1.2 (ROUGH_API), else as the conductivity at the band centre, as openEMS. --converge: the
FDTD layer's mesh levels, run on gfdtd.
"""
import os

from kicad2ems.api.solver import Option, AUTO
from .plugin import FDTD, MESH, RUN, CONVERGE

G = 'gfdtd'                                     # its section in kicad2ems's GUI
ENGINE_API = (1, 0)                             # the gfdtd API this connection is written against (gfdtd.API_VERSION)
LOSS_API = (1, 1)                               # from this gfdtd API on, conducting sheets are lossy (openEMS's model)
ROUGH_API = (1, 2)                              # ... and rough (ConductingSheet(roughness=), a wideband fit)
FINISH_API = (1, 3)                             # ... and finished, and thick copper's faces (finish=, face=)
# gfdtd's speed over openEMS's estimate (24 threads), per backend: the examples' runs on an RX 7800 XT and an
# i7-13700K, 8-20x on the GPU and 2-2.3x on the core (docs/gfdtd.md#speed); numpy ~10x slower than openEMS.
# Vulkan runs the same kernels within a few per cent of CuPy's speed there. START: seconds of setup a run (the GPU
# kernels, the core's threads)
SPEEDUP, START = dict(cupy=8.0, vulkan=8.0, core=2.0, numpy=0.1), 2.0
GPU_BACKENDS = ('cupy', 'vulkan')


class GFDTD(FDTD):
    api = (2, 9)                    # kicad2ems.api (2.4: resolve_finish, the finish row; 2.9: basic)
    name, aliases, label = 'gfdtd', ('gpu-fdtd',), 'gfdtd (GPU FDTD)'
    in_auto = True                  # auto: picked over openEMS where it can (no --drive; --sigma: LOSS_API)
    in_all = False                  # --solvers all: openEMS's model again, named on its own
    previewed = False
    order = 40                      # after openEMS and mom25d in kicad2ems's lists
    tag = '_gfdtd'
    capabilities = dict(FDTD.capabilities,
                        conductor_loss="yes: openEMS's conducting-sheet model (gfdtd's engine API 1.1 or later; PEC before)",
                        roughness="yes: Hammerstad over the band, a wideband fit of the sheets (gfdtd's engine API "
                                  "1.2 or later; else at the band centre)",
                        finish="yes with --thickness on: the outer faces' own sheets, a wideband fit; else the outer "
                               "layers' sheets weighted by the strip's finished share (gfdtd's engine API 1.3 or later; else "
                               "at the band centre)",
                        dispersion='yes: fitted at mid-band, or wideband Debye',
                        antenna='yes (a phased drive: a run per driven port)')
    options = MESH + CONVERGE + RUN + [
        Option('gfdtd_backend', 'auto', "gfdtd: where it runs - auto (the GPU when Vulkan or CuPy finds one, else the "
               "CPU), gpu (Vulkan through slangpy, else CuPy; CuPy first on NVIDIA CUDA) or cpu (the compiled core: "
               "python -m gfdtd.core.build; else numpy, slow); GFDTD_BACKEND=cupy|vulkan picks the GPU's; its "
               "mesh and run settings are the FDTD ones shared with openEMS (--res, --end-db, ...)", choices=('auto', 'gpu', 'cpu'), name='backend', gui='Runs on', group=G),
        Option('gfdtd_gpus', '', "gfdtd on Vulkan: the GPUs its excitations are shared out to, a process each - all "
               "(every discrete GPU with the kernels' features) or their indices (0,1); default: one GPU. The runs go "
               "by each GPU's speed measured on earlier runs (each to the GPU that would finish it first; one never "
               "measured is tried): a GPU too slow to help gets none, and one GPU alone runs them side by side", str,
               name='gpus',
               gui='GPUs for the excitations', group=G),
        Option('gfdtd_timestep', 'auto', "gfdtd's time step: auto (default: the operator's own leapfrog limit for "
               "models of 1 M cells or more, or one measured before; the vacuum Courant rule of the smallest cells "
               "otherwise), exact (the operator's limit always: 1.19-1.29 times the step on the examples, measured "
               "by Lanczos in 0.6-2 s once a model and kept; worth it on runs longer than ~10 s) or courant",
               choices=('auto', 'exact', 'courant'), name='timestep', gui='Time step', group=G),
    ]

    @property
    def conductor_loss(self):
        """--sigma reaches the model (conducting sheets) only when the installed gfdtd models their loss."""
        return has_conductor_loss()

    @property
    def roughness(self):
        """--rough as the sheets' own roughness when the installed gfdtd has it (ROUGH_API)."""
        return has_conductor_loss(ROUGH_API)

    @property
    def finish_wideband(self):
        """--finish as the sheets' own (a wideband fit) when the installed gfdtd has it (FINISH_API)."""
        return has_conductor_loss(FINISH_API)

    @property
    def face_sheets(self):
        return has_conductor_loss(FINISH_API)

    def check(self, s=None):
        try:
            from gfdtd.engine import gpu_info, core_info
        except ImportError as e:
            return False, f"gfdtd cannot be imported ({e}): python -m pip install 'kicad2ems-fdtd[gfdtd]'"
        why = engine_compatible()
        if why:
            return False, why
        gpu, what = gpu_info()
        want = getattr(s, 'gfdtd_backend', 'auto') if s is not None else 'auto'
        if want == 'gpu' and not gpu:
            return False, f"gfdtd_backend = gpu: {what}"
        if gpu and want != 'cpu':
            from gfdtd.engine import backend_info, resolve_backend
            return True, f"GPU: {backend_info(resolve_backend('gpu'))}"
        core, cwhat = core_info()
        return True, f"CPU: {cwhat if core else 'numpy (slow; build the core: python -m gfdtd.core.build)'}" + (
            '' if gpu else f"; {what}")

    def resources(self):
        """The GPU gfdtd's engine would run on (the best Vulkan GPU, else CuPy's first device;
        CuPy's first on CUDA), or why there is
        none."""
        from emsdev import gpu
        from gfdtd.engine import gpu_backend, gpu_info
        try:
            b = gpu_backend()
            if b == 'cupy':
                return dict(gpus=[gpu.cupy_device(0).as_dict()], why='')
            if b == 'vulkan':
                from gfdtd.engine import VULKAN_NEEDS
                return dict(gpus=[gpu.vulkan_device(need=VULKAN_NEEDS).as_dict()], why='')
        except gpu.Unavailable as e:
            return dict(gpus=[], why=str(e))
        return dict(gpus=[], why=gpu_info()[1])

    def details(self):
        ok, what = self.check()
        if not ok:
            return ok, [what]
        import gfdtd
        return ok, [f"gfdtd {gfdtd.__version__} ({os.path.dirname(gfdtd.__file__)}), engine API "
                    f"{'.'.join(map(str, gfdtd.API_VERSION))}", what]

    def validate(self, s):
        return self.validate_converge(s)

    def assess(self, run):
        """Named, gfdtd runs without what it lacks (with a note, prepare); for auto, those runs are openEMS's."""
        if AUTO not in run.plan:
            return []
        s = run.s
        return [f"{what}: openEMS's" for what, on in (("--sigma (this gfdtd's copper is PEC: update gfdtd)",
                                                     s.sigma and not self.conductor_loss),) if on]

    def estimate(self, run):
        """openEMS's estimate (the same mesh and timesteps), at gfdtd's speed on the backend it would use."""
        from gfdtd.engine import resolve_backend
        e = super().estimate(run)
        where = resolve_backend(self.engine(run.s).partition(':')[2])
        return dict(e, seconds=e['seconds'] / SPEEDUP[where] + START,
                    what=e['what'] + {'cupy': ' on the GPU', 'vulkan': ' on the GPU (Vulkan)',
                                      'core': ' on the CPU core'}.get(where, ' in numpy'))

    def jobs(self, run, p):
        """Runs at once: --jobs when given; on the GPU as many as its cache holds (engine.concurrent_runs), on the
        CPU one (the core's threads already share the cores). --gfdtd-gpus: one a GPU named, each in a process of
        its own bound to it (GPU_SLOTS; R-D9)."""
        from gfdtd.engine import concurrent_runs, resolve_backend
        global GPU_SLOTS
        GPU_SLOTS = None
        s = run.s
        where = resolve_backend(self.engine(s).partition(':')[2])
        slots = gpu_slots(getattr(s, 'gfdtd_gpus', '') or '', len(p.excite)) if where == 'vulkan' else None
        if slots:
            from gfdtd.engine import vulkan_gpus
            names = dict(vulkan_gpus())
            used = [k for k, v in share(slots, len(p.excite)).items() if v]
            if len(used) == 1:                                      # one GPU by the speeds: its runs side by side
                os.environ['EMSDEV_VULKAN_DEVICE'] = str(used[0])   # there, as without --gfdtd-gpus
                sp = gpu_speeds(slots)
                run.log(f"gfdtd: by the GPUs' measured speeds ({', '.join(f'{names.get(k, k)} {sp[k]:.0f}' for k in slots if sp[k])}"
                        f" M cells/s) the others would finish later: {names.get(used[0], used[0])} alone")
                slots = None
        if slots:
            GPU_SLOTS = slots
            run.log(f"gfdtd: the excitations shared out to {len(slots)} GPUs, a process each: "
                    + ', '.join(names.get(k, str(k)) for k in slots))
            return len(slots)
        if getattr(s, 'gfdtd_gpus', '') and where != 'vulkan':
            run.log("gfdtd: --gfdtd-gpus is for Vulkan's GPUs: one GPU used")
        if s.jobs > 1:
            return s.jobs
        if where not in GPU_BACKENDS:
            return 1
        return concurrent_runs(p.grid['info']['cells'], len(p.excite))

    def engine(self, s):
        return 'gfdtd:' + s.gfdtd_backend           # gpu: Vulkan's, else CuPy's; cpu: the core, else numpy

    def about(self, s):
        return f"gfdtd, 3D FDTD ({s.gfdtd_backend})"

    def prepare(self, run):
        p = super().prepare(run)
        s = run.s
        for what, on in (("--sigma (this gfdtd's copper is PEC: update gfdtd)", s.sigma and not self.conductor_loss),):
            if on:
                run.log(f"gfdtd: {what} not supported yet - ignored")
        self.summary(run)['backend'] = s.gfdtd_backend
        os.environ['GFDTD_TIMESTEP'] = getattr(s, 'gfdtd_timestep', 'auto') or 'auto'     # (R-D6; workers inherit it)
        run.log(f"gfdtd: {p.grid['info']['cells'] / 1e6:.2f} M cells, boundaries {','.join(p.bc)}, {self.check(s)[1]}")
        return p


GPU_SLOTS = None                    # [Vulkan GPU index] a process each, of the last plan (GFDTD.jobs), or None


def gpu_slots(spec, runs):
    """The GPUs --gfdtd-gpus names ('all', or indices '0,1'), at most `runs` of them, as [Vulkan adapter index];
    None: not asked, or fewer than two of them usable (ValueError for an index there is no such GPU of)."""
    spec = str(spec or '').strip().lower()
    if not spec:
        return None
    try:
        from gfdtd.engine import vulkan_gpus
    except ImportError:
        raise ValueError("--gfdtd-gpus needs a gfdtd with vulkan_gpus (update gfdtd)")
    usable = [k for k, _ in vulkan_gpus()]
    if spec == 'all':
        picked = usable
    else:
        try:
            picked = [int(x) for x in spec.replace(' ', '').split(',') if x]
        except ValueError:
            raise ValueError(f"--gfdtd-gpus {spec!r}: all, or GPU indices such as 0,1")
        bad = [k for k in picked if k not in usable]
        if bad:
            raise ValueError(f"--gfdtd-gpus: no usable Vulkan GPU {', '.join(map(str, bad))} (there are "
                             f"{', '.join(map(str, usable)) or 'none'})")
    picked = list(dict.fromkeys(picked))[:max(1, int(runs))]
    return picked if len(picked) > 1 else None


SPEED_FILE = 'gpu_speed.json'       # under gfdtd's cache folder: each GPU's measured M cell updates a second


def _speed_path():
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'gfdtd', SPEED_FILE)


def gpu_speeds(slots):
    """{slot: M cell updates / s} as measured on earlier runs (by the GPU's name), None where never measured."""
    import json
    try:
        from gfdtd.engine import vulkan_gpus
        names = dict(vulkan_gpus())
    except ImportError:
        names = {}
    try:
        with open(_speed_path(), encoding='utf-8') as fh:
            kept = json.load(fh)
    except (OSError, ValueError):
        kept = {}
    return {k: kept.get(names.get(k, str(k))) for k in slots}


def record_speed(slot, mcps):
    """A run's speed on GPU `slot`, kept by its name (the mean with what was kept, weighted 1:1)."""
    import json
    try:
        from gfdtd.engine import vulkan_gpus
        name = dict(vulkan_gpus()).get(slot, str(slot))
    except ImportError:
        name = str(slot)
    path = _speed_path()
    try:
        with open(path, encoding='utf-8') as fh:
            kept = json.load(fh)
    except (OSError, ValueError):
        kept = {}
    old = kept.get(name)
    kept[name] = round(float(mcps) if not old else 0.5 * (old + float(mcps)), 1)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(kept, fh, indent=1)
    except OSError:
        pass


def share(slots, n, speeds=None):
    """{slot: [run positions 0..n-1]}: each run to the GPU that would finish it first (its runs so far plus this one
    over its speed; the runs taken as alike). A GPU never measured counts as fast as the fastest (so it is measured);
    with no speeds at all, round the GPUs. A GPU too slow to finish a run before the others would is given none
    (R-D9: the 4.7 times slower one had taken the second of two runs)."""
    speeds = dict(speeds if speeds is not None else gpu_speeds(slots))
    known = [v for v in speeds.values() if v]
    top = max(known) if known else 1.0
    sp = {k: float(speeds.get(k) or top) for k in slots}
    out, done = {k: [] for k in slots}, {k: 0 for k in slots}
    for i in range(n):
        k = min(slots, key=lambda q: ((done[q] + 1) / sp[q], slots.index(q)))
        out[k].append(i)
        done[k] += 1
    return out


def bind_process(slots, counter, lock):
    """A worker process's initializer: the next slot's GPU, as EMSDEV_VULKAN_DEVICE (read when its runs pick one)."""
    with lock:
        k = counter.value
        counter.value += 1
    os.environ['EMSDEV_VULKAN_DEVICE'] = str(slots[k % len(slots)])


LABEL = 'gfdtd'                     # the engine's name in kicad2ems's log
IN_THREADS = True                   # one process: the runs share the GPU (a stream each)


def engine_compatible():
    """None when the installed gfdtd's API (gfdtd.API_VERSION) is one this connection can use: the major version
    of ENGINE_API and at least its minor. Else why not, and which package to update."""
    import gfdtd
    want = '.'.join(map(str, ENGINE_API))
    have = getattr(gfdtd, 'API_VERSION', None)
    if have is None:
        return f"gfdtd {gfdtd.__version__} has no API version (before 261028-r1): update gfdtd (kicad2ems-fdtd wants {want})"
    have = tuple(have)
    if have[0] == ENGINE_API[0] and have[1] >= ENGINE_API[1]:
        return None
    newer = have[0] > ENGINE_API[0]
    return (f"gfdtd {gfdtd.__version__} has engine API {'.'.join(map(str, have))}, kicad2ems-fdtd is written for "
            f"{want}: update {'kicad2ems-fdtd' if newer else 'gfdtd'}")


def has_conductor_loss(need=LOSS_API):
    """The installed gfdtd models conducting sheets' loss (engine API LOSS_API or later, same major; need:
    another API, ROUGH_API for their roughness)."""
    try:
        import gfdtd
    except ImportError:
        return False
    have = tuple(getattr(gfdtd, 'API_VERSION', (0, 0)))
    return have[0] == need[0] and have >= need


def api(backend=''):
    """(ContinuousStructure, openEMS) of gfdtd's engine (engines.py) on a backend: gpu, cupy, vulkan, core, numpy
    or auto. Refuses a gfdtd of another engine API (engine_compatible)."""
    import functools
    import gfdtd
    why = engine_compatible()
    if why:
        raise RuntimeError(why)
    return gfdtd.ContinuousStructure, functools.partial(gfdtd.openEMS, backend=backend or 'auto')
