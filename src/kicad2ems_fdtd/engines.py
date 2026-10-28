"""
FDTD engines: anything with openEMS's Python interface (a ContinuousStructure class and an openEMS class) runs
the model that model.build_fdtd makes. An engine is a module of this package (ENGINES) giving api(option) ->
(ContinuousStructure, openEMS), LABEL, its name in the log, and optionally IN_THREADS (True: the runs of one
solve share this process, as threads, gfdtd's on the GPU). It is named by a string, so runs in worker processes
can find it: '<name>' or '<name>:<option>' (gfdtd:cupy). The engines live here with their solvers, so an engine
and the FDTD layer that drives it are always the same version.
"""
import importlib

ENGINES = {'openems': 'kicad2ems_fdtd.openems.engine', 'gfdtd': 'kicad2ems_fdtd.gfdtd'}
DEFAULT = 'openems'


def engine_module(engine):
    """The module of an engine ('gfdtd:cupy'; None: the default), or None when there is none of that name."""
    name = (engine or DEFAULT).partition(':')[0]
    return importlib.import_module(ENGINES[name]) if name in ENGINES else None


def engine_api(engine):
    """(ContinuousStructure, openEMS) of an engine ('gfdtd:cupy'; None: the default)."""
    mod = engine_module(engine)
    if mod is None:
        raise ValueError(f"unknown FDTD engine {engine!r} (kicad2ems-fdtd has {', '.join(ENGINES)})")
    return mod.api((engine or DEFAULT).partition(':')[2])


def engine_name(engine):
    """The engine's folder under the output folder and its name in the log: ('openems', 'openEMS')."""
    name = (engine or DEFAULT).partition(':')[0]
    return name, getattr(engine_module(engine), 'LABEL', name)
