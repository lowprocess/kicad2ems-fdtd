"""
The FDTD solvers' common plugin: the board lowered to boundaries, a rectilinear mesh and the CSX model, run on an
engine with openEMS's interface (engines.py); antenna mode and field dumps. A solver on an FDTD engine subclasses
FDTD, names its engine and adds what is its own (openEMS: --converge and the import check; gfdtd: the backend).

    prepare  boundaries (bc), the mesh with the edge walls snapped onto it, the closed-box resonance check, the
             field-dump specs, the excitation length; the mesh preview (the first FDTD solver of the run draws it)
    solve    one run per driven port (--jobs at a time); in antenna mode one run with the far field
    outputs  field-dump maps, the CSX file, per-run times, antenna results and plots

The settings below are shared: every FDTD solver lists these same Option objects, so each is one setting.
"""
import os
from types import SimpleNamespace

import numpy as np

from kicad2ems.api.solver import Option, strlist, items, floatlist, G_DUMP
from kicad2ems.api.solver import Solver, Result, Deviation, registry
from .engines import engine_name


def _bc(v):
    return ",".join(strlist(v) or ['auto'])


def _air(v):
    """--air: one margin (mm) for every side, or six (xmin,xmax,ymin,ymax,zmin,zmax) as a tuple."""
    vals = floatlist(v) if v is not None else []
    if len(vals) not in (0, 1, 6):
        raise ValueError("one value or six (xmin,xmax,ymin,ymax,zmin,zmax)")
    return None if not vals else vals[0] if len(vals) == 1 else tuple(vals)


def mesh_zone(s):
    """auto_mesh's zone from --mesh-zone: False (off), True (on) or a distance in mm. Never in antenna mode: there
    the ground copper is often the radiator (a slot, an inverted-F's ground)."""
    v = str(getattr(s, 'mesh_zone', 'on') or 'on').lower()
    if v in ('off', 'no', 'false', '0') or getattr(s, 'antenna', False):
        return False
    if v in ('on', 'auto', 'yes', 'true'):
        return True
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"--mesh-zone {v!r}: on, off or a distance in mm") from None


# the GUI's sections for these settings (kicad2ems-gui's runner.py)
G_MESH, G_RUN = 'FDTD/Mesh', 'FDTD/Run'                                # the GUI's FDTD section (and G_DUMP)
G_ANT, G_SHARED_DUMP = 'antenna', 'field dumps'           # shared with mom25d: the GUI's Antenna and Field dumps

MESH = [
    Option('res', 20.0, "cells per wavelength at fmax in each medium (default {default})", float,
           gui='Cells per wavelength', group=G_MESH),
    Option('fine', None, "mm, cell size at copper edges (default: the narrowest port trace or twice the thinnest "
           "dielectric, whichever is smaller, / 4)", float, gui='Fine cell at copper edges (mm, empty: auto)',
           group=G_MESH),
    Option('air', None, "mm of air between the board and the boundary (default 5; antenna mode: a quarter "
           "wavelength at the band centre, at least 5), or six values xmin,xmax,ymin,ymax,zmin,zmax (ymax: the "
           "board's top edge in KiCad)", conv=_air, gui='Air margin, mm (1 or 6 values)', group=G_MESH),
    Option('ratio', 1.4, "max growth between neighbouring cells (default {default})", float, group=G_MESH),
    Option('mesh_zone', 'on', "mesh lines at the ground copper's edges only near the signal copper: on (default: "
           "within 1 mm or 3x the thickest dielectric; never in antenna mode), a distance in mm, or off (every "
           "copper edge). Large boards: the planes' pours, antipads and pads far from the signal no longer set "
           "the mesh",
           conv=lambda v: str(v).lower(), gui='Mesh zone around the signal (on / mm / off)', group=G_MESH),
    Option('nz', 4, "minimum cells through each dielectric layer (default {default})", int,
           gui='Cells through each dielectric', group=G_MESH),
    Option('z_sheet', 1.0, "first z cell either side of a copper layer with signal edges, as a multiple of the "
           "fine cell, graded into the dielectric (default {default}; 0: uniform, --nz cells)", float,
           gui='First z cell at signal copper (x fine)', group=G_MESH),
    Option('curve_res', 6.0, "over curved copper (arcs, fans) the largest cell is the dielectric's / this "
           "(default {default}; 1: no refinement)", float, gui='Refinement over curved copper', group=G_MESH),
    Option('curves', 'extremes', "mesh lines for rounded corners / arcs / circles: only the ends and extremes of "
           "each curve (default) or every vertex", choices=('extremes', 'vertices'), group=G_MESH),
    Option('vias', 'auto', "via mesh lines: auto = exact on signal nets, snap elsewhere; exact = centre and "
           "+-radius; centre = one line per axis; snap = move onto existing lines (within --via-snap)",
           choices=('auto', 'exact', 'centre', 'snap'), group=G_MESH),
    Option('via_snap', 0.2, "mm a snapped via may move (default {default})", float, group=G_MESH),
]
RUN = [
    Option('bc', 'auto', "boundaries: auto (PEC on a solid bottom plane, MUR elsewhere), MUR, PML_8 (any PML_n), "
           "PEC, PMC, or six values xmin,xmax,ymin,ymax,zmin,zmax", conv=_bc, group=G_RUN),
    Option('end_db', -40.0, "stop when the field energy has decayed by this much (default {default})", float,
           gui='Stop at (dB)', group=G_RUN),
    Option('end_on', 'ports', "ports (default): stop when the port signals have decayed by --end-db, or the field "
           "energy has; energy: only the field energy (openEMS's own criterion)", choices=('ports', 'energy'),
           gui='Stop on', group=G_RUN),
    Option('max_steps', 1_000_000, "timestep limit per run (default {default}); low frequencies and high-Q "
           "circuits need more", int, gui='Timestep limit', group=G_RUN),
    Option('diel', 'fixed', "dielectric loss: fixed = conductivity fitted at the band centre (default); wideband "
           "= Djordjevic-Sarkar model (er / tand constant-ish over the band) as Debye poles",
           choices=('fixed', 'wideband'), group=G_RUN),
    Option('diel_fref', 1.0, "GHz at which the stackup er / tand hold (wideband; default 1)", float, group=G_RUN),
    Option('excite', None, "comma-separated ports to excite (default all), e.g. P1; antenna mode: the antenna's "
           "port (openEMS and mom25d)", list, gui='Driven ports (empty: all)', group='solvers'),
    Option('threads', 0, "openEMS threads per run (default 0 = all cores; with --jobs above 1: half the cores, "
           "shared by the runs)", int, gui='Threads (0: openEMS default)', group=G_RUN),
    Option('jobs', 1, "excitations run in parallel (separate processes; default 1)", int, group=G_RUN),
    Option('openems_output', False, "show openEMS's own output instead of the progress / ETA line (it is always "
           "kept in openems.log of each run)", bool, group=G_RUN),
    Option('ff_freq', [], "antenna: frequency for the patterns (default: the resonance, minimum |S11|); also "
           "recorded", list, conv=floatlist, gui='Antenna: pattern frequencies (GHz; empty: the resonance)',
           group=G_ANT, cli=dict(metavar='GHz,...')),
    Option('ff_step', 2.0, "antenna: angle step of the pattern cuts (deg, default 2)", float, group=G_ANT),
    Option('dump', [], "field dump, repeatable: E|H|J @ layer[+-mm] | x=/y=/z= | volume | box=x0,y0,z0,x1,y1,z1 "
           "[@ frequencies in GHz -> frequency domain + PNG maps; none -> time domain VTK]. e.g. E@F.Cu+0.05@2,6 "
           "J@In2.Cu@4   E@y=7", list, conv=items, gui='Dumps (FIELD@REGION[@GHz]; ...)', group=G_DUMP,
           cli=dict(action='append', metavar='FIELD@REGION[@GHz,...]')),
    Option('dump_port', None, "ports whose excitation runs get the dumps (default: the first excited port)", list,
           gui='Runs that record dumps (empty: first driven)', group=G_SHARED_DUMP),
]

# --solvers auto (docs/architecture.md#choosing-solvers), fitted on openEMS runs of the 13 examples (24 cores):
# after the pulse the fields ring down for RING x |end_db| / 17.4 dB x the line model's peak group delay (a
# resonance's energy falls 17.4 dB per group delay; measured 0.4..3x that, median 1.7, and far longer where the
# board rings without the line model showing it: rf_gcpw, rf_stubnotch), or PULSES - 1 pulse lengths without a line
# model; CELLS_PER_CORE_S cell updates per second and core (13..35 M, median 21 M). Time within 2x on 9 of 13.
RING, PULSES, CELLS_PER_CORE_S = 1.7, 5.0, 21e6


def _peak_delay(run):
    """The line model's largest group delay (s) over its terms that carry signal: how long the board rings. None
    without a line model or a circuit solver for it."""
    from kicad2ems.api.solver import solve_lines as solve, lines_backend as resolve_backend
    if run.lm is None or not run.lm.paths:
        return None
    try:
        f = np.linspace(run.fmin, run.fmax, 801)
        S, _ = solve(run.lm, run.m.ports, f, backend=resolve_backend('auto', getattr(run.s, 'qucsator', None))[0])
    except Exception:                                               # no scikit-rf / qucsator, a failing netlist
        return None
    best = None
    for s in S.reshape(len(f), -1).T:
        if np.all(np.isfinite(s)) and np.abs(s).max() > 1e-3:
            g = -np.gradient(np.unwrap(np.angle(s)), 2 * np.pi * f)
            g = g[np.abs(s) > 0.1 * np.abs(s).max()]
            best = max(best or 0.0, float(g.max()))
    return best


class FDTD(Solver):
    """An FDTD solver. A subclass sets name, label, options (MESH + RUN + its own) and engine(s); tag names its
    files beside another FDTD solver's ('' for the first: <name>_antenna.json; '_gfdtd': <name>_gfdtd_antenna.json)."""
    consumes, rank = 'board', 2
    capabilities = dict(ports='yes', gap_ports='yes', diff_pairs='yes',
                        parts='approximated: parts with ESR / ESL split in two elements (openEMS 0.37)',
                        vias='approximated: off the signal nets moved onto mesh lines (--vias, --via-snap)',
                        outline='yes', planes='yes', edge_walls='yes: PEC sheets', thickness='ignored: conducting sheets',
                        conductor_loss='yes', dispersion='yes: fitted at mid-band, or wideband Debye', radiation='yes',
                        cavity='yes', antenna='yes')
    options = MESH + RUN
    tag = ''
    conductor_loss = True           # --sigma as conducting sheets (False: PEC copper)

    def engine(self, s):
        """The engine string of engines.py ('openems', 'gfdtd:cupy')."""
        raise NotImplementedError

    def summary(self, run):
        """Where its facts go in the run summary: its own table (the first FDTD solver: the top level)."""
        return run.summary.setdefault(self.name, {})

    def draws_mesh(self, run):
        """The first FDTD solver of the run draws the mesh preview (the others have the same mesh)."""
        reg = registry()
        return next((n for n in run.plan + run.previewed if isinstance(reg.get(n), FDTD)), self.name) == self.name

    # ---- what it changes, and what it holds -------------------------------------------------------------
    def deviations(self, run, p):
        """Vias moved onto mesh lines, parts split in two elements, edge walls moved onto mesh lines."""
        m, out = run.m, []
        for i, (v, q) in enumerate(zip(m.vias, p.grid.get('vias', m.vias))):
            d = float(np.hypot(q[0] - v[0], q[1] - v[1]))
            if d > 1e-9:
                out.append(Deviation(f"via:{i}", 'vias', 'moved onto the mesh', d, to=list(q)))
        for pt in m.parts:
            n = len(pt.pieces())
            if n > 1 and not pt.short:
                out.append(Deviation(f"part:{pt.ref}", 'parts', f"split in {n} elements (openEMS 0.37: a series R "
                                                                f"with L / C is a sheet of its own)", to=n))
        for i, (w, q) in enumerate(zip(m.walls, p.grid.get('walls') or m.walls)):
            if abs(q[1] - w[1]) > 1e-9:
                out.append(Deviation(f"wall:{i}", 'edge_walls', 'moved onto a mesh line', abs(q[1] - w[1]), to=list(q)))
        return out

    def readback(self, run, p):
        from .readback import readback
        return readback(run.m, p.grid, p.bc, sigma=run.s.sigma if self.conductor_loss else None)

    def levels(self, s):
        """Mesh levels solved (--converge: several)."""
        return 1

    def estimate(self, run):
        """Cells of the mesh x timesteps (the pulse and the ring-down, at most --max-steps) x the runs, at
        CELLS_PER_CORE_S per core."""
        from .boundaries import resolve_bc
        from .mesh import auto_mesh
        from .model import pulse_steps, port_index
        s, m = run.s, run.m
        bc = resolve_bc('PML_8' if s.antenna and s.bc == 'auto' else s.bc, m)
        g = auto_mesh(m, run.fmin, run.fmax, res=s.res, fine=s.fine, air=s.air, ratio=s.ratio, nz_min=s.nz, bc=bc,
                      curves=s.curves, vias=s.vias, via_snap=s.via_snap, z_sheet=s.z_sheet, curve_res=s.curve_res,
                      zone=mesh_zone(s), log=lambda *a: None)
        cells, dt = g['info']['cells'], g['info']['dt_ps'] * 1e-12
        pulse = pulse_steps(run.fmin, run.fmax, g['info']['dt_ps'])
        tg = _peak_delay(run)
        ring = RING * abs(s.end_db) / 17.4 * tg / dt if tg else (PULSES - 1) * pulse
        steps = int(min(s.max_steps, pulse + ring))
        runs = (1 if s.antenna else len(port_index(m, s.excite))) * self.levels(s)
        return dict(seconds=runs * cells * steps / (CELLS_PER_CORE_S * (os.cpu_count() or 1)), gb=g['info']['mem_gb'],
                    what=f"{cells / 1e6:.2f} M cells, {runs} run(s) of ~{steps:,} timesteps")

    # ---- the run ---------------------------------------------------------------------------------------
    def mesh(self, run, p, **kw):
        """The mesh that gives the result (openEMS: also the coarser --converge levels, in p.levels)."""
        from .mesh import auto_mesh
        return auto_mesh(run.m, run.fmin, run.fmax, res=run.s.res, fine=run.s.fine, **kw)

    def prepare(self, run):
        from .boundaries import resolve_bc, cavity_check
        from .model import pulse_steps
        from .antenna import exc_fmax
        from kicad2ems.api.solver import parse_dump
        s, m = run.s, run.m
        folder, who = engine_name(self.engine(s))
        first = self.draws_mesh(run)
        log = run.log if first else (lambda *a: None)               # the same mesh: said once
        p = SimpleNamespace(levels=None, coarse=[], dumps=[], result=None, excite=None, ant=None)
        air, bc = s.air, s.bc
        if s.antenna:
            from .antenna import antenna_air
            from .mesh import air_text
            air, bc = antenna_air(run.fmin, run.fmax, s.air), ('PML_8' if bc == 'auto' else bc)
            run.log(f"{who} antenna mode: boundaries {bc}, air margin {air_text(air)}")
        p.bc = resolve_bc(bc, m)
        p.grid = self.mesh(run, p, air=air, ratio=s.ratio, nz_min=s.nz, bc=p.bc, curves=s.curves, vias=s.vias,
                           via_snap=s.via_snap, z_sheet=s.z_sheet, curve_res=s.curve_res, zone=mesh_zone(s), log=log)
        if m.walls:
            from kicad2ems.api.solver import describe_edges as describe, snap_edges as snap
            for g in [p.grid] + [lm for _, lm in (p.levels or [])]:
                g['walls'] = snap(m.walls, g)
                g['info']['edge_walls'] = describe(m.walls, m).split(': ', 1)[1]
        out = self.summary(run)
        out['mesh'] = p.grid['info']
        if first:
            from kicad2ems.api.solver import save_mesh as save
            from kicad2ems.api.solver import plot_preview as preview
            run.summary['mesh'] = p.grid['info']
            run.wrote(f"{who} grid (3D view)", save(run, folder, 'grid', x=p.grid['x'], y=p.grid['y'], z=p.grid['z']))
            cavity_check(m, p.grid, p.bc, run.fmax, log=run.log)
            run.figure(preview(m, p.grid, run.name), '_mesh.png', 'mesh', 'model and mesh')
            run.log(f"wrote {run.path('_mesh.png')}")
        p.dumps = [parse_dump(sp, m, p.grid) for sp in s.dump]     # validated before any run
        for d in p.dumps:
            run.log("  " + d.describe(p.grid))
        ps = pulse_steps(run.fmin, run.fmax, p.grid['info']['dt_ps'], exc_fmax(run.fmin, run.fmax) if s.antenna else None)
        out['pulse_steps'] = ps
        log(f"excitation pulse ~{ps} timesteps (estimate); --max-steps {s.max_steps} allows {s.max_steps / ps:.0f} "
            f"pulse lengths" + (" - WARNING: few for the fields to ring down: raise --max-steps"
                                if s.max_steps < 20 * ps else ''))
        if s.diel == 'wideband':
            from .dielectric import describe as describe_diel
            log("\n".join(describe_diel(m.layers, s.diel_fref * 1e9, run.fmin, run.fmax)))
        return p

    def jobs(self, run, p):
        """Runs at once: --jobs, warned when they need more memory than the machine has."""
        from kicad2ems.api.solver import ram_gb
        jobs, ram, gb = max(1, min(run.s.jobs, len(p.excite))), ram_gb(), p.grid['info']['mem_gb']
        if jobs > 1 and ram and jobs * gb > 0.8 * ram:
            run.log(f"WARNING: {jobs} parallel runs need ~{jobs * gb:.1f} GB, {ram:.1f} GB RAM installed - use fewer --jobs")
        return jobs

    def coarse(self, run, p, jobs, run_kw):
        """Runs before the result's (openEMS: the coarser --converge levels)."""

    def solve(self, run, p):
        from .model import port_index
        from .run import run_excitations
        from kicad2ems.api.solver import dump_size_mb as estimate_size_mb
        from kicad2ems.api.solver import mixed_mode, fill_reciprocal
        s, m = run.s, run.m
        who = engine_name(self.engine(s))[1]
        p.excite = port_index(m, s.excite)
        p.drive = None
        if s.antenna and s.drive:                                   # a phased drive: a run per driven port
            from kicad2ems.api.solver import phased
            p.drive, _ = phased.settings_drive(s.drive, [q.name for q in m.ports], s.grid, s.steer, s.taper)
            p.excite = [j for j, v in enumerate(p.drive) if abs(v) > 0]
        elif s.antenna and len(p.excite) > 1:
            run.log(f"{who} antenna mode: driving {m.ports[p.excite[0]].name} only (one port per antenna run; "
                    f"--excite to choose)")
            p.excite = p.excite[:1]
        dump_runs = port_index(m, s.dump_port) if s.dump_port else None
        if p.dumps:
            n_runs = len(dump_runs) if dump_runs else 1
            run.log(f"field dumps: ~{estimate_size_mb(p.dumps, p.grid) * n_runs:.0f} MB on disk (rough), in "
                    f"{n_runs} run(s)")
        jobs = self.jobs(run, p)
        progress = dict(mode='raw' if s.openems_output else 'bar', on_status=run.status if run.events else None)
        run_kw = dict(end_db=s.end_db, end_on=s.end_on, sigma=s.sigma if self.conductor_loss else None,
                      threads=s.threads, diel=s.diel, max_steps=s.max_steps, diel_fref=s.diel_fref * 1e9,
                      progress=progress, log=run.log, engine=self.engine(s))
        self.coarse(run, p, jobs, run_kw)
        p.xml = run.path(f"{self.tag}.xml")
        if s.antenna and p.drive is not None:
            from . import antenna
            p.result = antenna.run_phased(m, p.grid, run.freq, p.excite, p.bc, s.out,
                                          ff_extra=[v * 1e9 for v in s.ff_freq], step=s.ff_step, xml_path=p.xml,
                                          dumps=p.dumps, jobs=jobs, **run_kw)
        elif s.antenna:
            from . import antenna
            k = p.excite[0]
            p.result = antenna.run_antenna(m, p.grid, run.freq, k, p.bc, s.out, ff_extra=[v * 1e9 for v in s.ff_freq],
                                           ff_pattern=s.ff_freq[0] * 1e9 if s.ff_freq else None, step=s.ff_step,
                                           xml_path=p.xml, dumps=p.dumps, **run_kw)
            p.ant = p.result.extra.get(m.ports[k].name)
        else:
            p.result = run_excitations(m, p.grid, run.freq, p.excite, p.bc, s.out, xml_path=p.xml, dumps=p.dumps,
                                       dump_runs=dump_runs, jobs=jobs, **run_kw)
        r = p.result
        missing = fill_reciprocal(r.S, p.excite)
        if missing:
            run.log(f"note: {', '.join(m.ports[k].name for k in missing)} not excited - their reflection and the "
                    f"terms between unexcited ports are unknown (written as 0)")
        if p.drive is not None:
            self.phased(run, p)
        S, names, zref = mixed_mode(r.S, m)
        seconds = sum(v or 0 for v in r.seconds.values())
        cells, steps = p.grid['info']['cells'], sum(v or 0 for v in r.steps.values())
        rate = f" ({cells * steps / max(seconds, 1e-9) / 1e6:.0f} M cell updates/s)" if steps else ''
        return Result(S, names, zref, seconds, comment=f"{who} FDTD ({run.name})",
                      info=dict(excited=[m.ports[k].name for k in p.excite], kpi=f"{seconds:.0f} s solver",
                                about=f"{self.about(s)}: {len(r.seconds)} run(s) on {cells / 1e6:.2f} M cells"
                                      + (f", ~{steps:,} timesteps" if steps else '') + f", {seconds:.0f} s{rate}."))

    def phased(self, run, p):
        """The phased drive's antenna from its runs (fdtd.antenna.combine_phased); --aim's phases at the pattern
        frequency the drive gives, then combined again."""
        from . import antenna
        from kicad2ems.api.solver import phased
        s, m = run.s, run.m
        names = [q.name for q in m.ports]
        pos = {q.name: (float(q.c[0]), float(q.c[1])) for q in m.ports}
        kw = dict(ff_pattern=s.ff_freq[0] * 1e9 if s.ff_freq else None, step=s.ff_step, pos=pos)
        a = p.drive
        if s.aim:
            ant, _, _ = antenna.combine_phased(m, run.freq, p.result, a, p.excite, log=lambda *_: None, **kw)
            a = phased.aim(a, names, pos, s.aim[0], s.aim[1], ant['f_pattern'])
        p.ant, p.array, p.sphere = antenna.combine_phased(m, run.freq, p.result, a, p.excite, log=run.log, **kw)
        if s.grid:
            p.ant['grid'] = phased.format_grid(phased.parse_grid(s.grid, names), names)
        if s.taper or s.steer or s.aim:                              # the GUI's Reset: these settings again
            p.ant['drive_base'] = phased.format_drive(phased.parse_drive(s.drive, names), names)
            p.ant['taper'] = s.taper or 'uniform'
            p.ant['steer'] = [float(x) for x in (list(s.steer or []) + [0.0, 0.0])[:2]]
            if s.aim:
                p.ant['aim'] = [float(x) for x in s.aim]

    def about(self, s):
        return f"{engine_name(self.engine(s))[1]}, 3D FDTD"

    def outputs(self, run, p, res):
        from .run import run_dir, log_path
        s, m, r = run.s, run.m, p.result
        folder, who = engine_name(self.engine(s))
        run.wrote(f"CSX geometry ({who}; openEMS XML, AppCSXCAD)", p.xml)
        self.dump_outputs(run, p)
        runs = {n: dict(seconds=r.seconds.get(n), steps=r.steps.get(n), ended=r.ended.get(n)) for n in r.seconds}
        for k in p.excite:
            runs.setdefault(m.ports[k].name, {})['log'] = run.rel(log_path(run_dir(m, k, s.out, self.engine(s))))
        self.summary(run)['runs'] = runs
        if p.ant:
            self.antenna(run, p)

    def dump_outputs(self, run, p):
        """The field maps (PNG, and animations with --dump-animate) of the frequency-domain plane dumps, named with
        the tag beside another FDTD solver's."""
        from kicad2ems.api.solver import plot_fd_dumps
        s, m, r = run.s, run.m, p.result
        prefix = self.tag.strip('_') + ' ' if self.tag else ''
        for (pname, dname), path in r.dump_files.items():
            d = next(x for x in p.dumps if x.name == dname)
            if d.freqs and os.path.exists(path):
                tag = prefix.replace(' ', '_') + pname.replace('+', 'p').replace('-', 'n')
                pngs = plot_fd_dumps(d, path, m, run.path(f"_{tag}"), norm=r.dump_norm.get(pname), port=pname,
                                     log=run.log)
                run.summary['dumps'] += [dict(label=f"{prefix}{dname} ({pname} driven)", png=run.rel(q)) for q in pngs]
                for q in pngs:
                    run.wrote(f"field map {prefix}{dname} ({pname})", q)
                if s.dump_animate and d.normal is not None:         # one period at each frequency
                    from kicad2ems.api.solver import DumpSource, save_animation
                    src = DumpSource(os.path.dirname(path), dname, 'FD')
                    for i, f in enumerate(src.freqs):
                        q = save_animation(src, f"{run.path(f'_{tag}')}_{dname}_{f / 1e9:g}GHz.{s.dump_animate}",
                                           d.normal, 0, 'mag', [i])
                        run.wrote(f"field animation {prefix}{dname} ({pname}, {f / 1e9:g} GHz)", q)
                run.log(f"  {prefix}{dname} ({pname} driven): {path}" + (
                    f" -> {len(pngs)} PNG map(s)" if pngs else "  (volume: the GUI's Fields tab or ParaView)"))
            else:
                run.log(f"  {prefix}{dname} ({pname} driven): {path}  (time domain: the GUI's Fields tab or ParaView)")

    def antenna(self, run, p):
        """<name><tag>_antenna.json (with the solver and its sphere file) and the antenna plots."""
        from .run import run_dir
        from kicad2ems.api.solver import save_antenna as save
        from kicad2ems.api.solver import plot_antenna, plot_pattern
        k, m, s = p.excite[0], run.m, run.s
        who = engine_name(self.engine(s))[1]
        tag = self.tag or '_' + self.name
        if p.drive is not None:                                     # phased: the combined sphere and the array
            from kicad2ems.api.solver import phased
            sph = run.path(f"{tag}_sphere.npz")
            np.savez_compressed(sph, **p.sphere)
            arr = run.path(f"{tag}_array.npz")
            phased.save(arr, **{k_: v for k_, v in p.array.items() if v is not None})
            run.wrote(f"antenna, each port's far field, {who} (beam steering)", arr)
            s11 = p.ant.pop('s11')
            ant = dict(p.ant, solver=self.name, sphere_file=os.path.basename(sph), array_file=os.path.basename(arr))
        else:
            sph = os.path.join(run_dir(m, k, s.out, self.engine(s)), 'nf2ff_sphere.h5')
            s11 = p.result.S[:, k, k]
            ant = dict(p.ant, solver=self.name, sphere_file=os.path.relpath(sph, s.out))
        key = '' if 'antenna' not in run.summary else self.tag          # the first solver's results come first
        by = f" ({who})" if self.tag else ''
        run.figure(plot_antenna(run.freq, s11, m.ports[k].z0, ant, f"{run.name} - antenna{by}"),
                   f"{self.tag}_antenna.png", f"antenna{key}", f"antenna{by}: match, gain, efficiency")
        run.figure(plot_pattern(ant, f"{run.name} - gain pattern{by}"), f"{self.tag}_pattern.png", f"pattern{key}",
                   f"antenna{by}: patterns")
        js = run.path(f"{self.tag}_antenna.json")
        save(ant, js)
        run.wrote(f"antenna results{by} (JSON)", js)
        run.wrote(f"antenna sphere{by}", sph)
        run.summary['antenna' + key] = {k_: v for k_, v in ant.items() if k_ != 'sphere'}
