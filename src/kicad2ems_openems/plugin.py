"""
openEMS as a kicad2ems solver plugin: kicad2ems's FDTD model (kicad2ems.fdtd.plugin) solved by openEMS's engine. Its own: the import check
(engine.py, venv.py) and the mesh convergence check (--converge: the coarser levels run first, converge.py).

What openEMS changes on the way (declared deviations, docs/architecture.md): vias snapped onto mesh lines
(--vias), parts with ESR / ESL split in two elements (openEMS 0.37), edge walls moved onto mesh lines.
"""
import os

from kicad2ems.api.solver import Option, floatlist, floats
from kicad2ems.api.solver import FDTD, MESH, RUN, G_MESH

CONVERGE = [
    Option('converge', None, "mesh convergence check: run on several mesh levels, level f with --res x f and "
           "--fine / f (e.g. 1,1.5,2); the finest level gives the result", list,
           conv=lambda v: floatlist(v) or None, gui='Convergence levels (e.g. 1,1.5)', group=G_MESH,
           cli=dict(metavar='F1,F2,...')),
    Option('converge_tol', (0.1, 2.0), "converged when the second-finest level is within this of the finest "
           "(default 0.1 dB, 2 deg)", tuple, conv=floats(2), group=G_MESH, cli=dict(metavar='DB,DEG')),
]


class OpenEMS(FDTD):
    api = (1, 0)                    # kicad2ems.api
    name, aliases, label = 'openems', ('fdtd',), 'openEMS (FDTD)'
    previewed = True
    order = 10                      # first in kicad2ems's lists: the reference
    options = MESH + CONVERGE + RUN

    def engine(self, s):
        return 'openems'

    def summary(self, run):
        return run.summary                                          # openEMS's facts: the top level

    def check(self, s=None):
        ok, lines = self.details()
        mods = {l.split()[1]: l.split(None, 2)[2] for l in lines if l.strip().startswith(('ok ', 'FAILED '))}
        if ok:
            return True, os.path.dirname(mods.get('openEMS', '')) or 'openEMS'
        unused = next((l.split('(not used: ', 1)[1][:-1] for l in lines
                       if l.startswith('openems_venv') and '(not used: ' in l), None)
        if unused:                                                  # the setting is why: say so first
            return False, unused
        return False, next((f"{m}: {e}" for m, e in mods.items() if any(l.strip().startswith(f'FAILED  {m}') for l in lines)),
                           'openEMS cannot be imported')

    def details(self):
        """openEMS's import check: (ok, report lines with the real errors and hints)."""
        from . import engine
        return engine.check()

    def validate(self, s):
        if s.converge is not None and (len(set(s.converge)) < 2 or min(s.converge) <= 0):
            return f"converge needs at least two positive refinement factors, e.g. 1,1.5,2 (got {s.converge})"
        if s.converge and s.antenna:
            return "converge is not available with mode = antenna"
        return None

    def levels(self, s):
        return max(1, len(s.converge or []))

    # ---- --converge: the mesh levels, run coarsest first --------------------------------------------------
    def mesh(self, run, p, **kw):
        if not run.s.converge:
            return super().mesh(run, p, **kw)
        from .converge import mesh_levels, describe_levels
        p.levels = mesh_levels(run.m, run.fmin, run.fmax, sorted(set(run.s.converge)), res=run.s.res,
                               fine=run.s.fine, **kw)
        run.log("\n".join(describe_levels(p.levels)))
        run.summary['levels'] = [dict(factor=f, cells=lm['info']['cells'], fine_mm=lm['info']['fine_mm'],
                                      dt_ps=lm['info']['dt_ps']) for f, lm in p.levels]
        return p.levels[-1][1]                                      # the finest level gives the result

    def coarse(self, run, p, jobs, run_kw):
        from kicad2ems.api.solver import run_excitations
        from kicad2ems.api.solver import mixed_mode, fill_reciprocal
        for f, lm in (p.levels or [])[:-1]:
            run.log(f"--- mesh level x{f:g} ({lm['info']['cells'] / 1e6:.2f} M cells) ---")
            r = run_excitations(run.m, lm, run.freq, p.excite, p.bc, os.path.join(run.s.out, 'converge', f"L{f:g}"),
                                jobs=jobs, **run_kw)
            fill_reciprocal(r.S, p.excite)
            p.coarse.append((f, mixed_mode(r.S, run.m)[0]))
        if p.levels:
            run.log(f"--- mesh level x{p.levels[-1][0]:g} (finest: the result) ---")

    def outputs(self, run, p, res):
        super().outputs(run, p, res)
        if p.levels:
            self._convergence(run, p, res)

    def _convergence(self, run, p, res):
        from .converge import compare_levels, verdict
        from kicad2ems.api.solver import plot_convergence
        tol_db, tol_deg = run.s.converge_tol
        dev = compare_levels([S_ for _, S_ in p.coarse] + [res.S], res.names) + [{}]    # finest: the reference
        lines, ok = verdict([f for f, _ in p.levels], dev[:-1], tol_db, tol_deg)
        run.log("\n".join(lines))
        if not ok:
            run.log("WARNING: mesh not converged (see the convergence table)")
        run.figure(plot_convergence(run.freq, [(f"x{f:g}", S_) for f, S_ in p.coarse] +
                                    [(f"x{p.levels[-1][0]:g}", res.S)], res.names, f"{run.name} - mesh convergence"),
                   '_converge.png', 'converge', 'mesh convergence')
        run.summary['convergence'] = dict(converged=ok, tol_db=tol_db, tol_deg=tol_deg,
                                          levels=[dict(factor=f, cells=lm['info']['cells'], **d)
                                                  for (f, lm), d in zip(p.levels, dev)])
