"""
openEMS as a kicad2ems solver plugin: the FDTD model (kicad2ems_fdtd.plugin) solved by openEMS's engine. Its own: the import check
(engine.py, venv.py). The mesh convergence check (--converge) is the FDTD layer's (plugin.py, converge.py).

What openEMS changes on the way (declared deviations, docs/architecture.md): vias snapped onto mesh lines
(--vias), parts with ESR / ESL split in two elements (openEMS 0.37), edge walls moved onto mesh lines.
"""
import os

from ..plugin import FDTD, MESH, RUN, CONVERGE


class OpenEMS(FDTD):
    api = (2, 9)                    # kicad2ems.api (2.4: resolve_finish, the finish row; 2.9: basic)
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
        return self.validate_converge(s)
