"""
openEMS as an FDTD engine (kicad2ems.fdtd.engines, through the 'kicad2ems.engines' entry point): its CSXCAD /
openEMS classes, from this Python or the openems_venv setting (venv.py), and the check of that import.
"""
import os

LABEL = 'openEMS'                   # the engine's name in kicad2ems's log


def api(_option=''):
    """(ContinuousStructure, openEMS): openEMS's own classes, the structure given AddDebyeMaterial where this
    CSXCAD lacks it (kicad2ems.fdtd.model builds wideband dielectrics with it)."""
    from .venv import activate
    activate()                                                  # an openEMS in another venv (openems_venv)
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    if hasattr(ContinuousStructure, 'AddDebyeMaterial'):
        return ContinuousStructure, openEMS

    class Structure(ContinuousStructure):
        def AddDebyeMaterial(self, name, **kw):
            from CSXCAD.CSProperties import CSPropDebyeMaterial
            mat = CSPropDebyeMaterial(self.GetParameterSet(), name=name, **kw)
            self.AddProperty(mat)
            return mat
    return Structure, openEMS


def check():
    """Can this Python import openEMS? Returns (ok, report lines) with the real errors and hints."""
    import importlib, shutil, sys
    from .venv import activate
    from kicad2ems.api.solver import sitecfg
    why = activate()
    lines = [f"python      {sys.executable}",
             f"environment {sys.prefix}" + ("  (virtual env)" if sys.prefix != getattr(sys, 'base_prefix', sys.prefix) else "  (no virtual env active)"),
             f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"]
    if sitecfg.get('openems_venv'):
        lines.append(f"openems_venv {sitecfg.get('openems_venv')}" + (f"  (not used: {why})" if why else ''))
    ok, hints = True, [why] if why else []
    for mod in ('numpy', 'h5py', 'CSXCAD', 'openEMS'):
        try:
            m = importlib.import_module(mod)
            lines.append(f"  ok      {mod:8s} {getattr(m, '__file__', '')}")
        except Exception as e:                                  # ImportError, OSError from the .so, ...
            ok = False
            msg = f"{type(e).__name__}: {e}"
            lines.append(f"  FAILED  {mod:8s} {msg}")
            if isinstance(e, ModuleNotFoundError) and e.name == mod and mod in ('CSXCAD', 'openEMS'):
                hints.append(f"{mod} is not installed for THIS python ({sys.executable}). If you installed it in a "
                             f"venv, run kicad2ems from that venv (python -m kicad2ems ...), or pip-install "
                             f"the {mod} python package (openEMS-Project/{mod}/python) with this python.")
            elif isinstance(e, ModuleNotFoundError):
                hints.append(f"missing dependency '{e.name}': python -m pip install {e.name}")
            elif '.so' in str(e) or 'shared object' in str(e) or isinstance(e, OSError):
                hints.append(f"{mod} is installed but cannot load the compiled openEMS/CSXCAD libraries: add the "
                             f"openEMS install's lib folder to LD_LIBRARY_PATH (e.g. export LD_LIBRARY_PATH="
                             f"$HOME/opt/openEMS/lib:$LD_LIBRARY_PATH)")
    exe = shutil.which('kicad2ems')
    if exe:
        try:
            with open(exe) as fh:
                first = fh.readline().strip()
        except (OSError, UnicodeDecodeError):
            first = ''
        launcher = first[2:].strip() if first.startswith('#!') else ''
        lines.append(f"command     {exe} -> {launcher or '?'}")
        # compare the bin folders, not resolved files: a venv's python is a symlink to the system one
        if launcher and os.path.dirname(os.path.abspath(launcher.split()[0])) != os.path.dirname(os.path.abspath(sys.executable)):
            hints.append(f"the 'kicad2ems' command on PATH runs {launcher}, not this python - reinstall it here "
                         f"(python -m pip install <kicad2ems folder>) or use: python -m kicad2ems ...")
    in_venv = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    if not ok and in_venv:
        cfg = os.path.join(sys.prefix, 'pyvenv.cfg')
        try:
            sysite = 'include-system-site-packages = true' in open(cfg).read().replace('True', 'true')
        except OSError:
            sysite = True
        if not sysite:
            hints.append("this venv does not see system packages: if openEMS came from your distribution "
                         "(dnf / apt), recreate the venv with  python3 -m venv --system-site-packages <dir>")
    if hints:
        lines.append("hints:")
        lines += [f"  - {h}" for h in dict.fromkeys(hints)]
    return ok, lines
