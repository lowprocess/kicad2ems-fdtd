"""
An openEMS installed in another Python environment (the GUI's Solvers and hardware window, or setup.sh): the
site setting openems_venv (sitecfg: a venv folder, or its python) puts that environment's site-packages at the end
of sys.path, so CSXCAD and openEMS come from there and everything else from kicad2ems's own environment. Their
compiled modules load only in the same Python version (3.13's do not in 3.14).

    venv.activate() -> None, or why the setting could not be used (a note for the check)
"""
import glob, os, sys

_done = None


def site_dirs(path):
    """The site-packages folders of a venv (its folder, or its python): [(python version 'X.Y', folder)]."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p):                                       # .../venv/bin/python
        p = os.path.dirname(os.path.dirname(p))
    out = []
    for d in sorted(glob.glob(os.path.join(p, 'lib*', 'python3.*', 'site-packages'))):
        out.append((os.path.basename(os.path.dirname(d))[len('python'):], d))
    return out


def activate(path=None):
    """The openems_venv setting (or path) onto sys.path, once; None when used (or none set), else the reason."""
    global _done
    if path is None:
        if _done is not None:
            return _done or None
        from kicad2ems.api.solver import sitecfg
        path = sitecfg.get('openems_venv')
        if not path:
            _done = ''
            return None
    want = f"{sys.version_info.major}.{sys.version_info.minor}"
    dirs = site_dirs(path)
    if not dirs:
        _done = f"openems_venv {path}: no Python environment there (no lib/python3.*/site-packages)"
        return _done
    mine = [d for v, d in dirs if v == want]
    if not mine:
        _done = (f"openems_venv {path} is a Python {', '.join(v for v, _ in dirs)} environment; kicad2ems runs on "
                 f"Python {want} ({sys.executable}), and openEMS's compiled modules load only in their own version")
        return _done
    for d in mine:
        if d not in sys.path:
            sys.path.append(d)                                  # last: kicad2ems's own packages come first
    _done = ''
    return None
