"""
An openEMS installed in another Python environment (the GUI's Solvers and hardware window, or setup.sh): the
site setting openems_venv (sitecfg: a venv folder, or its python) puts that environment's site-packages at the end
of sys.path, so CSXCAD and openEMS come from there and everything else from kicad2ems's own environment. Their
compiled modules load only in the same Python version (3.13's do not in 3.14). A Windows environment
(Scripts\\python.exe, Lib\\site-packages) names its version in pyvenv.cfg, or its python3XX.dll.

On Windows the openEMS and CSXCAD DLLs come from a folder named by the openems_dir setting (the unpacked openEMS
release; its bin folder too where it has one) or OPENEMS_INSTALL_PATH / CSXCAD_INSTALL_PATH: dll_dirs() adds them
to the DLL search (os.add_dll_directory), which is what LD_LIBRARY_PATH is on Linux.

    venv.activate() -> None, or why the setting could not be used (a note for the check)
    venv.dll_dirs() -> [folders added to Windows' DLL search]
"""
import glob, os, re, sys

_done = None


def site_dirs(path):
    """The site-packages folders of a venv (its folder, or its python): [(python version 'X.Y', folder)]."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(p):                                       # .../venv/bin/python, ...\\Scripts\\python.exe
        p = os.path.dirname(p)
        if os.path.basename(p).lower() in ('bin', 'scripts'):
            p = os.path.dirname(p)
    out = []
    for d in sorted(glob.glob(os.path.join(p, 'lib*', 'python3.*', 'site-packages'))):
        out.append((os.path.basename(os.path.dirname(d))[len('python'):], d))
    win = os.path.join(p, 'Lib', 'site-packages')               # Windows: a venv, or a Python's own folder
    if not out and os.path.isdir(win) and (ver := windows_version(p)):
        out.append((ver, win))
    return out


def windows_version(folder):
    """The 'X.Y' of a Windows Python environment: pyvenv.cfg's version (a venv), else its python3XX.dll (an
    installed Python); None where neither tells."""
    try:
        with open(os.path.join(folder, 'pyvenv.cfg'), encoding='utf-8') as fh:
            for line in fh:
                k, _, v = line.partition('=')
                if k.strip() in ('version', 'version_info') and (m := re.match(r'\s*(\d+)\.(\d+)', v)):
                    return f"{m.group(1)}.{m.group(2)}"
    except OSError:
        pass
    for dll in glob.glob(os.path.join(folder, 'python3*.dll')):
        if m := re.fullmatch(r'python3(\d+)\.dll', os.path.basename(dll), re.I):
            return f"3.{m.group(1)}"
    return None


_dll_dirs = None


def dll_dirs(dirs=None):
    """Windows: openEMS's DLL folders into the DLL search, once (the openems_dir setting, OPENEMS_INSTALL_PATH,
    CSXCAD_INSTALL_PATH; each with its bin folder where it has one). Returns the folders added; [] elsewhere."""
    global _dll_dirs
    if sys.platform != 'win32' or not hasattr(os, 'add_dll_directory'):
        return []
    if dirs is None and _dll_dirs is not None:
        return _dll_dirs
    if dirs is None:
        from kicad2ems.api.solver import sitecfg
        dirs = [sitecfg.get('openems_dir'), os.environ.get('OPENEMS_INSTALL_PATH'),
                os.environ.get('CSXCAD_INSTALL_PATH')]
    added = []
    for d in dirs:
        if not d:
            continue
        d = os.path.abspath(os.path.expanduser(d))
        for f in (os.path.join(d, 'bin'), d):
            if os.path.isdir(f) and f not in added and glob.glob(os.path.join(f, '*.dll')):
                os.add_dll_directory(f)
                added.append(f)
    _dll_dirs = added
    return added


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
        _done = (f"openems_venv {path}: no Python environment there (no lib/python3.*/site-packages, or "
                 f"Lib\\site-packages on Windows)")
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
