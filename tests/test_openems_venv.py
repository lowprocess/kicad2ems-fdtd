"""openems_venv: an openEMS in another Python environment, its site-packages put last on sys.path."""
import sys
from kicad2ems_fdtd.openems import venv


def _env(tmp_path, version):
    d = tmp_path / 'venv' / 'lib' / f'python{version}' / 'site-packages'
    d.mkdir(parents=True)
    (tmp_path / 'venv' / 'bin').mkdir()
    (tmp_path / 'venv' / 'bin' / 'python').write_text('', encoding='utf-8')
    return tmp_path / 'venv', d


def test_same_version_goes_last_on_sys_path(tmp_path, monkeypatch):
    mine = f"{sys.version_info.major}.{sys.version_info.minor}"
    root, site = _env(tmp_path, mine)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert venv.site_dirs(root / 'bin' / 'python') == [(mine, str(site))]           # the folder or its python
    assert venv.activate(str(root)) is None and sys.path[-1] == str(site)
    assert venv.activate(str(root)) is None and sys.path.count(str(site)) == 1        # once


def test_other_version_or_no_env_says_why(tmp_path, monkeypatch):
    root, site = _env(tmp_path, '3.9')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    why = venv.activate(str(root))
    assert 'Python 3.9 environment' in why and str(site) not in sys.path
    assert 'no Python environment' in venv.activate(str(tmp_path / 'nothing'))


def test_the_check_names_the_setting(tmp_path, monkeypatch):
    """The openEMS check reports the setting, and when it is why openEMS is missing, says so first."""
    from kicad2ems import sitecfg
    from kicad2ems_fdtd.openems.plugin import OpenEMS
    root, _ = _env(tmp_path, '3.9')
    monkeypatch.setattr(sitecfg, '_cache', {'openems_venv': str(root)})
    monkeypatch.setattr(venv, '_done', None)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    ok, lines = OpenEMS().details()
    assert any(l.startswith('openems_venv') and 'not used' in l for l in lines)
    if not ok:
        assert 'Python 3.9 environment' in OpenEMS().check()[1]
    monkeypatch.setattr(sitecfg, '_cache', None)
    monkeypatch.setattr(venv, '_done', None)


def test_windows_layout(tmp_path):
    """A Windows venv (Scripts\\python.exe, Lib\\site-packages): its version from pyvenv.cfg; an installed Python's
    own folder: from its python3XX.dll."""
    v = tmp_path / 'venv'
    (v / 'Lib' / 'site-packages').mkdir(parents=True)
    (v / 'Scripts').mkdir()
    (v / 'Scripts' / 'python.exe').write_text('', encoding='utf-8')
    (v / 'pyvenv.cfg').write_text('home = C:\\Python313\nversion = 3.13.1\n', encoding='utf-8')
    site = str(v / 'Lib' / 'site-packages')
    assert venv.site_dirs(str(v)) == [('3.13', site)] == venv.site_dirs(str(v / 'Scripts' / 'python.exe'))
    p = tmp_path / 'Python312'
    (p / 'Lib' / 'site-packages').mkdir(parents=True)
    (p / 'python312.dll').write_text('', encoding='utf-8')
    (p / 'python.exe').write_text('', encoding='utf-8')
    assert venv.site_dirs(str(p / 'python.exe')) == [('3.12', str(p / 'Lib' / 'site-packages'))]
    assert venv.site_dirs(str(tmp_path / 'nothing')) == []


def test_dll_dirs(tmp_path, monkeypatch):
    """Windows: the openEMS folder and its bin folder, where they hold DLLs, into the DLL search; nothing
    elsewhere."""
    import os
    monkeypatch.setattr(venv, '_dll_dirs', None)
    assert venv.dll_dirs([str(tmp_path)]) == [] or sys.platform == 'win32'
    added = []
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(os, 'add_dll_directory', added.append, raising=False)
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin' / 'openEMS.dll').write_text('', encoding='utf-8')
    (tmp_path / 'CSXCAD.dll').write_text('', encoding='utf-8')
    (tmp_path / 'empty').mkdir()
    assert venv.dll_dirs([str(tmp_path), str(tmp_path / 'empty'), None]) == [str(tmp_path / 'bin'), str(tmp_path)]
    assert added == [str(tmp_path / 'bin'), str(tmp_path)]
    monkeypatch.setattr(venv, '_dll_dirs', None)
