"""openems_venv: an openEMS in another Python environment, its site-packages put last on sys.path."""
import sys
from kicad2ems_openems import venv


def _env(tmp_path, version):
    d = tmp_path / 'venv' / 'lib' / f'python{version}' / 'site-packages'
    d.mkdir(parents=True)
    (tmp_path / 'venv' / 'bin').mkdir()
    (tmp_path / 'venv' / 'bin' / 'python').write_text('')
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
    from kicad2ems_openems.plugin import OpenEMS
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
