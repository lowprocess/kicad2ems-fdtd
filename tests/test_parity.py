"""Parity: each solver's native model read back against the board, minus what the solver declared. Every
difference must be declared; a deliberately undeclared change (a via moved 0.1 mm) fails. Here the FDTD model
(kicad2ems-fdtd's, openEMS's and gfdtd's); mom25d's planar model is checked the same way in mom25d's tests."""
import os
from kicad2ems.cli import settings_from_argv
from kicad2ems.pipeline import Job
from kicad2ems.solvers import registry
from kicad2ems.parity import check


def _prepared(tmp_path, base, *args):
    """A job with its board and line model, and the FDTD model prepared (the mesh of a preview)."""
    s, _ = settings_from_argv([base + '.kicad_pcb', '--pos', base + '-top.pos', '--preview', '--out', str(tmp_path),
                               *args])
    j = Job(s, echo=lambda *a: None)
    os.makedirs(s.out, exist_ok=True)
    j.model()
    j.lines()
    return j, registry()['openems'].prepare(j)


CASES = dict(transition=('demo', ['--nets', 'RF,GND', '--ground', 'B.Cu']),           # snapped and blind vias
             lowpass=('lowpass', []),                                                # parts with ESR / ESL
             launch=('launch', ['--edge-walls', 'all']))                             # edge walls, inner planes


def test_every_difference_is_declared(boards, tmp_path):
    for case, (name, args) in CASES.items():
        j, prep = _prepared(tmp_path / case, boards[name], *args)
        sv = registry()['openems']
        devs = sv.deviations(j, prep)
        assert check(j.m, sv.readback(j, prep), devs) == [], case
        seen = {d.feature for d in devs}
        if case == 'transition':
            assert 'vias' in seen                                   # the FDTD mesh snaps the stitching vias
        if case == 'lowpass':
            assert 'parts' in seen                                  # and splits the parts with ESR


def test_undeclared_changes_fail(boards, tmp_path):
    name, args = CASES['transition']
    j, prep = _prepared(tmp_path, boards[name], *args)
    sv = registry()['openems']
    devs = sv.deviations(j, prep)
    x, y, r, z0, z1 = prep.grid['vias'][0]
    prep.grid['vias'][0] = (x + 0.1, y, r, z0, z1)                  # moved after it was declared: not declared
    problems = check(j.m, sv.readback(j, prep), devs)
    assert len(problems) == 2 and all(p.startswith('via:') for p in problems)


def test_a_dropped_declaration_fails(boards, tmp_path):
    j, prep = _prepared(tmp_path, boards['lowpass'])
    sv = registry()['openems']
    devs = [d for d in sv.deviations(j, prep) if not d.item.startswith('part:')]       # parts left unsaid
    problems = check(j.m, sv.readback(j, prep), devs)
    assert any(p.startswith(('part', 'copper on F.Cu')) for p in problems), problems
