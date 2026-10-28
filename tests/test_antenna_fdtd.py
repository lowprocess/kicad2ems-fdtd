"""Antenna mode on the FDTD model: the air and recorded frequencies, gap ports (a centre-fed dipole) on the mesh and
in the CSX model, and a phased drive from the runs (moved from kicad2ems's test_antenna.py)."""
import numpy as np
import pytest
from kicad2ems.importers.kicad import KiCadBoard
from kicad2ems.board import Board
from kicad2ems_fdtd.mesh import auto_mesh
from kicad2ems_fdtd.antenna import antenna_air, record_freqs


def test_defaults_and_record_freqs():
    assert antenna_air(2e9, 3e9) == pytest.approx(299.792458 / 2.5 / 4)
    assert antenna_air(2e9, 3e9, air=12.0) == 12.0
    assert antenna_air(20e9, 30e9) == 5.0                            # at least 5 mm
    f = record_freqs(2e9, 3e9, [2.43e9])
    assert len(f) == 22 and 2.43e9 in f and f[0] == 2e9 and f[-1] == 3e9


def test_dipole_gap_port(boards):
    m = Board(KiCadBoard(boards['dipole'] + '.kicad_pcb'), log=lambda *a: None)
    p = m.ports[0]
    assert p.kind == 'gap' and p.direction == 'x' and p.z0 == 73
    assert p.gap == pytest.approx((34.5, 35.5)) and p.w == pytest.approx(2.0) and p.c[1] == pytest.approx(-10.0)
    s, e = p.box()
    assert s[2] == e[2] == m.cu_z['F.Cu'] and (s[0], e[0]) == pytest.approx((34.5, 35.5))
    mesh = auto_mesh(m, 1.5e9, 3.5e9, bc=('PML_8',) * 6, air=30, log=lambda *a: None)
    for v in p.gap:                                                    # both gap edges are mesh lines
        assert np.min(np.abs(mesh['x'] - v)) < 1e-9
    assert mesh['info']['lines_across_ports']['P1'] >= 2


def test_gap_port_direction_in_fdtd(boards, fake_openems, monkeypatch):
    from kicad2ems_fdtd import model
    m = Board(KiCadBoard(boards['dipole'] + '.kicad_pcb'), log=lambda *a: None)
    mesh = auto_mesh(m, 1.5e9, 3.5e9, bc=('PML_8',) * 6, air=30, log=lambda *a: None)
    seen = []
    import openEMS
    orig = openEMS.openEMS.AddLumpedPort
    monkeypatch.setattr(openEMS.openEMS, 'AddLumpedPort',
                        lambda self, nr, R, s, e, d, **kw: seen.append(d) or orig(self, nr, R, s, e, d, **kw))
    model.build_fdtd(m, mesh, 1.5e9, 3.5e9, 0, bc=('PML_8',) * 6)
    assert seen == ['x']


def test_fdtd_phased_drive():
    """An FDTD phased drive from its runs (fdtd.antenna: analyse_port, combine_phased) on stand-ins of the engine:
    each run's NF2FF gives a point source at its port, as openEMS lays it out ((theta, phi) per frequency), scaled
    by the run's incident voltage, which analyse_port divides out. The beam follows the phase step, the cuts and
    the sphere agree, the array file steers as mom25d's does, and a single port gives what its run alone gives."""
    from types import SimpleNamespace
    from shapely.geometry import box
    from kicad2ems import phased
    from kicad2ems_fdtd.antenna import analyse_port, combine_phased, record_freqs
    f0, d = 2.45e9, 61.2                                             # half a wavelength apart along y (mm)
    ports = [SimpleNamespace(name=f"P{i + 1}", z0=50.0, c=(0.0, i * d)) for i in range(2)]
    m = SimpleNamespace(ports=ports, outline=box(-10, -10, 10, 10 + d), thickness=1.6)
    freq = np.linspace(2e9, 3e9, 21)
    ffr = record_freqs(freq[0], freq[-1])
    k0 = 2 * np.pi * ffr / 299792458.0 * 1e-3

    class NF2FF:                                                     # a point source at port k, x its incident wave
        def __init__(self, k, scale):
            self.k, self.scale = k, scale

        def CalcNF2FF(self, path, fr, theta, phi, center=None, outfile=None):
            TH, PH = np.meshgrid(np.radians(theta), np.radians(phi), indexing='ij')     # (ntheta, nphi)
            y = ports[self.k].c[1]
            E = [s * np.exp(1j * kk * y * np.sin(TH) * np.sin(PH)) for kk, s in zip(k0, self.scale)]
            return SimpleNamespace(E_theta=E, E_phi=[np.zeros_like(e) for e in E])

    class Port:
        def __init__(self, uf):
            self.uf = uf

        def CalcPort(self, path, fr):
            self.uf_inc, self.Z_ref = self.uf, 50.0

    extra = {}
    for k in range(2):
        v = (0.3 + 0.1 * k) * np.exp(1j * (0.2 + k)) * np.ones(len(ffr))   # each run's own excitation level
        a_inc = v / np.sqrt(50.0)
        extra[ports[k].name] = analyse_port(m, ffr, 2.0, path='x', ports={k: Port(v)}, token=NF2FF(k, a_inc),
                                            k=k, col=None)
    res = SimpleNamespace(S=np.zeros((len(freq), 2, 2), complex), extra=extra, ffr=ffr)
    for step in (0, 90, -90):
        a = phased.parse_drive(['P1@0', f"P2@{step}"], ['P1', 'P2'])
        ant, data, sph = combine_phased(m, freq, res, a, [0, 1], ff_pattern=f0, log=lambda *_: None)
        pk = ant['peak']
        assert ant['f_pattern'] == pytest.approx(f0, rel=0.03) and ant['drive'] == phased.format_drive(a, ['P1', 'P2'])
        # isotropic sources: the beam is a cone round the y axis, sin(theta) sin(phi) = -step / 180 (half a wave)
        v = np.sin(np.radians(pk['theta'])) * np.sin(np.radians(pk['phi']))
        assert abs(v + step / 180) < 0.06, (step, pk)
        cut = ant['cuts']['yz (phi = 90)']
        assert max(cut['gain_dbi']) == pytest.approx(pk['gain_dbi'], abs=0.3)
        assert phased.combine(data, a)['peak'] == pk and data['S'].shape == (len(ffr), 2, 2)
    one, _, _ = combine_phased(m, freq, res, np.array([1, 0], complex), [0, 1], ff_pattern=f0, log=lambda *_: None)
    row = min(one['rows'], key=lambda r: abs(r['f'] - f0))
    assert row['D_dbi'] == pytest.approx(0.0, abs=0.1)              # one isotropic source
    pinc = 0.5                                                       # a unit incident wave: 1/2 W
    prad = 4 * np.pi / (2 * 376.730313668)                           # |r E| = 1 V over the sphere
    assert row['Prad'] == pytest.approx(prad, rel=0.02) and row['Pinc'] == pytest.approx(pinc)
