"""
Antenna mode on an FDTD engine: one excited port, absorbing boundaries, a quarter wavelength of air and a
near-field to far-field (NF2FF) box recorded at a set of frequencies; after the run, the far field analysed
(fields/antenna.py: match, gain, polarisation, patterns).

A phased drive (--drive several ports) is one run per driven port, the others terminated: an incident wave on that
port alone. Each run's far field divided by its incident wave is the field per unit wave, so any drive a is
sum_k a_k E_k, exact at every recorded frequency and with the coupling between the elements (phased.py's incident
basis). run_phased runs them; combine_phased gives the drive's antenna result and the array file the GUI steers.
"""
import functools
import numpy as np
from kicad2ems.api.solver import C0, ETA0
from kicad2ems.api.solver import match_summary, gain_table, add_polarisation, pattern_gain, describe_antenna as describe


def antenna_air(fmin, fmax, air=None):
    """Air margin (mm) of antenna runs: lambda / 4 at the band centre, at least 5 mm (unless given)."""
    return max(5.0, C0 / (0.5 * (fmin + fmax)) * 1e3 / 4) if air is None else air



def exc_fmax(fmin, fmax):
    """Upper edge of the antenna runs' excitation: 30 % of the band beyond fmax."""
    return fmax + 0.3 * (fmax - fmin)



def record_freqs(fmin, fmax, extra=(), n=21):
    """Frequencies (Hz) at which the NF2FF box records: n over the band plus any given ones."""
    return np.unique(np.concatenate([np.linspace(fmin, fmax, n), np.asarray(extra, float)]))



def nf2ff_box(freqs):
    """Run hook (simulation.run_excitations setup=): an NF2FF box recording at `freqs` (Hz)."""
    return functools.partial(_add_nf2ff, [float(f) for f in freqs])



def _add_nf2ff(freqs, FDTD, CSX):
    return FDTD.CreateNF2FFBox(frequency=freqs)



def analyse(m, freq, ffr, ff_pattern=None, step=2.0, log=print, *, path, ports, token, k, col):
    """Run hook (analyse=): input match, gain over frequency and the pattern cuts from a finished antenna
    run (token: the NF2FF box). Returns the result dict."""
    nf2ff, pk = token, ports[k]
    match = match_summary(freq, col[:, k], m.ports[k].z0)
    pk.CalcPort(path, ffr)
    p_acc, p_inc = np.atleast_1d(pk.P_acc), np.atleast_1d(pk.P_inc)
    ob = m.outline.bounds
    centre = [0.5 * (ob[0] + ob[2]) * 1e-3, 0.5 * (ob[1] + ob[3]) * 1e-3, 0.5 * m.thickness * 1e-3]
    # gain over frequency: Dmax / Prad on a coarse full sphere at every recorded frequency
    th, ph = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)
    full = nf2ff.CalcNF2FF(path, list(ffr), th, ph, center=centre, outfile='nf2ff_sphere.h5')
    rows = gain_table(ffr, np.atleast_1d(full.Dmax), np.atleast_1d(full.Prad), p_acc, p_inc)
    add_polarisation(rows, full.E_theta, full.E_phi, th, ph)
    ip = int(np.argmin(np.abs(ffr - (ff_pattern if ff_pattern else match['f_res']))))
    f_pat, eff, dmax = float(ffr[ip]), rows[ip]['eff'], float(np.atleast_1d(full.Dmax)[ip])
    e_max = float(np.max(full.E_norm[ip]))                       # one normalisation for all cuts
    tc = np.arange(-180, 180 + step / 2, step)
    cuts = {}
    for name, theta, phi in (('xz (phi = 0)', tc, [0.0]), ('yz (phi = 90)', tc, [90.0]),
                             ('xy (theta = 90)', [90.0], np.arange(0, 360 + step / 2, step))):
        r = nf2ff.CalcNF2FF(path, [f_pat], theta, phi, center=centre, outfile=f"nf2ff_cut_{name.split()[0]}.h5")
        g = (np.asarray(r.E_norm[0], float).ravel() / e_max) ** 2 * dmax * eff
        cuts[name] = dict(angle=list(map(float, theta if len(theta) > 1 else phi)),
                          gain_dbi=[float(v) for v in 10 * np.log10(np.maximum(g, 1e-6))])
    sphere = dict(theta=th, phi=ph, gain=pattern_gain(full.E_norm[ip], dmax, eff))
    for line in describe(match, rows, f_pat):
        log(line)
    return dict(match=match, rows=rows, f_pattern=f_pat, cuts=cuts, sphere=sphere, port=m.ports[k].name)



CUTS = (('xz (phi = 0)', 0.0), ('yz (phi = 90)', 90.0))      # theta cuts; and the xy cut at theta = 90


def cut_grids(step):
    """The pattern cuts' angles: theta from -180 to 180 (deg) in the xz and yz planes, phi round at theta 90."""
    return np.arange(-180, 180 + step / 2, step), np.arange(0, 360 + step / 2, step)


def analyse_port(m, ffr, step, *, path, ports, token, k, col):
    """Run hook of a phased drive's run (port k driven): its far field per unit incident wave at every recorded
    frequency, on the coarse sphere (Eth, Eph: (nf, nphi, ntheta)) and the cut planes (cuts: {name: (Et, Ep)},
    (nf, angles)). The incident wave is uf_inc / sqrt(Z_ref) at the port (CalcPort)."""
    nf2ff, pk = token, ports[k]
    pk.CalcPort(path, ffr)
    a_inc = np.atleast_1d(pk.uf_inc) / np.sqrt(np.real(pk.Z_ref))
    ob = m.outline.bounds
    centre = [0.5 * (ob[0] + ob[2]) * 1e-3, 0.5 * (ob[1] + ob[3]) * 1e-3, 0.5 * m.thickness * 1e-3]
    th, ph = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)
    full = nf2ff.CalcNF2FF(path, list(ffr), th, ph, center=centre, outfile='nf2ff_sphere.h5')
    norm = lambda E: np.array([np.asarray(e) / a for e, a in zip(E, a_inc)])
    out = dict(Eth=norm(full.E_theta).transpose(0, 2, 1), Eph=norm(full.E_phi).transpose(0, 2, 1), cuts={})
    tc, pa = cut_grids(step)
    for name, phi0 in CUTS:
        r = nf2ff.CalcNF2FF(path, list(ffr), tc, [phi0], center=centre, outfile=f"nf2ff_cut_{name.split()[0]}.h5")
        out['cuts'][name] = (norm(r.E_theta)[:, :, 0], norm(r.E_phi)[:, :, 0])
    r = nf2ff.CalcNF2FF(path, list(ffr), [90.0], pa, center=centre, outfile='nf2ff_cut_xy.h5')
    out['cuts']['xy (theta = 90)'] = (norm(r.E_theta)[:, 0, :], norm(r.E_phi)[:, 0, :])
    return out


def run_phased(m, mesh, freq, excite, bc, out_dir, ff_extra=(), step=2.0, log=print, jobs=1, **run_kw):
    """The runs of a phased drive: each port of excite driven once, with an NF2FF box (analyse_port). Returns the
    RunResult; RunResult.extra[port name] holds each run's far field per unit incident wave."""
    from .run import run_excitations
    from .engines import engine_name
    ffr = record_freqs(freq[0], freq[-1], ff_extra)
    log(f"{engine_name(run_kw.get('engine'))[1]}: phased drive, {len(excite)} runs ("
        f"{', '.join(m.ports[k].name for k in excite)} each driven alone), NF2FF box recording {len(ffr)} frequencies")
    res = run_excitations(m, mesh, freq, list(excite), bc, out_dir, jobs=jobs, exc_fmax=exc_fmax(freq[0], freq[-1]),
                          setup=nf2ff_box(ffr), log=log, analyse=functools.partial(analyse_port, m, ffr, step),
                          **run_kw)
    res.ffr = ffr
    return res


def _interp_S(freq, S, ffr):
    """S (nf, n, n) over freq (Hz) at the recorded frequencies ffr, real and imaginary parts interpolated."""
    n = S.shape[1]
    out = np.empty((len(ffr), n, n), complex)
    for i in range(n):
        for j in range(n):
            out[:, i, j] = np.interp(ffr, freq, S[:, i, j].real) + 1j * np.interp(ffr, freq, S[:, i, j].imag)
    return out


def combine_phased(m, freq, res, a, excite, ff_pattern=None, step=2.0, log=print, pos=None):
    """The antenna of drive a (incident waves, one per port of m.ports) from run_phased's runs: (result dict as
    analyse's, with drive, peak and active_db, and the array data for phased.save). The pattern frequency: the
    given one, else the resonance of the first driven port's active reflection."""
    from kicad2ems.api.solver import phased
    ffr, n = res.ffr, len(m.ports)
    S = np.where(np.isfinite(res.S), res.S, 0)                     # the ports not run: no column (not driven)
    names = [p.name for p in m.ports]
    zc = np.array([p.z0 for p in m.ports], float)
    th, ph = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)
    shape = (len(ffr), n, len(ph), len(th))
    Eth, Eph = np.zeros(shape, complex), np.zeros(shape, complex)
    for k in excite:
        x = res.extra[names[k]]
        Eth[:, k], Eph[:, k] = x['Eth'], x['Eph']
    excited = np.zeros(n, bool)
    excited[list(excite)] = True
    data = dict(theta=th, phi=ph, theta_sphere=th, freq=ffr, ports=names, zc=zc, S=_interp_S(freq, S, ffr),
                excited=excited, Eth=Eth, Eph=Eph, drive=np.asarray(a, complex), f_pattern=float(ffr[0]),
                pos=np.array([pos.get(nm, (np.nan, np.nan)) for nm in names]) if pos else None)
    driven = [j for j in range(n) if abs(a[j]) > 0]
    k = driven[0]
    s11 = (S @ np.asarray(a, complex))[:, k] / a[k]                    # the first driven port's active reflection
    match = match_summary(freq, s11, m.ports[k].z0)
    ip = int(np.argmin(np.abs(ffr - (ff_pattern if ff_pattern else match['f_res']))))
    data['f_pattern'] = float(ffr[ip])
    comb = phased.combine(data, a)
    rows, sph = comb['rows'], comb['sphere']
    pacc = rows[ip]['Pacc']
    tc, pa = cut_grids(step)
    cuts = {}
    for name in [c for c, _ in CUTS] + ['xy (theta = 90)']:
        Et = sum(a[j] * res.extra[names[j]]['cuts'][name][0][ip] for j in driven)
        Ep = sum(a[j] * res.extra[names[j]]['cuts'][name][1][ip] for j in driven)
        g = 4 * np.pi * (np.abs(Et) ** 2 + np.abs(Ep) ** 2) / (2 * ETA0) / pacc
        cuts[name] = dict(angle=list(map(float, tc if name != 'xy (theta = 90)' else pa)),
                          gain_dbi=[float(v) for v in 10 * np.log10(np.maximum(g, 1e-6))])
    e_pat = sph['e_norm'][ip]
    drive = phased.format_drive(a, names)
    for line in describe(match, rows, float(ffr[ip])):
        log(line)
    pk = comb['peak']
    act = {names[j]: float(20 * np.log10(max(abs(comb['active'][ip][j]), 1e-15))) for j in driven}
    log(f"  phased drive {', '.join(drive)}: beam at theta {pk['theta']:g}, phi {pk['phi']:g} deg; active S11 "
        + ', '.join(f"{nm} {v:.1f} dB" for nm, v in act.items()) + f" at {ffr[ip] / 1e9:.4g} GHz")
    ant = dict(match=match, rows=rows, f_pattern=float(ffr[ip]), cuts=cuts, port=', '.join(drive), drive=drive,
               sphere=dict(theta=th, phi=ph, gain=((e_pat / e_pat.max()) ** 2 * sph['dmax'][ip] * rows[ip]['eff']).T),
               peak=pk, active_db=act, s11=s11)
    return ant, data, sph


def run_antenna(m, mesh, freq, k, bc, out_dir, ff_extra=(), ff_pattern=None, step=2.0, log=print, **run_kw):
    """FDTD run with port k excited and an NF2FF box: simulation.run_excitations with the antenna hooks.
    The excitation reaches 30 % of the band beyond fmax: in the top ~15 % of a Gaussian's band (its -20 dB
    edge) the far field came out wrong (dipole: 6.3 dBi and 107 % efficiency at fmax). Returns the
    RunResult; RunResult.extra[port name] holds the antenna results."""
    from .run import run_excitations
    from .engines import engine_name
    ffr = record_freqs(freq[0], freq[-1], ff_extra)
    log(f"{engine_name(run_kw.get('engine'))[1]}: antenna run, {m.ports[k].name} driven, NF2FF box recording "
        f"{len(ffr)} frequencies")
    return run_excitations(m, mesh, freq, [k], bc, out_dir, jobs=1, exc_fmax=exc_fmax(freq[0], freq[-1]),
                           setup=nf2ff_box(ffr), log=log,
                           analyse=functools.partial(analyse, m, freq, ffr, ff_pattern, step, log), **run_kw)
