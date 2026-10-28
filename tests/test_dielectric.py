"""Wideband dielectric: Djordjevic-Sarkar model and its Debye-pole fit."""
import numpy as np
import pytest
from kicad2ems_fdtd.dielectric import djordjevic_sarkar, fit_debye, debye_eps


@pytest.mark.parametrize('er,tand', [(4.3, 0.02), (3.48, 0.0037)])
def test_ds_matches_datasheet_point(er, tand):
    e = djordjevic_sarkar(er, tand, 1e9, 1e9)
    assert e.real == pytest.approx(er) and -e.imag / e.real == pytest.approx(tand)
    hi = djordjevic_sarkar(er, tand, 1e9, 10e9)
    assert hi.real < er                                              # er falls slowly with frequency


@pytest.mark.parametrize('er,tand', [(4.3, 0.02), (3.48, 0.0037)])
@pytest.mark.parametrize('fmin,fmax', [(0.5e9, 10e9), (0, 40e9)])
def test_debye_fit_follows_ds(er, tand, fmin, fmax):
    e_inf, poles, kappa = fit_debye(er, tand, 1e9, fmin, fmax)
    assert e_inf >= 1 and kappa >= 0 and all(d > 0 and t > 0 for d, t in poles)
    f = np.linspace(max(fmin, fmax / 100), fmax, 101)
    ds, fit = djordjevic_sarkar(er, tand, 1e9, f), debye_eps(e_inf, poles, kappa, f)
    assert np.abs(fit.real / ds.real - 1).max() < 0.01
    assert np.abs((fit.imag / fit.real) / (ds.imag / ds.real) - 1).max() < 0.05


def test_debye_fit_drops_short_relaxation_times():
    _, poles, _ = fit_debye(4.3, 0.02, 1e9, 0.5e9, 10e9, tau_min=10e-12)
    assert poles and min(t for _, t in poles) >= 10e-12
