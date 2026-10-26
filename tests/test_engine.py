"""Validate the nuearth engine against MANGO's own public API.

    source env.sh && python -m pytest tests -q        (or: python tests/test_engine.py)
"""

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import engine  # noqa: E402
import mango  # noqa: E402
from mango import OscParams  # noqa: E402

E = np.logspace(0, 2, 24)
CZ = np.linspace(-1, 1, 17)
EARTH = dict(kind="prem", n_sub=2, h_atm_km=15.0)


def rotate(s_red, theta23, delta, anti=False):
    """S = O S' O^dagger with O = R23 . diag(1, 1, e^{i delta})  (delta -> -delta for anti)."""
    d = -delta if anti else delta
    c, s = np.cos(theta23), np.sin(theta23)
    ph = np.exp(1j * d)
    o = np.array([[1, 0, 0], [0, c, s * ph], [0, -s, c * ph]], dtype=complex)
    return o @ s_red @ o.conj().T


def full_prob(th12, th13, th23, dcp, dm21, dm31, anti):
    p = OscParams(*(jnp.asarray(v) for v in (th12, th13, th23, dcp, dm21, dm31)))
    return np.asarray(mango.probability_earth(p, E, CZ, n_sub=2, h_atm_km=15.0, anti=anti))


def test_reduced_basis_matches_mango():
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(4):
        th12, th13 = rng.uniform(0.5, 0.65), rng.uniform(0.12, 0.17)
        th23, dcp = rng.uniform(0.6, 1.0), rng.uniform(0, 2 * np.pi)
        dm21 = 7.4e-5
        dm31 = (1 if trial % 2 == 0 else -1) * rng.uniform(2.3e-3, 2.7e-3)
        osc = dict(theta12=th12, theta13=th13, dm21=dm21, dm31=dm31)
        for anti in (False, True):
            s_red = engine.amplitude_grid(osc, EARTH, E, CZ, anti=anti)
            p_rot = np.abs(rotate(s_red, th23, dcp, anti)) ** 2
            p_ref = full_prob(th12, th13, th23, dcp, dm21, dm31, anti)
            worst = max(worst, np.max(np.abs(p_rot - p_ref)))
    print(f"reduced basis vs mango.probability_earth: max |dP| = {worst:.2e}")
    assert worst < 1e-10


def test_path_endpoint_matches_grid():
    osc = dict(theta12=0.5836, theta13=0.1495, dm21=7.42e-5, dm31=2.515e-3)
    worst, worst_u = 0.0, 0.0
    for e_gev, cz in [(6.0, -0.75), (4.0, -0.95), (25.0, -0.3), (2.0, 0.4)]:
        for anti in (False, True):
            path = engine.path_evolution(osc, EARTH, e_gev, cz, n_points=200, anti=anti)
            s_end = path["S"][-1]
            s_grid = engine.amplitude_grid(osc, EARTH, [e_gev], [cz], anti=anti)[0, 0]
            worst = max(worst, np.max(np.abs(s_end - s_grid)))
            uu = np.einsum("nij,nkj->nik", path["S"], path["S"].conj()) - np.eye(3)
            worst_u = max(worst_u, np.max(np.abs(uu)))
            assert np.allclose(path["S"][0], np.eye(3))
    print(f"path end vs grid amplitude: max |dS| = {worst:.2e}; unitarity {worst_u:.1e}")
    assert worst < 1e-10 and worst_u < 1e-10


def test_density_derivative_matches_finite_difference():
    osc = dict(theta12=0.5836, theta13=0.1495, dm21=7.42e-5, dm31=2.515e-3)
    e, cz = np.array([3.0, 5.0]), np.array([-0.95, -0.9])
    for wrt, h in [("core_scale", 1e-5), ("dm31", 1e-9)]:
        s, ds = engine.amplitude_grid(osc, EARTH, e, cz, wrt=wrt)
        if wrt == "dm31":
            sp = engine.amplitude_grid({**osc, "dm31": osc["dm31"] + h}, EARTH, e, cz)
            sm = engine.amplitude_grid({**osc, "dm31": osc["dm31"] - h}, EARTH, e, cz)
        else:
            sp = engine.amplitude_grid(osc, {**EARTH, wrt: 1 + h}, e, cz)
            sm = engine.amplitude_grid(osc, {**EARTH, wrt: 1 - h}, e, cz)
        fd = (sp - sm) / (2 * h)
        rel = np.max(np.abs(ds - fd)) / np.max(np.abs(ds))
        print(f"dS'/d{wrt}: autodiff vs central FD, relative {rel:.1e}")
        assert rel < 1e-5


if __name__ == "__main__":
    test_reduced_basis_matches_mango()
    test_path_endpoint_matches_grid()
    test_density_derivative_matches_finite_difference()
