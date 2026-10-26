"""MANGO reference engine for nuearth.

The web app computes in the browser with mango-lite (web/js/mango-lite.js).  This
module computes the same quantities with MANGO itself
(github.com/pgranger23/mango-osc), and scripts/make_golden.py turns them into the
golden grids the port is tested against.  It uses MANGO's PREM chord geometry
(``mango.earth``), the matter potentials (``mango.constants``) and the layered
evolution operator (``mango.layers.propagate_layers``, cayley backend), and only
adds two things around them:

1. **Reduced basis.**  The matter term ``diag(V, 0, 0)`` commutes with
   ``O = R23 . diag(1, 1, e^{i delta})``, and the PDG PMNS factorises as
   ``U = O . R13 . R12``.  So for *any* matter profile

       S(theta23, delta) = O . S'(theta23 = 0, delta = 0) . O^dagger.

   ``S'`` is computed with MANGO and ``O`` is applied exactly afterwards, which
   makes theta23, delta_CP and the channel instantaneous in the app.
   Antineutrinos: ``S'`` with ``anti=True`` and ``O`` with ``delta -> -delta``.
   (Not valid with NSI or steriles.)

2. **Path-resolved evolution.**  The cumulative operator at points along one
   chord, for the "journey through the Earth" view.

Amplitudes follow MANGO's convention ``S[beta, alpha] = <nu_beta|S|nu_alpha>``.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import jax
import jax.numpy as jnp

import mango
from mango import OscParams
from mango import constants as C
from mango import earth as _earth
from mango.layers import propagate_layers
from mango.hamiltonian import matter_hamiltonian
from mango.propagator import propagator

R_EARTH_KM = C.R_EARTH_KM
CORE_RADIUS_KM = _earth.CORE_RADIUS_KM
YE_CORE = _earth.YE_CORE_DEFAULT
YE_MANTLE = _earth.YE_MANTLE_DEFAULT
INNER_CORE_KM = 1221.5


def reduced_params(theta12, theta13, dm21, dm31):
    """OscParams in the reduced basis (theta23 = delta = 0); angles in radians."""
    return OscParams(theta12=jnp.asarray(theta12), theta13=jnp.asarray(theta13),
                     theta23=jnp.asarray(0.0), deltacp=jnp.asarray(0.0),
                     dm21=jnp.asarray(dm21), dm31=jnp.asarray(dm31))


# --------------------------------------------------------------------------
# Earth models: (rho, ye, length_km) per segment, source -> detector.
# --------------------------------------------------------------------------

# Continuous inputs, all traced (a slider move never recompiles).  The first
# four are oscillation parameters (radians, eV^2 with signed dm31).
INPUTS = ("theta12", "theta13", "dm21", "dm31", "core_scale", "mantle_scale",
          "ye_core", "rho_uniform", "h_atm_km", "det_depth_km")
EARTH_DEFAULTS = dict(kind="prem", n_sub=2, core_scale=1.0, mantle_scale=1.0,
                      ye_core=YE_CORE, rho_uniform=5.51, h_atm_km=15.0,
                      det_depth_km=0.0)
KINDS = ("prem", "uniform", "vacuum")


def _segments(cz, kind, n_sub, x):
    """Chord segments for one cos(zenith).  ``kind`` ('prem' | 'uniform' |
    'vacuum') and ``n_sub`` are static; ``x`` maps INPUTS to (traced) values:
    PREM density multipliers ``core_scale`` / ``mantle_scale``, core electron
    fraction ``ye_core``, ``rho_uniform`` (g/cm^3), production height and
    detector depth (km)."""
    table = _earth.shell_table(n_sub)
    rho, ye, length = _earth.chord_segments(
        cz, table, h_atm_km=x["h_atm_km"], det_depth_km=x["det_depth_km"],
        ye_core=YE_CORE, ye_mantle=YE_MANTLE)
    # chord_segments puts ye_core exactly on the segments inside the core radius
    in_core = ye == YE_CORE
    in_earth = rho > 0.0
    if kind == "prem":
        rho = rho * jnp.where(in_core, x["core_scale"], x["mantle_scale"])
        ye = jnp.where(in_core, x["ye_core"], ye)
    elif kind == "uniform":
        rho = jnp.where(in_earth, x["rho_uniform"], 0.0)
        ye = jnp.where(in_earth, 0.5, ye)
    elif kind == "vacuum":
        rho = jnp.zeros_like(rho)
    else:
        raise ValueError(f"unknown Earth model {kind!r}")
    return rho, ye, length


def _amplitude(x, e_gev, cz, kind, n_sub, anti):
    p = reduced_params(x["theta12"], x["theta13"], x["dm21"], x["dm31"])
    rho, ye, length_km = _segments(cz, kind, n_sub, x)
    v_cc, _ = C.matter_potentials(rho, ye)
    return propagate_layers(p.pmns(), p.msquared(), e_gev * C.GEV_TO_EV,
                            v_cc, length_km * C.KM_TO_INV_EV, anti=anti,
                            backend="cayley")


def _grid(x, e, cz, kind, n_sub, anti):
    one = lambda c: jax.vmap(lambda ee: _amplitude(x, ee, c, kind, n_sub, anti))(e)
    return jax.vmap(one)(cz)


@functools.lru_cache(maxsize=None)
def _mesh():
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("d",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d")), len(jax.devices())


@functools.lru_cache(maxsize=None)
def _grid_fn(kind, n_sub, anti):
    return jax.jit(lambda xv, e, cz: _grid(dict(zip(INPUTS, xv)), e, cz, kind, n_sub, anti))


@functools.lru_cache(maxsize=None)
def _grid_jvp_fn(kind, n_sub, anti):
    def d(xv, tangent, e, cz):
        f = lambda v: _grid(dict(zip(INPUTS, v)), e, cz, kind, n_sub, anti)
        return jax.jvp(f, (xv,), (tangent,))
    return jax.jit(d)


def _inputs(osc, earth):
    vals = {**EARTH_DEFAULTS, **earth, **osc}
    return jnp.asarray([float(vals[k]) for k in INPUTS], dtype=jnp.float64)


def _sharded_cz(cos_zenith):
    shard, ndev = _mesh()
    cz = np.asarray(cos_zenith, dtype=np.float64)
    pad = (-len(cz)) % ndev
    czp = np.concatenate([cz, np.full(pad, cz[-1])])
    return jax.device_put(jnp.asarray(czp), shard), len(cz)


def amplitude_grid(osc, earth, energies, cos_zenith, anti=False, wrt=None):
    """Reduced-basis amplitudes S' on a (cz, E) grid, shape (n_cz, n_E, 3, 3).

    ``osc``: theta12, theta13 (rad), dm21, dm31 (eV^2, signed).  ``earth``:
    overrides of EARTH_DEFAULTS.  With ``wrt`` (one of INPUTS) also returns the
    forward-mode derivative dS'/d(wrt) (MANGO autodiff), as ``(S', dS')``.
    The cos-zenith axis is sharded across all JAX devices
    (``XLA_FLAGS=--xla_force_host_platform_device_count=N`` -> N CPU cores).
    """
    earth = {**EARTH_DEFAULTS, **earth}
    kind, n_sub = earth["kind"], int(earth["n_sub"])
    xv = _inputs(osc, earth)
    e = jnp.asarray(energies, dtype=jnp.float64)
    czp, n = _sharded_cz(cos_zenith)
    if wrt is None:
        return np.asarray(_grid_fn(kind, n_sub, bool(anti))(xv, e, czp))[:n]
    tangent = jnp.zeros_like(xv).at[INPUTS.index(wrt)].set(1.0)
    s, ds = _grid_jvp_fn(kind, n_sub, bool(anti))(xv, tangent, e, czp)
    return np.asarray(s)[:n], np.asarray(ds)[:n]


# --------------------------------------------------------------------------
# Path-resolved evolution along one chord
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _path_scan(anti):
    def run(th12, th13, dm21, dm31, e_gev, v_cc, l_inv_ev):
        p = reduced_params(th12, th13, dm21, dm31)
        u, msq, e_ev = p.pmns(), p.msquared(), e_gev * C.GEV_TO_EV

        def step(s, seg):
            v, l = seg
            h = matter_hamiltonian(u, msq, e_ev, v, anti=anti)
            s = propagator(h, l, backend="cayley") @ s
            return s, s

        _, hist = jax.lax.scan(step, jnp.eye(3, dtype=jnp.complex128), (v_cc, l_inv_ev))
        return hist

    return jax.jit(run)


def path_evolution(osc, earth, e_gev, cz, n_points=256, anti=False):
    """Cumulative reduced-basis operator S'(x) at ``n_points`` equally spaced
    distances along the chord (x = 0 at the production point).

    MANGO's segment list is cut at the sample points; each piece keeps its
    segment's constant density, so the final S'(L) equals the full-chord
    amplitude exactly.  Returns dict with x_km, rho, ye, radius_km, S (n,3,3).
    """
    earth = {**EARTH_DEFAULTS, **earth}
    x = {**earth, **{k: jnp.asarray(float(v)) for k, v in osc.items()}}
    rho, ye, length = (np.asarray(a) for a in _segments(
        jnp.asarray(float(cz)), earth["kind"], int(earth["n_sub"]), x))
    keep = length > 0
    rho, ye, length = rho[keep], ye[keep], length[keep]
    edges = np.concatenate([[0.0], np.cumsum(length)])
    total = edges[-1]
    xs = np.linspace(0.0, total, n_points)
    cuts = np.unique(np.concatenate([edges, xs]))
    mids = 0.5 * (cuts[:-1] + cuts[1:])
    seg_of = np.clip(np.searchsorted(edges, mids, side="right") - 1, 0, len(length) - 1)
    piece_len = np.diff(cuts)
    v_cc, _ = C.matter_potentials(rho[seg_of], ye[seg_of])
    # pad to a fixed bucket so the jitted scan compiles once per bucket size
    n = len(piece_len)
    bucket = 1 << int(np.ceil(np.log2(max(n, 2))))
    v_pad = np.zeros(bucket); v_pad[:n] = v_cc
    l_pad = np.zeros(bucket); l_pad[:n] = piece_len * C.KM_TO_INV_EV
    hist = np.asarray(_path_scan(bool(anti))(
        osc["theta12"], osc["theta13"], osc["dm21"], osc["dm31"],
        float(e_gev), jnp.asarray(v_pad), jnp.asarray(l_pad)))[:n]
    # operator after the piece that ends at each sample point (x = 0 -> identity)
    end_idx = np.searchsorted(cuts[1:], xs - 1e-9 * max(total, 1.0))
    s = np.where((xs > 0)[:, None, None], hist[np.clip(end_idx, 0, n - 1)],
                 np.eye(3)[None])
    # geometry: radius of each sample point (detector at r_det, source at r_prod)
    r_det = R_EARTH_KM - earth["det_depth_km"]
    seg_idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, len(length) - 1)
    from_det = total - xs
    # position relative to the Earth's centre, detector on +z axis
    sin_z = np.sqrt(max(0.0, 1.0 - cz * cz))
    px = from_det * sin_z
    pz = r_det + from_det * cz
    radius = np.hypot(px, pz)
    return dict(x_km=xs, total_km=total, rho=rho[seg_idx], ye=ye[seg_idx],
                radius_km=radius, S=s)


# --------------------------------------------------------------------------
# Static Earth info for drawing
# --------------------------------------------------------------------------

def prem_profile(n=400):
    r = np.linspace(0.0, R_EARTH_KM, n)
    return dict(r_km=r, rho=_earth.prem_density(r),
                core_km=CORE_RADIUS_KM, inner_core_km=INNER_CORE_KM,
                r_earth_km=R_EARTH_KM)


def critical_cz(det_depth_km=0.0):
    """cos(zenith) where the chord grazes the outer and inner core."""
    r_det = R_EARTH_KM - det_depth_km
    return dict(outer_core=-float(np.sqrt(1 - (CORE_RADIUS_KM / r_det) ** 2)),
                inner_core=-float(np.sqrt(1 - (INNER_CORE_KM / r_det) ** 2)))


def device_count():
    return len(jax.devices())


__all__ = ["amplitude_grid", "path_evolution", "prem_profile",
           "critical_cz", "reduced_params", "device_count"]

if os.environ.get("NUEARTH_DEBUG"):
    print("nuearth engine: devices", jax.devices())
