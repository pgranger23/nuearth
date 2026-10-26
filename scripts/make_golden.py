"""Golden reduced-basis grids from MANGO for the JS port test (tests/test_port.mjs)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import engine  # noqa: E402

E = np.logspace(np.log10(0.5), 2, 24)
CZ = np.linspace(-1, 1, 21)
NO = dict(theta12=np.arcsin(np.sqrt(0.307)), theta13=np.arcsin(np.sqrt(0.02215)),
          dm21=7.49e-5, dm31=2.438e-3 + 7.49e-5)
IO = dict(NO, dm31=-2.438e-3 + 7.49e-5)
CASES = [
    ("NO nu PREM", NO, dict(), False),
    ("IO nubar PREM", IO, dict(), True),
    ("NO nu PREM scaled", NO, dict(core_scale=1.4, mantle_scale=0.8, ye_core=0.45, h_atm_km=20.0), False),
    ("IO nu uniform", IO, dict(kind="uniform", rho_uniform=4.0), False),
    ("NO nubar vacuum", NO, dict(kind="vacuum"), True),
    ("NO nu PREM n_sub=4", NO, dict(n_sub=4), False),
    ("odd params", dict(theta12=0.55, theta13=0.2, dm21=9e-5, dm31=2.7e-3), dict(det_depth_km=1.5), False),
]
out = dict(E=E.tolist(), cz=CZ.tolist(), cases=[])
for name, osc, earth, anti in CASES:
    s = engine.amplitude_grid(osc, earth, E, CZ, anti=anti)
    out["cases"].append(dict(name=name, osc={k: float(v) for k, v in osc.items()}, earth=earth, anti=anti,
                             S=np.stack([s.real, s.imag], -1).reshape(len(CZ), len(E), 18).tolist()))
# path evolution
out["paths"] = []
for e_gev, cz, anti, osc, earth in [(6.0, -0.8, False, NO, {}), (4.0, -0.95, True, IO, {}),
                                     (2.0, -0.99, False, NO, dict(core_scale=1.3)), (3.0, 0.3, False, NO, {}),
                                     (8.0, -0.5, True, NO, dict(kind="uniform", rho_uniform=4.0))]:
    r = engine.path_evolution(osc, earth, e_gev, cz, n_points=50, anti=anti)
    out["paths"].append(dict(E=e_gev, cz=cz, anti=anti, osc={k: float(v) for k, v in osc.items()}, earth=earth,
                             x_km=r["x_km"].tolist(), rho=r["rho"].tolist(), radius_km=r["radius_km"].tolist(),
                             S=np.stack([r["S"].real, r["S"].imag], -1).reshape(-1, 18).tolist()))
path = os.path.join(os.path.dirname(__file__), "..", "tests", "golden", "mango_grids.json")
json.dump(out, open(path, "w"))
print("wrote", path, os.path.getsize(path) // 1024, "kB")
