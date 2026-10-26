import time, numpy as np, engine
osc = dict(theta12=0.5836, theta13=0.1495, dm21=7.42e-5, dm31=2.515e-3)
E = np.logspace(0, 2, 160); CZ = np.linspace(-1, 1, 160)
ref = engine.amplitude_grid(osc, dict(n_sub=8), E, CZ)
for ns in (1, 2, 3, 4):
    s = engine.amplitude_grid(osc, dict(n_sub=ns), E, CZ)
    t = time.time(); s = engine.amplitude_grid(osc, dict(n_sub=ns), E, CZ); dt = time.time() - t
    d = np.max(np.abs(np.abs(s) ** 2 - np.abs(ref) ** 2))
    print(f"n_sub={ns}: max|dP| vs n_sub=8 = {d:.1e}; 160x160 grid {dt*1e3:.0f} ms on {engine.device_count()} devices")
