"""Measure what an offline (static-site) precomputation would cost.

    source env.sh
    XLA_FLAGS=--xla_force_host_platform_device_count=16 python scripts/estimate_precompute.py

Reports: seconds per (E, cos zenith) grid of S' (forward, vacuum, jvp), bytes per grid
under several encodings + their probability error, and the cost of path (journey) data.
"""

import gzip
import lzma
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import engine  # noqa: E402

N = int(os.environ.get("NGRID", 160))
E = np.logspace(0, 2, N)
CZ = np.linspace(-1, 1, N)
OSC = dict(theta12=np.arcsin(np.sqrt(0.307)), theta13=np.arcsin(np.sqrt(0.02215)),
           dm21=7.49e-5, dm31=2.438e-3 + 7.49e-5)


def timed(fn, reps):
    fn(0)
    t = time.time()
    for k in range(reps):
        fn(k + 1)
    return (time.time() - t) / reps


def rotate_prob(s, th23, delta, a, b, anti=False):
    d = -delta if anti else delta
    c, sn, ph = np.cos(th23), np.sin(th23), np.exp(1j * d)
    o = np.array([[1, 0, 0], [0, c, sn * ph], [0, -sn, c * ph]])
    amp = np.einsum("i,...ij,j->...", o[b], s, o[a].conj())
    return np.abs(amp) ** 2


def main():
    ndev = engine.device_count()
    print(f"== {N}x{N} grid, n_sub=2, {ndev} JAX CPU devices")
    t_fwd = timed(lambda k: engine.amplitude_grid({**OSC, "dm31": OSC["dm31"] + k * 1e-6}, {}, E, CZ), 8)
    t_vac = timed(lambda k: engine.amplitude_grid({**OSC, "dm31": OSC["dm31"] + k * 1e-6},
                                                  {"kind": "vacuum"}, E, CZ), 4)
    t_jvp = timed(lambda k: engine.amplitude_grid({**OSC, "dm31": OSC["dm31"] + k * 1e-6}, {}, E, CZ,
                                                  wrt="core_scale"), 4)
    print(f"forward grid  {t_fwd:6.3f} s   ({t_fwd * ndev:5.2f} core-s)")
    print(f"vacuum grid   {t_vac:6.3f} s")
    print(f"jvp grid      {t_jvp:6.3f} s   (returns S' and dS')")

    s = engine.amplitude_grid(OSC, {}, E, CZ)
    raw = np.stack([s.real, s.imag], -1)
    rng = np.random.default_rng(0)
    trials = [(rng.uniform(0.6, 1.0), rng.uniform(0, 2 * np.pi)) for _ in range(6)]
    p_ref = [rotate_prob(s, t, d, a, b) for t, d in trials for a, b in [(1, 0), (1, 1), (0, 0)]]

    def report(name, blob, decoded):
        sd = decoded[..., 0] + 1j * decoded[..., 1]
        p = [rotate_prob(sd, t, d, a, b) for t, d in trials for a, b in [(1, 0), (1, 1), (0, 0)]]
        err = max(np.max(np.abs(x - y)) for x, y in zip(p, p_ref))
        gz = len(gzip.compress(blob, 9))
        xz = len(lzma.compress(blob, preset=9))
        print(f"{name:34s} raw {len(blob) / 1e6:5.2f} MB  gzip {gz / 1e6:5.2f}  xz {xz / 1e6:5.2f}   max|dP| {err:.1e}")

    print("== storage per grid (one parameter point, one of nu/nubar)")
    f32 = raw.astype("<f4")
    report("float32, 18 reals", f32.tobytes(), f32.astype(float))
    f16 = raw.astype("<f2")
    report("float16, 18 reals", f16.tobytes(), f16.astype(float))
    q = np.round(raw * 32767).astype("<i2")
    report("int16 (x/32767), 18 reals", q.tobytes(), q / 32767.0)
    up = CZ <= 0.2
    q_up = q[up]
    print(f"   cos zenith in [-1, 0.2] only: x{up.mean():.2f} of the rows")
    # byte-plane shuffle (what blosc/zstd filters do) helps generic compressors
    sh = np.frombuffer(q.tobytes(), np.uint8).reshape(-1, 2).T.copy().tobytes()
    print(f"int16 + byte shuffle               gzip {len(gzip.compress(sh, 9)) / 1e6:5.2f}  "
          f"xz {len(lzma.compress(sh, preset=9)) / 1e6:5.2f}")
    _ = q_up

    print("== journey (path) data")
    for n_pts in (64, 128):
        def one(k):
            cz = -0.05 - 0.9 * ((k * 0.37) % 1.0)
            return engine.path_evolution(OSC, {}, 2.0 + k % 20, cz, n_points=n_pts)
        for k in range(6):
            one(k)  # warm every scan bucket
        t_path = timed(one, 30)
        print(f"{n_pts} points: {t_path * 1e3:6.1f} ms per trajectory (unbatched), "
              f"{n_pts * 18 * 2 / 1e3:.1f} kB per trajectory as int16")


if __name__ == "__main__":
    main()
