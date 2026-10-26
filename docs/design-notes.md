# Design notes

How the engine was chosen, kept for the record. The app first ran MANGO on a FastAPI
backend (removed once the JavaScript port was validated); the numbers below come from
that stage.

## Engine options considered (option 4 was chosen)

1. **MANGO on a server + reduced basis in the browser** (the first prototype). It is literally
   MANGO, and the most-played sliders are instant. It needs a running server (CERN PaaS,
   as for pylarevd and nuosc-explorer), and the MANGO-side sliders have 0.3–1 s latency.
2. **MANGO in the browser.** Not possible today. JAX/jaxlib has no WebAssembly/Pyodide
   build, and the TF.js and ONNX export paths have no float64 or complex128, which MANGO
   requires.
3. **Fully static site with MANGO-precomputed tables.** Grids of S' for a lattice of
   |Δm²32| values and orderings (≈1.5 MB each) would give GitHub Pages / offline use,
   with θ23/δCP still exact. But the path view needs either a lattice of trajectories
   (tens of MB) or a solver in JS.
4. **Port MANGO's layered cayley propagator to JS/WASM**, validated against MANGO the way
   NuGlass cross-checks its two engines. This gives full NuGlass-style interactivity
   offline, but it is a second implementation to maintain.

## Cost of a fully precomputed (static) version (not pursued)

Measured with `scripts/estimate_precompute.py` (160×160, n_sub=2):

* **One grid** (one parameter point, ν or ν̄):
  * compute: 0.38 s on 8 cores, 0.15 s on 32, i.e. 3–5 core-seconds. A derivative
    grid (jvp) costs 2.2× that.
  * storage: as int16 (amplitude × 32767), max |ΔP| = 6e-5 against float64. That is
    0.92 MB raw, **0.48 MB gzipped**, 0.32 MB with byte-shuffle + xz. float16 is worse
    in both size (0.68 MB gzipped) and error (7e-4).
* **One parameter point:** 4 matter grids (NO/IO × ν/ν̄) and 2 vacuum grids. In the
  reduced basis the vacuum ν̄ grid equals the ν grid (checked: identical). Points that
  only change the Earth model need just the 4 matter grids. Total ≈ 2.9 MB gzipped,
  ≈ 25 core-seconds.
* **Journey data:** 23 ms per trajectory (unbatched); 2.3 kB (64 points) or 4.6 kB (128
  points) as int16, times 6 per trajectory (matter ν/ν̄ × NO/IO, plus vacuum).

| what is precomputed | grids | size (gzip) | compute |
|---|---|---|---|
| A. best-fit point only (θ23, δCP, channel, ν/ν̄, NO/IO stay live) + 12 derivative maps | 18 | ~9 MB | ~1 core-min |
| A + journey lattice (61 cos θz × 41 E, adaptive points along the path) | — | +15–40 MB | ~6 min on one core |
| B. one-at-a-time slider lattices: \|Δm²32\| 41 values, sin²θ13 16, core/mantle scale 21 each, core Yₑ 16, uniform ρ 27, θ12/Δm²21 11 each | ~810 | ~390 MB | ~1 core-hour (~2 min on 32 cores) |
| C. 2-D product \|Δm²32\| × sin²θ13 (41 × 16) | ~3900 | ~1.9 GB | ~4.5 core-hours |
| D. full product of all MANGO sliders | ~2×10⁷ | ~9 TB | ~2,000 core-days |

Each slider move in B fetches 4–6 grids (2–3 MB). Journey data for every point of B
would add several GB, so in a static build the journey view would stay at the best-fit
parameters. Further factors are available: storing only the cos θz ≤ 0.2 rows (×0.6),
exploiting unitarity (13 instead of 18 reals, ×0.72), and serving brotli (≈ xz, ×0.75).
For reference, GitHub Pages limits a site to 1 GB.

## Performance of MANGO on a server (CERN host, CPU, float64)

MANGO's layered propagation is essentially single-threaded on CPU: XLA runs a `lax.scan`
of tiny 3×3 complex ops. A 100×100 PREM grid takes 1.2 s on one core with `n_sub=4`
(0.7 s with `n_sub=2`). Sharding the cos θz axis over host devices scales almost
linearly:

| devices | 1 | 4 | 8 | 16 | 48 |
|---|---|---|---|---|---|
| 96×100 grid, n_sub=4 | 1182 ms | 502 | 264 | 145 | 73 |

Defaults are a 160×160 grid with `n_sub=2`: 0.23 s per grid on 16 devices and 0.37 s on 8.
One parameter change loads 5–6 grids (NO/IO × ν/ν̄, vacuum, optional derivative); the
first appears after a single grid time. Each grid is 1.8 MB, 1.47 MB gzipped. A warm path
request takes 0.12 s. The first request per shape pays about 1.5–2.5 s of JIT compilation.
