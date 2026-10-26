// Client-side half of the reduced-basis trick (see server/engine.py).
// The engine (mango-lite) computes S' = S(theta23 = 0, delta = 0); here
//   S = O S' O^dagger,  O = R23 . diag(1, 1, e^{i delta})   (delta -> -delta for nubar)
// is applied exactly, so theta23, delta_CP and the channel never need a recompute.
// Flavor indices follow MANGO: 0 = e, 1 = mu, 2 = tau; S[beta][alpha] = <nu_beta|S|nu_alpha>.

export const FLAVOR = ['e', 'μ', 'τ'];

export const CHANNELS = {
  mumu: { a: 1, b: 1 },
  mue: { a: 1, b: 0 },
  ee: { a: 0, b: 0 },
  emu: { a: 0, b: 1 },
  mutau: { a: 1, b: 2 },
};

export const DEG = Math.PI / 180;

// O as rows of [re, im].
export function rotation(th23, delta) {
  const c = Math.cos(th23), s = Math.sin(th23), cd = Math.cos(delta), sd = Math.sin(delta);
  return [
    [[1, 0], [0, 0], [0, 0]],
    [[0, 0], [c, 0], [s * cd, s * sd]],
    [[0, 0], [-s, 0], [c * cd, c * sd]],
  ];
}

// K_ij = O[b][i] conj(O[a][j]) so that S_ba = sum_ij K_ij S'_ij  (18 floats, [re, im]).
function kernelFrom(Ob, Oa) {
  const K = new Float64Array(18);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const [p, q] = Ob[i], [r, t] = Oa[j]; // (p + iq)(r - it)
    K[(i * 3 + j) * 2] = p * r + q * t;
    K[(i * 3 + j) * 2 + 1] = q * r - p * t;
  }
  return K;
}

// Kernel for one channel a -> b (antineutrinos: delta -> -delta).
export function kernel(th23, delta, a, b, anti = false) {
  const O = rotation(th23, anti ? -delta : delta);
  return kernelFrom(O[b], O[a]);
}

// Amplitude at pixel p of a packed S' array (18 floats per pixel).
function amp(S, off, K) {
  let re = 0, im = 0;
  for (let n = 0; n < 9; n++) {
    const sr = S[off + 2 * n], si = S[off + 2 * n + 1], kr = K[2 * n], ki = K[2 * n + 1];
    re += kr * sr - ki * si;
    im += kr * si + ki * sr;
  }
  return [re, im];
}

export function probAt(S, p, K) {
  const [re, im] = amp(S, p * 18, K);
  return re * re + im * im;
}

// P over a whole grid.
export function probMap(S, K, out) {
  const n = S.length / 18;
  for (let p = 0; p < n; p++) {
    const [re, im] = amp(S, p * 18, K);
    out[p] = re * re + im * im;
  }
  return out;
}

// Cell-averaged grids store <v_p conj(v_q)> (mango-lite averagedGrid, 45 floats/pixel):
// v = S'[:, 0] for an initial nu_e, v_p = S'[i][j] (j = 1, 2) for an initial nu_mu.
// The channel enters through weights w with A = sum_p w_p v_p, so that
//   <P> = sum_p |w_p|^2 M_pp + 2 Re sum_{p<q} w_p conj(w_q) M_pq.
const NM = 45;
export function channelWeights(th23, delta, a, b, anti = false) {
  const O = rotation(th23, anti ? -delta : delta);
  if (a === 0) return { off: 0, n: 3, w: Float64Array.from(O[b].flat()) };
  if (a !== 1) throw new Error('only nu_e and nu_mu initial states are stored');
  const w = new Float64Array(12);
  let p = 0;
  for (let i = 0; i < 3; i++) for (let j = 1; j <= 2; j++) {
    const [x, y] = O[b][i], [u, v] = O[1][j]; // O_bi conj(O_mu,j)
    w[2 * p] = x * u + y * v; w[2 * p + 1] = y * u - x * v; p++;
  }
  return { off: 9, n: 6, w };
}

export function probAvgAt(M, pix, W) {
  const base = pix * NM + W.off, n = W.n, w = W.w;
  let s = 0;
  for (let p = 0; p < n; p++) s += (w[2 * p] * w[2 * p] + w[2 * p + 1] * w[2 * p + 1]) * M[base + p];
  let k = base + n;
  for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
    const ar = w[2 * p] * w[2 * q] + w[2 * p + 1] * w[2 * q + 1];
    const ai = w[2 * p + 1] * w[2 * q] - w[2 * p] * w[2 * q + 1];
    s += 2 * (ar * M[k] - ai * M[k + 1]);
    k += 2;
  }
  return s;
}

export function probAvgMap(M, W, out) {
  const n = M.length / NM;
  for (let p = 0; p < n; p++) out[p] = probAvgAt(M, p, W);
  return out;
}

// All three final-flavor probabilities for initial flavor a, at every point of a path.
// path: packed S' along the chord (18 floats per point, as the grids).
export function flavorFractions(path, th23, delta, a, anti) {
  const Ks = [0, 1, 2].map((b) => kernel(th23, delta, a, b, anti));
  const n = path.length / 18;
  const out = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  for (let k = 0; k < n; k++) for (let b = 0; b < 3; b++) out[b][k] = probAt(path, k, Ks[b]);
  return out;
}
