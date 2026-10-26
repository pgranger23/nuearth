// mango-lite: a line-by-line JavaScript port of the part of MANGO the app uses
// (github.com/pgranger23/mango-osc): PREM shell table + chord geometry
// (mango/earth.py), matter potential (mango/constants.py), analytic eigenvalues
// (mango/eigensolve.py) and the cayley propagator (mango/propagator.py), chained
// source -> detector as in mango/layers.py.  It computes the reduced-basis
// amplitudes S' (theta23 = delta = 0), where the Hamiltonian is real symmetric;
// physics.js then applies theta23 / delta exactly.  MANGO stays the reference:
// tests/test_port.mjs compares against MANGO-generated golden grids.

// ---- constants (mango/constants.py) ----
const GF_EV = 1.1663787e-23;
const NA = 6.02214076e23;
const HBARC_EV_M = 197.3269804 * 1.0e6 * 1.0e-15;
const M_TO_INV_EV = 1.0 / HBARC_EV_M;
export const KM_TO_INV_EV = M_TO_INV_EV * 1.0e3;
const CM_TO_INV_EV = M_TO_INV_EV * 1.0e-2;
const V_FACTOR_EV = Math.SQRT2 * GF_EV * (NA / CM_TO_INV_EV ** 3);
export const R_EARTH_KM = 6371.0;
const GEV_TO_EV = 1.0e9;

// ---- PREM (mango/earth.py) ----
const PREM = [
  [1221.5, [13.0885, 0.0, -8.8381, 0.0]],
  [3480.0, [12.5815, -1.2638, -3.6426, -5.5281]],
  [5701.0, [7.9565, -6.4761, 5.5283, -3.0807]],
  [5771.0, [5.3197, -1.4836, 0.0, 0.0]],
  [5971.0, [11.2494, -8.0298, 0.0, 0.0]],
  [6151.0, [7.1089, -3.8045, 0.0, 0.0]],
  [6346.6, [2.6910, 0.6924, 0.0, 0.0]],
  [6356.0, [2.9000, 0.0, 0.0, 0.0]],
  [6368.0, [2.6000, 0.0, 0.0, 0.0]],
  [R_EARTH_KM, [1.0200, 0.0, 0.0, 0.0]],
];
export const CORE_RADIUS_KM = 3480.0;
export const YE_CORE = 0.4656;
export const YE_MANTLE = 0.4957;

const poly = (c, x) => c[0] + x * (c[1] + x * (c[2] + x * c[3]));

// prem_density_jax: regions (prev, outer], first one includes r = 0, 0 above R_E
export function premDensity(r) {
  const x = r / R_EARTH_KM;
  let prev = -1.0;
  for (const [outer, c] of PREM) {
    if (r > prev && r <= outer) return poly(c, x);
    prev = outer;
  }
  return 0.0;
}

// Python's round(): half to even
const pyRound = (v) => {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
};

export function shellTable(nSub) {
  const dr = R_EARTH_KM / (10 * nSub);
  const outer = [], inner = [], core = [];
  let lo = 0.0;
  for (const [hi] of PREM) {
    const n = Math.max(1, pyRound((hi - lo) / dr));
    for (let k = 0; k < n; k++) {
      // np.linspace(lo, hi, n + 1) edges
      const a = lo + ((hi - lo) * k) / n, b = k === n - 1 ? hi : lo + ((hi - lo) * (k + 1)) / n;
      inner.push(a); outer.push(b); core.push(0.5 * (a + b) < CORE_RADIUS_KM);
    }
    lo = hi;
  }
  return { outer, inner, core };
}

const SQRT_FLOOR = 1e-9;
const dcoord = (x) => (x > SQRT_FLOOR ? Math.sqrt(x) : 0.0); // _d
const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

// chord_segments + the nuearth Earth-model options (server/engine.py _segments).
// Returns {rho, ye, len} (Float64Array, km) ordered source -> detector.
export function chordSegments(cz, table, earth) {
  const { hAtm = 15, detDepth = 0, kind = 'prem', coreScale = 1, mantleScale = 1,
    yeCore = YE_CORE, rhoUniform = 5.51 } = earth;
  const rDet = R_EARTH_KM - detDepth, rProd = R_EARTH_KM + hAtm;
  const up = cz < 0.0;
  const s2 = 1.0 - cz * cz;
  const rmin2 = rDet * rDet * (s2 >= 0.0 ? s2 : 0.0);
  const sProd = dcoord(rProd * rProd - rmin2);
  const sDet = rDet * cz;
  const dLow = up ? 0.0 : sDet;
  const sDetAbs = Math.abs(sDet);
  const n = table.outer.length;
  const rho = new Float64Array(2 * n + 1), ye = new Float64Array(2 * n + 1), len = new Float64Array(2 * n + 1);
  const dRe = dcoord(R_EARTH_KM * R_EARTH_KM - rmin2);
  len[0] = sProd - clip(dRe, dLow, sProd);
  rho[0] = 0.0; ye[0] = YE_MANTLE;
  const model = (r, isCore) => {
    const base = premDensity(r);
    if (kind === 'prem') return [base * (isCore ? coreScale : mantleScale), isCore ? yeCore : YE_MANTLE];
    if (kind === 'uniform') return [base > 0 ? rhoUniform : 0.0, base > 0 ? 0.5 : (isCore ? YE_CORE : YE_MANTLE)];
    return [0.0, isCore ? YE_CORE : YE_MANTLE];
  };
  for (let i = 0; i < n; i++) {
    const dOut = dcoord(table.outer[i] ** 2 - rmin2), dIn = dcoord(table.inner[i] ** 2 - rmin2);
    // descending leg (stored reversed: outer shells first)
    const hiD = clip(dOut, dLow, sProd), loD = clip(dIn, dLow, sProd);
    const rD = Math.sqrt(rmin2 + (0.5 * (hiD + loD)) ** 2);
    const coreD = rD < CORE_RADIUS_KM;
    const jd = n - i;
    [rho[jd], ye[jd]] = model(rD, coreD);
    len[jd] = hiD - loD;
    // ascending leg
    const hiA = clip(dOut, 0.0, sDetAbs), loA = clip(dIn, 0.0, sDetAbs);
    const rA = Math.sqrt(rmin2 + (0.5 * (hiA + loA)) ** 2);
    const ja = n + 1 + i;
    [rho[ja], ye[ja]] = model(rA, rA < CORE_RADIUS_KM);
    len[ja] = up ? hiA - loA : 0.0;
  }
  // palindromic Earth part (see amplitude): up-going, detector at the surface
  let palindromic = up && detDepth === 0;
  for (let i = 1; palindromic && i <= n; i++) {
    const a = n + 1 - i, b = n + i; // descending and ascending pass of the same shell
    palindromic = len[a] === len[b] && rho[a] === rho[b] && ye[a] === ye[b];
  }
  return { rho, ye, len, palindromic };
}

// ---- reduced-basis vacuum Hamiltonian numerator: U' diag(0, dm21, dm31) U'^T / 2 ----
export function vacuumKernel(th12, th13, dm21, dm31) {
  const c12 = Math.cos(th12), s12 = Math.sin(th12), c13 = Math.cos(th13), s13 = Math.sin(th13);
  // U' = R13 . R12  (MANGO plane_rotation, delta = 0, theta23 = 0)
  const U = [
    [c13 * c12, c13 * s12, s13],
    [-s12, c12, 0],
    [-s13 * c12, -s13 * s12, c13],
  ];
  const m = [0, dm21, dm31];
  const A = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += U[a][k] * m[k] * U[b][k];
    A[a * 3 + b] = 0.5 * s;
  }
  return A;
}

// ---- eigvalsh3 for a real symmetric 3x3 (mango/eigensolve.py) ----
const EPS = 1e-12;
function eigvalsh3(M, out) {
  const p1 = M[1] * M[1] + M[2] * M[2] + M[5] * M[5];
  const d0 = M[0], d1 = M[4], d2 = M[8];
  const q = (d0 + d1 + d2) / 3.0;
  const p2 = (d0 - q) ** 2 + (d1 - q) ** 2 + (d2 - q) ** 2 + 2.0 * p1;
  if (!(p2 > EPS)) {
    const s = [d0, d1, d2].sort((x, y) => x - y);
    out[0] = s[0]; out[1] = s[1]; out[2] = s[2];
    return;
  }
  const p = Math.sqrt(p2 / 6.0);
  const a = d0 - q, e = d1 - q, i = d2 - q, b = M[1], c = M[2], f = M[5];
  const det = a * (e * i - f * f) - b * (b * i - f * c) + c * (b * f - e * c);
  let r = det / (2.0 * p * p * p);
  r = Math.min(Math.max(r, -1.0 + EPS), 1.0 - EPS);
  const phi = Math.acos(r) / 3.0;
  const hi = q + 2.0 * p * Math.cos(phi);
  const lo = q + 2.0 * p * Math.cos(phi + (2.0 * Math.PI) / 3.0);
  out[0] = lo; out[1] = 3.0 * q - hi - lo; out[2] = hi;
}

// ---- propagator_cayley on M = H L (real symmetric): S = f0 I + d01 A + d012 A B ----
const DEGEN_EPS = 1e-7;
const mu = new Float64Array(3), AB = new Float64Array(9);
function propagator(M, S) {
  eigvalsh3(M, mu);
  const m0 = mu[0], m1 = mu[1], m2 = mu[2];
  const f0r = Math.cos(m0), f0i = -Math.sin(m0);
  const f1r = Math.cos(m1), f1i = -Math.sin(m1);
  const f2r = Math.cos(m2), f2i = -Math.sin(m2);
  const dd = (fa_r, fa_i, fb_r, fb_i, a, b) => {
    const d = a - b;
    if (Math.abs(d) > DEGEN_EPS) return [(fa_r - fb_r) / d, (fa_i - fb_i) / d];
    return [-Math.sin(a), -Math.cos(a)]; // f'(a) = -i e^{-ia}
  };
  const [d01r, d01i] = dd(f0r, f0i, f1r, f1i, m0, m1);
  const [d12r, d12i] = dd(f1r, f1i, f2r, f2i, m1, m2);
  const span = m2 - m0;
  let d012r, d012i;
  if (Math.abs(span) > DEGEN_EPS) { d012r = (d12r - d01r) / span; d012i = (d12i - d01i) / span; }
  else { d012r = -0.5 * f0r; d012i = -0.5 * f0i; }
  // A = M - m0 I, B = M - m1 I  (real); AB real symmetric
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += (M[r * 3 + k] - (r === k ? m0 : 0)) * (M[k * 3 + c] - (k === c ? m1 : 0));
    AB[r * 3 + c] = s;
  }
  for (let k = 0; k < 9; k++) {
    const a = M[k] - (k % 4 === 0 ? m0 : 0);
    S[2 * k] = d01r * a + d012r * AB[k] + (k % 4 === 0 ? f0r : 0);
    S[2 * k + 1] = d01i * a + d012i * AB[k] + (k % 4 === 0 ? f0i : 0);
  }
}

// complex 3x3 C = A . B (interleaved re/im, row-major)
function cmul(A, B, C) {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    let re = 0, im = 0;
    for (let k = 0; k < 3; k++) {
      const ar = A[2 * (r * 3 + k)], ai = A[2 * (r * 3 + k) + 1];
      const br = B[2 * (k * 3 + c)], bi = B[2 * (k * 3 + c) + 1];
      re += ar * br - ai * bi; im += ar * bi + ai * br;
    }
    C[2 * (r * 3 + c)] = re; C[2 * (r * 3 + c) + 1] = im;
  }
}

const Mbuf = new Float64Array(9), P = new Float64Array(18), T = new Float64Array(18), SCR = new Float64Array(18);

// Reduced-basis S' for one (E, segments).  out: Float64Array/Float32Array, 18 floats at offset.
// Up-going chords with the detector at the surface are palindromic (atmosphere, then
// shells 1..n descending, then n..1 ascending, with equal rho and length), and each
// shell's S is complex symmetric (H' is real symmetric), so S_earth = Q^T Q with
// Q = A_n ... A_1: half the propagators (NuFast-Earth's symmetric-trajectory trick).
const Q = new Float64Array(18), QT = new Float64Array(18);
function step(Avac, invE, sign, seg, j, s) {
  const L = seg.len[j] * KM_TO_INV_EV;
  if (L === 0) return; // zero-length segment: exact identity
  const V = sign * V_FACTOR_EV * seg.rho[j] * seg.ye[j];
  for (let k = 0; k < 9; k++) Mbuf[k] = Avac[k] * invE * L;
  Mbuf[0] += V * L;
  propagator(Mbuf, P);
  cmul(P, s, T);
  s.set(T);
}
export function amplitude(Avac, eGeV, seg, anti, out, off = 0) {
  const s = SCR;
  s.fill(0);
  s[0] = 1; s[8] = 1; s[16] = 1;
  const invE = 1.0 / (eGeV * GEV_TO_EV);
  const sign = anti ? -1 : 1;
  const nSeg = seg.len.length, n = (nSeg - 1) / 2;
  if (seg.palindromic) {
    step(Avac, invE, sign, seg, 0, s); // atmosphere
    Q.fill(0); Q[0] = 1; Q[8] = 1; Q[16] = 1;
    for (let j = 1; j <= n; j++) step(Avac, invE, sign, seg, j, Q);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      QT[2 * (r * 3 + c)] = Q[2 * (c * 3 + r)]; QT[2 * (r * 3 + c) + 1] = Q[2 * (c * 3 + r) + 1];
    }
    cmul(Q, s, T); cmul(QT, T, s);
  } else {
    for (let j = 0; j < nSeg; j++) step(Avac, invE, sign, seg, j, s);
  }
  for (let k = 0; k < 18; k++) out[off + k] = s[k];
  return out;
}

// Whole (cz, E) grid, packed like the server: [nC][nE][3][3][re, im].
export function amplitudeGrid(osc, earth, E, CZ, anti = false, nSub = 2, Out = Float32Array) {
  const table = shellTable(nSub);
  const A = vacuumKernel(osc.theta12, osc.theta13, osc.dm21, osc.dm31);
  const out = new Out(CZ.length * E.length * 18);
  for (let c = 0; c < CZ.length; c++) {
    const seg = chordSegments(CZ[c], table, earth);
    for (let e = 0; e < E.length; e++) amplitude(A, E[e], seg, anti, out, (c * E.length + e) * 18);
  }
  return out;
}

// ---- cell averaging -------------------------------------------------------------------
// P = |sum K_ij S'_ij|^2 is bilinear in S', so the average of P over a cell only needs
// the averaged products <S' S'*>, and theta23 / delta can still be applied afterwards.
// Atmospheric channels start from nu_e or nu_mu, so two blocks suffice (45 floats/pixel):
//   e-block  (9):  v_i = S'[i][0]                    -> <v_p conj(v_q)>, 3x3 Hermitian
//   mu-block (36): v_p = S'[i][j], j = 1, 2 (p = 2i + j - 1) -> 6x6 Hermitian
// each packed as the real diagonal followed by (re, im) of p < q in row-major order.
export const NM = 45;
const VE = new Float64Array(6), VMU = new Float64Array(12);

function hermAdd(v, n, acc, off, wt) {
  for (let p = 0; p < n; p++) acc[off + p] += wt * (v[2 * p] * v[2 * p] + v[2 * p + 1] * v[2 * p + 1]);
  let k = off + n;
  for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
    acc[k] += wt * (v[2 * p] * v[2 * q] + v[2 * p + 1] * v[2 * q + 1]);
    acc[k + 1] += wt * (v[2 * p + 1] * v[2 * q] - v[2 * p] * v[2 * q + 1]);
    k += 2;
  }
}

export function accumulateBilinear(S, acc, off = 0, wt = 1) {
  for (let i = 0; i < 3; i++) { VE[2 * i] = S[6 * i]; VE[2 * i + 1] = S[6 * i + 1]; }
  let p = 0;
  for (let i = 0; i < 3; i++) for (let j = 1; j <= 2; j++) {
    VMU[2 * p] = S[2 * (i * 3 + j)]; VMU[2 * p + 1] = S[2 * (i * 3 + j) + 1]; p++;
  }
  hermAdd(VE, 3, acc, off, wt);
  hermAdd(VMU, 6, acc, off + 9, wt);
}

export function baselineKm(cz, hAtm = 15) {
  return Math.sqrt((R_EARTH_KM + hAtm) ** 2 - R_EARTH_KM ** 2 * (1 - cz * cz)) - R_EARTH_KM * cz;
}
// chord length inside a sphere of radius r (detector at the surface), up-going only
const chordInside = (cz, r) => (cz < 0 ? 2 * Math.sqrt(Math.max(0, r * r - R_EARTH_KM ** 2 * (1 - cz * cz))) : 0);
const INNER_CORE_KM = 1221.5;

// Cell-averaged bilinear products on a (cz, E) grid: [nC][nE][45].
// cell = { dlnE, dcz }: the grid spacing (cells are centred on the grid points, clipped
// to |cos zenith| <= 1); fast: one sample at the centre; otherwise each cell is sampled
// with nE x nC midpoints chosen so that the oscillation phase changes by at most hTarget
// radians between samples, capped at nMax per axis.  Phase estimates: along E, the
// vacuum phase 2*Delta31 = 2.534 dm2 L / E; along cos zenith, the change across the
// cell of the total path at 2.534 (dm2 / E + 1e-3) rad/km (the 1e-3 eV^2/GeV bounds the
// matter splitting, 1.5e-4 rho Y_e E eV^2 with rho Y_e <= 6.5), plus the path moved
// between mantle and core (it jumps like a square root at the core-grazing angles and
// near cos zenith = -1), which only matters through the density contrast: 2.534 x 1.5e-3.
export function averagedGrid(osc, earth, E, CZ, anti = false, nSub = 2, cell = {}) {
  const { dlnE = 0, dcz = 0, fast = true, hTarget = 1.5, nMax = 24 } = cell;
  const table = shellTable(nSub);
  const A = vacuumKernel(osc.theta12, osc.theta13, osc.dm21, osc.dm31);
  const dm = Math.abs(osc.dm31), hAtm = earth.hAtm ?? 15;
  const out = new Float32Array(CZ.length * E.length * NM);
  const acc = new Float64Array(NM), S = new Float64Array(18);
  for (let c = 0; c < CZ.length; c++) {
    const lo = Math.max(-1, CZ[c] - dcz / 2), hi = Math.min(1, CZ[c] + dcz / 2);
    const L = baselineKm(CZ[c], hAtm);
    const dPath = Math.abs(baselineKm(hi, hAtm) - baselineKm(lo, hAtm));
    const dCore = Math.abs(chordInside(hi, CORE_RADIUS_KM) - chordInside(lo, CORE_RADIUS_KM))
      + Math.abs(chordInside(hi, INNER_CORE_KM) - chordInside(lo, INNER_CORE_KM));
    const segCache = new Map();
    const segsFor = (n) => {
      if (!segCache.has(n)) {
        segCache.set(n, n === 1 ? [chordSegments(CZ[c], table, earth)]
          : Array.from({ length: n }, (_, k) => chordSegments(lo + ((hi - lo) * (k + 0.5)) / n, table, earth)));
      }
      return segCache.get(n);
    };
    for (let e = 0; e < E.length; e++) {
      let nE = 1, nC = 1;
      if (!fast) {
        nE = Math.min(nMax, Math.max(1, Math.ceil((2.534 * dm * L * dlnE) / (E[e] * hTarget))));
        nC = Math.min(nMax, Math.max(1, Math.ceil((2.534 * ((dm / E[e] + 1e-3) * dPath + 1.5e-3 * dCore)) / hTarget)));
      }
      acc.fill(0);
      const segs = segsFor(nC), wt = 1 / (nE * nC);
      for (let k = 0; k < nE; k++) {
        const Ek = nE === 1 ? E[e] : E[e] * Math.exp(dlnE * ((k + 0.5) / nE - 0.5));
        for (const seg of segs) {
          amplitude(A, Ek, seg, anti, S);
          accumulateBilinear(S, acc, 0, wt);
        }
      }
      out.set(acc, (c * E.length + e) * NM);
    }
  }
  return out;
}

// Cumulative S'(x) at nPoints equally spaced distances along one chord (x = 0 at
// production), as server/engine.py path_evolution: within segment k,
// S'(x) = exp(-i H_k (x - X_k)) S'(X_k), so S'(L) is the full-chord amplitude.
export function pathEvolution(osc, earth, eGeV, cz, nPoints = 240, anti = false, nSub = 2) {
  const seg = chordSegments(cz, shellTable(nSub), earth);
  const keep = [];
  for (let j = 0; j < seg.len.length; j++) if (seg.len[j] > 0) keep.push(j);
  const edges = new Float64Array(keep.length + 1);
  keep.forEach((j, k) => { edges[k + 1] = edges[k] + seg.len[j]; });
  const total = edges[keep.length];
  const xs = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) xs[i] = (total * i) / (nPoints - 1);
  const A = vacuumKernel(osc.theta12, osc.theta13, osc.dm21, osc.dm31);
  const invE = 1.0 / (eGeV * GEV_TO_EV), sign = anti ? -1 : 1;
  const S = new Float64Array(nPoints * 18);
  let cur = new Float64Array(18);
  cur[0] = 1; cur[8] = 1; cur[16] = 1;
  S.set(cur, 0);
  const Hk = new Float64Array(9), M = new Float64Array(9), Pk = new Float64Array(18), nxt = new Float64Array(18);
  let i = 1;
  const tol = 1e-9 * Math.max(total, 1);
  keep.forEach((j, k) => {
    const V = sign * V_FACTOR_EV * seg.rho[j] * seg.ye[j];
    for (let q = 0; q < 9; q++) Hk[q] = A[q] * invE;
    Hk[0] += V;
    const step = (dxKm, out) => {
      const L = dxKm * KM_TO_INV_EV;
      for (let q = 0; q < 9; q++) M[q] = Hk[q] * L;
      propagator(M, Pk);
      cmul(Pk, cur, out);
    };
    while (i < nPoints && xs[i] <= edges[k + 1] + tol) {
      const out = S.subarray(i * 18, (i + 1) * 18);
      step(xs[i] - edges[k], out);
      i++;
    }
    step(edges[k + 1] - edges[k], nxt);
    cur.set(nxt);
  });
  // density and radius at each sample (segment chosen as searchsorted(..., 'right') - 1)
  const rho = new Float64Array(nPoints), radius = new Float64Array(nPoints);
  const rDet = R_EARTH_KM - (earth.detDepth ?? 0), sz = Math.sqrt(Math.max(0, 1 - cz * cz));
  let k = 0;
  for (let n = 0; n < nPoints; n++) {
    while (k < keep.length - 1 && edges[k + 1] <= xs[n]) k++;
    rho[n] = seg.rho[keep[k]];
    const d = total - xs[n];
    radius[n] = Math.hypot(d * sz, rDet + d * cz);
  }
  return { x_km: xs, total_km: total, rho, radius_km: radius, S };
}
