// JS port (web/js/mango-lite.js) vs MANGO golden grids, plus timing.
//   node tests/test_port.mjs           (regenerate goldens: python scripts/make_golden.py)
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { amplitude, amplitudeGrid, averagedGrid, accumulateBilinear, NM, chordSegments, pathEvolution, shellTable, vacuumKernel } from '../web/js/mango-lite.js';
import { kernel, probAt, channelWeights, probAvgAt, CHANNELS } from '../web/js/physics.js';

const gold = JSON.parse(readFileSync(new URL('./golden/mango_grids.json', import.meta.url)));
const earthOpts = (e) => ({
  kind: e.kind ?? 'prem', coreScale: e.core_scale ?? 1, mantleScale: e.mantle_scale ?? 1,
  yeCore: e.ye_core ?? 0.4656, rhoUniform: e.rho_uniform ?? 5.51, hAtm: e.h_atm_km ?? 15,
  detDepth: e.det_depth_km ?? 0,
});

let worstAll = 0;
for (const cs of gold.cases) {
  const nSub = cs.earth.n_sub ?? 2;
  const table = shellTable(nSub);
  const A = vacuumKernel(cs.osc.theta12, cs.osc.theta13, cs.osc.dm21, cs.osc.dm31);
  const buf = new Float64Array(18);
  let worst = 0, worstP = 0;
  gold.cz.forEach((cz, c) => {
    const seg = chordSegments(cz, table, earthOpts(cs.earth));
    gold.E.forEach((E, e) => {
      amplitude(A, E, seg, cs.anti, buf);
      const ref = cs.S[c][e];
      for (let k = 0; k < 18; k++) worst = Math.max(worst, Math.abs(buf[k] - ref[k]));
      for (let k = 0; k < 9; k++) {
        const p = buf[2 * k] ** 2 + buf[2 * k + 1] ** 2, pr = ref[2 * k] ** 2 + ref[2 * k + 1] ** 2;
        worstP = Math.max(worstP, Math.abs(p - pr));
      }
    });
  });
  worstAll = Math.max(worstAll, worst);
  console.log(`${cs.name.padEnd(22)} max|dS'| = ${worst.toExponential(1)}   max|dP| = ${worstP.toExponential(1)}`);
}
let ok = worstAll < 1e-10;

// path evolution
for (const p of gold.paths) {
  const r = pathEvolution(p.osc, earthOpts(p.earth), p.E, p.cz, p.x_km.length, p.anti, 2);
  let ws = 0, wx = 0, wr = 0;
  p.S.forEach((ref, n) => ref.forEach((v, k) => { ws = Math.max(ws, Math.abs(r.S[n * 18 + k] - v)); }));
  p.x_km.forEach((v, n) => { wx = Math.max(wx, Math.abs(r.x_km[n] - v)); wr = Math.max(wr, Math.abs(r.rho[n] - p.rho[n])); });
  ok &&= ws < 1e-10 && wx < 1e-6 && wr < 1e-9;
  console.log(`path E=${p.E} cz=${p.cz}${p.anti ? ' nubar' : ''}: max|dS'| = ${ws.toExponential(1)}, |dx| = ${wx.toExponential(1)} km, |drho| = ${wr.toExponential(1)}`);
}
// cell averaging (1): a one-sample grid holds the bilinear products of MANGO's S'
{
  const cs = gold.cases[1];
  const M = averagedGrid(cs.osc, earthOpts(cs.earth), gold.E, gold.cz, cs.anti, 2, { fast: true });
  const acc = new Float64Array(NM), s18 = new Float64Array(18);
  let worst = 0;
  gold.cz.forEach((_, c) => gold.E.forEach((_, e) => {
    s18.set(cs.S[c][e]); acc.fill(0); accumulateBilinear(s18, acc);
    for (let k = 0; k < NM; k++) worst = Math.max(worst, Math.abs(M[(c * gold.E.length + e) * NM + k] - acc[k]));
  }));
  ok &&= worst < 1e-6;
  console.log(`averaged grid, one sample/cell vs products of MANGO S': max diff ${worst.toExponential(1)} (float32 storage)`);
}
// cell averaging (2): <P> from averaged products and channel weights == mean of P over the same samples
{
  const osc = gold.cases[0].osc, table = shellTable(2), A = vacuumKernel(osc.theta12, osc.theta13, osc.dm21, osc.dm31);
  const acc = new Float64Array(NM), S = new Float64Array(18);
  const samples = [];
  for (let k = 0; k < 40; k++) {
    const cz = -1 + 1.2 * Math.random(), E = 0.1 * 10 ** (2.5 * Math.random());
    amplitude(A, E, chordSegments(cz, table, earthOpts({})), k % 2 === 1, S);
    samples.push(Float64Array.from(S));
    accumulateBilinear(S, acc, 0, 1 / 40);
  }
  let worst = 0;
  for (let t = 0; t < 20; t++) {
    const th23 = 0.6 + 0.4 * Math.random(), dcp = 2 * Math.PI * Math.random(), anti = t % 2 === 1;
    for (const { a, b } of Object.values(CHANNELS)) {
      const direct = samples.reduce((m, s) => m + probAt(s, 0, kernel(th23, dcp, a, b, anti)), 0) / samples.length;
      worst = Math.max(worst, Math.abs(direct - probAvgAt(acc, 0, channelWeights(th23, dcp, a, b, anti))));
    }
  }
  ok &&= worst < 1e-12;
  console.log(`<P> from averaged S' products vs mean of P, all channels, random theta23/delta: max diff ${worst.toExponential(1)}`);
}
// cell averaging (3): convergence of the adaptive sampling (0.1-30 GeV, 60x60 cells)
{
  const n = 60, osc = gold.cases[0].osc;
  const E = Array.from({ length: n }, (_, i) => 0.1 * 300 ** (i / (n - 1)));
  const CZ = Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1));
  const cell = { dlnE: Math.log(300) / (n - 1), dcz: 2 / (n - 1), fast: false };
  const coarse = averagedGrid(osc, earthOpts({}), E, CZ, false, 2, { ...cell, hTarget: 1.5 });
  const fine = averagedGrid(osc, earthOpts({}), E, CZ, false, 2, { ...cell, hTarget: 0.5, nMax: 72 });
  const raw = averagedGrid(osc, earthOpts({}), E, CZ, false, 2, { fast: true });
  let dCoarse = 0, dRaw = 0;
  for (const { a, b } of Object.values(CHANNELS)) {
    const W = channelWeights(0.75, 3.7, a, b, false);
    for (let p = 0; p < n * n; p++) {
      const pf = probAvgAt(fine, p, W);
      dCoarse = Math.max(dCoarse, Math.abs(probAvgAt(coarse, p, W) - pf));
      dRaw = Math.max(dRaw, Math.abs(probAvgAt(raw, p, W) - pf));
    }
  }
  ok &&= dCoarse < 0.06;
  console.log(`cell averages, default sampling vs 3x finer: max |dP| = ${dCoarse.toFixed(3)} (one sample per cell: ${dRaw.toFixed(2)})`);
}
console.log(ok ? 'PASS: port agrees with MANGO' : 'FAIL');

// timing: one 160x160 grid, single thread
const N = 160;
const E = Array.from({ length: N }, (_, i) => 10 ** (2 * i / (N - 1)));
const CZ = Array.from({ length: N }, (_, i) => -1 + (2 * i) / (N - 1));
const osc = gold.cases[0].osc;
for (const nSub of [2, 4]) {
  amplitudeGrid(osc, {}, E, CZ, false, nSub); // warm up the JIT
  const t0 = performance.now();
  const reps = 3;
  for (let r = 0; r < reps; r++) amplitudeGrid(osc, {}, E, CZ, r % 2 === 1, nSub);
  console.log(`160x160 grid, n_sub=${nSub}: ${((performance.now() - t0) / reps).toFixed(0)} ms (single thread, node ${process.version})`);
}
{
  const Elo = Array.from({ length: N }, (_, i) => 0.1 * 300 ** (i / (N - 1)));
  const cell = { dlnE: Math.log(300) / (N - 1), dcz: 2 / (N - 1) };
  for (const fast of [true, false]) {
    const t0 = performance.now();
    averagedGrid(osc, {}, Elo, CZ, false, 2, { ...cell, fast });
    console.log(`160x160 averaged-format grid, 0.1-30 GeV, ${fast ? 'one sample per cell' : 'cell-averaged'}: ${(performance.now() - t0).toFixed(0)} ms (single thread)`);
  }
}
process.exit(ok ? 0 : 1);
