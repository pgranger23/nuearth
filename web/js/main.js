import { CHANNELS, FLAVOR, DEG, channelWeights, probAvgAt, probAvgMap, flavorFractions } from './physics.js';
import { maps, css } from './colors.js';
import { LinePlot, swatch, tooltip } from './plot.js';
import { Oscillogram, baselineKm } from './oscillogram.js';
import { EarthView } from './earth.js';
import { computeGrid, computePath, earthMeta, hasGrid, cancelExcept, isCancelled, gridKey, N_WORKERS } from './compute.js';

// NuFIT 6.0-like normal-ordering point (approximate; IC24 with SK).  Δm²32 is the
// atmospheric parameter (MaCh3 convention): Δm²31 = ±|Δm²32| + Δm²21.
const DEFAULTS = { s23: 0.470, dcp: 212, dm32: 2.438, s13: 0.02215, s12: 0.307, dm21: 7.49 };
const H_ATM_KM = 15; // production height; the Earth itself is MANGO's PREM, fixed

const S = {
  channel: 'mue', anti: false, ordering: 'NO',
  erange: [0.1, 30], res: 160, smooth: true,
  sel: { E: 6.0, cz: -0.8 },
  t: 1,
  ...DEFAULTS,
};

const nuName = (f, anti) => `${anti ? 'ν̄' : 'ν'}${FLAVOR[f]}`;
const chLabel = (ch = S.channel, anti = S.anti) => {
  const { a, b } = CHANNELS[ch];
  return `P(${nuName(a, anti)}→${nuName(b, anti)})`;
};
const asinDeg = (s2) => Math.asin(Math.sqrt(s2)) / DEG;
const th23 = () => Math.asin(Math.sqrt(S.s23));

// ---------------------------------------------------------------- controls

const LIVE = {
  s23: { label: 'sin²θ₂₃', min: 0.35, max: 0.65, step: 0.005, fmt: (v) => `${v.toFixed(3)}  (θ₂₃ ${asinDeg(v).toFixed(1)}°)` },
  dcp: { label: 'δ<sub>CP</sub>', min: 0, max: 360, step: 1, fmt: (v) => `${v.toFixed(0)}°` },
};
const RECOMPUTED = {
  dm32: { label: '|Δm²₃₂|', min: 2.2, max: 2.8, step: 0.005, fmt: (v) => `${v.toFixed(3)}×10⁻³ eV²` },
  s13: { label: 'sin²θ₁₃', min: 0.015, max: 0.03, step: 0.0002, fmt: (v) => v.toFixed(4) },
  s12: { label: 'sin²θ₁₂', min: 0.25, max: 0.36, step: 0.002, fmt: (v) => v.toFixed(3) },
  dm21: { label: 'Δm²₂₁', min: 6, max: 9, step: 0.02, fmt: (v) => `${v.toFixed(2)}×10⁻⁵ eV²` },
};
const sliderEls = {};
function buildSliders(container, defs, onChange) {
  for (const [key, d] of Object.entries(defs)) {
    const wrap = document.createElement('div');
    wrap.className = 'slider';
    wrap.innerHTML = `<div class="row"><span>${d.label}</span><b></b></div>
      <input type="range" min="${d.min}" max="${d.max}" step="${d.step}" aria-label="${d.label.replace(/<[^>]+>/g, '')}">`;
    const input = wrap.querySelector('input'), out = wrap.querySelector('b');
    input.value = S[key];
    out.textContent = d.fmt(S[key]);
    input.addEventListener('input', () => { S[key] = +input.value; out.textContent = d.fmt(S[key]); onChange(key, false); });
    input.addEventListener('change', () => onChange(key, true));
    sliderEls[key] = { wrap, input, out, d };
    container.appendChild(wrap);
  }
}
function syncSlider(key) {
  const el = sliderEls[key];
  if (!el) return;
  el.input.value = S[key];
  el.out.textContent = el.d.fmt(S[key]);
}

const ordName = (o) => (o === 'NO' ? 'normal' : 'inverted');
// The three oscillogram panels: P, and the two differences that carry the physics.
const PANELS = {
  P: {
    title: () => `${chLabel()} · ${ordName(S.ordering)} ordering`,
    caption: () => `Chance that a ${nuName(CHANNELS[S.channel].a, S.anti)} made in the atmosphere arrives as ${nuName(CHANNELS[S.channel].b, S.anti)}, versus energy and direction. Below the horizon it has crossed the Earth (up to 12 700 km); the dashed lines mark where paths start crossing the outer and the inner core. Click or drag on any panel to follow a neutrino.`,
    short: 'P',
  },
  noio: {
    title: () => `${chLabel()} · normal − inverted`,
    caption: () => 'Same |Δm²₃₂|, opposite sign. The Earth enhances νμ→νe for neutrinos if the ordering is normal, and for antineutrinos if it is inverted (MSW resonance at ~3–8 GeV in the mantle, extra structure for core-crossing paths). Toggle ν/ν̄ to see the sign flip.',
    short: 'ΔP',
  },
  vac: {
    title: () => `${chLabel()} · Earth − vacuum`,
    caption: () => 'The same baselines without matter subtracted: everything left is the Earth\'s matter effect. Down-going neutrinos (above the horizon) only cross the atmosphere, so the two agree there.',
    short: 'ΔP',
  },
};
const EXPLAIN = {
  journey: () => 'The selected neutrino, from its production in the atmosphere to the detector, drawn on a cut through the PREM Earth. The ribbon\'s three bands are the νe, νμ and ντ fractions at each point; the plot on the right shows the same fractions against the distance travelled (dashed: the same baseline in vacuum), with the density along the path below it. Drag along the path or press travel; sweep zenith rotates the arrival direction.',
  smooth: () => 'Below ~1 GeV the oscillations are faster than one grid cell, so one sample per cell gives moiré patterns. With this option each cell shows the average probability over the cell, from enough sub-samples to follow the phase (more at low energy and long baselines), much as a detector\'s energy and angle resolution would. θ₂₃ and δCP stay exact: the averages are taken over S′ products, before the rotation.',
  slices: () => 'Below the fold: P versus energy at the selected direction and P versus direction at the selected energy, for normal (solid) and inverted (dashed) ordering and for vacuum (dotted).',
};
const tipText = (key) => (PANELS[key] ?? { caption: EXPLAIN[key] }).caption();

let explainKey = null;
function renderExplain() {
  const key = `${S.channel}:${S.anti}:${S.ordering}`;
  if (key === explainKey) return;
  explainKey = key;
  const rows = [['Probability', 'P'], ['Normal − inverted', 'noio'], ['Earth − vacuum', 'vac'],
    ['Journey', 'journey'], ['Slices', 'slices']];
  document.getElementById('explain').innerHTML = rows
    .map(([name, key]) => `<dt>${name}</dt><dd>${tipText(key)}</dd>`).join('');
}

// ---------------------------------------------------------------- data

const grids = {};      // name -> grid payload for the current parameters
let gen = 0;           // bumps whenever a recompute is needed
let path = null;

function engineParams(ordering, { anti = S.anti, vacuum = false } = {}) {
  const dm31 = (ordering === 'NO' ? 1 : -1) * S.dm32 * 1e-3 + S.dm21 * 1e-5;
  return {
    th12: asinDeg(S.s12), th13: asinDeg(S.s13), dm21: S.dm21 * 1e-5, dm31,
    kind: vacuum ? 'vacuum' : 'prem', h_atm: H_ATM_KM,
    emin: S.erange[0], emax: S.erange[1], n_e: S.res, n_cz: S.res,
    // in vacuum the reduced-basis S' of nubar equals that of nu (U' real, V = 0):
    // one grid serves both, the nubar sign of delta is applied by the rotation
    anti: vacuum ? false : anti,
  };
}

const PREFETCH = N_WORKERS > 3;

// q: 'fast' (one sample per cell) or 'avg' (cell-averaged, see mango-lite averagedGrid)
function wanted(q) {
  const o = S.ordering, other = o === 'NO' ? 'IO' : 'NO';
  const pn = (anti) => (anti ? 'nubar' : 'nu');
  const withQ = (p) => ({ ...p, q });
  const list = [
    [`${o}:${pn(S.anti)}`, withQ(engineParams(o))],
    [`${other}:${pn(S.anti)}`, withQ(engineParams(other))],
    ['vac', withQ(engineParams(o, { vacuum: true }))],
  ];
  // prefetched so that the nu/nubar toggle is instant, except on machines with few
  // cores, where it would double the time to "ready" (the toggle then recomputes)
  if (PREFETCH) list.push(
    [`${o}:${pn(!S.anti)}`, withQ(engineParams(o, { anti: !S.anti }))],
    [`${other}:${pn(!S.anti)}`, withQ(engineParams(other, { anti: !S.anti }))],
  );
  return list;
}

const statusEl = document.getElementById('status');
function setStatus(text, busy) {
  statusEl.textContent = text;
  statusEl.classList.toggle('busy', !!busy);
}

const NICE = { nu: 'ν', nubar: 'ν̄', vac: 'vacuum' };
const nice = (name) => name.split(':').map((t) => NICE[t] ?? t).join(' ');

// Two passes: first whatever shows up fastest (the cell average if it is cached, else a
// one-sample-per-cell grid, ~30 ms), then the cell averages replace them.
async function refreshGrids() {
  const my = ++gen;
  for (const k of Object.keys(grids)) delete grids[k];
  const fast = wanted('fast'), avg = S.smooth ? wanted('avg') : [];
  cancelExcept(new Set([...fast, ...avg].map(([, p]) => gridKey(p))));
  const fail = (err) => {
    if (my === gen && !isCancelled(err)) setStatus(`engine error: ${err.message}`, false);
  };
  let lastFast = null, lastAvg = null;
  for (let i = 0; i < fast.length; i++) {
    const [name, pf] = fast[i];
    const params = avg.length && hasGrid(avg[i][1]) ? avg[i][1] : pf;
    const cached = hasGrid(params);
    if (!cached) setStatus(`computing ${i + 1}/${fast.length}: ${nice(name)} (${S.res}×${S.res})…`, true);
    try {
      const g = await computeGrid(params);
      if (my !== gen) return;
      grids[name] = g;
      if (!cached) lastFast = g.seconds;
      render();
    } catch (err) { fail(err); return; }
  }
  for (let i = 0; i < avg.length; i++) {
    const [name, pa] = avg[i];
    if (grids[name]?.quality === 'avg') continue;
    setStatus(`averaging within cells ${i + 1}/${avg.length}: ${nice(name)}…`, true);
    try {
      const g = await computeGrid(pa);
      if (my !== gen) return;
      grids[name] = g;
      lastAvg = g.seconds;
      render();
    } catch (err) { fail(err); return; }
  }
  const t = lastAvg ?? lastFast;
  const timing = t !== null ? `${(1000 * t).toFixed(0)} ms per grid on ${N_WORKERS} threads` : 'cached';
  setStatus(`ready · ${fast.length} grids${S.smooth ? ', averaged within cells' : ''}, ${timing}`, false);
}

// one trajectory, matter and vacuum, nu and nubar: a few ms, so it runs synchronously.
// Sampling follows the fast (Δm²31) oscillation, ~16 points per vacuum oscillation length
// (2.48 km × E[GeV] / Δm²[eV²]): about 100 km at 0.1 GeV.
function refreshPath() {
  const lOsc = (2.48 * S.sel.E) / (S.dm32 * 1e-3);
  const n = Math.min(4000, Math.max(240, Math.ceil((16 * baselineKm(S.sel.cz, H_ATM_KM)) / lOsc)));
  path = computePath(engineParams(S.ordering), S.sel.E, S.sel.cz, n);
  renderPath();
}

let paramTimer = null;
function paramsChanged(now) {
  clearTimeout(paramTimer);
  paramTimer = setTimeout(() => { refreshGrids(); refreshPath(); }, now ? 0 : 30);
}

// ---------------------------------------------------------------- views

const meta = earthMeta();
const GUIDES = [
  { cz: meta.critical_cz.outer_core, label: 'outer core' },
  { cz: meta.critical_cz.inner_core, label: 'inner core' },
  { cz: 0, label: 'horizon' },
];

function select(E, cz) {
  S.sel = { E, cz };
  for (const p of Object.values(oscs)) p.select(E, cz);
  renderSlices();
  refreshPath();
}
const oscs = {};
for (const id of Object.keys(PANELS)) {
  oscs[id] = new Oscillogram(document.getElementById(`osc-${id}`), { onSelect: select });
  oscs[id].guides = GUIDES;
}
const earth = new EarthView(document.getElementById('earth'), {
  onScrub: (t) => { stopAnim('travel'); S.t = t; renderPathMarker(); },
});
earth.setMeta(meta);
const sliceE = new LinePlot(document.getElementById('sliceE'), { xlog: true, xlabel: 'E [GeV]', ylabel: 'P' });
const sliceC = new LinePlot(document.getElementById('sliceC'), { xlabel: 'cos θ_z', ylabel: 'P' },
  { onClick: (cz) => select(S.sel.E, cz) });
const pathFlavor = new LinePlot(document.getElementById('pathFlavor'), { xlabel: '', ylabel: 'fraction', margin: { l: 46, r: 40, t: 8, b: 22 } },
  { onHover: (x) => { if (x !== null && path && !anim.travel) { S.t = Math.max(0, Math.min(1, x / path.total_km)); earth.setT(S.t); } } });
const pathRho = new LinePlot(document.getElementById('pathRho'), { xlabel: 'distance travelled from production [km]', ylabel: 'ρ', margin: { l: 46, r: 40, t: 6, b: 34 } });

function axesArrays(g) {
  const E = Array.from({ length: g.nE }, (_, i) => S.erange[0] * (S.erange[1] / S.erange[0]) ** (i / (g.nE - 1)));
  const cz = Array.from({ length: g.nC }, (_, i) => -1 + (2 * i) / (g.nC - 1));
  return { E, cz };
}

function robustMax(v) {
  const a = Array.from(v, Math.abs).filter(Number.isFinite).sort((x, y) => x - y);
  const m = a[Math.floor(0.995 * (a.length - 1))] || 1e-12;
  const mag = 10 ** Math.floor(Math.log10(m));
  return [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((k) => k * mag).find((x) => x >= m);
}

function gridName(ordering, anti) { return `${ordering}:${anti ? 'nubar' : 'nu'}`; }

// Colour scales follow each panel's range but stay frozen while theta23 / delta move,
// so a sweep shows the change instead of a rescaled picture.
const scales = {};
function frozenScale(id, values, cap) {
  const key = `${gen}:${S.channel}:${S.anti}:${S.ordering}`;
  if (scales[id]?.key !== key) scales[id] = { key, vmax: Math.min(cap, robustMax(values)) };
  return scales[id].vmax;
}

function renderOscs() {
  const base = grids[gridName(S.ordering, S.anti)];
  if (!base) return;
  const { a, b } = CHANNELS[S.channel];
  const W = channelWeights(th23(), S.dcp * DEG, a, b, S.anti);
  const n = base.nC * base.nE;
  const { E, cz } = axesArrays(base);
  const prob = (g) => probAvgMap(g.M, W, new Float32Array(n));
  const values = { P: prob(base) };
  const gNO = grids[gridName('NO', S.anti)], gIO = grids[gridName('IO', S.anti)];
  if (gNO && gIO) {
    const pNO = gNO === base ? values.P : prob(gNO), pIO = gIO === base ? values.P : prob(gIO);
    values.noio = pNO.map((v, i) => v - pIO[i]);
  }
  if (grids.vac) {
    const pv = prob(grids.vac);
    values.vac = values.P.map((v, i) => v - pv[i]);
  }
  for (const [id, panel] of Object.entries(PANELS)) {
    document.getElementById(`title-${id}`).textContent = panel.title();
    const v = values[id];
    if (!v) continue;
    const seq = id === 'P';
    oscs[id].set({ values: v, nE: base.nE, nC: base.nC, E, cz, kind: seq ? 'seq' : 'div',
      vmax: frozenScale(id, v, 1), label: `${chLabel()}${{ P: '', noio: ' NO − IO', vac: ' Earth − vacuum' }[id]}`,
      fmt: (x) => (seq ? x.toFixed(3) : x.toFixed(4)), hAtm: H_ATM_KM, short: panel.short });
    oscs[id].select(S.sel.E, S.sel.cz);
  }
}

function legendHTML(items) {
  return items.map(([color, dash, text]) => `<span class="item">${swatch(color, dash)}${text}</span>`).join('');
}

function renderSlices() {
  const base = grids[gridName(S.ordering, S.anti)];
  if (!base) return;
  const { a, b } = CHANNELS[S.channel];
  const W = channelWeights(th23(), S.dcp * DEG, a, b, S.anti);
  const { E, cz } = axesArrays(base);
  const c = Math.round(((S.sel.cz + 1) / 2) * (base.nC - 1));
  const e = Math.round((Math.log10(S.sel.E / S.erange[0]) / Math.log10(S.erange[1] / S.erange[0])) * (base.nE - 1));
  const col = maps().flavor[b], muted = css('--text-muted');
  const lines = [['NO', grids[gridName('NO', S.anti)], col, false], ['IO', grids[gridName('IO', S.anti)], col, true],
    [`vacuum (${S.ordering})`, grids.vac, muted, 'dot']];
  const serE = [], serC = [];
  for (const [label, g, color, dash] of lines) {
    if (!g) continue;
    serE.push({ x: E, y: E.map((_, i) => probAvgAt(g.M, c * g.nE + i, W)), color, dash, label, width: label === S.ordering ? 2.5 : 2 });
    serC.push({ x: cz, y: cz.map((_, j) => probAvgAt(g.M, j * g.nE + e, W)), color, dash, label, width: label === S.ordering ? 2.5 : 2 });
  }
  const L = baselineKm(cz[c], H_ATM_KM);
  document.getElementById('sliceE-title').textContent = `${chLabel()} vs energy at cos θ_z = ${cz[c].toFixed(2)} (L ≈ ${L.toFixed(0)} km)`;
  document.getElementById('sliceC-title').textContent = `${chLabel()} vs direction at E = ${E[e].toFixed(E[e] < 10 ? 2 : 1)} GeV`;
  const leg = legendHTML([[col, false, 'normal ordering'], [col, true, 'inverted ordering'], [muted, true, 'vacuum']]);
  document.getElementById('slice-legend').innerHTML = leg;
  document.getElementById('slice-legend2').innerHTML = leg;
  const fmtE = (x) => `E = ${x.toFixed(x < 10 ? 2 : 1)} GeV`;
  sliceE.set({ xlim: S.erange, ylim: [0, 1], series: serE, vlines: [{ x: E[e], color: css('--text-muted'), dash: true }], fmtX: fmtE });
  const guides = meta ? [meta.critical_cz.outer_core, meta.critical_cz.inner_core].map((x) => ({ x, color: css('--axis'), dash: true })) : [];
  sliceC.set({ xlim: [-1, 1], ylim: [0, 1], series: serC, fmtX: (x) => `cos θ_z = ${x.toFixed(3)} (L ≈ ${baselineKm(x, H_ATM_KM).toFixed(0)} km)`,
    vlines: [...guides, { x: cz[c], color: css('--text-muted'), width: 1.5 }] });
}

let frac = null, fracVac = null;
function renderPath() {
  if (!path) return;
  const { a, b } = CHANNELS[S.channel];
  const key = S.anti ? 'nubar' : 'nu';
  frac = flavorFractions(path[key], th23(), S.dcp * DEG, a, S.anti);
  fracVac = flavorFractions(path[`${key}_vac`], th23(), S.dcp * DEG, a, S.anti);
  earth.set(path, frac, { cz: path.cz, a, anti: S.anti });
  earth.setT(S.t);
  const x = path.x_km, n = x.length;
  const { flavor } = maps();
  const cum1 = frac[0].map((v, i) => v + frac[1][i]);
  const v1 = fracVac[0], v2 = fracVac[0].map((v, i) => v + fracVac[1][i]);
  const zeros = new Float64Array(n), ones = new Float64Array(n).fill(1);
  const last = n - 1;
  const texts = [];
  const mid = [frac[0][last] / 2, frac[0][last] + frac[1][last] / 2, cum1[last] + frac[2][last] / 2];
  for (let f = 0; f < 3; f++) {
    const share = [frac[0], frac[1], frac[2]][f][last];
    if (share > 0.07) texts.push({ x: x[last], y: mid[f], text: `ν${FLAVOR[f]}`, dx: 4, color: css('--text-secondary') });
  }
  const muted = css('--text-primary');
  pathFlavor.set({
    xlim: [0, path.total_km], ylim: [0, 1],
    bands: [
      { x, y0: zeros, y1: frac[0], color: flavor[0], label: nuName(0, S.anti) },
      { x, y0: frac[0], y1: cum1, color: flavor[1], label: nuName(1, S.anti) },
      { x, y0: cum1, y1: ones, color: flavor[2], label: nuName(2, S.anti) },
    ],
    series: [
      { x, y: v1, color: muted, dash: true, width: 1.25 },
      { x, y: v2, color: muted, dash: true, width: 1.25 },
    ],
    vlines: [{ x: S.t * path.total_km, color: css('--text-primary'), width: 1.5 }],
    texts, fmtX: (v) => `${v.toFixed(0)} km from production`, fmtY: (v) => `${(100 * v).toFixed(1)}%`,
  });
  pathRho.set({
    xlim: [0, path.total_km], ylim: [0, 14],
    bands: [{ x, y0: zeros, y1: path.rho, color: css('--grid') }],
    series: [{ x, y: path.rho, color: css('--text-muted'), width: 1.5, label: 'density [g/cm³]' }],
    vlines: [{ x: S.t * path.total_km, color: css('--text-primary'), width: 1.5 }],
    fmtX: (v) => `${v.toFixed(0)} km`, fmtY: (v) => v.toFixed(2), yticks: [0, 5, 10],
  });
  document.getElementById('path-legend').innerHTML = `<span class="item"><b>Flavor content along the path</b></span>` + legendHTML([
    [flavor[0], false, nuName(0, S.anti)], [flavor[1], false, nuName(1, S.anti)], [flavor[2], false, nuName(2, S.anti)],
    [muted, true, 'same baseline in vacuum']]);
  const R = meta?.r_earth_km ?? 6371;
  const depth = path.cz < 0 ? R - R * Math.sqrt(1 - path.cz * path.cz) : 0;
  const crosses = path.cz >= 0 ? 'down-going, atmosphere only'
    : depth > R - (meta?.inner_core_km ?? 1221.5) ? 'crosses the inner core'
      : depth > R - (meta?.core_km ?? 3480) ? 'crosses the outer core' : 'mantle only';
  document.getElementById('journey-info').textContent = `E = ${path.E.toFixed(1)} GeV · cos θ_z = ${path.cz.toFixed(2)} · ${path.total_km.toFixed(0)} km · deepest ${depth.toFixed(0)} km · ${crosses}`;
}

function renderPathMarker() {
  if (!path) return;
  earth.setT(S.t);
  for (const pl of [pathFlavor, pathRho]) {
    if (!pl.data) continue;
    pl.data.vlines = [{ x: S.t * path.total_km, color: css('--text-primary'), width: 1.5 }];
    pl.draw();
  }
}

function render() {
  renderExplain();
  renderOscs();
  renderSlices();
  renderPath();
}

// ---------------------------------------------------------------- animation

const anim = { travel: false, dcp: false, cz: false };
let lastT = null, czDir = 1, czAccum = 0;
function stopAnim(k) { anim[k] = false; updateAnimButtons(); }
function updateAnimButtons() {
  document.getElementById('travel').textContent = anim.travel ? '❚❚ travel' : '▶ travel';
  document.getElementById('sweep-dcp').innerHTML = anim.dcp ? '❚❚ stop δ<sub>CP</sub> sweep' : '▶ sweep δ<sub>CP</sub>';
  document.getElementById('sweep-cz').textContent = anim.cz ? '❚❚ stop sweep' : '▶ sweep zenith';
}
function loop(ts) {
  const dt = lastT === null ? 0 : Math.min(0.1, (ts - lastT) / 1000);
  lastT = ts;
  if (anim.travel && path) {
    S.t += dt / 4;
    if (S.t > 1.25) S.t = 0;
    const t = S.t; S.t = Math.min(1, t);
    renderPathMarker();
    S.t = t;
  }
  if (anim.dcp) {
    S.dcp = (S.dcp + 60 * dt) % 360;
    syncSlider('dcp');
    renderOscs(); renderSlices(); renderPath();
  }
  if (anim.cz) {
    czAccum += dt;
    let cz = S.sel.cz + czDir * 0.12 * dt;
    if (cz > 0.05) { cz = 0.05; czDir = -1; }
    if (cz < -1) { cz = -1; czDir = 1; }
    S.sel = { ...S.sel, cz };
    for (const p of Object.values(oscs)) p.select(S.sel.E, cz);
    if (czAccum > 0.03) { czAccum = 0; renderSlices(); refreshPath(); }
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- wiring

function buildHeader() {
  const nav = document.getElementById('channels');
  for (const ch of Object.keys(CHANNELS)) {
    const bt = document.createElement('button');
    bt.dataset.ch = ch;
    nav.appendChild(bt);
    bt.addEventListener('click', () => { S.channel = ch; syncHeader(); render(); });
  }
  for (const bt of document.querySelectorAll('#particle button')) {
    bt.addEventListener('click', () => { S.anti = bt.dataset.v === 'nubar'; syncHeader(); refreshGrids(); render(); });
  }
  for (const bt of document.querySelectorAll('#ordering button')) {
    bt.addEventListener('click', () => { S.ordering = bt.dataset.v; syncHeader(); refreshGrids(); refreshPath(); });
  }
  document.getElementById('theme').addEventListener('click', () => {
    const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem('nuearth-theme', t); } catch { /* private mode */ }
    for (const p of Object.values(oscs)) p.paintImage();
    earth.cache.key = null; render();
  });
}
function syncHeader() {
  for (const bt of document.querySelectorAll('#channels button')) {
    const { a, b } = CHANNELS[bt.dataset.ch];
    bt.textContent = `${nuName(a, S.anti)} → ${nuName(b, S.anti)}`;
    bt.classList.toggle('on', bt.dataset.ch === S.channel);
  }
  for (const bt of document.querySelectorAll('#particle button')) bt.classList.toggle('on', (bt.dataset.v === 'nubar') === S.anti);
  for (const bt of document.querySelectorAll('#ordering button')) bt.classList.toggle('on', bt.dataset.v === S.ordering);
}

window.addEventListener('error', (e) => { (window.__errs ||= []).push(String(e.message)); });
window.addEventListener('unhandledrejection', (e) => { (window.__errs ||= []).push(String(e.reason)); });

// the dashboard layout sizes itself against the (possibly wrapped) header height
function syncHeaderHeight() {
  document.documentElement.style.setProperty('--header-h', `${document.getElementById('top').offsetHeight}px`);
}

function initInfoTips() {
  for (const el of document.querySelectorAll('.info')) {
    const show = () => {
      const r = el.getBoundingClientRect();
      tooltip.show(tipText(el.dataset.tip), r.left, r.bottom);
    };
    el.addEventListener('pointerenter', show);
    el.addEventListener('focus', show);
    el.addEventListener('click', show);
    el.addEventListener('pointerleave', () => tooltip.hide());
    el.addEventListener('blur', () => tooltip.hide());
  }
}

function init() {
  try { const t = localStorage.getItem('nuearth-theme'); if (t) document.documentElement.dataset.theme = t; } catch { /* */ }
  if (!localStorage.getItem?.('nuearth-theme') && window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    document.documentElement.dataset.theme = 'dark';
  }
  buildHeader(); syncHeader();
  buildSliders(document.getElementById('live-sliders'), LIVE, () => render());
  buildSliders(document.getElementById('osc-sliders'), RECOMPUTED, (_, now) => paramsChanged(now));
  document.getElementById('erange').addEventListener('change', (ev) => {
    S.erange = ev.target.value.split(',').map(Number);
    S.sel.E = Math.min(Math.max(S.sel.E, S.erange[0]), S.erange[1]);
    paramsChanged(true);
  });
  document.getElementById('res').addEventListener('change', (ev) => { S.res = +ev.target.value; paramsChanged(true); });
  document.getElementById('smooth').addEventListener('change', (ev) => { S.smooth = ev.target.checked; refreshGrids(); });
  document.getElementById('reset').addEventListener('click', () => {
    Object.assign(S, DEFAULTS);
    for (const k of Object.keys(sliderEls)) syncSlider(k);
    paramsChanged(true);
  });
  document.getElementById('travel').addEventListener('click', () => { anim.travel = !anim.travel; if (anim.travel && S.t >= 1) S.t = 0; updateAnimButtons(); });
  document.getElementById('sweep-dcp').addEventListener('click', () => { anim.dcp = !anim.dcp; updateAnimButtons(); });
  document.getElementById('sweep-cz').addEventListener('click', () => { anim.cz = !anim.cz; updateAnimButtons(); });
  syncHeaderHeight();
  initInfoTips();
  window.addEventListener('resize', () => { syncHeaderHeight(); requestAnimationFrame(render); });
  document.getElementById('engine-info').textContent = `Engine: mango-lite, a JavaScript port of MANGO (PREM Earth, cayley propagator) that agrees with it to 5×10⁻¹⁴; running on ${N_WORKERS} worker threads in this browser.`;
  refreshGrids();
  refreshPath();
  requestAnimationFrame(loop);
}

init();
window.nuearth = { S, grids, render, select, get path() { return path; } };
