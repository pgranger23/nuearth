// In-browser engine: mango-lite (JS port of MANGO) on a pool of Web Workers.
// Grids are split into blocks of cos(zenith) rows; queued blocks of grids that are
// no longer wanted are dropped as soon as the parameters move.  A small LRU keeps
// recent grids.
import { pathEvolution, premDensity, NM, R_EARTH_KM, CORE_RADIUS_KM } from './mango-lite.js';

const DEG = Math.PI / 180;
const INNER_CORE_KM = 1221.5;
export const N_WORKERS = Math.max(1, Math.min(12, (navigator.hardwareConcurrency || 4) - 1));
const ROWS_PER_TASK = 4;
const MAX_GRIDS = 24; // ~4.6 MB each at 160x160 (45 floats per pixel)

function query(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) q.set(k, typeof v === 'number' ? String(+v.toPrecision(9)) : String(v));
  return q.toString();
}
export const gridKey = query;

function inputs(p) {
  return {
    osc: { theta12: p.th12 * DEG, theta13: p.th13 * DEG, dm21: p.dm21, dm31: p.dm31 },
    earth: { kind: p.kind, hAtm: p.h_atm, detDepth: 0 }, // PREM as in MANGO, or vacuum
    nSub: p.n_sub ?? 2,
  };
}

export function gridAxes(p) {
  const E = new Float64Array(p.n_e), CZ = new Float64Array(p.n_cz);
  for (let i = 0; i < p.n_e; i++) E[i] = p.emin * (p.emax / p.emin) ** (i / (p.n_e - 1));
  for (let i = 0; i < p.n_cz; i++) CZ[i] = -1 + (2 * i) / (p.n_cz - 1);
  return { E, CZ };
}

class Cancelled extends Error {}

// ------------------------------------------------------------------ worker pool
const workers = [];
const queue = [];          // pending tasks
const running = new Map(); // id -> task
let nextId = 1;

function pump() {
  for (const w of workers) {
    if (w.busy || queue.length === 0) continue;
    const task = queue.shift();
    w.busy = true;
    task.worker = w;
    running.set(task.id, task);
    w.postMessage({ id: task.id, ...task.msg });
  }
}

function startPool() {
  if (workers.length) return;
  for (let i = 0; i < N_WORKERS; i++) {
    const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    w.busy = false;
    w.onmessage = (ev) => {
      const task = running.get(ev.data.id);
      running.delete(ev.data.id);
      w.busy = false;
      if (task) {
        if (ev.data.error) task.reject(new Error(ev.data.error));
        else task.resolve(ev.data);
      }
      pump();
    };
    w.onerror = (ev) => {
      const t = [...running.values()].find((x) => x.worker === w);
      if (t) { running.delete(t.id); t.reject(new Error(ev.message || 'worker error')); }
      w.busy = false;
      pump();
    };
    workers.push(w);
  }
}

function runTask(msg, owner) {
  return new Promise((resolve, reject) => {
    queue.push({ id: nextId++, msg, owner, resolve, reject });
    pump();
  });
}

// ------------------------------------------------------------------ grids
const cache = new Map();     // key -> Promise<grid>
const pending = new Map();   // key -> true while computing

export function hasGrid(params) {
  const q = query(params);
  return cache.has(q) && !pending.has(q);
}

export function computeGrid(params) {
  startPool();
  const q = query(params);
  if (cache.has(q)) {
    const p = cache.get(q);
    cache.delete(q); cache.set(q, p);
    return p;
  }
  const { osc, earth, nSub } = inputs(params);
  const { E, CZ } = gridAxes(params);
  const nC = CZ.length, nE = E.length;
  // quality 'fast': one sample per cell; 'avg': adaptive oversampling (see averagedGrid)
  const cell = { dlnE: Math.log(params.emax / params.emin) / (nE - 1), dcz: 2 / (nC - 1), fast: params.q !== 'avg' };
  const t0 = performance.now();
  const blocks = [];
  for (let r = 0; r < nC; r += ROWS_PER_TASK) {
    const rows = CZ.slice(r, Math.min(nC, r + ROWS_PER_TASK));
    blocks.push(runTask({ osc, earth, E, CZ: rows, anti: !!params.anti, nSub, cell }, q)
      .then((res) => ({ r, res })));
  }
  pending.set(q, true);
  const p = Promise.all(blocks).then((parts) => {
    const M = new Float32Array(nC * nE * NM);
    for (const { r, res } of parts) M.set(res.M, r * nE * NM);
    pending.delete(q);
    return { M, nC, nE, quality: params.q ?? 'fast', seconds: (performance.now() - t0) / 1000 };
  });
  p.catch(() => { pending.delete(q); if (cache.get(q) === p) cache.delete(q); });
  cache.set(q, p);
  while (cache.size > MAX_GRIDS) cache.delete(cache.keys().next().value);
  return p;
}

// Drop queued work for grids not in `keep` (a set of keys).  Their promises reject
// with Cancelled and leave the cache synchronously, so a new request recomputes.
export function cancelExcept(keep) {
  const dropped = new Set();
  for (let i = queue.length - 1; i >= 0; i--) {
    const t = queue[i];
    if (!keep.has(t.owner)) { queue.splice(i, 1); dropped.add(t.owner); t.reject(new Cancelled('cancelled')); }
  }
  for (const k of dropped) { cache.delete(k); pending.delete(k); }
}
export const isCancelled = (err) => err instanceof Cancelled;

// ------------------------------------------------------------------ paths and meta
// One trajectory: matter and vacuum, nu and nubar (main thread, ~1 ms each).
export function computePath(params, E, cz, nPoints = 240) {
  const { osc, earth, nSub } = inputs(params);
  const vac = { ...earth, kind: 'vacuum' };
  const nu = pathEvolution(osc, earth, E, cz, nPoints, false, nSub);
  return {
    x_km: nu.x_km, total_km: nu.total_km, rho: nu.rho, radius_km: nu.radius_km,
    nu: nu.S,
    nubar: pathEvolution(osc, earth, E, cz, nPoints, true, nSub).S,
    nu_vac: pathEvolution(osc, vac, E, cz, nPoints, false, nSub).S,
    nubar_vac: pathEvolution(osc, vac, E, cz, nPoints, true, nSub).S,
    cz, E,
  };
}

export function earthMeta() {
  const n = 400;
  const r_km = Array.from({ length: n }, (_, i) => (R_EARTH_KM * i) / (n - 1));
  const crit = (r) => -Math.sqrt(1 - (r / R_EARTH_KM) ** 2);
  return {
    prem: { r_km, rho: r_km.map(premDensity) },
    core_km: CORE_RADIUS_KM, inner_core_km: INNER_CORE_KM, r_earth_km: R_EARTH_KM,
    critical_cz: { outer_core: crit(CORE_RADIUS_KM), inner_core: crit(INNER_CORE_KM) },
  };
}
