// Earth cross-section: PREM shells shaded by density, the chord of the selected
// trajectory as a ribbon whose three stacked bands are the flavor fractions along
// the way, and a neutrino marker riding it.
import { setupCanvas, FONT, tooltip } from './plot.js';
import { css, maps } from './colors.js';
import { FLAVOR } from './physics.js';

export class EarthView {
  constructor(canvas, { onScrub } = {}) {
    this.canvas = canvas;
    this.onScrub = onScrub;
    this.meta = null; this.path = null; this.frac = null; this.t = 1;
    this.cache = { key: null, canvas: document.createElement('canvas') };
    let dragging = false;
    const scrub = (ev) => {
      const s = this.nearest(ev);
      if (s !== null) this.onScrub?.(s);
    };
    canvas.addEventListener('pointerdown', (ev) => { if (this.nearest(ev) !== null) { dragging = true; canvas.setPointerCapture(ev.pointerId); scrub(ev); } });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('pointermove', (ev) => { if (dragging) scrub(ev); this.hover(ev); });
    canvas.addEventListener('pointerleave', () => tooltip.hide());
  }
  setMeta(meta) { this.meta = meta; this.cache.key = null; }
  // path: server path payload; frac: [Pe, Pmu, Ptau] arrays; info: {cz, hAtm, a, anti}
  set(path, frac, info) { this.path = path; this.frac = frac; this.info = info; this.draw(); }
  setT(t) { this.t = t; this.draw(); }
  geometry(w, h) {
    const R = 0.43 * Math.min(w, h);
    return { R, cx: w / 2, cy: h / 2 + 0.03 * h, k: R / this.meta.r_earth_km };
  }
  points() {
    // sample positions in Earth coordinates (km), detector on +z, source along (sin z, cos z)
    const p = this.path, cz = this.info.cz, sz = Math.sqrt(Math.max(0, 1 - cz * cz));
    const rDet = this.meta.r_earth_km;
    return Array.from(p.x_km, (x) => { const d = p.total_km - x; return [d * sz, rDet + d * cz]; });
  }
  earthLayer(w, h) {
    const key = `${w}x${h}:${document.documentElement.dataset.theme}`;
    const c = this.cache;
    if (c.key === key) return c.canvas;
    const dpr = window.devicePixelRatio || 1;
    c.canvas.width = Math.round(w * dpr); c.canvas.height = Math.round(h * dpr);
    const ctx = c.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { R, cx, cy, k } = this.geometry(w, h);
    const { earth } = maps();
    const { r_km, rho } = this.meta.prem;
    const rhoMax = 13.1;
    for (let i = r_km.length - 1; i >= 0; i--) {
      const t = Math.sqrt(rho[i] / rhoMax);
      const kk = Math.round(Math.max(0, Math.min(1, t)) * 255);
      ctx.fillStyle = `rgb(${earth[kk * 3]},${earth[kk * 3 + 1]},${earth[kk * 3 + 2]})`;
      ctx.beginPath(); ctx.arc(cx, cy, Math.max(0.5, r_km[i] * k), 0, 2 * Math.PI); ctx.fill();
    }
    ctx.strokeStyle = css('--axis'); ctx.lineWidth = 1;
    for (const r of [this.meta.inner_core_km, this.meta.core_km, this.meta.r_earth_km]) {
      ctx.beginPath(); ctx.arc(cx, cy, r * k, 0, 2 * Math.PI); ctx.stroke();
    }
    if (R >= 110) { // layer names only when there is room for them
      ctx.font = FONT; ctx.fillStyle = css('--text-secondary'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const lab = (txt, r) => ctx.fillText(txt, cx - r * k * 0.87, cy + r * k * 0.5);
      lab('inner core', 0);
      lab('outer core', (this.meta.inner_core_km + this.meta.core_km) / 2);
      lab('mantle', (this.meta.core_km + this.meta.r_earth_km) / 2);
    }
    c.key = key;
    return c.canvas;
  }
  draw() {
    const { ctx, w, h } = setupCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    if (!this.meta) return;
    ctx.drawImage(this.earthLayer(w, h), 0, 0, w, h);
    const { R, cx, cy, k } = this.geometry(w, h);
    const toPx = ([x, z]) => [cx + x * k, cy - z * k];
    // detector
    const [dX, dY] = toPx([0, this.meta.r_earth_km]);
    ctx.fillStyle = css('--text-primary');
    ctx.fillRect(dX - 5, dY - 5, 10, 10);
    ctx.font = FONT; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('detector', dX + 9, dY - 4);
    if (!this.path || !this.frac) return;
    const pts = this.points().map(toPx);
    const n = pts.length;
    // unit normal of the (straight) chord
    const [x0, y0] = pts[0], [x1, y1] = pts[n - 1];
    const len = Math.hypot(x1 - x0, y1 - y0) || 1;
    const nx = -(y1 - y0) / len, ny = (x1 - x0) / len;
    const W = 16;
    const { flavor } = maps();
    const [Pe, Pm] = this.frac;
    const edge = (i, f) => [pts[i][0] + nx * W * (f - 0.5), pts[i][1] + ny * W * (f - 0.5)];
    const bounds = [(i) => 0, (i) => Pe[i], (i) => Pe[i] + Pm[i], (i) => 1];
    for (let b = 0; b < 3; b++) {
      ctx.fillStyle = flavor[b];
      ctx.beginPath();
      for (let i = 0; i < n; i++) { const [X, Y] = edge(i, bounds[b + 1](i)); ctx.lineTo(X, Y); }
      for (let i = n - 1; i >= 0; i--) { const [X, Y] = edge(i, bounds[b](i)); ctx.lineTo(X, Y); }
      ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = css('--surface'); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const [X, Y] = edge(i, 0); ctx.lineTo(X, Y); }
    ctx.stroke();
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const [X, Y] = edge(i, 1); ctx.lineTo(X, Y); }
    ctx.stroke();
    // production point
    ctx.fillStyle = css('--text-primary'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.beginPath(); ctx.arc(x0, y0, 4, 0, 2 * Math.PI); ctx.fill();
    const below = y0 > cy;
    ctx.textBaseline = below ? 'top' : 'bottom';
    const plabel = R >= 110 ? `${this.info.anti ? 'ν̄' : 'ν'}${FLAVOR[this.info.a]} produced in the atmosphere`
      : `${this.info.anti ? 'ν̄' : 'ν'}${FLAVOR[this.info.a]} produced`;
    const half = ctx.measureText(plabel).width / 2;
    ctx.fillText(plabel, Math.min(w - half - 4, Math.max(half + 4, x0)), y0 + (below ? 8 : -8));
    // neutrino marker
    const i = Math.round(this.t * (n - 1));
    const [mx, my] = pts[i];
    ctx.lineWidth = 3; ctx.strokeStyle = css('--surface');
    ctx.fillStyle = css('--text-primary');
    ctx.beginPath(); ctx.arc(mx, my, 7, 0, 2 * Math.PI); ctx.stroke(); ctx.fill();
    // readout box
    const x = this.path.x_km[i];
    const lines = [`${x.toFixed(0)} km travelled · ρ = ${this.path.rho[i].toFixed(2)} g/cm³`,
      `νe ${(100 * this.frac[0][i]).toFixed(0)}%  ·  νμ ${(100 * this.frac[1][i]).toFixed(0)}%  ·  ντ ${(100 * this.frac[2][i]).toFixed(0)}%`];
    ctx.font = FONT;
    const bw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16, bh = 40;
    const bx = 6, by = h - bh - 6;
    ctx.fillStyle = css('--surface'); ctx.strokeStyle = css('--axis'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 6); ctx.fill(); ctx.stroke();
    ctx.fillStyle = css('--text-primary'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(lines[0], bx + 8, by + 6);
    ctx.fillText(lines[1], bx + 8, by + 22);
  }
  nearest(ev) {
    if (!this.path) return null;
    const r = this.canvas.getBoundingClientRect();
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    const { cx, cy, k } = this.geometry(r.width, r.height);
    const pts = this.points().map(([x, z]) => [cx + x * k, cy - z * k]);
    let best = -1, bd = 400;
    pts.forEach(([X, Y], i) => { const d = (X - px) ** 2 + (Y - py) ** 2; if (d < bd) { bd = d; best = i; } });
    return best < 0 ? null : best / (pts.length - 1);
  }
  hover(ev) {
    const s = this.nearest(ev);
    if (s === null) { tooltip.hide(); return; }
    const i = Math.round(s * (this.path.x_km.length - 1));
    tooltip.show(`<div class="tt-head">${this.path.x_km[i].toFixed(0)} km from production</div>
      radius ${this.path.radius_km[i].toFixed(0)} km · ρ ${this.path.rho[i].toFixed(2)} g/cm³<br>
      νe ${(100 * this.frac[0][i]).toFixed(1)}% · νμ ${(100 * this.frac[1][i]).toFixed(1)}% · ντ ${(100 * this.frac[2][i]).toFixed(1)}%<br>
      <span class="muted">drag along the path to move the neutrino</span>`, ev.clientX, ev.clientY);
  }
}
