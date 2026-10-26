// Oscillogram: a heatmap over (log E, cos zenith) with the core-crossing guides,
// a colorbar, hover readout and click/drag selection of one trajectory.
import { Axes, setupCanvas, tooltip, FONT, niceTicks, fmtTick } from './plot.js';
import { css, maps } from './colors.js';

export function baselineKm(cz, hAtm = 15, R = 6371) {
  return Math.sqrt((R + hAtm) ** 2 - R * R * (1 - cz * cz)) - R * cz;
}

export class Oscillogram {
  constructor(canvas, { onSelect }) {
    this.canvas = canvas;
    this.onSelect = onSelect;
    this.axes = new Axes({ xlog: true, xlabel: 'E [GeV]', ylabel: 'cos θ_z',
      margin: { l: 46, r: 58, t: 20, b: 36 } });
    this.off = document.createElement('canvas');
    this.map = null;
    this.sel = null;
    this.guides = [];
    let dragging = false;
    const pick = (ev) => {
      const { x, y } = this.local(ev);
      if (!this.map || !this.axes.inside(x, y)) return false;
      this.onSelect?.(this.axes.px2x(x), this.axes.px2y(y));
      return true;
    };
    canvas.addEventListener('pointerdown', (ev) => { dragging = pick(ev); if (dragging) canvas.setPointerCapture(ev.pointerId); });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('pointermove', (ev) => { if (dragging) pick(ev); this.hover(ev); });
    canvas.addEventListener('pointerleave', () => tooltip.hide());
  }
  local(ev) {
    const r = this.canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }
  // map: { values, nE, nC, E, cz, kind: 'seq' | 'div', vmax, label, fmt, hAtm }
  set(map) { this.map = map; this.paintImage(); this.draw(); }
  select(E, cz) { this.sel = { E, cz }; this.draw(); }
  paintImage() {
    const m = this.map;
    if (!m) return;
    const { seq, div } = maps();
    const table = m.kind === 'div' ? div : seq;
    this.off.width = m.nE; this.off.height = m.nC;
    const ctx = this.off.getContext('2d');
    const img = ctx.createImageData(m.nE, m.nC);
    for (let c = 0; c < m.nC; c++) {
      const row = m.nC - 1 - c; // cz increases upwards
      for (let e = 0; e < m.nE; e++) {
        const v = m.values[c * m.nE + e];
        const t = m.kind === 'div' ? 0.5 + 0.5 * v / m.vmax : v / m.vmax;
        const k = Math.max(0, Math.min(255, Math.round(t * 255)));
        const o = (row * m.nE + e) * 4;
        img.data[o] = table[k * 3]; img.data[o + 1] = table[k * 3 + 1]; img.data[o + 2] = table[k * 3 + 2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  draw() {
    const m = this.map;
    const { ctx, w, h } = setupCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    if (!m) return;
    const A = this.axes;
    A.layout(w, h, [m.E[0], m.E[m.nE - 1]], [m.cz[0], m.cz[m.nC - 1]]);
    const b = A.box;
    // pixel centres sit on grid points: extend the image by half a cell
    const dx = b.w / (m.nE - 1), dy = b.h / (m.nC - 1);
    ctx.save();
    ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.off, b.x - dx / 2, b.y - dy / 2, b.w + dx, b.h + dy);
    // guides: core-grazing directions and the horizon
    ctx.font = FONT;
    for (const g of this.guides) {
      if (g.cz < m.cz[0] || g.cz > m.cz[m.nC - 1]) continue;
      const y = Math.round(A.y2px(g.cz)) + 0.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 3;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y); ctx.stroke();
      ctx.strokeStyle = 'rgba(11,11,11,0.75)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y); ctx.stroke();
      ctx.setLineDash([]);
    }
    // selection crosshair
    if (this.sel) {
      const X = A.x2px(this.sel.E), Y = A.y2px(this.sel.cz);
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(X, Y, 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = 'rgba(11,11,11,0.9)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X, Y, 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([2, 3]); ctx.strokeStyle = 'rgba(11,11,11,0.6)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(b.x, Y); ctx.lineTo(X - 8, Y); ctx.moveTo(X + 8, Y); ctx.lineTo(b.x + b.w, Y);
      ctx.moveTo(X, b.y); ctx.lineTo(X, Y - 8); ctx.moveTo(X, Y + 8); ctx.lineTo(X, b.y + b.h); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
    const yt = niceTicks(m.cz[0], m.cz[m.nC - 1], Math.max(3, Math.min(8, Math.round(b.h / 34))));
    A.frame(ctx, { grid: false, yticks: yt });
    // guide labels inside the plot, just above their line, on a halo
    ctx.font = FONT; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.lineJoin = 'round';
    for (const g of this.guides) {
      if (g.cz < m.cz[0] || g.cz > m.cz[m.nC - 1]) continue;
      const y = A.y2px(g.cz) - 3;
      ctx.strokeStyle = css('--surface'); ctx.lineWidth = 3;
      ctx.strokeText(g.label, b.x + 5, y);
      ctx.fillStyle = css('--text-primary');
      ctx.fillText(g.label, b.x + 5, y);
    }
    this.drawColorbar(ctx, b);
  }
  drawColorbar(ctx, b) {
    const m = this.map;
    const { seq, div } = maps();
    const table = m.kind === 'div' ? div : seq;
    const x0 = b.x + b.w + 8, wbar = 9, y0 = b.y, hbar = b.h;
    for (let i = 0; i < hbar; i++) {
      const k = Math.round((1 - i / (hbar - 1)) * 255);
      ctx.fillStyle = `rgb(${table[k * 3]},${table[k * 3 + 1]},${table[k * 3 + 2]})`;
      ctx.fillRect(x0, y0 + i, wbar, 1.5);
    }
    ctx.strokeStyle = css('--axis'); ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, wbar, hbar);
    const lo = m.kind === 'div' ? -m.vmax : 0, hi = m.vmax;
    ctx.fillStyle = css('--text-muted'); ctx.font = FONT; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (const t of niceTicks(lo, hi, 4)) {
      const y = y0 + (1 - (t - lo) / (hi - lo)) * hbar;
      ctx.fillText(fmtTick(t), x0 + wbar + 4, y);
    }
    ctx.fillStyle = css('--text-secondary'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(m.short ?? '', x0, b.y - 5);
  }
  hover(ev) {
    const m = this.map;
    if (!m) return;
    const { x, y } = this.local(ev);
    const A = this.axes;
    if (!A.box || !A.inside(x, y)) { tooltip.hide(); return; }
    const E = A.px2x(x), cz = A.px2y(y);
    const e = Math.round((Math.log10(E / m.E[0]) / Math.log10(m.E[m.nE - 1] / m.E[0])) * (m.nE - 1));
    const c = Math.round(((cz - m.cz[0]) / (m.cz[m.nC - 1] - m.cz[0])) * (m.nC - 1));
    const v = m.values[c * m.nE + e];
    const L = baselineKm(cz, m.hAtm);
    tooltip.show(`<div class="tt-head">E = ${E.toFixed(E < 10 ? 2 : 1)} GeV · cos θ_z = ${cz.toFixed(3)}</div>
      baseline ≈ ${L.toFixed(0)} km<br>${m.label}: <b>${m.fmt(v)}</b><br><span class="muted">click to follow this neutrino</span>`,
    ev.clientX, ev.clientY);
  }
}
