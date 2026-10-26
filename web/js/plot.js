// Minimal canvas plotting: axes with hairline grid, lines, stacked bands,
// crosshair + tooltip hover.  Text always uses ink tokens; color marks carry identity.
import { css } from './colors.js';

export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(10, r.width), h = Math.max(10, r.height);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

export function logTicks(lo, hi) {
  const out = [];
  for (let k = Math.floor(Math.log10(lo)); k <= Math.ceil(Math.log10(hi)); k++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** k;
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
    }
  }
  return out;
}

export function fmtTick(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(0).replace('e+', 'e');
  return String(+v.toPrecision(3));
}

export const FONT = '12px system-ui, -apple-system, "Segoe UI", sans-serif';

export class Axes {
  constructor({ xlog = false, xlabel = '', ylabel = '', margin = { l: 46, r: 12, t: 10, b: 34 } } = {}) {
    Object.assign(this, { xlog, xlabel, ylabel, margin });
  }
  layout(w, h, xlim, ylim) {
    const m = this.margin;
    this.box = { x: m.l, y: m.t, w: w - m.l - m.r, h: h - m.t - m.b };
    this.xlim = xlim; this.ylim = ylim;
  }
  fx(x) { return this.xlog ? Math.log10(x) : x; }
  x2px(x) {
    const [a, b] = this.xlim.map((v) => this.fx(v));
    return this.box.x + ((this.fx(x) - a) / (b - a)) * this.box.w;
  }
  y2px(y) {
    const [a, b] = this.ylim;
    return this.box.y + (1 - (y - a) / (b - a)) * this.box.h;
  }
  px2x(px) {
    const [a, b] = this.xlim.map((v) => this.fx(v));
    const v = a + ((px - this.box.x) / this.box.w) * (b - a);
    return this.xlog ? 10 ** v : v;
  }
  px2y(py) {
    const [a, b] = this.ylim;
    return a + (1 - (py - this.box.y) / this.box.h) * (b - a);
  }
  inside(px, py) {
    const b = this.box;
    return px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h;
  }
  frame(ctx, { xticks, yticks, grid = true } = {}) {
    const b = this.box;
    xticks = xticks ?? (this.xlog ? logTicks(...this.xlim) : niceTicks(...this.xlim, Math.max(2, Math.round(b.w / 80))));
    yticks = yticks ?? niceTicks(...this.ylim, Math.max(2, Math.round(b.h / 45)));
    ctx.save();
    ctx.font = FONT;
    ctx.lineWidth = 1;
    if (grid) {
      ctx.strokeStyle = css('--grid');
      ctx.beginPath();
      for (const t of xticks) { const x = Math.round(this.x2px(t)) + 0.5; ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); }
      for (const t of yticks) { const y = Math.round(this.y2px(t)) + 0.5; ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y); }
      ctx.stroke();
    }
    ctx.strokeStyle = css('--axis');
    ctx.strokeRect(Math.round(b.x) + 0.5, Math.round(b.y) + 0.5, Math.round(b.w), Math.round(b.h));
    ctx.fillStyle = css('--text-muted');
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of xticks) ctx.fillText(fmtTick(t), this.x2px(t), b.y + b.h + 4);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of yticks) ctx.fillText(fmtTick(t), b.x - 5, this.y2px(t));
    ctx.fillStyle = css('--text-secondary');
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(this.xlabel, b.x + b.w / 2, b.y + b.h + 32);
    ctx.save();
    ctx.translate(12, b.y + b.h / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle';
    ctx.fillText(this.ylabel, 0, 0);
    ctx.restore();
    ctx.restore();
  }
}

export const tooltip = (() => {
  let el;
  return {
    show(html, clientX, clientY) {
      if (!el) { el = document.createElement('div'); el.className = 'tooltip'; document.body.appendChild(el); }
      el.innerHTML = html;
      el.style.display = 'block';
      const r = el.getBoundingClientRect();
      let x = clientX + 14, y = clientY + 14;
      if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
      if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
      el.style.left = `${x}px`; el.style.top = `${y}px`;
    },
    hide() { if (el) el.style.display = 'none'; },
  };
})();

export function swatch(color, dash) {
  const style = dash ? `border-top: 2px dashed ${color}; background: none; height: 0;` : `background:${color};`;
  return `<span class="sw" style="${style}"></span>`;
}

function interp(xs, ys, x) {
  const n = xs.length;
  if (n === 0) return NaN;
  const asc = xs[n - 1] >= xs[0];
  let lo = 0, hi = n - 1;
  if ((asc && (x <= xs[0] || x >= xs[n - 1])) || (!asc && (x >= xs[0] || x <= xs[n - 1]))) {
    return Math.abs(x - xs[0]) < Math.abs(x - xs[n - 1]) ? ys[0] : ys[n - 1];
  }
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if ((xs[m] < x) === asc) lo = m; else hi = m;
  }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}

// data = { xlim, ylim, xticks?, yticks?, series: [{x, y, color, dash, width, label, param?}],
//          bands: [{x, y0, y1, color, label}], vlines: [{x, color, dash, label}],
//          hlines: [{y, color, dash}], diag: bool, texts: [{x, y, text, color, align}],
//          dots: [{x, y, color, r}], fmtX, fmtY, hoverMode: 'x' | 'nearest' }
export class LinePlot {
  constructor(canvas, axesOpts, { onHover, onClick } = {}) {
    this.canvas = canvas;
    this.axes = new Axes(axesOpts);
    this.onHover = onHover; this.onClick = onClick;
    this.hoverPx = null;
    canvas.addEventListener('pointermove', (ev) => this.hover(ev));
    canvas.addEventListener('pointerleave', () => { this.hoverPx = null; tooltip.hide(); this.draw(); this.onHover?.(null); });
    canvas.addEventListener('click', (ev) => {
      const { x, y } = this.local(ev);
      if (this.data && this.axes.inside(x, y)) this.onClick?.(this.axes.px2x(x), this.axes.px2y(y));
    });
  }
  local(ev) {
    const r = this.canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }
  set(data) { this.data = data; this.draw(); }
  draw() {
    const d = this.data;
    const { ctx, w, h } = setupCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    if (!d) return;
    const A = this.axes;
    A.layout(w, h, d.xlim, d.ylim);
    A.frame(ctx, { xticks: d.xticks, yticks: d.yticks });
    const b = A.box;
    ctx.save();
    ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    if (d.diag) {
      ctx.strokeStyle = css('--axis'); ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      const lo = Math.max(d.xlim[0], d.ylim[0]), hi = Math.min(d.xlim[1], d.ylim[1]);
      ctx.beginPath(); ctx.moveTo(A.x2px(lo), A.y2px(lo)); ctx.lineTo(A.x2px(hi), A.y2px(hi)); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const band of d.bands ?? []) {
      ctx.fillStyle = band.color;
      ctx.beginPath();
      for (let i = 0; i < band.x.length; i++) ctx.lineTo(A.x2px(band.x[i]), A.y2px(band.y1[i]));
      for (let i = band.x.length - 1; i >= 0; i--) ctx.lineTo(A.x2px(band.x[i]), A.y2px(band.y0[i]));
      ctx.closePath(); ctx.fill();
    }
    if ((d.bands ?? []).length > 1) {
      // 2px surface gap between stacked fills
      ctx.strokeStyle = css('--surface'); ctx.lineWidth = 2;
      for (const band of d.bands.slice(0, -1)) {
        ctx.beginPath();
        for (let i = 0; i < band.x.length; i++) ctx.lineTo(A.x2px(band.x[i]), A.y2px(band.y1[i]));
        ctx.stroke();
      }
    }
    for (const hl of d.hlines ?? []) {
      ctx.strokeStyle = hl.color; ctx.lineWidth = 1; ctx.setLineDash(hl.dash ? [4, 4] : []);
      ctx.beginPath(); ctx.moveTo(b.x, A.y2px(hl.y)); ctx.lineTo(b.x + b.w, A.y2px(hl.y)); ctx.stroke();
    }
    for (const vl of d.vlines ?? []) {
      ctx.strokeStyle = vl.color; ctx.lineWidth = vl.width ?? 1; ctx.setLineDash(vl.dash ? [4, 4] : []);
      const x = A.x2px(vl.x);
      ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const s of d.series ?? []) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width ?? 2;
      ctx.setLineDash(s.dash === 'dot' ? [2, 3] : s.dash ? [6, 4] : []);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < s.x.length; i++) {
        const X = A.x2px(s.x[i]), Y = A.y2px(s.y[i]);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      if (s.closed) ctx.closePath();
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const dot of d.dots ?? []) {
      ctx.fillStyle = dot.color; ctx.strokeStyle = css('--surface'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(A.x2px(dot.x), A.y2px(dot.y), dot.r ?? 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
    ctx.font = FONT;
    for (const t of d.texts ?? []) {
      ctx.fillStyle = t.color ?? css('--text-secondary');
      ctx.textAlign = t.align ?? 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(t.text, A.x2px(t.x) + (t.dx ?? 0), A.y2px(t.y) + (t.dy ?? 0));
    }
    if (this.hoverPx && d.hoverMode !== 'nearest') {
      ctx.strokeStyle = css('--text-muted'); ctx.lineWidth = 1;
      const x = Math.round(this.hoverPx.x) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
    }
  }
  hover(ev) {
    const d = this.data;
    if (!d) return;
    const { x, y } = this.local(ev);
    const A = this.axes;
    if (!A.box || !A.inside(x, y)) { this.hoverPx = null; tooltip.hide(); this.draw(); this.onHover?.(null); return; }
    this.hoverPx = { x, y };
    const fx = d.fmtX ?? ((v) => fmtTick(v)), fy = d.fmtY ?? ((v) => v.toFixed(3));
    let html;
    if (d.hoverMode === 'nearest') {
      let best = null;
      for (const s of d.series ?? []) {
        if (!s.param) continue;
        for (let i = 0; i < s.x.length; i++) {
          const dd = (A.x2px(s.x[i]) - x) ** 2 + (A.y2px(s.y[i]) - y) ** 2;
          if (!best || dd < best.dd) best = { dd, s, i };
        }
      }
      if (!best || best.dd > 900) { tooltip.hide(); this.draw(); return; }
      html = `${swatch(best.s.color, best.s.dash)}<b>${best.s.label}</b><br>${best.s.param(best.i)}`;
    } else {
      const xv = A.px2x(x);
      const rows = [];
      for (const s of d.series ?? []) if (s.label) rows.push(`${swatch(s.color, s.dash)}${s.label}: <b>${fy(interp(s.x, s.y, xv))}</b>`);
      for (const bd of [...(d.bands ?? [])].reverse()) if (bd.label) {
        rows.push(`${swatch(bd.color)}${bd.label}: <b>${fy(interp(bd.x, bd.y1, xv) - interp(bd.x, bd.y0, xv))}</b>`);
      }
      html = `<div class="tt-head">${fx(xv)}</div>${rows.join('<br>')}`;
      this.onHover?.(xv);
    }
    this.draw();
    tooltip.show(html, ev.clientX, ev.clientY);
  }
}
