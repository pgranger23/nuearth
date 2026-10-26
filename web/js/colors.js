// Theme-aware colors.  Hexes come from the dataviz reference palette; maps are
// built per theme (dark mode re-anchors, it is not an automatic flip).

const SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
  '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];

const DIV = {
  light: ['#104281', '#2a78d6', '#86b6ef', '#f0efec', '#f1a3a2', '#e34948', '#a3302f'],
  dark: ['#9ec5f4', '#3987e5', '#1c5cab', '#383835', '#8f3434', '#e66767', '#f4b3b3'],
};

export const FLAVOR_COLORS = {
  light: ['#eb6834', '#2a78d6', '#1baf7a'], // nu_e orange, nu_mu blue, nu_tau aqua
  dark: ['#d95926', '#3987e5', '#199e70'],
};

const EARTH_GRAY = {
  light: ['#f4f3f0', '#d6d4cc', '#a9a79f'],
  dark: ['#26262a', '#3a3a3a', '#5a5955'],
};

function hexToRgb(h) {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function lut(stops, n = 256) {
  const rgb = stops.map(hexToRgb);
  const out = new Uint8ClampedArray(n * 3);
  for (let k = 0; k < n; k++) {
    const t = (k / (n - 1)) * (rgb.length - 1);
    const i = Math.min(Math.floor(t), rgb.length - 2), f = t - i;
    for (let c = 0; c < 3; c++) out[k * 3 + c] = rgb[i][c] + f * (rgb[i + 1][c] - rgb[i][c]);
  }
  return out;
}

const cache = {};
export function theme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

export function maps() {
  const t = theme();
  if (!cache[t]) {
    cache[t] = {
      seq: lut(t === 'dark' ? ['#1a1a19', ...[...SEQ_BLUE].reverse()] : ['#fcfcfb', ...SEQ_BLUE]),
      div: lut(DIV[t]),
      earth: lut(EARTH_GRAY[t]),
      flavor: FLAVOR_COLORS[t],
    };
  }
  return cache[t];
}

export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function lutColor(table, t) {
  const k = Math.max(0, Math.min(255, Math.round(t * 255)));
  return `rgb(${table[k * 3]},${table[k * 3 + 1]},${table[k * 3 + 2]})`;
}
