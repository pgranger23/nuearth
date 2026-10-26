// Grid worker: computes a block of cos(zenith) rows of the cell-averaged bilinear
// products of S' (see mango-lite averagedGrid) off the main thread.
import { averagedGrid } from './mango-lite.js';

self.onmessage = (ev) => {
  const { id, osc, earth, E, CZ, anti, nSub, cell } = ev.data;
  try {
    const M = averagedGrid(osc, earth, E, CZ, anti, nSub, cell);
    self.postMessage({ id, M }, [M.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
