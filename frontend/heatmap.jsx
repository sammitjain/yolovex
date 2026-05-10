// Procedural viridis heatmaps — used to stub feature-map activations.
// Each block + channel has a deterministic seed so the textures are consistent across renders.

// Viridis colormap (5-stop approximation)
const VIRIDIS = [
  [68, 1, 84],     // 0.0
  [59, 82, 139],   // 0.25
  [33, 145, 140],  // 0.5
  [94, 201, 98],   // 0.75
  [253, 231, 37],  // 1.0
];

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const i = t * 4;
  const lo = Math.floor(i);
  const hi = Math.min(4, lo + 1);
  const f = i - lo;
  const a = VIRIDIS[lo], b = VIRIDIS[hi];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// Tiny seedable RNG (mulberry32)
function rng(seed) {
  let t = seed >>> 0;
  return function() {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Smooth value noise — combines a few low-freq octaves with a "feature" mask
function makeHeatmap(canvas, opts) {
  const { seed = 1, blockType = 'Conv', resolution = 1, edges = false, accent = 0.5 } = opts;
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);

  const r = rng(seed);

  // Generate a few hotspots at random positions
  const numSpots = 2 + Math.floor(r() * 4);
  const spots = [];
  for (let i = 0; i < numSpots; i++) {
    spots.push({
      cx: r() * w,
      cy: r() * h,
      sigma: w * (0.15 + r() * 0.35) / Math.max(1, resolution * 0.6),
      amp: 0.4 + r() * 0.6,
    });
  }

  // Add a "subject" hotspot biased toward the center-right (where the person is)
  if (accent > 0.4) {
    spots.push({
      cx: w * (0.55 + r() * 0.15),
      cy: h * (0.55 + r() * 0.15),
      sigma: w * 0.25,
      amp: 0.7 + r() * 0.3,
    });
  }

  // Edge-detector style — emphasize gradients around vertical bands
  const phase = r() * Math.PI * 2;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0.05 + r() * 0.05;

      // Spot field
      for (const s of spots) {
        const dx = x - s.cx, dy = y - s.cy;
        v += s.amp * Math.exp(-(dx * dx + dy * dy) / (2 * s.sigma * s.sigma));
      }

      // Edge-y blocks: add some vertical-band structure (mimics edge detectors)
      if (edges) {
        v += 0.15 * Math.abs(Math.sin((x / w) * 12 + phase)) * (1 - Math.abs(y - h/2) / h);
      }

      // Conv early layers: pixelate by resolution factor
      v = Math.max(0, Math.min(1, v));

      const [rC, gC, bC] = viridis(v);
      const i = (y * w + x) * 4;
      img.data[i] = rC;
      img.data[i + 1] = gC;
      img.data[i + 2] = bC;
      img.data[i + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
}

// Per-class detection score heatmap — coarse cells, percentile-clipped
function makeClassHeatmap(canvas, opts) {
  const { seed = 1, gridW = 16, gridH = 20, peakX = 0.5, peakY = 0.5, peakStrength = 0.9, noise = 0.15 } = opts;
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);

  const r = rng(seed);

  // Generate cell values
  const cells = new Array(gridW * gridH);
  for (let cy = 0; cy < gridH; cy++) {
    for (let cx = 0; cx < gridW; cx++) {
      const dx = (cx + 0.5) / gridW - peakX;
      const dy = (cy + 0.5) / gridH - peakY;
      const d = Math.sqrt(dx * dx + dy * dy);
      let v = peakStrength * Math.exp(-d * d / 0.04) + noise * (r() - 0.3);
      cells[cy * gridW + cx] = Math.max(0, Math.min(1, v));
    }
  }

  // Percentile clip — find 95th percentile and renormalize
  const sorted = [...cells].sort((a, b) => a - b);
  const p95 = sorted[Math.floor(sorted.length * 0.95)] || 1;
  const clip = Math.max(0.2, p95);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const cx = Math.floor((x / w) * gridW);
      const cy = Math.floor((y / h) * gridH);
      const v = Math.min(1, cells[cy * gridW + cx] / clip);
      const [rC, gC, bC] = viridis(v);
      const i = (y * w + x) * 4;
      img.data[i] = rC;
      img.data[i + 1] = gC;
      img.data[i + 2] = bC;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

window.YV = window.YV || {};
window.YV.viridis = viridis;
window.YV.makeHeatmap = makeHeatmap;
window.YV.makeClassHeatmap = makeClassHeatmap;
