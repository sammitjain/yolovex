// yolovex — data model.
//
// The architecture (block list, dimensions, internal hierarchy) and the
// block-to-block wiring are read from window.YV_SPEC — the generated
// spec-viewer library (frontend/spec-data.js). Nothing about the model is
// hand-coded here.
//
// The ONLY hand-authored thing below is PRESENTATION layout: which column /
// vertical slot / role bucket each block is drawn in. That's a canvas
// decision, not architecture — ported from the original frontend/arch.jsx so
// this code reproduces the exact yolovex.html positions. The one deliberate change:
// blocks 9 (SPPF) and 10 (C2PSA) move into the "Neck" role bucket.

// idx -> { col, vpos?, role }. vpos is only used for column 1 (FPN-up),
// which is laid out in explicit reverse-flow order.
const PRESENTATION = {
  0:  { col: 0, role: 'Backbone' },
  1:  { col: 0, role: 'Backbone' },
  2:  { col: 0, role: 'Backbone' },
  3:  { col: 0, role: 'Backbone' },
  4:  { col: 0, role: 'Backbone' },
  5:  { col: 0, role: 'Backbone' },
  6:  { col: 0, role: 'Backbone' },
  7:  { col: 0, role: 'Backbone' },
  8:  { col: 0, role: 'Backbone' },
  9:  { col: 0, role: 'Neck' },        // SPPF — positioned in backbone column, but part of the neck
  10: { col: 0, role: 'Neck' },        // C2PSA — same
  11: { col: 1, vpos: 5, role: 'Neck' },
  12: { col: 1, vpos: 4, role: 'Neck' },
  13: { col: 1, vpos: 3, role: 'Neck' },
  14: { col: 1, vpos: 2, role: 'Neck' },
  15: { col: 1, vpos: 1, role: 'Neck' },
  16: { col: 1, vpos: 0, role: 'Neck' },
  17: { col: 2, role: 'Neck' },
  18: { col: 2, role: 'Neck' },
  19: { col: 2, role: 'Neck' },
  20: { col: 2, role: 'Neck' },
  21: { col: 2, role: 'Neck' },
  22: { col: 2, role: 'Neck' },
  23: { col: 3, role: 'Head' },
};

// Per-block learner-facing one-liners. Presentation copy, not architecture.
const BLOCK_DESC = {
  0:  'Stem 3×3 stride-2 conv. First downsample of the input image.',
  1:  'Second stride-2 conv. Halves spatial size, doubles channels.',
  2:  'Cross-stage partial bottleneck — feature mixing at this resolution.',
  3:  'Stride-2 conv to /8 resolution. The P3 scale begins here.',
  4:  'P3 feature stage. Skip-jumps forward to the FPN-up Concat at 15.',
  5:  'Stride-2 conv to /16 resolution. P4 scale begins.',
  6:  'P4 feature stage. Skip-connects forward to the FPN-up Concat at 12.',
  7:  'Stride-2 conv to /32 resolution. P5 scale begins.',
  8:  'Deepest backbone feature stage at /32.',
  9:  'Spatial Pyramid Pooling — Fast. Multi-scale receptive field.',
  10: 'Cross-stage Partial Self-Attention. Adds long-range context. Skips to Concat 21.',
  11: '2× nearest upsample of block 10. Brings /32 features up to /16.',
  12: 'Concatenates block 11 (upsampled deep) with block 6 (P4 skip).',
  13: 'Mixes merged P4-scale features. Skip-connects forward to Concat 18.',
  14: '2× nearest upsample. Brings features up to /8 (P3 scale).',
  15: 'Concatenates block 14 with block 4 (P3 backbone skip).',
  16: 'Final P3 (small-object) feature map. Feeds Detect head P3 branch.',
  17: 'Stride-2 downsample. Pulls P3 features back toward P4 resolution.',
  18: 'Concatenates block 17 with block 13 (FPN-up skip).',
  19: 'Final P4 (medium-object) feature map. Feeds Detect head P4 branch.',
  20: 'Stride-2 downsample. Pulls P4 features back toward P5 resolution.',
  21: 'Concatenates block 20 with block 10 (deep backbone skip).',
  22: 'Final P5 (large-object) feature map. Feeds Detect head P5 branch.',
  23: 'Multi-scale anchor-free detection head. Predicts boxes + classes at P3/P4/P5.',
};

// Palettes. Light keeps the calm-pastel feel but punchier; dark is neon-on-deep
// so the blocks pop against the dark canvas. Both palettes ship; the active
// one is mirrored into window.YV.LAYOUT_SETTINGS.{TYPE_PALETTE,ROLE_PALETTE}
// where the Settings drawer can override individual colors at runtime.
// Light: ~88% lightness pastels with stronger chroma — calm but with presence,
// not washed out. Borders go deeper (saturation up) and text reads at ~25%
// lightness so the block label survives against the fill.
// Dark: ~28–36% lightness "colored glass" fills against the #0c1118 canvas,
// with bright borders (~50–60% lightness) and near-white tinted text — reads
// as an actual block in low light, not a near-black void.
// ── Central style config (single source of truth) ──────────────────────────
// Keyed by node *type* (class name), spanning both L1 outer blocks AND the leaf
// modules revealed on expansion (Conv2d / BatchNorm2d / SiLU / …). Each entry:
//   { fill, border, text, textSize }
// Light values are seeded from the recovered L2 palette (git 29edfd0^). Dark
// values follow the existing recipe: deep "colored-glass" fills against the
// #0c1118 canvas, bright borders, near-white tinted text. textSize is the SVG
// label font-size (px) — outer blocks read a touch larger than inner leaves.
//
// Non-module nodes (pure fx ops: cat / split / arith / io / shape / attr) are
// NOT in here — they're coloured by *subkind* via the op fallback table
// (INNER_PALETTE, seeded in expand.jsx). getNodeStyle() resolves type first,
// then falls back to the op table. To recolour a block family, edit one row
// here; every renderer (collapsed node, expanded region, nesting container,
// leaf sub-node) pulls from it through getNodeStyle / TYPE_COLORS.
const TYPE_PALETTES = {
  light: {
    // Outer (L1) blocks
    Conv:        { fill: '#dcfce7', border: '#86efac', text: '#14532d', textSize: 13 },
    C3k2:        { fill: '#fce7f3', border: '#f9a8d4', text: '#831843', textSize: 13 },
    Upsample:    { fill: '#fae8ff', border: '#e9a8f5', text: '#701a75', textSize: 13 },
    Concat:      { fill: '#ede9fe', border: '#c4b5fd', text: '#4c1d95', textSize: 13 },
    SPPF:        { fill: '#fef9c3', border: '#fde047', text: '#713f12', textSize: 13 },
    C2PSA:       { fill: '#dbeafe', border: '#93c5fd', text: '#1e3a8a', textSize: 13 },
    Detect:      { fill: '#bbf7d0', border: '#22c55e', text: '#14532d', textSize: 13 },
    DetectHead:  { fill: '#bbf7d0', border: '#22c55e', text: '#14532d', textSize: 13 },
    // Leaf / sub-modules (revealed on expansion)
    Conv2d:      { fill: '#d1fae5', border: '#6ee7b7', text: '#064e3b', textSize: 12 },
    BatchNorm2d: { fill: '#fef3c7', border: '#fcd34d', text: '#78350f', textSize: 12 },
    SiLU:        { fill: '#ffe4e6', border: '#fda4af', text: '#881337', textSize: 12 },
    MaxPool2d:   { fill: '#e0f2fe', border: '#7dd3fc', text: '#0c4a6e', textSize: 12 },
    Bottleneck:  { fill: '#fce7f3', border: '#f9a8d4', text: '#831843', textSize: 12 },
    C3k:         { fill: '#fce7f3', border: '#f9a8d4', text: '#831843', textSize: 12 },
    Sequential:  { fill: '#fce7f3', border: '#f9a8d4', text: '#831843', textSize: 12 },
    PSABlock:    { fill: '#dbeafe', border: '#93c5fd', text: '#1e3a8a', textSize: 12 },
    // Types L2 didn't define — judgement / family reuse
    Attention:   { fill: '#e0e7ff', border: '#a5b4fc', text: '#312e81', textSize: 12 },
    PSA:         { fill: '#dbeafe', border: '#93c5fd', text: '#1e3a8a', textSize: 12 },
    DWConv:      { fill: '#d1fae5', border: '#6ee7b7', text: '#064e3b', textSize: 12 },
    Identity:    { fill: '#f1f5f9', border: '#cbd5e1', text: '#475569', textSize: 12 },
  },
  dark: {
    // Outer (L1) blocks
    Conv:        { fill: '#1d3e2c', border: '#48b870', text: '#c8f0d8', textSize: 13 },
    C3k2:        { fill: '#3e1e26', border: '#d86870', text: '#f5d4d8', textSize: 13 },
    Upsample:    { fill: '#2c1e42', border: '#a870d4', text: '#e8d4f8', textSize: 13 },
    Concat:      { fill: '#1e2244', border: '#6068c8', text: '#d4d8f8', textSize: 13 },
    SPPF:        { fill: '#3a2c10', border: '#c89428', text: '#f5e8c0', textSize: 13 },
    C2PSA:       { fill: '#102838', border: '#3a98d8', text: '#c4dff5', textSize: 13 },
    Detect:      { fill: '#102e28', border: '#38a888', text: '#c0f0e4', textSize: 13 },
    DetectHead:  { fill: '#102e28', border: '#38a888', text: '#c0f0e4', textSize: 13 },
    // Leaf / sub-modules
    Conv2d:      { fill: '#173a28', border: '#5ac888', text: '#c8f0d8', textSize: 12 },
    BatchNorm2d: { fill: '#332811', border: '#d4a83a', text: '#f5e4b8', textSize: 12 },
    SiLU:        { fill: '#38161c', border: '#e07880', text: '#f5d0d4', textSize: 12 },
    MaxPool2d:   { fill: '#0e2838', border: '#4aa8e0', text: '#bce0f5', textSize: 12 },
    Bottleneck:  { fill: '#3e1e26', border: '#d86870', text: '#f5d4d8', textSize: 12 },
    C3k:         { fill: '#3e1e26', border: '#d86870', text: '#f5d4d8', textSize: 12 },
    Sequential:  { fill: '#3e1e26', border: '#d86870', text: '#f5d4d8', textSize: 12 },
    PSABlock:    { fill: '#102838', border: '#3a98d8', text: '#c4dff5', textSize: 12 },
    Attention:   { fill: '#1a1e3e', border: '#7888d8', text: '#d4daf8', textSize: 12 },
    PSA:         { fill: '#102838', border: '#3a98d8', text: '#c4dff5', textSize: 12 },
    DWConv:      { fill: '#173a28', border: '#5ac888', text: '#c8f0d8', textSize: 12 },
    Identity:    { fill: '#252a34', border: '#485060', text: '#98a4b0', textSize: 12 },
  },
};

// Role containers (Backbone / Neck / Head) — recovered L2 hues. Dark variants
// lifted for contrast against the dark canvas.
const ROLE_PALETTES = {
  light: { Backbone: '#22c55e', Neck: '#3b82f6', Head: '#ef4444' },
  dark:  { Backbone: '#3aaa64', Neck: '#4878c8', Head: '#d04848' },
};

const ACCENTS = { light: '#c8682e', dark: '#e07840' };

// Canvas (graph) background — mirrors index.html's --graph-bg per theme.
// Seeded into LAYOUT_SETTINGS.GRAPH_BG so the Settings drawer can expose it.
const GRAPH_BGS = { light: '#fdfdfc', dark: '#0c1118' };

// Seed LAYOUT_SETTINGS with the light palette + getters so consumers (graph)
// can read TYPE_COLORS / ROLE_COLORS / ACCENT and pick up Settings overrides
// without any extra plumbing.
window.YV = window.YV || {};
window.YV.TYPE_PALETTES = TYPE_PALETTES;
window.YV.ROLE_PALETTES = ROLE_PALETTES;
window.YV.ACCENTS = ACCENTS;
window.YV.GRAPH_BGS = GRAPH_BGS;
// IMPORTANT: don't capture a const reference to window.YV.LAYOUT_SETTINGS
// here — layout.jsx loads AFTER us and reassigns LAYOUT_SETTINGS to a fresh
// object (so it can merge in its defaults). If we held a stale reference, the
// proxy below would read from the orphaned object and the Settings drawer's
// edits (which target the live LAYOUT_SETTINGS) would never be visible.
window.YV.LAYOUT_SETTINGS = window.YV.LAYOUT_SETTINGS || {};
const clonePalette = (p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, { ...v }]));
window.YV.LAYOUT_SETTINGS.TYPE_PALETTE = clonePalette(TYPE_PALETTES.light);
window.YV.LAYOUT_SETTINGS.ROLE_PALETTE = { ...ROLE_PALETTES.light };
window.YV.LAYOUT_SETTINGS.ACCENT_COLOR = ACCENTS.light;
window.YV.LAYOUT_SETTINGS.GRAPH_BG = GRAPH_BGS.light;

// Live getters that re-resolve LAYOUT_SETTINGS on every access, so:
//   1. arch's load-order ordering with layout.jsx doesn't matter
//   2. The Settings drawer's mutations of LAYOUT_SETTINGS.TYPE_PALETTE are
//      visible to graph on the very next render.
const liveLS = () => window.YV.LAYOUT_SETTINGS;
const TYPE_COLORS = new Proxy({}, {
  get: (_t, k) => liveLS().TYPE_PALETTE[k] || liveLS().TYPE_PALETTE.Conv,
  ownKeys: () => Object.keys(liveLS().TYPE_PALETTE),
  getOwnPropertyDescriptor: (_t, k) => ({ enumerable: true, configurable: true, value: liveLS().TYPE_PALETTE[k] }),
});
const ROLE_COLORS = new Proxy({}, {
  get: (_t, k) => liveLS().ROLE_PALETTE[k] || '#64748b',
  ownKeys: () => Object.keys(liveLS().ROLE_PALETTE),
  getOwnPropertyDescriptor: (_t, k) => ({ enumerable: true, configurable: true, value: liveLS().ROLE_PALETTE[k] }),
});
// Same pattern for ACCENT — exported via a getter so consumers always see fresh.
Object.defineProperty(window.YV, 'ACCENT', { get: () => liveLS().ACCENT_COLOR, configurable: true });
const ACCENT = liveLS().ACCENT_COLOR;  // kept for the module-bottom export

// Resolver — the one entry point every renderer should use to colour a node.
// Resolution order:
//   1. type-keyed BLOCK_STYLES (TYPE_PALETTE) — outer blocks + leaf modules
//   2. op fallback (INNER_PALETTE, seeded in expand.jsx) — pure fx ops keyed by
//      subkind (cat / split / arith / io / shape / attr / op / mod)
//   3. neutral default
// Reads LAYOUT_SETTINGS live, so Settings-drawer edits + theme swaps apply on
// the next render. `type` is the node class (e.g. 'Conv2d'); `subkind` is the
// expand.jsx bucket used only when no type entry exists.
function getNodeStyle(type, subkind) {
  const LS = liveLS();
  const tp = LS.TYPE_PALETTE || {};
  if (type && tp[type]) return tp[type];
  const ip = LS.INNER_PALETTE || {};
  if (subkind && ip[subkind]) return ip[subkind];
  return tp.Conv || { fill: '#f1f5f9', border: '#cbd5e1', text: '#475569', textSize: 12 };
}
window.YV.getNodeStyle = getNodeStyle;

// Build the L1 block list by joining spec-data instances with presentation.
function buildArch() {
  const spec = window.YV_SPEC;
  if (!spec) throw new Error('spec-data.js (window.YV_SPEC) not loaded');
  return spec.instances
    .slice()
    .sort((a, b) => a.idx - b.idx)
    .map(inst => {
      const pres = PRESENTATION[inst.idx] || { col: 0, role: 'Backbone' };
      return {
        idx: inst.idx,
        type: inst.class_name,
        col: pres.col,
        vpos: pres.vpos,
        role: pres.role,
        inputShape: inst.input_shape,
        outputShape: inst.output_shape,
        params: inst.params,
        specId: inst.spec_id,
        desc: BLOCK_DESC[inst.idx] || '',
      };
    });
}

// Block-to-block edges, straight from the pipeline ({src, dst, is_skip}).
function buildEdges() {
  return (window.YV_SPEC && window.YV_SPEC.edges) || [];
}

window.YV = window.YV || {};
window.YV.buildArch = buildArch;
window.YV.buildEdges = buildEdges;
window.YV.TYPE_COLORS = TYPE_COLORS;
window.YV.ROLE_COLORS = ROLE_COLORS;
window.YV.ACCENT = ACCENT;
