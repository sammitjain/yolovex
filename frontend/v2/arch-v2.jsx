// yolovex v2 — data model.
//
// The architecture (block list, dimensions, internal hierarchy) and the
// block-to-block wiring are read from window.YV_SPEC — the generated
// spec-viewer library (frontend/spec-data.js). Nothing about the model is
// hand-coded here.
//
// The ONLY hand-authored thing below is PRESENTATION layout: which column /
// vertical slot / role bucket each block is drawn in. That's a canvas
// decision, not architecture — ported from the original frontend/arch.jsx so
// v2 reproduces the exact yolovex.html positions. The one deliberate change:
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

const TYPE_COLORS = {
  Conv:     { fill: '#dcfce7', border: '#86efac', text: '#14532d' },
  C3k2:     { fill: '#fce7f3', border: '#f9a8d4', text: '#831843' },
  Upsample: { fill: '#fae8ff', border: '#e9a8f5', text: '#701a75' },
  Concat:   { fill: '#ede9fe', border: '#c4b5fd', text: '#4c1d95' },
  SPPF:     { fill: '#fef9c3', border: '#fde047', text: '#713f12' },
  C2PSA:    { fill: '#dbeafe', border: '#93c5fd', text: '#1e3a8a' },
  Detect:   { fill: '#bbf7d0', border: '#22c55e', text: '#14532d' },
};

const ROLE_COLORS = {
  Backbone: '#22c55e',
  Neck:     '#3b82f6',
  Head:     '#ef4444',
};

const ACCENT = '#fb923c';

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

window.YVV2 = window.YVV2 || {};
window.YVV2.buildArch = buildArch;
window.YVV2.buildEdges = buildEdges;
window.YVV2.TYPE_COLORS = TYPE_COLORS;
window.YVV2.ROLE_COLORS = ROLE_COLORS;
window.YVV2.ACCENT = ACCENT;
