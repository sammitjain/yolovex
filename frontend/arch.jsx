// YOLO26 architecture topology — 24 blocks
// Each block: {idx, type, column, role, shape, params, desc}
// shape format: (1, C, H, W) where input image is 640x480

const ARCH = [
  // ========== BACKBONE (column 0, top-to-bottom) ==========
  { idx: 0,  type: 'Conv',     col: 0, role: 'Backbone', shape: [1, 16, 320, 240],  params: '448',     desc: 'Stem 3×3 stride-2 conv. First downsample of the input image.' },
  { idx: 1,  type: 'Conv',     col: 0, role: 'Backbone', shape: [1, 32, 160, 120],  params: '4,640',   desc: 'Second stride-2 conv. Reduces spatial size while doubling channels.' },
  { idx: 2,  type: 'C3k2',     col: 0, role: 'Backbone', shape: [1, 64, 160, 120],  params: '6,520',   desc: 'Cross-stage partial bottleneck — main feature-mixing unit at this resolution.' },
  { idx: 3,  type: 'Conv',     col: 0, role: 'Backbone', shape: [1, 64, 80, 60],    params: '36,928',  desc: 'Stride-2 conv. Drops to /8 resolution; the P3 scale begins here.' },
  { idx: 4,  type: 'C3k2',     col: 0, role: 'Backbone', shape: [1, 128, 80, 60],   params: '25,840',  desc: 'P3 feature stage. This output skip-jumps to the FPN-up Concat at 15.', skip: true },
  { idx: 5,  type: 'Conv',     col: 0, role: 'Backbone', shape: [1, 128, 40, 30],   params: '147,584', desc: 'Stride-2 conv to /16 resolution. P4 scale begins.' },
  { idx: 6,  type: 'C3k2',     col: 0, role: 'Backbone', shape: [1, 128, 40, 30],   params: '86,528',  desc: 'P4 feature stage. Skip-connects forward to FPN-up Concat at 12.', skip: true },
  { idx: 7,  type: 'Conv',     col: 0, role: 'Backbone', shape: [1, 256, 20, 15],   params: '295,168', desc: 'Stride-2 conv to /32 resolution. P5 scale begins.' },
  { idx: 8,  type: 'C3k2',     col: 0, role: 'Backbone', shape: [1, 256, 20, 15],   params: '345,088', desc: 'Deepest backbone feature stage at /32.' },
  { idx: 9,  type: 'SPPF',     col: 0, role: 'Backbone', shape: [1, 256, 20, 15],   params: '164,224', desc: 'Spatial Pyramid Pooling — Fast. Pools at multiple kernel sizes for multi-scale receptive field.' },
  { idx: 10, type: 'C2PSA',    col: 0, role: 'Backbone', shape: [1, 256, 20, 15],   params: '248,320', desc: 'Cross-stage Partial Self-Attention. Adds long-range context. Skip-connects to PAN-down Concat at 21.', skip: true },

  // ========== NECK FPN-up (column 1) ==========
  { idx: 11, type: 'Upsample', col: 1, role: 'Neck', shape: [1, 256, 40, 30],  params: '0',       desc: '2× nearest upsample of block 10. Brings /32 feature map up to /16.', vpos: 5 },
  { idx: 12, type: 'Concat',   col: 1, role: 'Neck', shape: [1, 384, 40, 30],  params: '0',       desc: 'Concatenates block 11 (upsampled deep) with block 6 (P4 backbone skip).', vpos: 4 },
  { idx: 13, type: 'C3k2',     col: 1, role: 'Neck', shape: [1, 128, 40, 30],  params: '119,296', desc: 'Mixes the merged P4-scale features. Skip-connects forward to PAN-down Concat at 18.', skip: true, vpos: 3 },
  { idx: 14, type: 'Upsample', col: 1, role: 'Neck', shape: [1, 128, 80, 60],  params: '0',       desc: '2× nearest upsample. Brings features up to /8 (P3 scale).', vpos: 2 },
  { idx: 15, type: 'Concat',   col: 1, role: 'Neck', shape: [1, 256, 80, 60],  params: '0',       desc: 'Concatenates block 14 with block 4 (P3 backbone skip).', vpos: 1 },
  { idx: 16, type: 'C3k2',     col: 1, role: 'Neck', shape: [1, 64, 80, 60],   params: '34,048',  desc: 'Final P3 (small-object) feature map. Feeds the Detect head P3 branch.', vpos: 0 },

  // ========== NECK PAN-down (column 2) ==========
  { idx: 17, type: 'Conv',     col: 2, role: 'Neck', shape: [1, 64, 40, 30],   params: '36,928',  desc: 'Stride-2 downsample. Pulls P3 features back down toward P4 resolution.' },
  { idx: 18, type: 'Concat',   col: 2, role: 'Neck', shape: [1, 192, 40, 30],  params: '0',       desc: 'Concatenates block 17 with block 13 (FPN-up skip).' },
  { idx: 19, type: 'C3k2',     col: 2, role: 'Neck', shape: [1, 128, 40, 30],  params: '94,720',  desc: 'Final P4 (medium-object) feature map. Feeds the Detect head P4 branch.' },
  { idx: 20, type: 'Conv',     col: 2, role: 'Neck', shape: [1, 128, 20, 15],  params: '147,584', desc: 'Stride-2 downsample. Pulls P4 features back down toward P5 resolution.' },
  { idx: 21, type: 'Concat',   col: 2, role: 'Neck', shape: [1, 384, 20, 15],  params: '0',       desc: 'Concatenates block 20 with block 10 (deep backbone skip).' },
  { idx: 22, type: 'C3k2',     col: 2, role: 'Neck', shape: [1, 256, 20, 15],  params: '461,504', desc: 'Final P5 (large-object) feature map. Feeds the Detect head P5 branch.' },

  // ========== HEAD (column 3) ==========
  { idx: 23, type: 'Detect',   col: 3, role: 'Head', shape: 'multi', params: '153,564', desc: 'Multi-scale anchor-free detection head. Three branches predict boxes + classes at P3 / P4 / P5 resolutions.' },
];

const EDGES = [
  [0,1,'forward'],[1,2,'forward'],[2,3,'forward'],[3,4,'forward'],[4,5,'forward'],
  [5,6,'forward'],[6,7,'forward'],[7,8,'forward'],[8,9,'forward'],[9,10,'forward'],
  [10,11,'cross'],
  [11,12,'forward'],[12,13,'forward'],[13,14,'forward'],[14,15,'forward'],[15,16,'forward'],
  [16,17,'cross'],
  [17,18,'forward'],[18,19,'forward'],[19,20,'forward'],[20,21,'forward'],[21,22,'forward'],
  [4,15,'skip'],
  [6,12,'skip'],
  [10,21,'skip'],
  [13,18,'skip'],
  [16,23,'detect-p3'],
  [19,23,'detect-p4'],
  [22,23,'detect-p5'],
];

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

window.YV = window.YV || {};
window.YV.ARCH = ARCH;
window.YV.EDGES = EDGES;
window.YV.TYPE_COLORS = TYPE_COLORS;
window.YV.ROLE_COLORS = ROLE_COLORS;
window.YV.ACCENT = ACCENT;
