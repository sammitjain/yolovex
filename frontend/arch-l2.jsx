// L2 architecture — parent containers with sub-nodes.
// Built programmatically from window.YV_DATA.blocks at module-init time,
// with a hand-coded ordered template that fixes ordering and provides metadata.

// ---- Sub-node templates per parent type ----
// Each entry: { path, label }
// These define display order; actual type is read from YV_DATA.blocks at render time.

// Sub-node display order per parent type. We keep paths here so we can resolve
// the sub-node id; the on-canvas label is derived from `blockData.type` (and
// shape) at render time, not from a hard-coded label.
const SUBNODE_TEMPLATES = {
  Conv:     ['conv', 'bn', 'act'],
  C3k2:     ['cv1', 'm.0', 'concat', 'cv2'],
  SPPF:     ['cv1', 'm.0', 'm.1', 'm.2', 'concat', 'cv2'],
  C2PSA:    ['cv1', 'm.0', 'cv2'],
  Detect:   ['head_p3', 'head_p4', 'head_p5'],
  Concat:   [],   // atomic
  Upsample: [],   // atomic
};

// Friendly per-path overrides for cases where the raw type string isn't
// the most useful label on its own (e.g. SPPF reuses one MaxPool2d 3 times —
// distinguishing pool×1/×2/×3 helps the user; the three Detect heads need
// scale labels because they share the type "DetectHead").
const PATH_LABEL_HINTS = {
  'm.0': null,    // resolved per parent below
  'm.1': null,
  'm.2': null,
};

function labelForSub(parentType, path, dataType) {
  // SPPF: differentiate the three pool calls
  if (parentType === 'SPPF' && path.startsWith('m.')) {
    const i = parseInt(path.split('.')[1], 10);
    return `MaxPool ×${i + 1}`;
  }
  // Detect heads: scale label
  if (parentType === 'Detect') {
    if (path === 'head_p3') return 'P3 head';
    if (path === 'head_p4') return 'P4 head';
    if (path === 'head_p5') return 'P5 head';
  }
  // Default: use the data type itself ("Conv2d", "BatchNorm2d", "SiLU",
  // "Bottleneck", "C3k", "PSABlock", "Concat", etc.)
  return dataType || path;
}

// Build ARCH_L2 from the L1 ARCH ordering, enriching with sub-nodes from data.
function buildArchL2() {
  const L1 = window.YV.ARCH;
  const blocks = window.YV_DATA?.blocks || {};

  return L1.map(b => {
    const tmpl = SUBNODE_TEMPLATES[b.type] || [];
    const sub = tmpl
      .map(path => {
        const id = `${b.idx}.${path}`;
        const blockData = blocks[id];
        if (!blockData) return null;
        return {
          id,
          path,
          type: blockData.type,
          label: labelForSub(b.type, path, blockData.type),
          shape: blockData.shape || null,
        };
      })
      .filter(Boolean);

    return {
      idx:  b.idx,
      type: b.type,
      role: b.role,
      col:  b.col,
      vpos: b.vpos,
      sub,
    };
  });
}

// Build EDGES_L2.
// Inter-block edges: use sub-node endpoints (last sub of source → first sub of target).
// Skip-connection edges land on the target container's top edge (handled in layout).
// Intra-container sequential edges are computed in the layout/graph step, not here.
function buildEdgesL2(arch_l2) {
  const L1_EDGES = window.YV.EDGES;

  // Helper: last sub-node id for a container (or null if atomic)
  const lastSub = (idx) => {
    const c = arch_l2.find(a => a.idx === idx);
    if (!c || c.sub.length === 0) return null;
    return c.sub[c.sub.length - 1].id;
  };

  // Helper: first sub-node id for a container (or null if atomic)
  const firstSub = (idx) => {
    const c = arch_l2.find(a => a.idx === idx);
    if (!c || c.sub.length === 0) return null;
    return c.sub[0].id;
  };

  return L1_EDGES.map(([from, to, kind]) => {
    // Skip-connections land on the target container itself (topPort), not a sub-node.
    // detect-pX edges also land on the specific detect sub-node.
    if (kind === 'skip') {
      // source: last sub of 'from'; target: container idx (topPort)
      return { from: lastSub(from) || String(from), to: to, kind: 'skip', interContainer: true };
    }
    if (kind.startsWith('detect-')) {
      const scale = kind.split('-')[1]; // 'p3', 'p4', 'p5'
      const headPath = `head_p${scale.slice(1)}`; // 'head_p3' etc
      const c = arch_l2.find(a => a.idx === to);
      const headNode = c?.sub?.find(s => s.path === headPath);
      return {
        from: lastSub(from) || String(from),
        to: headNode ? headNode.id : to,
        kind,
        interContainer: true,
      };
    }
    // forward / cross: last sub → first sub (or container if atomic)
    const f = lastSub(from) || String(from);
    const t = firstSub(to) || String(to);
    return { from: f, to: t, kind, interContainer: true };
  });
}

// Intra-container edges. Most containers are simple sequential chains, but
// C3k2 and SPPF have a true fan-IN at their internal Concat node — we want
// to draw arrows from EVERY tap source into that Concat, not just from the
// previous sub-node. Topology references (Ultralytics source):
//
//   C3k2.forward:  y = list(cv1(x).chunk(2,1)); y.extend(m.0(y[-1])); cv2(cat(y))
//                  → concat takes BOTH halves of cv1 + m.0 output
//
//   SPPF.forward:  y = [cv1(x)]; y += [m(y[-1]) for _ in range(3)]; cv2(cat(y))
//                  → concat takes cv1 + each of m.0/m.1/m.2 outputs
//
// We tag the fan-in edges with kind='fanin' so the layout can route them
// distinctly (curving outward to land on different points along the Concat
// node's top edge), instead of overdrawing on top of the sequential chain.
function buildIntraEdges(arch_l2) {
  const edges = [];
  const push = (from, to, kind = 'intra') => {
    if (!from || !to) return;
    edges.push({ from, to, kind, interContainer: false });
  };

  for (const container of arch_l2) {
    const subs = container.sub;
    if (!subs.length) continue;
    const idOf = (path) => subs.find(s => s.path === path)?.id;

    if (container.type === 'C3k2') {
      // sequential: cv1 → m.0  (m.0 receives the second half of cv1's output)
      push(idOf('cv1'), idOf('m.0'));
      // fan-in: cv1 → concat (split halves both go in), m.0 → concat
      push(idOf('cv1'), idOf('concat'), 'fanin');
      push(idOf('m.0'), idOf('concat'), 'fanin');
      // sequential: concat → cv2
      push(idOf('concat'), idOf('cv2'));
    } else if (container.type === 'SPPF') {
      // sequential maxpool chain: cv1 → m.0 → m.1 → m.2
      push(idOf('cv1'), idOf('m.0'));
      push(idOf('m.0'), idOf('m.1'));
      push(idOf('m.1'), idOf('m.2'));
      // fan-in: every tap → concat
      push(idOf('cv1'), idOf('concat'), 'fanin');
      push(idOf('m.0'), idOf('concat'), 'fanin');
      push(idOf('m.1'), idOf('concat'), 'fanin');
      push(idOf('m.2'), idOf('concat'), 'fanin');
      // sequential: concat → cv2
      push(idOf('concat'), idOf('cv2'));
    } else {
      // Default: simple top-to-bottom (or bottom-to-top, layout decides)
      for (let i = 0; i < subs.length - 1; i++) {
        push(subs[i].id, subs[i + 1].id);
      }
    }
  }
  return edges;
}

// TYPE_COLORS and ROLE_COLORS — extend L1's with sub-node types
const TYPE_COLORS_L2 = {
  // Inherit L1 colors for top-level types
  Conv:        { fill: '#dcfce7', border: '#86efac', text: '#14532d' },
  C3k2:        { fill: '#fce7f3', border: '#f9a8d4', text: '#831843' },
  Upsample:    { fill: '#fae8ff', border: '#e9a8f5', text: '#701a75' },
  Concat:      { fill: '#ede9fe', border: '#c4b5fd', text: '#4c1d95' },
  SPPF:        { fill: '#fef9c3', border: '#fde047', text: '#713f12' },
  C2PSA:       { fill: '#dbeafe', border: '#93c5fd', text: '#1e3a8a' },
  Detect:      { fill: '#bbf7d0', border: '#22c55e', text: '#14532d' },
  DetectHead:  { fill: '#bbf7d0', border: '#22c55e', text: '#14532d' },
  // Sub-node leaf types
  Conv2d:      { fill: '#d1fae5', border: '#6ee7b7', text: '#064e3b' },
  BatchNorm2d: { fill: '#fef3c7', border: '#fcd34d', text: '#78350f' },
  SiLU:        { fill: '#ffe4e6', border: '#fda4af', text: '#881337' },
  MaxPool2d:   { fill: '#e0f2fe', border: '#7dd3fc', text: '#0c4a6e' },
  Bottleneck:  { fill: '#fce7f3', border: '#f9a8d4', text: '#831843' },
  C3k:         { fill: '#fce7f3', border: '#f9a8d4', text: '#831843' },
  Sequential:  { fill: '#fce7f3', border: '#f9a8d4', text: '#831843' },
  PSABlock:    { fill: '#dbeafe', border: '#93c5fd', text: '#1e3a8a' },
};

function getTypeColor(type) {
  return TYPE_COLORS_L2[type] || { fill: '#f1f5f9', border: '#cbd5e1', text: '#475569' };
}

// Lazily initialize ARCH_L2 / EDGES_L2 on first access so data-l2.js is loaded first.
let _archL2 = null;
let _edgesInterL2 = null;
let _edgesIntraL2 = null;

function getArchL2() {
  if (!_archL2) _archL2 = buildArchL2();
  return _archL2;
}
function getEdgesInterL2() {
  if (!_edgesInterL2) _edgesInterL2 = buildEdgesL2(getArchL2());
  return _edgesInterL2;
}
function getEdgesIntraL2() {
  if (!_edgesIntraL2) _edgesIntraL2 = buildIntraEdges(getArchL2());
  return _edgesIntraL2;
}

window.YV = window.YV || {};
window.YV.getArchL2 = getArchL2;
window.YV.getEdgesInterL2 = getEdgesInterL2;
window.YV.getEdgesIntraL2 = getEdgesIntraL2;
window.YV.getTypeColor = getTypeColor;
window.YV.TYPE_COLORS_L2 = TYPE_COLORS_L2;
