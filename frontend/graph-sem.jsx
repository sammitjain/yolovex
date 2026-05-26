// yolovex — shared graph SEMANTICS.
//
// The pure graph-semantic transforms (preprocess / aggregate / subkind
// classification), the node-sizing rules + region padding, and the sub-kind
// colour palettes. These are layout-engine agnostic: ELK (expand-elk.jsx)
// consumes them via window.YV._graphSem, and the renderer / side panel read the
// palettes + shape formatter. The geometry/edge-routing back-half that used to
// live alongside this (legacy buildExpansion) was retired with the old renderer.
//
// Scope: depth-1 aggregation and deeper (recursive) via the `expansions` set.

// ============================ ported constants ============================
const ARITH_OP_NAMES  = new Set(['add', 'mul', 'sub', 'truediv']);
const STRUCT_OP_NAMES = new Set(['getitem']);
const SHAPE_OP_NAMES  = new Set(['view', 'permute', 'transpose', 'reshape', 'flatten', 'squeeze', 'unsqueeze', 'contiguous']);

function opVisibilityForLevel(level) {
  return { hideShapeOps: false, hideGetAttr: true };
}

function opShortName(target) {
  if (!target) return '';
  const last = String(target).split('.').pop();
  return last.replace(/^_+|_+$/g, '');
}

// ===================== ported: preprocessGraph =============================
function preprocessGraph(graph, visibility, shapes) {
  const dropped = new Set();
  const hidden = new Set();
  const hiddenLabelPart = new Map();

  for (const n of graph.nodes) {
    if (n.op === 'placeholder' || n.op === 'output') continue;
    if (shapes && shapes[n.name] === null) {
      dropped.add(n.name);
      continue;
    }
    const opShort = opShortName(n.target);
    if (n.op === 'call_function' && opShort === 'getitem') {
      hidden.add(n.name);
      if (Array.isArray(n.args) && n.args.length >= 2) {
        hiddenLabelPart.set(n.name, String(n.args[1]));
      }
      continue;
    }
    if (visibility.hideGetAttr && n.op === 'get_attr') {
      hidden.add(n.name);
      continue;
    }
    if (visibility.hideShapeOps && (n.op === 'call_function' || n.op === 'call_method')) {
      if (SHAPE_OP_NAMES.has(opShort)) {
        hidden.add(n.name);
        hiddenLabelPart.set(n.name, opShort);
      }
    }
  }

  const succs = new Map();
  for (const n of graph.nodes) succs.set(n.name, []);
  for (const [s, t] of graph.edges) {
    succs.get(s)?.push(t);
  }

  function liveSuccs(name, label, visited) {
    const out = [];
    for (const dst of succs.get(name) || []) {
      if (visited.has(dst)) continue;
      if (dropped.has(dst)) continue;
      if (hidden.has(dst)) {
        const part = hiddenLabelPart.get(dst);
        const newLabel = part ? (label ? `${label}→${part}` : part) : label;
        const inner = new Set(visited);
        inner.add(dst);
        out.push(...liveSuccs(dst, newLabel, inner));
      } else {
        out.push({ dst, label });
      }
    }
    return out;
  }

  const newEdges = [];
  for (const n of graph.nodes) {
    if (hidden.has(n.name) || dropped.has(n.name)) continue;
    const visited = new Set([n.name]);
    for (const { dst, label } of liveSuccs(n.name, null, visited)) {
      if (label != null) newEdges.push([n.name, dst, label]);
      else               newEdges.push([n.name, dst]);
    }
  }

  const newNodes = graph.nodes.filter(n => !hidden.has(n.name) && !dropped.has(n.name));
  return { nodes: newNodes, edges: newEdges };
}

// ===================== aggregateWithExpansions =============================
//
// Generalises spec-viewer's aggregateAtDepth: instead of a single uniform
// depth, you pass `expansions: Set<string>` — the set of path keys (joined
// with '/') that should be peeled open. '' always represents the block root,
// so `expansions = new Set([''])` reproduces the old depth-1 behaviour. Adding
// 'm' to the set reveals m's depth-2 children individually; adding
// 'm/0_PSABlock' reveals its depth-3 children; and so on.
//
// For a node with visible_path V we compute k = largest such that V[:k]
// joined is in `expansions`. The group is V[:k] (aggregated) if k < V.length,
// else V (individual, fully revealed). Functions / methods absorb into their
// enclosing module unless the module has been expanded past _op's index.
function aggregateWithExpansions(graph, shapes, expansions, pathClasses) {
  const nameToGroup = {};
  const groups = new Map();
  const groupKind = new Map();
  const groupOrder = [];
  const groupK = new Map();     // group key -> the k value (group depth) — used by containerPath

  for (const n of graph.nodes) {
    const vpath = n.visible_path || n.path || [];
    let key, kind, k = 0;

    if (n.op === 'placeholder' || n.op === 'output') {
      key = n.name; kind = 'io';
    } else if (n.op === 'get_attr') {
      key = n.name; kind = 'attr';
    } else {
      // Advance k while every prefix is expanded.
      while (k < vpath.length) {
        const probe = vpath.slice(0, k).join('/');
        if (!expansions.has(probe)) break;
        k++;
      }
      if (n.op === 'call_module') {
        if (k >= vpath.length) { key = n.name; kind = 'mod'; }
        else                   { key = vpath.slice(0, k).join('/'); kind = 'mod'; }
      } else if (n.op === 'call_function' || n.op === 'call_method') {
        const opIdx = vpath.indexOf('_op');
        if (opIdx >= 0 && opIdx >= k) {
          // Absorbed into enclosing module's group at depth k.
          if (k >= vpath.length) { key = n.name; kind = 'mod'; }
          else                   { key = vpath.slice(0, k).join('/'); kind = 'mod'; }
        } else {
          key = n.name; kind = 'op';
        }
      } else {
        key = n.name; kind = 'mod';
      }
    }

    nameToGroup[n.name] = key;
    if (!groups.has(key)) {
      groups.set(key, []);
      groupKind.set(key, kind);
      groupK.set(key, k);
      groupOrder.push(key);
    }
    groups.get(key).push(n);
  }

  const aggNodes = [];
  for (const key of groupOrder) {
    const members = groups.get(key);
    const last = members[members.length - 1];
    const shape = shapes ? shapes[last.name] : null;
    const first = members[0];
    const fvpath = first.visible_path || first.path || [];
    const fkind = groupKind.get(key);
    const k = groupK.get(key);
    let label;

    if (members.length === 1 && members[0].name === key) {
      const n = members[0];
      if (n.op === 'call_module') {
        label = `${n.target_class || n.target}`;
      } else if (n.op === 'call_function') {
        label = `fn:${opShortName(n.target)}`;
      } else if (n.op === 'call_method') {
        label = `.${n.target}()`;
      } else {
        label = n.name;
      }
    } else {
      const cls = pathClasses ? pathClasses[key] : null;
      const lastSeg = key.split('/').pop() || '';
      const containerMatch = lastSeg.match(/^(\d+)_(.+)$/);
      if (containerMatch) {
        label = containerMatch[2];
      } else if (cls) {
        label = `${cls}`;
      } else {
        label = lastSeg || key;
      }
    }

    let containerPath = null;
    if (first.op !== 'placeholder' && first.op !== 'output' && first.op !== 'get_attr') {
      if (fkind === 'mod') {
        containerPath = fvpath.slice(0, Math.max(0, k - 1));
      } else if (fkind === 'op') {
        const opIdx = fvpath.indexOf('_op');
        containerPath = opIdx >= 0 ? fvpath.slice(0, opIdx) : [];
      }
    }

    // Expandable: an aggregated module group with internals to reveal.
    const expandable = fkind === 'mod' && members.length > 1;

    aggNodes.push({
      id: key,
      label,
      kind: fkind,
      shape,
      members: members.map(m => m.name),
      containerPath,
      expandable,
      pathKey: key,
    });
  }

  const edgesMerged = new Map();
  for (const edge of graph.edges) {
    const [s, t, lbl] = edge;
    const sg = nameToGroup[s], tg = nameToGroup[t];
    if (sg == null || tg == null || sg === tg) continue;
    const k = `${sg}\x00${tg}`;
    if (!edgesMerged.has(k)) edgesMerged.set(k, { s: sg, t: tg, labels: [] });
    if (lbl != null && lbl !== '') {
      const arr = edgesMerged.get(k).labels;
      if (!arr.includes(lbl)) arr.push(lbl);
    }
  }
  const aggEdges = [];
  for (const { s, t, labels } of edgesMerged.values()) {
    const merged = labels.length > 0 ? labels.join(', ') : null;
    if (merged) aggEdges.push([s, t, merged]); else aggEdges.push([s, t]);
  }

  for (const n of aggNodes) {
    const orig = graph.nodes.find(g => g.name === n.id);
    n.subkind = classifySubkind(orig, n);
    // Carry the nn.Module class through so the renderer can colour every module
    // node by type (Conv2d / BatchNorm2d / SiLU / …) via getNodeStyle. Three
    // sources, in order:
    //   1. leaf fx node          → orig.target_class
    //   2. aggregated mod group  → pathClasses[key]  (e.g. cv1 → 'Conv')
    //   3. last-resort           → the group label IF it's a module group
    //      (label is already the class name for mod groups: 'Conv',
    //      'Bottleneck', 'C3k', …). Pure ops / io / attr stay null and fall
    //      back to subkind colouring.
    n.targetClass = (orig && orig.target_class)
      || (pathClasses && pathClasses[n.id])
      || (n.kind === 'mod' ? n.label : null);
    // A binary arith op against a constant (one tensor arg + one numeric
    // literal) draws only one edge — the scalar isn't a node. Surface the
    // literal so the canvas can show it (e.g. attention's score scaling
    // mul × 0.1768 = 1/√d_k), instead of an unexplained one-input × circle.
    if (n.subkind === 'arith' && orig && Array.isArray(orig.args)) {
      const nums = orig.args.filter(a => typeof a === 'number');
      if (nums.length === 1) n.scalarOperand = nums[0];
    }
  }

  return { nodes: aggNodes, edges: aggEdges };
}

// ===================== ported: classifySubkind =============================
function classifySubkind(orig, agg) {
  if (!orig) return agg.kind === 'op' ? 'op' : agg.kind;
  if (orig.op === 'placeholder' || orig.op === 'output') return 'io';
  if (orig.op === 'get_attr') return 'attr';
  if (orig.op === 'call_module') return 'module';
  if (orig.op === 'call_function') {
    const name = opShortName(orig.target);
    if (name === 'cat') return 'cat';
    if (ARITH_OP_NAMES.has(name)) return 'arith';
    if (STRUCT_OP_NAMES.has(name)) return 'struct';
    if (SHAPE_OP_NAMES.has(name)) return 'shape';
    return 'op';
  }
  if (orig.op === 'call_method') {
    const name = String(orig.target);
    if (name === 'chunk' || name === 'split') return 'split';
    if (SHAPE_OP_NAMES.has(name)) return 'shape';
    return 'op';
  }
  return agg.kind;
}

// ===================== sizing rules =========================
const SUB_NODE_W = 150;
const SUB_NODE_H = 60;
const ARITH_R    = 18;
const SMALL_W    = 120;
const SMALL_H    = 32;

function nodeSize(node, staircaseSpan) {
  if (node.subkind === 'arith') return { w: 2 * ARITH_R, h: 2 * ARITH_R };
  if (node.subkind === 'shape' || node.subkind === 'attr' || node.subkind === 'struct') {
    return { w: SMALL_W, h: SMALL_H };
  }
  if (node.subkind === 'cat' && staircaseSpan != null) {
    return { w: staircaseSpan, h: SUB_NODE_H };
  }
  return { w: SUB_NODE_W, h: SUB_NODE_H };
}

// ===================== region padding =========================
const REGION_PAD_X      = 22;
// Title (block label) sits near y≈19; give the first sub-node extra clearance
// below it so the expansion doesn't look cramped under the title.
const REGION_PAD_TOP    = 44;
const REGION_PAD_BOTTOM = 22;

// ===================== render styling (shared with the renderer) ===========
// Sub-kind palettes — one per theme. mod / module coordinate with the Conv
// family (green); op is warm orange (matches accent hue); cat is amber; split
// is blue. Dark variants mute the fills and keep borders bright enough to read.
const SUB_KIND_COLORS_LIGHT = {
  io:     { fill: '#eeede8', border: '#8a8680', text: '#3a3630' },
  mod:    { fill: '#d8f0e4', border: '#2a9450', text: '#123c22' },
  module: { fill: '#d8f0e4', border: '#2a9450', text: '#123c22' },
  op:     { fill: '#feeee0', border: '#c86838', text: '#5a2008' },
  cat:    { fill: '#fef6c8', border: '#b08814', text: '#4c3800' },
  split:  { fill: '#d8e8fc', border: '#4070c8', text: '#0c2658' },
};

const SUB_KIND_COLORS_DARK = {
  io:     { fill: '#252a34', border: '#485060', text: '#98a4b0' },
  mod:    { fill: '#1a2e24', border: '#34784a', text: '#90d4a8' },
  module: { fill: '#1a2e24', border: '#34784a', text: '#90d4a8' },
  op:     { fill: '#2c1e14', border: '#b86030', text: '#f0b888' },
  cat:    { fill: '#2a2410', border: '#a07c1c', text: '#f0d890' },
  split:  { fill: '#16203e', border: '#3860a8', text: '#88b0e0' },
};

// Back-compat alias — any direct reference (or destructure from window.YV)
// gets the light set. The renderer's ExpandedNode switches to _DARK by theme.
const SUB_KIND_COLORS = SUB_KIND_COLORS_LIGHT;

function subFormatShape(sh) {
  if (!sh) return '';
  if (Array.isArray(sh) && sh.length && typeof sh[0] === 'number') return sh.join('×');
  return '';
}

window.YV = window.YV || {};
// Pure graph-SEMANTIC transforms reused by the ELK layout path (expand-elk.jsx)
// without duplicating logic or touching anything geometric.
window.YV._graphSem = {
  preprocessGraph,
  aggregateWithExpansions,
  classifySubkind,
  opShortName,
  opVisibilityForLevel,
  // sizing rules + region padding, reused so ELK node sizes / region framing
  // match the intended look.
  nodeSize,
  SUB_NODE_W, SUB_NODE_H, ARITH_R, SMALL_W, SMALL_H,
  REGION_PAD_X, REGION_PAD_TOP, REGION_PAD_BOTTOM,
};
window.YV.SUB_KIND_COLORS      = SUB_KIND_COLORS_LIGHT;  // alias for back-compat
window.YV.SUB_KIND_COLORS_LIGHT = SUB_KIND_COLORS_LIGHT;
window.YV.SUB_KIND_COLORS_DARK  = SUB_KIND_COLORS_DARK;
// Bundle both theme palettes for the Settings drawer + theme switcher to
// mirror into LAYOUT_SETTINGS.INNER_PALETTE (same pattern as TYPE_PALETTES).
window.YV.INNER_PALETTES = { light: SUB_KIND_COLORS_LIGHT, dark: SUB_KIND_COLORS_DARK };
window.YV.subFormatShape       = subFormatShape;
window.YV.opShortName          = opShortName;

// Seed LAYOUT_SETTINGS.INNER_PALETTE with a deep-cloned light preset so the
// Settings drawer can edit individual sub-kind colors without mutating the
// shared default. The renderer's ExpandedNode reads from LS at render time.
const _cloneInner = (p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, { ...v }]));
window.YV.LAYOUT_SETTINGS = window.YV.LAYOUT_SETTINGS || {};
if (!('INNER_PALETTE' in window.YV.LAYOUT_SETTINGS)) {
  window.YV.LAYOUT_SETTINGS.INNER_PALETTE = _cloneInner(SUB_KIND_COLORS_LIGHT);
}
