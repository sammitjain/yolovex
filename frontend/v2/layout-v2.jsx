// yolovex v2 — pixel positions, role containers, and edge paths.
//
// Positions are ported verbatim from the original frontend/layout.jsx so v2
// reproduces the exact yolovex.html canvas. New here: role-container shapes —
// Backbone/Head are plain rects, the Neck is a mirrored-L polygon because its
// members (SPPF 9, C2PSA 10) sit in the backbone column while the rest of the
// neck sits in columns 1-2.

// ===================== TUNABLE SPACING — tweak these ======================
// Everything that controls how the canvas is spaced out lives here.
const ROW_GAP        = 110;  // vertical gap between consecutive blocks in a column
const COL_GAP        = 360;  // horizontal gap between the two neck columns (FPN-up / PAN-down)
const CONTAINER_GAP  = 100;  // uniform horizontal gap between the Backbone | Neck | Head containers
const NECK_Y_OFFSET  = 120;  // how far down the whole neck region sits — breathing room below the backbone
const DETECT_GAP     = 100;   // vertical gap between the P3 / P4 / P5 detect boxes (taller Head container)

// Edge bezier flat-tail leads — each edge is a flat tail out of the source,
// a short bend, then a flat tail into the target. Bigger lead = flatter edge
// with a sharper corner. Horizontal applies to cross / skip / detect edges;
// vertical applies to forward (same-column) edges.
const H_ENTRY = 30;  // horizontal: flat tail leaving the source
const H_EXIT  = 36;  // horizontal: flat tail entering the target
const V_ENTRY = 8;   // vertical: flat tail leaving the source
const V_EXIT  = 14;  // vertical: flat tail entering the target
// ===========================================================================

const NODE_W = 158;
const NODE_H = 56;
const COL_TOP = 80;

const CONTAINER_PAD   = 18;
const CONTAINER_PAD_T = 28;

// Column x-positions. COL_X[0] is fixed; the rest are derived so the gaps
// between the three role containers come out uniform (= CONTAINER_GAP), while
// COL_GAP independently controls the spacing between the two neck columns.
// With NODE_W=158, CONTAINER_PAD=18: backbone right edge = 296.
//   neck-body-left = COL_X[1] - CONTAINER_PAD ; want = 296 + CONTAINER_GAP
//      -> COL_X[1] = 314 + CONTAINER_GAP
//   COL_X[2]   = COL_X[1] + COL_GAP
//   head-left  = neck-right + CONTAINER_GAP   -> COL_X[3] = COL_X[2] + 194 + CONTAINER_GAP
const COL_X = (() => {
  const c1 = 314 + CONTAINER_GAP;
  const c2 = c1 + COL_GAP;
  const c3 = c2 + 194 + CONTAINER_GAP;
  return [120, c1, c2, c3];
})();

// --- Node positions --------------------------------------------------------
// Same column scheme as the original layout.jsx, plus NECK_Y_OFFSET which
// pushes the whole neck region down (its col-0 leg blocks 9 & 10, and the
// col-1 / col-2 bodies) for breathing room below the backbone.
//
// Each column is placed as a flow-order WALK that accumulates position — with
// no expansions this reproduces the exact original positions (ROW_GAP / NODE_H
// are uniform), but when a block is expanded into a region it consumes
// `regionH` instead of NODE_H and every later block in the column slides along
// the flow direction. `expansionMap` is `{ idx: <buildExpansion result> }`.
//
// Width: a region also grows rightward by `regionW - NODE_W`; every column to
// the right of an expanded column is shifted right by the widest such delta.
const GAP = ROW_GAP - NODE_H;   // actual visual gap between stacked boxes

function layoutGraph(arch, expansionMap) {
  expansionMap = expansionMap || {};
  const nodes = {};
  const exOf = idx => expansionMap[idx] || null;

  // --- Per-column rightward shift from expanded-region widths.
  const maxDW = { 0: 0, 1: 0, 2: 0, 3: 0 };
  for (const b of arch) {
    const ex = exOf(b.idx);
    if (ex) maxDW[b.col] = Math.max(maxDW[b.col], ex.regionW - NODE_W);
  }
  const xShift = {};
  let accW = 0;
  for (const c of [0, 1, 2, 3]) { xShift[c] = accW; accW += maxDW[c]; }

  const mk = (b, x, y) => {
    const ex = exOf(b.idx);
    if (ex) {
      return {
        x, y, w: ex.regionW, h: ex.regionH,
        expanded: true, region: ex,
      };
    }
    return { x, y, w: NODE_W, h: NODE_H };
  };

  // Column 0 (backbone column): blocks 0..10, walked top-to-bottom. The first
  // Neck-role block (SPPF 9) gets pushed down by NECK_Y_OFFSET.
  {
    let y = COL_TOP;
    let addedNeckOffset = false;
    arch.filter(b => b.col === 0).forEach(b => {
      if (b.role === 'Neck' && !addedNeckOffset) { y += NECK_Y_OFFSET; addedNeckOffset = true; }
      nodes[b.idx] = mk(b, COL_X[0] + xShift[0], y);
      y += nodes[b.idx].h + GAP;
    });
  }

  // Column 1 (FPN-up): dataflow runs upward (high vpos at the bottom is the
  // entry, low vpos at the top is the exit). For visual consistency with the
  // other columns we walk TOP-DOWN here too — an expanded block keeps its top
  // edge anchored and blocks below it (higher vpos) slide further down. The
  // internal sub-graph is still flipped (bottom-to-top), so the entry sub-node
  // sits at the region's bottom and connects cleanly to the predecessor below.
  const fpnTop = COL_TOP + 1.5 * ROW_GAP + NECK_Y_OFFSET;
  {
    const col1 = arch.filter(b => b.col === 1).sort((a, b) => a.vpos - b.vpos);
    let y = fpnTop;   // block with vpos=0 sits at the top
    col1.forEach(b => {
      nodes[b.idx] = mk(b, COL_X[1] + xShift[1], y);
      y += nodes[b.idx].h + GAP;
    });
  }

  // Column 2 (PAN-down): walked top-to-bottom from the same top.
  const panTop = COL_TOP + 1.5 * ROW_GAP + NECK_Y_OFFSET;
  {
    let y = panTop;
    arch.filter(b => b.col === 2).forEach(b => {
      nodes[b.idx] = mk(b, COL_X[2] + xShift[2], y);
      y += nodes[b.idx].h + GAP;
    });
  }

  // Detect head — three separate, normal-sized boxes (P3 / P4 / P5) stacked
  // with DETECT_GAP between them. The node's `detect` array carries each
  // box's vertical offset; `h` spans all three so the Head container wraps
  // them. Nothing is stretched — every box is NODE_W x NODE_H like the rest.
  const colSpan = COL_TOP + 10 * ROW_GAP + NECK_Y_OFFSET + NODE_H;  // bottom of block 10
  const detect = arch.find(b => b.col === 3);
  if (detect) {
    const span = 3 * NODE_H + 2 * DETECT_GAP;
    const scales = [
      { scale: 'p3', label: 'P3', size: 'small' },
      { scale: 'p4', label: 'P4', size: 'medium' },
      { scale: 'p5', label: 'P5', size: 'large' },
    ];
    nodes[detect.idx] = {
      x: COL_X[3] + xShift[3],
      y: (colSpan - span) / 2 + 20,
      w: NODE_W,
      h: span,
      detect: scales.map((s, i) => ({ ...s, relY: i * (NODE_H + DETECT_GAP) })),
    };
  }

  // Canvas size from the actual extent of every node, plus a margin.
  let maxX = 0, maxY = 0;
  Object.values(nodes).forEach(n => {
    maxX = Math.max(maxX, n.x + n.w);
    maxY = Math.max(maxY, n.y + n.h);
  });
  return { nodes, totalW: maxX + 120, totalH: maxY + 60 };
}

// --- Ports -----------------------------------------------------------------
function leftPort(n)  { return { x: n.x,           y: n.y + n.h / 2 }; }
function rightPort(n) { return { x: n.x + n.w,     y: n.y + n.h / 2 }; }
function topPort(n)   { return { x: n.x + n.w / 2, y: n.y }; }
function botPort(n)   { return { x: n.x + n.w / 2, y: n.y + n.h }; }

// Left-edge port of the P3 / P4 / P5 detect box for a given scale.
function detectPort(n, scale) {
  const boxes = n.detect || [];
  const box = boxes.find(b => b.scale === scale) || boxes[boxes.length - 1];
  const relY = box ? box.relY : 0;
  return { x: n.x, y: n.y + relY + NODE_H / 2 };
}

// --- Ports for expanded blocks --------------------------------------------
// When a block is expanded, edges connect to its internal entry / exit
// sub-node (the "first layer" / "last layer") rather than the region box —
// `side` is the face the edge meets ('top' | 'bottom' | 'left' | 'right').
// The region's subNodes carry LOCAL coords; add the node's x / y to absolute.
function entryPort(node, side) {
  const en = node.region && node.region.entryNodes && node.region.entryNodes[0];
  if (!en) return leftPort(node);
  const ax = node.x + en.x, ay = node.y + en.y;
  if (side === 'top')    return { x: ax + en.w / 2, y: ay };
  if (side === 'bottom') return { x: ax + en.w / 2, y: ay + en.h };
  return { x: ax, y: ay + en.h / 2 };   // left
}
function exitPort(node, side) {
  const xn = node.region && node.region.exitNodes && node.region.exitNodes[0];
  if (!xn) return rightPort(node);
  const ax = node.x + xn.x, ay = node.y + xn.y;
  if (side === 'top')    return { x: ax + xn.w / 2, y: ay };
  if (side === 'bottom') return { x: ax + xn.w / 2, y: ay + xn.h };
  return { x: ax + xn.w, y: ay + xn.h / 2 };   // right
}

// --- Edge metadata ---------------------------------------------------------
// The pipeline gives us {src, dst, is_skip}. Rendering kind is derived here:
//   detect  — dst is the Detect head (assigned a p3/p4/p5 scale port)
//   forward — same column, immediate predecessor
//   cross   — different column, immediate predecessor
//   skip    — anything else (a long-range connection)
function buildEdgeMeta(edges, arch) {
  const headIdx = arch.find(b => b.role === 'Head')?.idx;
  const detectSrcs = edges
    .filter(e => e.dst === headIdx)
    .map(e => e.src)
    .sort((a, b) => a - b);
  const scaleBySrc = {};
  ['p3', 'p4', 'p5'].forEach((s, i) => { if (detectSrcs[i] != null) scaleBySrc[detectSrcs[i]] = s; });

  return edges.map(e => {
    const from = arch[e.src], to = arch[e.dst];
    let kind;
    if (e.dst === headIdx) kind = 'detect';
    else if (from.col === to.col) kind = 'forward';
    else if (!e.is_skip) kind = 'cross';
    else kind = 'skip';
    return { ...e, kind, scale: scaleBySrc[e.src] || null };
  });
}

// --- Edge path -------------------------------------------------------------
// Routing rules:
//   detect / cross / skip — horizontal flat-tail bezier (rightPort -> leftPort
//                           or detect port)
//   forward (same column) — vertical flat-tail bezier; the bottom-to-top vs
//                           top-to-bottom direction falls out of comparing y
// Port-on-a-sub-node (used when an expanded block has multiple entry/exit
// internal nodes — e.g. SPPF, whose placeholder feeds both `cv1` and the
// residual `add`).
function portOnSub(node, subNode, side) {
  const ax = node.x + subNode.x, ay = node.y + subNode.y;
  if (side === 'top')    return { x: ax + subNode.w / 2, y: ay };
  if (side === 'bottom') return { x: ax + subNode.w / 2, y: ay + subNode.h };
  if (side === 'left')   return { x: ax, y: ay + subNode.h / 2 };
  return { x: ax + subNode.w, y: ay + subNode.h / 2 };
}

// All source ports for the L1 edge `meta`. An expanded block with K exits
// (rare — most blocks have one output) emits K source ports.
function sourcePorts(meta, a, side) {
  if (a.expanded && a.region.exitNodes && a.region.exitNodes.length) {
    return a.region.exitNodes.map(xn => portOnSub(a, xn, side));
  }
  if (side === 'right')  return [rightPort(a)];
  if (side === 'bottom') return [botPort(a)];
  if (side === 'top')    return [topPort(a)];
  return [leftPort(a)];
}

// All target ports for the L1 edge `meta`. An expanded block with K entries
// (common — e.g. SPPF's residual `add`) emits K target ports so the upstream
// connection lands on EVERY first layer that consumes it.
function targetPorts(meta, b, side) {
  if (b.expanded && b.region.entryNodes && b.region.entryNodes.length) {
    return b.region.entryNodes.map(en => portOnSub(b, en, side));
  }
  if (side === 'left')   return [leftPort(b)];
  if (side === 'top')    return [topPort(b)];
  if (side === 'bottom') return [botPort(b)];
  return [rightPort(b)];
}

// Returns an array of SVG path strings for one L1 edge. Multi-entry / multi-
// exit expansions fan out — graph-v2 renders one path per element.
function edgePaths(meta, nodes, arch) {
  const a = nodes[meta.src];
  const b = nodes[meta.dst];
  const from = arch[meta.src];
  const to = arch[meta.dst];

  if (meta.kind === 'detect') {
    const srcs = sourcePorts(meta, a, 'right');
    const dst = detectPort(b, meta.scale || 'p5');
    return srcs.map(p1 => flatBezier(p1, dst, H_ENTRY, H_EXIT));
  }
  if (from.col === to.col) {
    const down = a.y < b.y;
    const srcs = sourcePorts(meta, a, down ? 'bottom' : 'top');
    const dsts = targetPorts(meta, b, down ? 'top' : 'bottom');
    const out = [];
    for (const p1 of srcs) for (const p2 of dsts) out.push(flatBezierVertical(p1, p2, V_ENTRY, V_EXIT));
    return out;
  }
  // cross + skip
  const srcs = sourcePorts(meta, a, 'right');
  const dsts = targetPorts(meta, b, 'left');
  const out = [];
  for (const p1 of srcs) for (const p2 of dsts) out.push(flatBezier(p1, p2, H_ENTRY, H_EXIT));
  return out;
}

// Back-compat: edgePath returns the first of edgePaths. Kept so the existing
// graph-v2 call site keeps working if it doesn't switch to edgePaths.
function edgePath(meta, nodes, arch) {
  return edgePaths(meta, nodes, arch)[0];
}

// --- Flat-tail beziers (ported from layout-l2.jsx) -------------------------
// Each edge is a dead-flat tail of length `entry` out of p1, a short cubic
// bezier that does the actual bending, then a dead-flat tail of length `exit`
// into p2. Bigger leads => flatter edge with a crisper corner. `fitTails`
// scales the two leads down proportionally if they'd otherwise cross.
const BEND_EPS = 0.5;

function fitTails(entry, exit, span) {
  const max = span - 2 * BEND_EPS;
  const total = entry + exit;
  if (total <= max) return [entry, exit];
  const k = max / total;
  return [entry * k, exit * k];
}

function flatBezier(p1, p2, entry, exit) {
  const dx = p2.x - p1.x;
  const adx = Math.abs(dx);
  if (adx < 1) return `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y}`;
  const [e, x] = fitTails(entry, exit, adx);
  const dir = dx >= 0 ? 1 : -1;
  const q1x = p1.x + dir * e;
  const q2x = p2.x - dir * x;
  const midX = (q1x + q2x) / 2;
  return `M ${p1.x} ${p1.y} L ${q1x} ${p1.y} C ${midX} ${p1.y} ${midX} ${p2.y} ${q2x} ${p2.y} L ${p2.x} ${p2.y}`;
}

function flatBezierVertical(p1, p2, entry, exit) {
  const dy = p2.y - p1.y;
  const ady = Math.abs(dy);
  if (ady < 1) return `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y}`;
  const [e, x] = fitTails(entry, exit, ady);
  const dir = dy >= 0 ? 1 : -1;
  const q1y = p1.y + dir * e;
  const q2y = p2.y - dir * x;
  const midY = (q1y + q2y) / 2;
  return `M ${p1.x} ${p1.y} L ${p1.x} ${q1y} C ${p1.x} ${midY} ${p2.x} ${midY} ${p2.x} ${q2y} L ${p2.x} ${p2.y}`;
}

// --- Rounded orthogonal polygon --------------------------------------------
// Quadratic-bezier corner rounding; works for both convex and concave corners
// (the Neck L-polygon has one concave corner).
function roundedPolyPath(pts, r) {
  const n = pts.length;
  const dist = (p, q) => Math.hypot(q.x - p.x, q.y - p.y);
  let d = '';
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const cur = pts[i];
    const next = pts[(i + 1) % n];
    const inLen = dist(prev, cur), outLen = dist(cur, next);
    const e1 = Math.min(r, inLen / 2), e2 = Math.min(r, outLen / 2);
    const v1 = { x: (cur.x - prev.x) / inLen, y: (cur.y - prev.y) / inLen };
    const v2 = { x: (next.x - cur.x) / outLen, y: (next.y - cur.y) / outLen };
    const pStart = { x: cur.x - v1.x * e1, y: cur.y - v1.y * e1 };
    const pEnd   = { x: cur.x + v2.x * e2, y: cur.y + v2.y * e2 };
    d += (i === 0 ? `M ${pStart.x} ${pStart.y}` : ` L ${pStart.x} ${pStart.y}`);
    d += ` Q ${cur.x} ${cur.y} ${pEnd.x} ${pEnd.y}`;
  }
  return d + ' Z';
}

// --- Role containers -------------------------------------------------------
// Backbone / Head: plain rounded rects. Neck: mirrored-L polygon — the full
// bounding box of its members with the top-left corner notched out (that
// corner is where Backbone blocks 3-8 live).
function computeContainers(arch, nodes) {
  const byRole = { Backbone: [], Neck: [], Head: [] };
  arch.forEach(b => byRole[b.role].push(b.idx));

  const rawBBox = idxs => {
    const ns = idxs.map(i => nodes[i]);
    return {
      x:  Math.min(...ns.map(n => n.x)),
      y:  Math.min(...ns.map(n => n.y)),
      x2: Math.max(...ns.map(n => n.x + n.w)),
      y2: Math.max(...ns.map(n => n.y + n.h)),
    };
  };

  const out = [];

  // Plain rectangles.
  for (const role of ['Backbone', 'Head']) {
    const bb = rawBBox(byRole[role]);
    out.push({
      role, kind: 'rect',
      x: bb.x - CONTAINER_PAD,
      y: bb.y - CONTAINER_PAD_T,
      w: (bb.x2 - bb.x) + 2 * CONTAINER_PAD,
      h: (bb.y2 - bb.y) + CONTAINER_PAD_T + CONTAINER_PAD,
      labelX: bb.x - CONTAINER_PAD + 12,
      labelY: bb.y - CONTAINER_PAD_T - 8,
    });
  }

  // Neck — mirrored-L polygon.
  const neckIdx = byRole.Neck;
  const footIdx = neckIdx.filter(i => arch[i].col === 0);   // 9, 10
  const bodyIdx = neckIdx.filter(i => arch[i].col !== 0);   // 11..22
  const foot = rawBBox(footIdx);
  const body = rawBBox(bodyIdx);
  const backbone = rawBBox(byRole.Backbone);

  const left   = foot.x - CONTAINER_PAD;
  const right  = body.x2 + CONTAINER_PAD;
  const top    = body.y - CONTAINER_PAD_T;
  const bottom = foot.y2 + CONTAINER_PAD;
  // Neck body's left edge hugs the first neck column (just CONTAINER_PAD of
  // breathing room) — independent of CONTAINER_GAP, so widening the gaps
  // between containers no longer opens dead space inside the neck body. The
  // foot simply widens to meet this edge (its extra width is where the
  // SPPF/C2PSA skip + cross edges route out).
  const xDivide = body.x - CONTAINER_PAD;          // between col-0 leg and col-1 body
  const yDivide = foot.y - CONTAINER_PAD_T;        // foot top sits CONTAINER_PAD_T above SPPF — uniform with the body's top padding

  // Clockwise from the body's top-left; the notch is the top-left rectangle.
  const pts = [
    { x: xDivide, y: top },
    { x: right,   y: top },
    { x: right,   y: bottom },
    { x: left,    y: bottom },
    { x: left,    y: yDivide },
    { x: xDivide, y: yDivide },
  ];
  out.push({
    role: 'Neck', kind: 'poly',
    path: roundedPolyPath(pts, 12),
    labelX: xDivide + 12,
    labelY: top - 8,
  });

  return out;
}

window.YVV2 = window.YVV2 || {};
window.YVV2.layoutGraph = layoutGraph;
window.YVV2.buildEdgeMeta = buildEdgeMeta;
window.YVV2.edgePath = edgePath;
window.YVV2.edgePaths = edgePaths;
window.YVV2.computeContainers = computeContainers;
window.YVV2.detectPort = detectPort;
window.YVV2.NODE_W = NODE_W;
window.YVV2.NODE_H = NODE_H;
