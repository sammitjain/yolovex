// L2 layout — computes pixel positions for containers and sub-nodes.

// ---- Layout constants (matched to the SPPF test canvas) ----
const COL_GAP_L2 = 520;
const COL_X_L2 = [120, 120 + COL_GAP_L2, 120 + COL_GAP_L2 * 2, 120 + COL_GAP_L2 * 3];

// Every sub-node is exactly TILE_W wide regardless of which block it lives
// in — so single-stack blocks (Conv / C2PSA / Detect) stay compact, and the
// staircase blocks (C3k2 / SPPF) use the same tile size for their taps.
const TILE_W = 120;
const TILE_H = 46;
const STAIR_FRAC = 0.65;          // each tap shifts right by this fraction of TILE_W

const SUB_ROW_GAP = 78;            // top-to-top distance between rows
const POST_MERGE_GAP = 78;         // gap between merge row and post-merge rows

const CONTAINER_PAD_TOP = 30;
const CONTAINER_PAD_BOTTOM = 18;
const CONTAINER_PAD_X = 16;
const CONTAINER_ROW_GAP = 36;

// Atomic containers (Concat / Upsample at the block level — no sub-nodes
// inside) are sized so their center lines up with the single-tile center of
// adjacent containers (innerLeft + TILE_W / 2).
const ATOMIC_W = TILE_W + 2 * CONTAINER_PAD_X;   // 152
const ATOMIC_H = TILE_H + 24;                    // 70
const CONTAINER_W = ATOMIC_W;                    // back-compat alias used in a few places

const SEQ_BEND = 22;               // lead length for the right-angle bezier between consecutive taps

// ---- Per-parent layout ----
//
// Two flavours:
//   stack     — simple top-to-bottom column. Every sub at innerLeft, w=TILE_W.
//   taps+merge — staircase: each tap[i] shifts right by i*STAIR_FRAC*TILE_W;
//                merge tile spans the full inner width; postMerge tiles are
//                left-aligned at innerLeft (so they line up with the next
//                block's first sub-node).
const SUBNODE_LAYOUTS_L2 = {
  Conv:   { stack: ['conv', 'bn', 'act'] },
  C3k2:   { taps: ['cv1', 'm.0'],                merge: 'concat', postMerge: ['cv2'] },
  SPPF:   { taps: ['cv1', 'm.0', 'm.1', 'm.2'],  merge: 'concat', postMerge: ['cv2'] },
  C2PSA:  { stack: ['cv1', 'm.0', 'cv2'] },
  Detect: { stack: ['head_p3', 'head_p4', 'head_p5'] },
};

// Compute the {w, h} a container needs given its parent type.
//   - stack mode: width = TILE_W + 2*pad
//   - staircase mode: inner width = mergeW = TILE_W * (1 + STAIR_FRAC*(N-1))
function containerSize(parentType, nSubs) {
  if (nSubs === 0) return { w: ATOMIC_W, h: ATOMIC_H };
  const layout = SUBNODE_LAYOUTS_L2[parentType];
  if (layout?.taps) {
    const N = layout.taps.length;
    const M = (layout.postMerge || []).length;
    const mergeW = TILE_W * (1 + STAIR_FRAC * (N - 1));
    const cw = mergeW + 2 * CONTAINER_PAD_X;
    // Rows: N taps + 1 merge + M post-merge
    const totalRows = N + 1 + M;
    const ch = CONTAINER_PAD_TOP + (totalRows - 1) * SUB_ROW_GAP + TILE_H + CONTAINER_PAD_BOTTOM;
    return { w: cw, h: ch };
  }
  // stack mode (or unknown — fall back)
  const stack = layout?.stack || new Array(nSubs).fill('?');
  const cw = TILE_W + 2 * CONTAINER_PAD_X;
  const ch = CONTAINER_PAD_TOP + (stack.length - 1) * SUB_ROW_GAP + TILE_H + CONTAINER_PAD_BOTTOM;
  return { w: cw, h: ch };
}

// Helper used by edge routing — does this sub-node sit in a staircase tap row?
function isTapSub(parentType, path) {
  const layout = SUBNODE_LAYOUTS_L2[parentType];
  return !!(layout?.taps && layout.taps.includes(path));
}

function layoutGraphL2(arch_l2) {
  const containers = {};   // idx -> {x, y, w, h, label}
  const subnodes = {};     // id  -> {x, y, w, h, parentIdx, type}

  const colTop = 80;

  // --- Backbone (col 0): stack top-to-bottom ---
  let y0 = colTop;
  arch_l2.filter(b => b.col === 0).forEach(b => {
    const { w, h } = containerSize(b.type, b.sub.length);
    // Anchor on column center so wider containers (SPPF, C3k2) stay aligned.
    const x = COL_X_L2[0];   // left-align (block N's left edge sits on the column line)
    containers[b.idx] = { x, y: y0, w, h, label: `[${b.idx}] ${b.type}` };
    placeSubnodes(b, x, y0, w, subnodes);
    y0 += h + CONTAINER_ROW_GAP;
  });

  // --- FPN-up (col 1): in L1, lower vpos = higher on screen, so block 16
  // (vpos 0) is at the top and block 11 (vpos 5) is at the bottom. The data
  // FLOWS bottom-to-top through this column (11 → 12 → 13 → 14 → 15 → 16).
  // To preserve that semantics in L2: place containers in vpos order (top-to-
  // bottom on screen), AND place each container's sub-nodes in REVERSE order
  // so the data inside each container also reads bottom-to-top.
  const fpnBlocks = arch_l2.filter(b => b.col === 1);
  const fpnByVpos = [...fpnBlocks].sort((a, b) => a.vpos - b.vpos);
  let fpnY = colTop + 138;
  fpnByVpos.forEach(b => {
    const { w, h } = containerSize(b.type, b.sub.length);
    const x = COL_X_L2[1];   // left-align
    containers[b.idx] = { x, y: fpnY, w, h, label: `[${b.idx}] ${b.type}` };
    placeSubnodes(b, x, fpnY, w, subnodes, /* reverse */ true);
    fpnY += h + CONTAINER_ROW_GAP;
  });

  // --- PAN-down (col 2): stack top-to-bottom ---
  let y2 = colTop + 138;
  arch_l2.filter(b => b.col === 2).forEach(b => {
    const { w, h } = containerSize(b.type, b.sub.length);
    const x = COL_X_L2[2];   // left-align
    containers[b.idx] = { x, y: y2, w, h, label: `[${b.idx}] ${b.type}` };
    placeSubnodes(b, x, y2, w, subnodes);
    y2 += h + CONTAINER_ROW_GAP;
  });

  // --- Head (col 3): Detect container (idx 23) — align so it sits roughly
  // at the *vertical mid-point of the neck columns*, slightly biased upward.
  // Centering against ALL containers drags the head too far down because the
  // backbone column extends much further than the neck columns.
  const detectBlock = arch_l2.find(b => b.col === 3);
  if (detectBlock) {
    const { w, h } = containerSize(detectBlock.type, detectBlock.sub.length);
    const neckContainers = arch_l2
      .filter(b => b.col === 1 || b.col === 2)
      .map(b => containers[b.idx])
      .filter(Boolean);
    let centerY;
    if (neckContainers.length) {
      const neckTop = Math.min(...neckContainers.map(c => c.y));
      const neckBot = Math.max(...neckContainers.map(c => c.y + c.h));
      const mid = (neckTop + neckBot) / 2 - (neckBot - neckTop) * 0.15;
      centerY = mid - h / 2;
    } else {
      centerY = colTop;
    }
    const x = COL_X_L2[3];   // left-align
    containers[detectBlock.idx] = { x, y: centerY, w, h, label: `[${detectBlock.idx}] Detect` };
    placeSubnodes(detectBlock, x, centerY, w, subnodes);
  }

  // Compute total dimensions
  const allContainers = Object.values(containers);
  const totalW = Math.max(...allContainers.map(c => c.x + c.w)) + 120;
  const totalH = Math.max(...allContainers.map(c => c.y + c.h)) + 80;

  return { containers, subnodes, totalW, totalH };
}

function placeSubnodes(container, containerX, containerY, containerW, subnodes, reverse) {
  if (container.sub.length === 0) return;
  const layout = SUBNODE_LAYOUTS_L2[container.type];
  if (!layout) return;

  const subByPath = {};
  container.sub.forEach(s => { subByPath[s.path] = s; });
  const innerLeft = containerX + CONTAINER_PAD_X;

  // Build a forward-order row list with placement info, then reverse if needed.
  // Each entry: { path, x, w, role: 'tap'|'merge'|'post'|'stack' }
  const rows = [];
  if (layout.taps) {
    const N = layout.taps.length;
    const mergeW = TILE_W * (1 + STAIR_FRAC * (N - 1));
    layout.taps.forEach((path, i) => {
      rows.push({
        path,
        role: 'tap',
        x: innerLeft + i * STAIR_FRAC * TILE_W,
        w: TILE_W,
      });
    });
    rows.push({
      path: layout.merge,
      role: 'merge',
      x: innerLeft,
      w: mergeW,
    });
    (layout.postMerge || []).forEach(path => {
      rows.push({
        path,
        role: 'post',
        x: innerLeft,            // left-aligned so it lines up with the next block's first sub
        w: TILE_W,
      });
    });
  } else {
    layout.stack.forEach(path => {
      rows.push({
        path,
        role: 'stack',
        x: innerLeft,
        w: TILE_W,
      });
    });
  }

  const orderedRows = reverse ? [...rows].reverse() : rows;

  orderedRows.forEach((row, rowIdx) => {
    const s = subByPath[row.path];
    if (!s) return;
    const subY = containerY + CONTAINER_PAD_TOP + rowIdx * SUB_ROW_GAP;
    subnodes[s.id] = {
      x: row.x,
      y: subY,
      w: row.w,
      h: TILE_H,
      parentIdx: container.idx,
      type: s.type,
      shape: s.shape,
      role: row.role,
    };
  });
}

// ---- Port helpers (copied from layout.jsx) ----

function leftPortL2(n)  { return { x: n.x,        y: n.y + n.h / 2 }; }
function rightPortL2(n) { return { x: n.x + n.w,  y: n.y + n.h / 2 }; }
function topPortL2(n)   { return { x: n.x + n.w/2, y: n.y }; }
function botPortL2(n)   { return { x: n.x + n.w/2, y: n.y + n.h }; }

function detectPortL2(n, scale) {
  // n is the sub-node rect for the detect head (head_p3/p4/p5)
  return leftPortL2(n);
}

// ---- Bezier (copied from layout.jsx) ----
function bezierL2(p1, p2, mode) {
  if (!mode) mode = 'horizontal';
  if (mode === 'vertical') {
    const dy = p2.y - p1.y;
    const c1y = p1.y + dy * 0.5;
    const c2y = p2.y - dy * 0.5;
    return `M ${p1.x} ${p1.y} C ${p1.x} ${c1y} ${p2.x} ${c2y} ${p2.x} ${p2.y}`;
  }
  if (mode === 'skip') {
    const dx = p2.x - p1.x;
    const sag = Math.min(140, Math.abs(dx) * 0.45);
    const bow = Math.min(80, Math.abs(dx) * 0.18);
    return `M ${p1.x} ${p1.y} C ${p1.x + sag} ${p1.y - bow} ${p2.x - sag} ${p2.y} ${p2.x} ${p2.y}`;
  }
  const dx = p2.x - p1.x;
  return `M ${p1.x} ${p1.y} C ${p1.x + dx * 0.5} ${p1.y} ${p2.x - dx * 0.5} ${p2.y} ${p2.x} ${p2.y}`;
}

// ---- Edge path computation ----

function edgePathL2(edge, containers, subnodes) {
  const { from, to, kind } = edge;

  // Resolve node positions — 'from'/'to' can be string sub-node id or numeric container idx
  function resolveNode(id) {
    // Try subnodes first (string ids like "4.cv1")
    if (typeof id === 'string' && subnodes[id]) return subnodes[id];
    // Try containers (numeric or string of a number)
    const idx = typeof id === 'number' ? id : parseInt(id, 10);
    if (!isNaN(idx) && containers[idx]) return containers[idx];
    return null;
  }

  const a = resolveNode(from);
  const b = resolveNode(to);
  if (!a || !b) return null;

  if (kind === 'skip') {
    // Design rule: arrows that approach a node from a different column should
    // land on its LEFT edge — they're crossing horizontally, not stacking
    // vertically. We bias the landing point a little above center to keep the
    // skip arc visually distinct from the regular forward inputs.
    const containerIdx = typeof to === 'number' ? to : parseInt(to, 10);
    const targetContainer = containers[containerIdx] || b;
    const port = leftPortL2(targetContainer);   // dead center of the left edge
    return skipBezier(rightPortL2(a), port);
  }

  if (kind.startsWith('detect-')) {
    // detect-pX edges go to the specific detect sub-node left port
    return bezierL2(rightPortL2(a), leftPortL2(b), 'horizontal');
  }

  if (kind === 'fanin') {
    // Tap → merge: pure-vertical straight line from the tap's bottom-mid
    // (forward direction) or top-mid (FPN-up reversed) to the merge's top
    // or bottom edge at the SAME x as the tap's center. No curve.
    if (a.y < b.y) {
      const p1 = botPortL2(a);
      const p2 = { x: p1.x, y: b.y };
      return `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y}`;
    }
    const p1 = topPortL2(a);
    const p2 = { x: p1.x, y: b.y + b.h };
    return `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y}`;
  }

  if (kind === 'intra') {
    // Three sub-cases:
    //   1. Both endpoints are STAIRCASE TAP rows → right-angle bezier
    //      (departs horizontally, arrives vertically).
    //   2. Same row (rare) → horizontal bezier.
    //   3. Otherwise (stack-mode chain, or merge → post-merge) → vertical
    //      bezier with horizontal-tangent ends; degenerates to a straight
    //      line when source and target share x.
    const sameRow = Math.abs(a.y - b.y) < 1;
    if (sameRow) {
      return bezierL2(rightPortL2(a), leftPortL2(b), 'horizontal');
    }
    if (a.role === 'tap' && b.role === 'tap') {
      if (a.y < b.y) return rightAngleBezier(rightPortL2(a), topPortL2(b));
      return rightAngleBezier(rightPortL2(a), botPortL2(b));
    }
    if (a.y < b.y) return bezierL2(botPortL2(a), topPortL2(b), 'vertical');
    return bezierL2(topPortL2(a), botPortL2(b), 'vertical');
  }

  // Inter-container edges: kind is 'forward' (same column) or 'cross'
  // (different column). With every single-tile sub-node left-aligned at
  // innerLeft and every container's left edge sitting on COL_X, forward
  // arrows collapse to a clean vertical line because the source's last sub
  // and the target's first sub share an x.
  if (kind === 'cross') {
    return bezierL2(rightPortL2(a), leftPortL2(b), 'horizontal');
  }
  if (kind === 'forward') {
    if (a.y < b.y) return bezierL2(botPortL2(a), topPortL2(b), 'vertical');
    return bezierL2(topPortL2(a), botPortL2(b), 'vertical');
  }

  return bezierL2(rightPortL2(a), leftPortL2(b), 'horizontal');
}

// Skip-edge bezier with a long FLAT approach to the target. The last control
// point sits a generous distance to the left of p2 at exactly the same y, so
// the curve enters the arrowhead along a near-perfect horizontal tangent —
// and the arrowhead lines up symmetrically with the line.
function skipBezier(p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  // First leg: source leaves to the right, with a vertical bow that matches
  // the y-distance between p1 and p2 so the curve actually clears intermediate
  // nodes when source is far above/below the target.
  const leg = Math.max(80, Math.abs(dx) * 0.35);
  const bow = Math.min(60, Math.abs(dy) * 0.18);
  const c1 = { x: p1.x + leg, y: p1.y + (dy < 0 ? -bow : bow) * 0.0 };  // mostly horizontal departure
  // Second leg: SAME y as p2, nice and far back from p2 so the arrow lands flat.
  const flat = Math.max(60, Math.abs(dx) * 0.30);
  const c2 = { x: p2.x - flat, y: p2.y };
  return `M ${p1.x} ${p1.y} C ${c1.x} ${c1.y} ${c2.x} ${c2.y} ${p2.x} ${p2.y}`;
}

// (FANIN_ROUTES / fanInRoute removed — the staircase layout means each tap
// already sits at a unique x position, so the tap → merge connection is
// just a straight vertical line at the tap's center x. No special routing,
// no bus detour, no per-tap landing offsets.)

// Right-angle bezier — leaves p1 horizontally to the right, arrives at p2
// vertically (going down if p2 is below p1, going up if p2 is above). Used
// for the sequential connections between consecutive staircase taps.
function rightAngleBezier(p1, p2, lead = SEQ_BEND) {
  const dy = p2.y - p1.y;
  const c1 = { x: p1.x + lead, y: p1.y };
  const c2 = { x: p2.x, y: p2.y + (dy >= 0 ? -lead : lead) };
  return `M ${p1.x} ${p1.y} C ${c1.x} ${c1.y} ${c2.x} ${c2.y} ${p2.x} ${p2.y}`;
}

window.YV = window.YV || {};
window.YV.layoutGraphL2 = layoutGraphL2;
window.YV.edgePathL2 = edgePathL2;
window.YV.COL_X_L2 = COL_X_L2;
window.YV.CONTAINER_W = CONTAINER_W;
window.YV.SUB_NODE_W = TILE_W;
window.YV.SUB_NODE_H = TILE_H;
window.YV.TILE_W = TILE_W;
window.YV.TILE_H = TILE_H;
