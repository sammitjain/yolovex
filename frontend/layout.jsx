// Computes pixel positions for nodes and Bezier paths for edges.

const NODE_W = 158;
const NODE_H = 56;
const COL_GAP = 230;
const ROW_GAP = 92;       // bumped up for more vertical breathing room
const COL_X = [120, 120 + COL_GAP, 120 + COL_GAP * 2, 120 + COL_GAP * 3];

function layoutGraph(arch) {
  const nodes = {};
  const colTop = 80;

  // Backbone: 11 nodes
  arch.filter(b => b.col === 0).forEach((b, i) => {
    nodes[b.idx] = { x: COL_X[0], y: colTop + i * ROW_GAP, w: NODE_W, h: NODE_H };
  });

  // FPN-up: 6 nodes — vpos 0 = top, vpos 5 = bottom; centered on backbone
  const fpnTop = colTop + 1.5 * ROW_GAP;
  arch.filter(b => b.col === 1).forEach(b => {
    nodes[b.idx] = { x: COL_X[1], y: fpnTop + b.vpos * ROW_GAP, w: NODE_W, h: NODE_H };
  });

  // PAN-down: 6 nodes
  const panTop = colTop + 1.5 * ROW_GAP;
  arch.filter(b => b.col === 2).forEach((b, i) => {
    nodes[b.idx] = { x: COL_X[2], y: panTop + i * ROW_GAP, w: NODE_W, h: NODE_H };
  });

  const totalH = colTop + 10 * ROW_GAP + NODE_H;
  // Taller Detect node so the P3/P4/P5 sub-rects breathe and the three input
  // arrows from blocks 16/19/22 land at clearly distinct ports.
  const detectH = NODE_H * 3.6;
  nodes[23] = {
    x: COL_X[3],
    y: (totalH - detectH) / 2 + 20,
    w: NODE_W,
    h: detectH,
  };

  return { nodes, totalW: COL_X[3] + NODE_W + 120, totalH: totalH + 60 };
}

function leftPort(n)  { return { x: n.x,        y: n.y + n.h / 2 }; }
function rightPort(n) { return { x: n.x + n.w,  y: n.y + n.h / 2 }; }
function topPort(n)   { return { x: n.x + n.w/2,y: n.y }; }
function botPort(n)   { return { x: n.x + n.w/2,y: n.y + n.h }; }

function detectPort(n, scale) {
  const subH = n.h / 3;
  const yOff = scale === 'p3' ? subH * 0.5 : scale === 'p4' ? subH * 1.5 : subH * 2.5;
  return { x: n.x, y: n.y + yOff };
}

function edgePath(from, to, kind, nodes, arch) {
  const fromBlock = arch[from];
  const toBlock = arch[to];
  const a = nodes[from];
  const b = nodes[to];

  if (kind === 'detect-p3') return bezier(rightPort(a), detectPort(b, 'p3'), 'horizontal');
  if (kind === 'detect-p4') return bezier(rightPort(a), detectPort(b, 'p4'), 'horizontal');
  if (kind === 'detect-p5') return bezier(rightPort(a), detectPort(b, 'p5'), 'horizontal');

  if (fromBlock.col === toBlock.col) {
    if (fromBlock.col === 1) {
      return bezier(topPort(a), botPort(b), 'vertical');
    }
    return bezier(botPort(a), topPort(b), 'vertical');
  }

  if (kind === 'cross') {
    return bezier(rightPort(a), leftPort(b), 'horizontal');
  }

  return bezier(rightPort(a), leftPort(b), 'skip');
}

// Bezier with end-tangent guaranteed horizontal so the arrowhead lines up with the target.
function bezier(p1, p2, mode = 'horizontal') {
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
    // c1: leaves outward + slightly up, c2: arrives HORIZONTALLY (same y as p2) so the arrowhead is straight
    return `M ${p1.x} ${p1.y} C ${p1.x + sag} ${p1.y - bow} ${p2.x - sag} ${p2.y} ${p2.x} ${p2.y}`;
  }
  // horizontal default — both control points on the y-line of their endpoints, so tangents are horizontal at both ends
  const dx = p2.x - p1.x;
  return `M ${p1.x} ${p1.y} C ${p1.x + dx * 0.5} ${p1.y} ${p2.x - dx * 0.5} ${p2.y} ${p2.x} ${p2.y}`;
}

window.YV = window.YV || {};
window.YV.layoutGraph = layoutGraph;
window.YV.edgePath = edgePath;
window.YV.NODE_W = NODE_W;
window.YV.NODE_H = NODE_H;
window.YV.leftPort = leftPort;
window.YV.rightPort = rightPort;
window.YV.detectPort = detectPort;
