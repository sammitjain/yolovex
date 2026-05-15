// yolovex v2 — in-place block expansion.
//
// Self-contained port of the spec-graph machinery from frontend/spec-viewer.jsx
// (preprocess / aggregate / staircase-detect / auto-layout / edge-routing),
// plus buildExpansion() which turns one L1 block into a laid-out internal
// component sub-graph ready to drop into the v2 canvas.
//
// Scope for now: depth-1 aggregation only (the "first decomposition" —
// e.g. Conv -> conv/bn/act). Deeper/recursive expansion can layer on later.

// ============================ ported constants ============================
const STAIR_FRAC = 0.65;
const SEQ_BEND   = 22;

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

// ===================== ported: detectStaircases ============================
function detectStaircases(nodes, edges) {
  const inMap = new Map(), outMap = new Map();
  for (const n of nodes) { inMap.set(n.id, []); outMap.set(n.id, []); }
  for (const e of edges) {
    const [s, t] = e;
    inMap.get(t)?.push(s);
    outMap.get(s)?.push(t);
  }

  const stairs = [];
  const tapOffset = new Map();
  const tapInStaircase = new Set();
  const mergeIds = new Set();

  for (const n of nodes) {
    if (n.subkind !== 'cat') continue;
    const preds = inMap.get(n.id) || [];
    if (preds.length < 2) continue;
    const predSet = new Set(preds);
    let start = null;
    for (const p of preds) {
      const internalPreds = (inMap.get(p) || []).filter(pp => predSet.has(pp));
      if (internalPreds.length === 0) { start = p; break; }
    }
    if (!start) continue;
    const ordered = [start];
    const used = new Set([start]);
    let cur = start;
    while (true) {
      const next = (outMap.get(cur) || []).find(s => predSet.has(s) && !used.has(s));
      if (!next) break;
      ordered.push(next);
      used.add(next);
      cur = next;
    }
    if (ordered.length < preds.length) continue;
    if (ordered.length < 2) continue;

    let strictIncoming = true;
    for (let i = 1; i < ordered.length; i++) {
      const ip = inMap.get(ordered[i]) || [];
      if (ip.length !== 1 || ip[0] !== ordered[i - 1]) { strictIncoming = false; break; }
    }
    if (!strictIncoming) continue;

    stairs.push({ mergeId: n.id, taps: ordered });
    ordered.forEach((id, i) => {
      tapOffset.set(id, i);
      tapInStaircase.add(id);
    });
    mergeIds.add(n.id);
  }

  return { stairs, tapOffset, tapInStaircase, mergeIds };
}

// ===================== ported: fan-in / fan-out detection ==================
function detectFanOutSources(nodes, edges, rankByName) {
  const outMap = new Map();
  nodes.forEach(n => outMap.set(n.id, new Set()));
  for (const [s, t] of edges) outMap.get(s)?.add(t);

  const sources = new Map();
  const tapEdgeKeys = new Set();
  for (const n of nodes) {
    const succs = [...(outMap.get(n.id) || [])];
    const adjacent = succs.filter(s => (rankByName[s] - rankByName[n.id]) === 1);
    if (adjacent.length < 2) continue;
    sources.set(n.id, adjacent);
    for (const a of adjacent) tapEdgeKeys.add(`${n.id}->${a}`);
  }
  return { sources, tapEdgeKeys };
}

function detectFanInMerges(nodes, edges, staircaseMergeIds, rankByName, multiNodeRanks) {
  const inMap = new Map();
  nodes.forEach(n => inMap.set(n.id, []));
  for (const [s, t] of edges) inMap.get(t)?.push(s);

  const merges = new Map();
  const tapEdgeKeys = new Set();
  for (const n of nodes) {
    if (n.subkind !== 'cat') continue;
    if (staircaseMergeIds.has(n.id)) continue;
    const preds = inMap.get(n.id) || [];
    if (preds.length < 2) continue;
    let triggered = false;
    for (const p of preds) {
      const gap = (rankByName[n.id] ?? 0) - (rankByName[p] ?? 0);
      if (gap >= 2 && multiNodeRanks.has(rankByName[p])) {
        triggered = true;
        break;
      }
    }
    if (!triggered) continue;
    merges.set(n.id, preds);
    for (const p of preds) tapEdgeKeys.add(`${p}->${n.id}`);
  }
  return { merges, tapEdgeKeys };
}

// ===================== ported: sizing + autoLayout =========================
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

const SKIP_SHIFT   = 90;
const SKIP_OFFSET  = 20;
const SKIP_LANE_W  = 35;
const ROW_GAP_PX   = 70;
const COL_GAP_PX   = 50;
const MARGIN_PX    = 40;

function autoLayout(nodes, edges, stairs) {
  const ROW_GAP = ROW_GAP_PX, COL_GAP = COL_GAP_PX, MARGIN = MARGIN_PX;
  const NODE_W = SUB_NODE_W, NODE_H = SUB_NODE_H;

  if (nodes.length === 0) {
    return { positions: {}, totalW: 600, totalH: 200, skipEdges: new Set(), corridorX: 0 };
  }

  const inMap = new Map(), outMap = new Map();
  nodes.forEach(n => { inMap.set(n.id, []); outMap.set(n.id, []); });
  edges.forEach(([s, t]) => {
    inMap.get(t)?.push(s);
    outMap.get(s)?.push(t);
  });
  // Stable, expansion-invariant order: position in the aggregated node list
  // follows fx trace order (= forward execution order). Used as the secondary
  // sort key for same-rank peers so left/right branch assignment doesn't flip
  // when an unrelated sub-node further down the graph is expanded.
  const nodeIndex = new Map(nodes.map((n, i) => [n.id, i]));

  const rank = {};
  const visiting = new Set();
  function computeRank(id) {
    if (id in rank) return rank[id];
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const preds = inMap.get(id) || [];
    rank[id] = preds.length === 0 ? 0 : Math.max(...preds.map(computeRank)) + 1;
    return rank[id];
  }
  nodes.forEach(n => computeRank(n.id));

  const maxRank = Math.max(...Object.values(rank));
  nodes.forEach(n => {
    if ((n.subkind === 'io' || n.kind === 'io') && (n.id === 'output' || n.label === 'output')) {
      rank[n.id] = maxRank;
    }
  });

  const tapOffset = stairs.tapOffset;
  const mergeSpanByMerge = new Map();
  const mergeStaircase  = new Map();
  for (const s of stairs.stairs) {
    const N = s.taps.length;
    const span = NODE_W + (N - 1) * STAIR_FRAC * NODE_W;
    mergeSpanByMerge.set(s.mergeId, span);
    mergeStaircase.set(s.mergeId, s.taps);
  }

  const rows = new Map();
  for (const n of nodes) {
    if (!rows.has(rank[n.id])) rows.set(rank[n.id], []);
    rows.get(rank[n.id]).push(n);
  }
  const rankKeys = [...rows.keys()].sort((a, b) => a - b);

  const positions = {};
  const sizes = {};
  for (const n of nodes) {
    sizes[n.id] = nodeSize(n, mergeSpanByMerge.get(n.id));
  }

  let maxRowW = 0;
  for (const r of rankKeys) {
    const row = rows.get(r);
    let w = 0;
    if (row.length === 1) {
      const n = row[0];
      const sz = sizes[n.id];
      const off = tapOffset.has(n.id) ? tapOffset.get(n.id) * STAIR_FRAC * NODE_W : 0;
      w = off + sz.w;
    } else {
      w = row.reduce((acc, n) => acc + sizes[n.id].w + COL_GAP, -COL_GAP);
    }
    if (w > maxRowW) maxRowW = w;
  }

  const multiNodeRanks = new Set();
  for (const [r, nodesAtR] of rows) {
    if (nodesAtR.length > 1) multiNodeRanks.add(r);
  }
  const fanIns = detectFanInMerges(nodes, edges, stairs.mergeIds, rank, multiNodeRanks);
  const fanOuts = detectFanOutSources(nodes, edges, rank);

  const skipEdges = new Set();
  for (const e of edges) {
    const [s, t] = e;
    if (rank[t] != null && rank[s] != null && (rank[t] - rank[s]) >= 2) {
      const isStaircaseInternal =
        stairs.tapInStaircase.has(s) && stairs.mergeIds.has(t);
      const isFanInTap = fanIns.tapEdgeKeys.has(`${s}->${t}`);
      const isFanOutTap = fanOuts.tapEdgeKeys.has(`${s}->${t}`);
      if (!isStaircaseInternal && !isFanInTap && !isFanOutTap) {
        skipEdges.add(`${s}->${t}`);
      }
    }
  }
  const spineX = MARGIN;

  const skipsList = [];
  for (const key of skipEdges) {
    const [s, t] = key.split('->');
    if (rank[s] == null || rank[t] == null) continue;
    skipsList.push({ src: s, tgt: t, srcRank: rank[s], tgtRank: rank[t], key });
  }
  const skipLane = new Map();
  const bySrc = new Map();
  for (const sk of skipsList) {
    if (!bySrc.has(sk.src)) bySrc.set(sk.src, []);
    bySrc.get(sk.src).push(sk);
  }
  for (const [, group] of bySrc) {
    group.sort((a, b) => (b.tgtRank - b.srcRank) - (a.tgtRank - a.srcRank));
    group.forEach((sk, i) => skipLane.set(sk.key, i));
  }
  const skipDepth = new Map();
  for (const sk of skipsList) {
    let d = 0;
    for (const other of skipsList) {
      if (other.key === sk.key) continue;
      if (other.srcRank < sk.srcRank && other.tgtRank > sk.tgtRank) d++;
    }
    skipDepth.set(sk.key, d);
  }
  const shiftByRank = new Map();
  for (const sk of skipsList) {
    const d = skipDepth.get(sk.key) || 0;
    const want = SKIP_SHIFT + d * SKIP_LANE_W;
    for (let r = sk.srcRank + 1; r < sk.tgtRank; r++) {
      shiftByRank.set(r, Math.max(shiftByRank.get(r) || 0, want));
    }
  }

  const skipsByTarget = new Map();
  for (const sk of skipsList) {
    if (!skipsByTarget.has(sk.tgt)) skipsByTarget.set(sk.tgt, []);
    skipsByTarget.get(sk.tgt).push(sk);
  }
  const fanOutGap = (n) => {
    let g = 0;
    for (const succ of outMap.get(n.id) || []) {
      const dg = (rank[succ] ?? 0) - (rank[n.id] ?? 0);
      if (dg > g) g = dg;
    }
    return g;
  };

  const FANIN_PAD = 28;
  const positionedX = new Map();
  for (const r of rankKeys) {
    const row = rows.get(r);
    const preferredX = new Map();
    const centeredSet = new Set();
    for (const n of row) {
      let want;
      if (fanIns.merges.has(n.id)) {
        const taps = fanIns.merges.get(n.id);
        let minCx = Infinity, maxCx = -Infinity;
        for (const tapId of taps) {
          const tp = positions[tapId];
          if (!tp) continue;
          const cx = tp.x + tp.w / 2;
          if (cx < minCx) minCx = cx;
          if (cx > maxCx) maxCx = cx;
        }
        if (minCx === Infinity) {
          want = spineX + (shiftByRank.get(r) || 0);
        } else {
          want = minCx - FANIN_PAD;
          const w = Math.max(NODE_W, (maxCx - minCx) + 2 * FANIN_PAD);
          sizes[n.id] = { w, h: NODE_H };
        }
      } else if (tapOffset.has(n.id)) {
        const shift = shiftByRank.get(r) || 0;
        want = spineX + shift + tapOffset.get(n.id) * STAIR_FRAC * NODE_W;
      } else {
        const incoming = skipsByTarget.get(n.id) || [];
        if (incoming.length > 0) {
          let align = 0;
          for (const sk of incoming) {
            const srcX = positionedX.get(sk.src) ?? spineX;
            const cand = srcX + (skipLane.get(sk.key) ?? 0) * SKIP_LANE_W;
            if (cand > align) align = cand;
          }
          want = align;
        } else {
          want = spineX + (shiftByRank.get(r) || 0);
          centeredSet.add(n.id);
        }
      }
      preferredX.set(n.id, want);
    }
    const sortedRow = [...row].sort((a, b) => {
      const pa = preferredX.get(a.id);
      const pb = preferredX.get(b.id);
      if (pa !== pb) return pa - pb;
      // Break ties by fx trace order, not fanOutGap. fanOutGap can flip the
      // left/right assignment of parallel branches when one branch is expanded
      // (its downstream chain grows, its gap shrinks) — trace order is stable
      // across every expansion combination because aggregation never reorders
      // groupOrder relative to the original graph.
      return (nodeIndex.get(a.id) ?? 0) - (nodeIndex.get(b.id) ?? 0);
    });
    let curX = -Infinity;
    for (const n of sortedRow) {
      const sz = sizes[n.id];
      let slotLeft = preferredX.get(n.id);
      if (slotLeft < curX) slotLeft = curX;
      let x = slotLeft;
      if (centeredSet.has(n.id) && sz.w < NODE_W) x += (NODE_W - sz.w) / 2;
      positions[n.id] = { x, y: r * (NODE_H + ROW_GAP) + MARGIN, ...sz };
      positionedX.set(n.id, x);
      curX = slotLeft + Math.max(sz.w, NODE_W) + COL_GAP;
    }
  }

  const FANOUT_PAD = FANIN_PAD;
  for (const [sourceId, children] of fanOuts.sources) {
    const sp = positions[sourceId];
    if (!sp) continue;
    let minCx = Infinity, maxCx = -Infinity;
    for (const c of children) {
      const cp = positions[c];
      if (!cp) continue;
      const cx = cp.x + cp.w / 2;
      if (cx < minCx) minCx = cx;
      if (cx > maxCx) maxCx = cx;
    }
    if (minCx === Infinity) continue;
    const left  = Math.min(sp.x, minCx - FANOUT_PAD);
    const right = Math.max(sp.x + sp.w, maxCx + FANOUT_PAD);
    sp.x = left;
    sp.w = right - left;
    positionedX.set(sourceId, left);
  }

  let totalW = 0;
  for (const p of Object.values(positions)) {
    if (p.x + p.w > totalW) totalW = p.x + p.w;
  }

  return {
    positions,
    totalW: totalW + MARGIN,
    totalH: (Math.max(...rankKeys) + 1) * (NODE_H + ROW_GAP) + MARGIN,
    skipEdges,
    skipLane,
    fanInMerges: fanIns.merges,
    fanInTapEdges: fanIns.tapEdgeKeys,
    fanOutSources: fanOuts.sources,
    fanOutTapEdges: fanOuts.tapEdgeKeys,
    corridorX: spineX,
  };
}

// ===================== ported: edge-routing helpers ========================
function rightAngleEdge(p1, p2, lead = SEQ_BEND) {
  const dy = p2.y - p1.y;
  const c1 = { x: p1.x + lead, y: p1.y };
  const c2 = { x: p2.x, y: p2.y + (dy >= 0 ? -lead : lead) };
  return `M ${p1.x} ${p1.y} C ${c1.x} ${c1.y} ${c2.x} ${c2.y} ${p2.x} ${p2.y}`;
}

function subFlatEdge(p1, p2, entry = 14, exit = 18) {
  const dy = p2.y - p1.y;
  const ady = Math.abs(dy);
  if (ady < 1) {
    const adx = Math.abs(p2.x - p1.x);
    if (adx < 1) return `M ${p1.x} ${p1.y}`;
    const dir = p2.x >= p1.x ? 1 : -1;
    const total = entry + exit;
    const max = adx - 1;
    let e = entry, x = exit;
    if (total > max) { e = entry * max / total; x = exit * max / total; }
    const q1x = p1.x + dir * e;
    const q2x = p2.x - dir * x;
    const midX = (q1x + q2x) / 2;
    return `M ${p1.x} ${p1.y} L ${q1x} ${p1.y} C ${midX} ${p1.y} ${midX} ${p2.y} ${q2x} ${p2.y} L ${p2.x} ${p2.y}`;
  }
  const dir = dy >= 0 ? 1 : -1;
  const total = entry + exit;
  const max = ady - 1;
  let e = entry, x = exit;
  if (total > max) { e = entry * max / total; x = exit * max / total; }
  const q1y = p1.y + dir * e;
  const q2y = p2.y - dir * x;
  const midY = (q1y + q2y) / 2;
  return `M ${p1.x} ${p1.y} L ${p1.x} ${q1y} C ${p1.x} ${midY} ${p2.x} ${midY} ${p2.x} ${q2y} L ${p2.x} ${p2.y}`;
}

// A pure-vertical segment between two boxes, direction-aware: if the source
// sits below the target (flipped FPN-up flow), it leaves the source's top and
// enters the target's bottom; otherwise it's the usual top-to-bottom drop.
function vSegment(pa, pb, x) {
  const srcBelow = pa.y > pb.y;
  const y1 = srcBelow ? pa.y : pa.y + pa.h;
  const y2 = srcBelow ? pb.y + pb.h : pb.y;
  return { d: `M ${x} ${y1} L ${x} ${y2}`, mid: { x, y: (y1 + y2) / 2 } };
}

// Compute the rendered path for one internal sub-edge. Returns
// { path, accent, label, labelPos } (labelPos may be null). All coordinates
// are local to the expanded region (0-origin).
function subEdgePath(edge, layout, stairs, nodesById) {
  const [a, b, label] = edge;
  const pa = layout.positions[a], pb = layout.positions[b];
  if (!pa || !pb) return null;
  const aNode = nodesById.get(a);
  const bNode = nodesById.get(b);
  const aIsTap = stairs.tapInStaircase.has(a);
  const bIsTap = stairs.tapInStaircase.has(b);
  const bIsMerge = stairs.mergeIds.has(b);
  const isSkip = layout.skipEdges?.has(`${a}->${b}`);
  const isFanInTap = layout.fanInTapEdges?.has(`${a}->${b}`);
  const isFanOutTap = layout.fanOutTapEdges?.has(`${a}->${b}`);

  if (isFanInTap) {
    const seg = vSegment(pa, pb, pa.x + pa.w / 2);
    return { path: seg.d, accent: false, label: null, labelPos: null };
  }
  if (isFanOutTap) {
    const seg = vSegment(pa, pb, pb.x + pb.w / 2);
    return { path: seg.d, accent: false, label, labelPos: { x: seg.mid.x + 6, y: seg.mid.y + 4 } };
  }
  if (isSkip) {
    const lane = layout.skipLane?.get(`${a}->${b}`) ?? 0;
    const seg = vSegment(pa, pb, pa.x + SKIP_OFFSET + lane * SKIP_LANE_W);
    return { path: seg.d, accent: true, label: null, labelPos: null };
  }
  if (aIsTap && bIsMerge) {
    const seg = vSegment(pa, pb, pa.x + pa.w / 2);
    return { path: seg.d, accent: false, label: null, labelPos: null };
  }
  if (aIsTap && bIsTap) {
    const off1 = stairs.tapOffset.get(a) ?? 0;
    const off2 = stairs.tapOffset.get(b) ?? 0;
    if (off2 > off1) {
      const p1 = { x: pa.x + pa.w, y: pa.y + pa.h / 2 };
      const p2 = { x: pb.x + pb.w / 2, y: pb.y > pa.y ? pb.y : pb.y + pb.h };
      return { path: rightAngleEdge(p1, p2), accent: false, label: null, labelPos: null };
    }
  }
  if (stairs.mergeIds.has(a) && !bIsTap) {
    const seg = vSegment(pa, pb, pb.x + pb.w / 2);
    return { path: seg.d, accent: false, label: null, labelPos: null };
  }

  const aBottom = { x: pa.x + pa.w / 2, y: pa.y + pa.h };
  const bTop    = { x: pb.x + pb.w / 2, y: pb.y };
  const aRight  = { x: pa.x + pa.w, y: pa.y + pa.h / 2 };
  const bLeft   = { x: pb.x, y: pb.y + pb.h / 2 };
  let p1, p2;
  if (Math.abs(pa.y - pb.y) < 1) {
    if (pb.x > pa.x) { p1 = aRight; p2 = bLeft; }
    else             { p1 = { x: pa.x, y: pa.y + pa.h / 2 }; p2 = { x: pb.x + pb.w, y: pb.y + pb.h / 2 }; }
  } else if (pb.y > pa.y) {
    p1 = aBottom; p2 = bTop;
  } else {
    p1 = { x: pa.x + pa.w / 2, y: pa.y };
    p2 = { x: pb.x + pb.w / 2, y: pb.y + pb.h };
  }
  return {
    path: subFlatEdge(p1, p2),
    accent: false,
    label,
    labelPos: label != null && label !== '' ? { x: (p1.x + p2.x) / 2 + 6, y: (p1.y + p2.y) / 2 + 4 } : null,
  };
}

// ===================== render styling (shared with graph-v2) ===============
const SUB_KIND_COLORS = {
  io:     { fill: '#e2e8f0', border: '#94a3b8', text: '#1e293b' },
  mod:    { fill: '#dcfce7', border: '#22c55e', text: '#14532d' },
  module: { fill: '#dcfce7', border: '#22c55e', text: '#14532d' },
  op:     { fill: '#fed7aa', border: '#fb923c', text: '#7c2d12' },
  cat:    { fill: '#fde68a', border: '#f59e0b', text: '#78350f' },
  split:  { fill: '#dbeafe', border: '#3b82f6', text: '#1e3a8a' },
};

function subFormatShape(sh) {
  if (!sh) return '';
  if (Array.isArray(sh) && sh.length && typeof sh[0] === 'number') return sh.join('×');
  return '';
}

// ===================== inner containers (ported) ===========================
// When a sub-node is recursively expanded, its children should live inside a
// labeled rectangle so the user can see the grouping (and click the label to
// collapse). This is spec-viewer's `computeContainers`, lightly adapted.
const INNER_PAD_X      = 14;
const INNER_PAD_TOP    = 22;
const INNER_PAD_BOTTOM = 12;

function _arrayEq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function _containerLabel(path, pathClasses) {
  if (path.length === 0) return null;
  const key = path.join('/');
  if (pathClasses && pathClasses[key]) return pathClasses[key];
  const last = path[path.length - 1];
  const m = last.match(/^(\d+)_(.+)$/);
  return m ? m[2] : last;
}

function computeInnerContainers(aggNodes, positions, pathClasses) {
  const pathSet = new Map();
  let any = false;
  for (const n of aggNodes) {
    if (!n.containerPath || n.containerPath.length === 0) continue;
    any = true;
    let p = n.containerPath.slice();
    pathSet.set(p.join('/'), p);
    while (p.length > 1) {
      p = p.slice(0, -1);
      pathSet.set(p.join('/'), p);
    }
  }
  if (!any) return [];

  const sorted = [...pathSet.values()].sort((a, b) => b.length - a.length);
  const bboxByKey = new Map();

  for (const p of sorted) {
    const key = p.join('/');
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of aggNodes) {
      if (!n.containerPath) continue;
      if (!_arrayEq(n.containerPath, p)) continue;
      const pos = positions[n.id];
      if (!pos) continue;
      minX = Math.min(minX, pos.x); minY = Math.min(minY, pos.y);
      maxX = Math.max(maxX, pos.x + pos.w); maxY = Math.max(maxY, pos.y + pos.h);
    }
    for (const [otherKey, ob] of bboxByKey) {
      const op = otherKey === '' ? [] : otherKey.split('/');
      if (op.length !== p.length + 1) continue;
      let isChild = true;
      for (let i = 0; i < p.length; i++) {
        if (op[i] !== p[i]) { isChild = false; break; }
      }
      if (!isChild) continue;
      minX = Math.min(minX, ob.x); minY = Math.min(minY, ob.y);
      maxX = Math.max(maxX, ob.x + ob.w); maxY = Math.max(maxY, ob.y + ob.h);
    }
    if (minX === Infinity) continue;
    bboxByKey.set(key, {
      x: minX - INNER_PAD_X,
      y: minY - INNER_PAD_TOP,
      w: (maxX - minX) + 2 * INNER_PAD_X,
      h: (maxY - minY) + INNER_PAD_TOP + INNER_PAD_BOTTOM,
      path: p,
      pathKey: key,
      depth: p.length,
      label: _containerLabel(p, pathClasses) || key,
    });
  }
  return [...bboxByKey.values()].sort((a, b) => a.depth - b.depth);
}

// ============================ buildExpansion ===============================
// idx              — L1 block index to expand
// opts.flip        — col-1 (FPN-up) mirrors internal flow bottom-to-top.
// opts.expansions  — Set<string> of sub-paths to recursively expand
//                    (e.g. {"m"} reveals m's children individually).
//
// Returns null when the block can't be expanded (no fx spec, e.g. Detect).
// Otherwise returns the laid-out internal sub-graph in LOCAL coords (0-origin):
//   { subNodes, subEdges, innerContainers, entryNodes, exitNodes, regionW, regionH, flip }
const REGION_PAD_X      = 22;
const REGION_PAD_TOP    = 30;
const REGION_PAD_BOTTOM = 22;

function buildExpansion(idx, opts) {
  const spec = window.YV_SPEC;
  if (!spec) return null;
  const instance = spec.instances.find(i => i.idx === idx);
  if (!instance) return null;
  const blockSpec = spec.specs[instance.spec_id];
  if (!blockSpec || blockSpec.derivation_method !== 'fx') return null;

  const flip = !!(opts && opts.flip);

  // '' is always expanded — that's depth-1. Add any sub-paths the caller
  // wants peeled deeper.
  const expansions = new Set(['']);
  if (opts && opts.expansions) for (const p of opts.expansions) expansions.add(p);

  const visibility = opVisibilityForLevel(2);
  const pre = preprocessGraph(blockSpec.graph, visibility, instance.shapes_by_node || {});
  const agg = aggregateWithExpansions(pre, instance.shapes_by_node || {}, expansions, blockSpec.path_classes);
  const stairs = detectStaircases(agg.nodes, agg.edges);
  const layout = autoLayout(agg.nodes, agg.edges, stairs);

  // Identify the I/O nodes. placeholder = io node with no incoming agg edge;
  // output = io node with no outgoing agg edge.
  const ioNodes = agg.nodes.filter(n => (n.subkind || n.kind) === 'io');
  const inDeg = new Map(), outDeg = new Map();
  agg.nodes.forEach(n => { inDeg.set(n.id, 0); outDeg.set(n.id, 0); });
  agg.edges.forEach(([s, t]) => {
    outDeg.set(s, (outDeg.get(s) || 0) + 1);
    inDeg.set(t, (inDeg.get(t) || 0) + 1);
  });
  let placeholderId = null, outputId = null;
  for (const n of ioNodes) {
    if ((inDeg.get(n.id) || 0) === 0) placeholderId = n.id;
    else if ((outDeg.get(n.id) || 0) === 0) outputId = n.id;
  }
  const ioIds = new Set(ioNodes.map(n => n.id));

  // entry = internal nodes the placeholder fed into; exit = internal nodes
  // feeding the output.
  const entryIds = [];
  const exitIds = [];
  for (const [s, t] of agg.edges) {
    if (s === placeholderId && !ioIds.has(t)) entryIds.push(t);
    if (t === outputId && !ioIds.has(s)) exitIds.push(s);
  }

  // Internal nodes + edges (I/O stripped).
  const internalNodes = agg.nodes.filter(n => !ioIds.has(n.id));
  const internalEdges = agg.edges.filter(([s, t]) => !ioIds.has(s) && !ioIds.has(t));

  if (internalNodes.length === 0) return null;

  // Normalize: translate so the internal bbox sits at (REGION_PAD_X,
  // REGION_PAD_TOP). positions in `layout` are mutated copies — clone first.
  const pos = {};
  for (const n of internalNodes) {
    const p = layout.positions[n.id];
    pos[n.id] = { x: p.x, y: p.y, w: p.w, h: p.h };
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of internalNodes) {
    const p = pos[n.id];
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x + p.w); maxY = Math.max(maxY, p.y + p.h);
  }
  const bboxW = maxX - minX, bboxH = maxY - minY;
  const regionW = bboxW + 2 * REGION_PAD_X;
  const regionH = bboxH + REGION_PAD_TOP + REGION_PAD_BOTTOM;
  for (const n of internalNodes) {
    const p = pos[n.id];
    p.x = p.x - minX + REGION_PAD_X;
    p.y = p.y - minY + REGION_PAD_TOP;
  }

  // Flip: mirror y so internal flow runs bottom-to-top.
  if (flip) {
    for (const n of internalNodes) {
      const p = pos[n.id];
      p.y = regionH - p.y - p.h;
    }
  }

  // Sub-edge paths — recompute against the final (normalized + maybe flipped)
  // positions. subEdgePath reads layout.positions, so point it at `pos`.
  const routingLayout = {
    positions: pos,
    skipEdges: layout.skipEdges,
    skipLane: layout.skipLane,
    fanInTapEdges: layout.fanInTapEdges,
    fanOutTapEdges: layout.fanOutTapEdges,
  };
  const nodesById = new Map(internalNodes.map(n => [n.id, n]));

  const subNodes = internalNodes.map(n => ({
    id: n.id,
    label: n.label,
    subkind: n.subkind || n.kind,
    shape: n.shape,
    pathKey: n.pathKey,
    expandable: !!n.expandable,
    x: pos[n.id].x, y: pos[n.id].y, w: pos[n.id].w, h: pos[n.id].h,
  }));

  const subEdges = [];
  for (const edge of internalEdges) {
    const routed = subEdgePath(edge, routingLayout, stairs, nodesById);
    if (!routed) continue;
    subEdges.push({ src: edge[0], dst: edge[1], ...routed });
  }

  // Inner containers — rectangles around recursively-expanded sub-paths.
  // Compute from the already-normalized & flipped sub-node positions so they
  // sit in the same local coordinate system as everything else.
  const subPosForContainers = {};
  for (const sn of subNodes) {
    subPosForContainers[sn.id] = { x: sn.x, y: sn.y, w: sn.w, h: sn.h };
  }
  const innerContainers = computeInnerContainers(internalNodes, subPosForContainers, blockSpec.path_classes);

  const byId = new Map(subNodes.map(n => [n.id, n]));
  const entryNodes = entryIds.map(id => byId.get(id)).filter(Boolean);
  const exitNodes = exitIds.map(id => byId.get(id)).filter(Boolean);

  return {
    subNodes, subEdges, innerContainers,
    entryNodes, exitNodes,
    regionW, regionH,
    flip,
  };
}

window.YVV2 = window.YVV2 || {};
window.YVV2.buildExpansion   = buildExpansion;
window.YVV2.SUB_KIND_COLORS  = SUB_KIND_COLORS;
window.YVV2.subFormatShape   = subFormatShape;
window.YVV2.opShortName      = opShortName;
