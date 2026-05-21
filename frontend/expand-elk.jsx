// yolovex — ELK-driven in-Region layout (ADR-0001).
//
// Parallel to expand.jsx's buildExpansion(), but the GEOMETRY back-half
// (autoLayout / subEdgePath / computeInnerContainers / the INNER_PAD inset
// hack / the bespoke beziers) is replaced by ELK (elkjs, loaded as the global
// `ELK` from vendor/elk.bundled.js). The graph-SEMANTIC front-half
// (preprocess / aggregate / classify / sizing) is reused verbatim via
// window.YV._graphSem.
//
// Returns the SAME contract as buildExpansion so graph-elk.jsx renders it with
// no change:
//   { subNodes, subEdges, innerContainers, entryNodes, exitNodes,
//     regionW, regionH, flip }
// ...but ASYNC (ELK's layout() is promise-based). graph-elk.jsx awaits it.

// One ELK instance for the whole page.
const _elk = (typeof ELK !== 'undefined') ? new ELK() : null;

// Nesting padding for inner containers (mirror expand.jsx's INNER_PAD_*).
const ELK_INNER_PAD_X = 14;
const ELK_INNER_PAD_TOP = 30;
const ELK_INNER_PAD_BOTTOM = 12;

function _elkContainerLabel(path, pathClasses) {
  if (!path.length) return null;
  const key = path.join('/');
  if (pathClasses && pathClasses[key]) return pathClasses[key];
  const last = path[path.length - 1];
  const m = last.match(/^(\d+)_(.+)$/);
  return m ? m[2] : last;
}

// Edge label text: prefer the aggregate's own label (e.g. chunk/getitem index),
// else annotate dimension-bearing sources (split / shape / struct) with their
// output shape. Modest by design — we don't label every edge.
function _elkEdgeLabel(lbl, sNode) {
  if (lbl != null && lbl !== '') return String(lbl);
  if (sNode && (sNode.subkind === 'split' || sNode.subkind === 'shape' || sNode.subkind === 'struct')) {
    const sh = window.YV.subFormatShape && window.YV.subFormatShape(sNode.shape);
    if (sh) return sh;
  }
  return null;
}

async function buildExpansionELK(idx, opts) {
  const spec = window.YV_SPEC;
  if (!spec || !_elk) return null;
  const gs = window.YV._graphSem;
  if (!gs) return null;

  const instance = spec.instances.find(i => i.idx === idx);
  if (!instance) return null;
  const blockSpec = spec.specs[instance.spec_id];
  if (!blockSpec || blockSpec.derivation_method !== 'fx') return null;

  const flip = !!(opts && opts.flip);
  const expansions = new Set(['']);
  if (opts && opts.expansions) for (const p of opts.expansions) expansions.add(p);

  // ── Reused semantic front-half ──────────────────────────────────────────
  const visibility = gs.opVisibilityForLevel(2);
  const pre = gs.preprocessGraph(blockSpec.graph, visibility, instance.shapes_by_node || {});
  const agg = gs.aggregateWithExpansions(
    pre, instance.shapes_by_node || {}, expansions, blockSpec.path_classes,
  );

  // I/O + entry/exit + internal (identical logic to legacy buildExpansion).
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
  const entryIds = [], exitIds = [];
  for (const [s, t] of agg.edges) {
    if (s === placeholderId && !ioIds.has(t)) entryIds.push(t);
    if (t === outputId && !ioIds.has(s)) exitIds.push(s);
  }
  const internalNodes = agg.nodes.filter(n => !ioIds.has(n.id));
  const internalEdges = agg.edges.filter(([s, t]) => !ioIds.has(s) && !ioIds.has(t));
  if (internalNodes.length === 0) return null;

  const byId = new Map(internalNodes.map(n => [n.id, n]));

  // ── Build the ELK hierarchy (compound layout replaces the inset hack) ─────
  const root = {
    id: 'root', _path: [], children: [], edges: [],
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': flip ? 'UP' : 'DOWN',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      'elk.edgeRouting': 'ORTHOGONAL',
      // Left-align the spine (branches extend rightward) instead of centering —
      // Brandes-Köpf fixed alignment. RIGHTDOWN is the corner that puts the
      // spine on the LEFT in our DOWN orientation (LEFTUP mirrored it).
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.layered.nodePlacement.bk.fixedAlignment': 'RIGHTDOWN',
      'elk.layered.spacing.nodeNodeBetweenLayers': '44',
      'elk.spacing.nodeNode': '40',
      'elk.spacing.edgeNode': '18',
      'elk.spacing.edgeEdge': '12',
      'elk.layered.spacing.edgeNodeBetweenLayers': '18',
      'elk.padding': `[top=${gs.REGION_PAD_TOP},left=${gs.REGION_PAD_X},bottom=${gs.REGION_PAD_BOTTOM},right=${gs.REGION_PAD_X}]`,
    },
  };

  const containerNodes = new Map();   // path-key -> elk container node
  const ensureContainer = (path) => {
    if (!path || path.length === 0) return root;
    const key = path.join('/');
    if (containerNodes.has(key)) return containerNodes.get(key);
    const node = {
      id: 'C::' + key, _path: path.slice(), children: [], edges: [],
      layoutOptions: {
        'elk.direction': flip ? 'UP' : 'DOWN',
        'elk.padding': `[top=${ELK_INNER_PAD_TOP},left=${ELK_INNER_PAD_X},bottom=${ELK_INNER_PAD_BOTTOM},right=${ELK_INNER_PAD_X}]`,
      },
    };
    containerNodes.set(key, node);
    ensureContainer(path.slice(0, -1)).children.push(node);
    return node;
  };

  // In-degree, to widen merge-style nodes (#3) so multiple incoming edges enter
  // across a wider top edge — fewer overlapping verticals, neater offsets. Same
  // idea as the legacy staircase span, but ELK does the edge spreading.
  const FIXED_SUBKINDS = new Set(['arith', 'shape', 'attr', 'struct']);
  const WIDEN_STEP = 64;
  // Widen ONLY merge nodes (in-degree ≥ 2) so multiple inputs spread across the
  // top edge. Fan-out widening is intentionally NOT done — it made the maxpool
  // chain visually uneven; revisit later.
  const indeg = new Map();
  for (const [, t] of internalEdges) indeg.set(t, (indeg.get(t) || 0) + 1);

  for (const n of internalNodes) {
    const sz = gs.nodeSize(n, null);
    let w = sz.w;
    const din = indeg.get(n.id) || 0;
    if (din >= 2 && !FIXED_SUBKINDS.has(n.subkind)) w = sz.w + (din - 1) * WIDEN_STEP;
    ensureContainer(n.containerPath || []).children.push({ id: n.id, width: w, height: sz.h });
  }

  // Longest common prefix of two container paths — the LCA container an edge
  // should live in (proper ELK hierarchical-edge placement; fixes #5/#6 where
  // intra-container edges declared at root were dropped / mis-routed).
  const lcaPath = (a, b) => {
    a = a || []; b = b || [];
    const out = [];
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (a[i] === b[i]) out.push(a[i]); else break;
    }
    return out;
  };

  let ei = 0;
  for (const e of internalEdges) {
    const [s, t, lbl] = e;
    const sNode = byId.get(s), tNode = byId.get(t);
    const container = ensureContainer(lcaPath(sNode && sNode.containerPath, tNode && tNode.containerPath));
    const edge = { id: `e${ei++}`, sources: [s], targets: [t] };
    const text = _elkEdgeLabel(lbl, sNode);
    if (text) edge.labels = [{ text, width: Math.max(10, text.length * 6 + 4), height: 12 }];
    (container.edges = container.edges || []).push(edge);
  }

  // ── Run ELK ───────────────────────────────────────────────────────────────
  let res;
  try {
    res = await _elk.layout(root);
  } catch (err) {
    console.error(`[yolovex/elk] layout failed for block ${idx}:`, err);
    return null;   // degrade gracefully — block stays collapsed, no crash
  }

  // Accumulate absolute coords (ELK child + edge coords are relative to their
  // containing node's origin). We collect sub-edges from EVERY node in the tree
  // — not just root — because ELK re-parents intra-container edges down into
  // their container, offsetting each by that container's absolute origin.
  const abs = new Map();
  const containerAbs = [];
  const subEdges = [];
  const walk = (node, ox, oy) => {
    const x = (node.x || 0) + ox, y = (node.y || 0) + oy;
    if (node._path && node._path.length > 0) {
      containerAbs.push({ path: node._path, x, y, w: node.width, h: node.height });
    }
    if (byId.has(node.id)) abs.set(node.id, { x, y, w: node.width, h: node.height });
    for (const e of (node.edges || [])) {
      const sec = e.sections && e.sections[0];
      if (!sec) continue;
      const pts = [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint];
      const d = 'M ' + pts.map(pt => `${pt.x + x} ${pt.y + y}`).join(' L ');
      let label = null, labelPos = null;
      if (e.labels && e.labels[0] && e.labels[0].text) {
        label = e.labels[0].text;
        const lb = e.labels[0];
        if (lb.x != null) labelPos = { x: lb.x + x, y: lb.y + y };
      }
      subEdges.push({ src: e.sources[0], dst: e.targets[0], path: d, accent: false, label, labelPos });
    }
    if (node.children) for (const c of node.children) walk(c, x, y);
  };
  walk(res, 0, 0);

  const subNodes = internalNodes.map(n => {
    const sz = gs.nodeSize(n, null);
    const p = abs.get(n.id) || { x: 0, y: 0, w: sz.w, h: sz.h };
    return {
      id: n.id, label: n.label, subkind: n.subkind || n.kind,
      targetClass: n.targetClass || null, shape: n.shape, pathKey: n.pathKey,
      expandable: !!n.expandable, members: n.members || [n.id],
      x: p.x, y: p.y, w: p.w, h: p.h,
    };
  });

  const innerContainers = containerAbs.map(c => ({
    x: c.x, y: c.y, w: c.w, h: c.h,
    path: c.path, pathKey: c.path.join('/'), depth: c.path.length,
    label: _elkContainerLabel(c.path, blockSpec.path_classes) || c.path.join('/'),
  })).sort((a, b) => a.depth - b.depth);

  const byIdSub = new Map(subNodes.map(n => [n.id, n]));
  const entryNodes = entryIds.map(id => byIdSub.get(id)).filter(Boolean);
  const exitNodes = exitIds.map(id => byIdSub.get(id)).filter(Boolean);

  return {
    subNodes, subEdges, innerContainers, entryNodes, exitNodes,
    regionW: res.width, regionH: res.height, flip,
  };
}

window.YV = window.YV || {};
window.YV.buildExpansionELK = buildExpansionELK;
window.YV._elk = _elk;
