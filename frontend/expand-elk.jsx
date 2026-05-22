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
  // Shared layout knobs. CRITICAL: under `hierarchyHandling: INCLUDE_CHILDREN`,
  // ELK does NOT inherit these into child containers — a nested Region falls back
  // to ELK defaults unless we set them on it too. So this single object is spread
  // into the root AND every container (see ensureContainer), giving identical
  // layout at every nesting depth. Tune spacing/placement/order HERE, once.
  // (Per-node bits — `elk.padding`, and the graph-global `elk.algorithm` /
  // `elk.hierarchyHandling` — are set separately below.)
  const ELK_LAYOUT_OPTS = {
    'elk.direction': flip ? 'UP' : 'DOWN',
    'elk.edgeRouting': 'ORTHOGONAL',
    // Brandes-Köpf node placement. `fixedAlignment` picks the alignment corner
    // and controls which way a branching chain LEANS as it descends. Cycle
    // these to flip the maxpool/branch drift left↔right:
    //   LEFTDOWN | RIGHTDOWN | LEFTUP | RIGHTUP | BALANCED
    'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
    'elk.layered.nodePlacement.bk.fixedAlignment': 'LEFTUP',
    // Respect input (fx execution) order when choosing in-layer left/right, so
    // a Split's first-declared branch (the processing chain — e.g. SPPF's
    // maxpools, C3k2's C3k) sits on the LEFT instead of ELK freely swapping it
    // to minimise crossings. ELK re-routes edges automatically to match.
    // Set to 'NONE' to revert to pure crossing-minimised order.
    'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
    'elk.layered.spacing.nodeNodeBetweenLayers': '50',   // vertical gap between Layers
    'elk.spacing.nodeNode': '40',                         // horizontal gap, same-Layer nodes
    // Minimum gap between adjacent ports on a node side. Raising this spreads
    // multiple Merge inputs further apart across the node's edge (ELK widens the
    // node if needed). Default is ~10 — bump to taste.
    'elk.spacing.portPort': '10',
    'elk.spacing.edgeNode': '18',
    'elk.spacing.edgeEdge': '12',
    'elk.layered.spacing.edgeNodeBetweenLayers': '18',
  };

  const root = {
    id: 'root', _path: [], children: [], edges: [],
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      ...ELK_LAYOUT_OPTS,
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
        ...ELK_LAYOUT_OPTS,
        'elk.padding': `[top=${ELK_INNER_PAD_TOP},left=${ELK_INNER_PAD_X},bottom=${ELK_INNER_PAD_BOTTOM},right=${ELK_INNER_PAD_X}]`,
      },
    };
    containerNodes.set(key, node);
    ensureContainer(path.slice(0, -1)).children.push(node);
    return node;
  };

  // Widen branching nodes so their multiple edges spread across a wider top /
  // bottom edge — fewer overlapping verticals, neater offsets, and ELK does the
  // edge spreading. Two independent knobs (tune these to adjust branch spacing):
  const FIXED_SUBKINDS = new Set(['arith', 'shape', 'attr', 'struct']);
  const WIDEN_STEP_IN = 135;    // fan-IN: per extra input on a merge (in-degree ≥ 2)
  const WIDEN_STEP_OUT = 30;   // fan-OUT: per extra output on a split (out-degree ≥ 2)
  const indeg = new Map(), outdeg = new Map();
  for (const [s, t] of internalEdges) {
    indeg.set(t, (indeg.get(t) || 0) + 1);
    outdeg.set(s, (outdeg.get(s) || 0) + 1);
  }

  for (const n of internalNodes) {
    const sz = gs.nodeSize(n, null);
    let w = sz.w;
    if (!FIXED_SUBKINDS.has(n.subkind)) {
      const din = indeg.get(n.id) || 0;
      const dout = outdeg.get(n.id) || 0;
      const boost = Math.max(
        din >= 2 ? (din - 1) * WIDEN_STEP_IN : 0,
        dout >= 2 ? (dout - 1) * WIDEN_STEP_OUT : 0,
      );
      w = sz.w + boost;
    }
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

  // elkjs NPEs ("Cannot read properties of undefined (reading 'a')") when
  // `considerModelOrder` is set on a CONTAINER whose boundary a cross-hierarchy
  // edge crosses — i.e. an edge (declared at an ancestor LCA) with an endpoint
  // inside the container. That boundary dummy/port breaks model-order indexing
  // for ANY strategy value. Such containers are common (every expanded sub-block
  // connects to the outside; an edge-less container is just the degenerate case).
  // Strip MO from every boundary-crossed container; keep it on the root (its
  // boundary can't be crossed) and on fully-internal containers so their branch
  // ordering still works. Verified against the real C2PSA+cv1 graph.
  const MODEL_ORDER_KEY = 'elk.layered.considerModelOrder.strategy';
  const stripMO = new Set();
  for (const [s, t] of internalEdges) {
    const sCP = (byId.get(s) || {}).containerPath || [];
    const tCP = (byId.get(t) || {}).containerPath || [];
    const lca = lcaPath(sCP, tCP).length;
    for (const cp of [sCP, tCP]) {
      for (let d = lca + 1; d <= cp.length; d++) stripMO.add(cp.slice(0, d).join('/'));
    }
  }
  for (const [key, node] of containerNodes) {
    if (stripMO.has(key) || !node.edges || node.edges.length === 0) {
      delete node.layoutOptions[MODEL_ORDER_KEY];
    }
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
