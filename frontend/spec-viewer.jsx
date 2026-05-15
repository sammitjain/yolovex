// Block Spec Viewer — renders one block's derived graph at a chosen fidelity
// level. Reads window.YV_SPEC (written by src/yolovex/block_spec.py) and runs
// the same path-based aggregation as block_spec.aggregate_at_depth, in JS.

const { useState, useMemo, useRef, useEffect } = React;

// === Staircase / op constants (design language from layout-l2.jsx) ===
const STAIR_FRAC = 0.65;   // tap[i].x = base + i * STAIR_FRAC * NODE_W
const SEQ_BEND   = 22;     // lead for right-angle bezier between consecutive taps

// fx targets we want to treat specially in rendering
const ARITH_OP_NAMES  = new Set(['add', 'mul', 'sub', 'truediv']);
const STRUCT_OP_NAMES = new Set(['getitem']);                       // collapse to edge labels (always)
const SHAPE_OP_NAMES  = new Set(['view', 'permute', 'transpose', 'reshape', 'flatten', 'squeeze', 'unsqueeze', 'contiguous']);

// === Op visibility per fidelity level ===
// Shape ops (view/permute/reshape/transpose/...) are shown as their own boxes
// (small dashed rectangles). They're real intermediate states a learner
// benefits from seeing, especially in Attention. getitem is still collapsed
// to an edge index label, get_attr is dropped entirely (it's metadata).
function opVisibilityForLevel(level) {
  return { hideShapeOps: false, hideGetAttr: true };
}

function opShortName(target) {
  if (!target) return '';
  const last = String(target).split('.').pop();
  return last.replace(/^_+|_+$/g, '');
}

// === Preprocessor: hide noise ops, collapse getitem into edge labels ===
//
// Operates on the raw fx graph. Drops every node we don't want to visualise
// (shape ops, get_attr, getitem) and reconnects edges over them, so a chain
// `A -> view -> B` collapses to `A -> B` and `chunk -> getitem(0) -> X`
// becomes `chunk --[0]--> X`. The output is a graph with the SAME node
// identifiers but fewer nodes and rewired edges.
function preprocessGraph(graph, visibility, shapes) {
  // Two distinct kinds of removal:
  //
  //  `dropped` — non-tensor / metadata nodes (ShapeProp shape === null).
  //    These are fx artifacts: `x.shape` access, integer extraction of
  //    B/C/H/W, scalar arithmetic feeding view/reshape's size args. They
  //    produce no tensor data flow and DON'T belong as edges or labels —
  //    we erase them and their incident edges completely. Without this,
  //    Attention's L4 view fills with scalar-args-as-skip-arrows that look
  //    like dangling skips to a tensor-data-flow reader.
  //
  //  `hidden` — tensor ops we collapse INTO edge labels: getitem (index)
  //    and shape ops (view/transpose/reshape/...). Their label parts
  //    accumulate as we walk past them ("2→reshape").
  const dropped = new Set();
  const hidden = new Set();
  const hiddenLabelPart = new Map();

  for (const n of graph.nodes) {
    if (n.op === 'placeholder' || n.op === 'output') continue;
    // Non-tensor: drop entirely.
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

  // Adjacency
  const succs = new Map();
  for (const n of graph.nodes) succs.set(n.name, []);
  for (const [s, t] of graph.edges) {
    succs.get(s)?.push(t);
  }

  // Walk past hidden nodes to find live successors, accumulating label parts.
  // Dropped nodes are full dead-ends — we don't traverse them or emit any edge
  // to or through them.
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

// === Aggregation by visible_path depth ===
//
// depth = level - 1. At depth d:
//   - call_module nodes with visible_path.length > d collapse to visible_path[:d]
//   - call_function/call_method nodes get an "enclosing depth" from where
//     '_op' appears in visible_path. If that depth >= d, the op is absorbed
//     into its enclosing module's group; otherwise it stays individual.
//   - I/O always stays individual.
//
// Top-level ops have visible_path = ['_op', ...], so opIdx = 0; they're
// individual at any depth >= 1 and only collapse at depth 0 (the L1 case,
// handled specially upstream).
function aggregateAtDepth(graph, shapes, depth, pathClasses) {
  const nameToGroup = {};
  const groups = new Map();
  const groupKind = new Map();
  const groupOrder = [];

  for (const n of graph.nodes) {
    const vpath = n.visible_path || n.path || [];
    let key, kind;

    if (n.op === 'call_module') {
      if (vpath.length > depth) {
        key = vpath.slice(0, depth).join('/');
        kind = 'mod';
      } else {
        key = n.name;
        kind = 'mod';
      }
    } else if (n.op === 'call_function' || n.op === 'call_method') {
      const opIdx = vpath.indexOf('_op');
      if (opIdx >= 0 && opIdx >= depth) {
        // Absorbed into the enclosing module's group at this depth.
        key = vpath.slice(0, depth).join('/');
        kind = 'mod';
      } else {
        key = n.name;
        kind = 'op';
      }
    } else if (n.op === 'placeholder' || n.op === 'output') {
      key = n.name;
      kind = 'io';
    } else if (n.op === 'get_attr') {
      key = n.name;
      kind = 'attr';
    } else {
      key = n.name;
      kind = 'mod';
    }

    nameToGroup[n.name] = key;
    if (!groups.has(key)) {
      groups.set(key, []);
      groupKind.set(key, kind);
      groupOrder.push(key);
    }
    groups.get(key).push(n);
  }

  const aggNodes = [];
  for (const key of groupOrder) {
    const members = groups.get(key);
    const last = members[members.length - 1];
    const shape = shapes ? shapes[last.name] : null;
    let label;

    if (members.length === 1 && members[0].name === key) {
      // Single-node group. Show the TYPE — the class for modules, the op name
      // for functions/methods — not the fx instance name (noise for learners).
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
      // Aggregated module group: label with its class (the type), falling back
      // to the path's last segment when pathClasses has no entry.
      const cls = pathClasses ? pathClasses[key] : null;
      const lastSeg = key.split('/').pop() || '';
      // Container children carry "<index>_<ClassName>" — strip the index.
      const containerMatch = lastSeg.match(/^(\d+)_(.+)$/);
      if (containerMatch) {
        label = containerMatch[2];
      } else if (cls) {
        label = `${cls}`;
      } else {
        label = lastSeg || key;
      }
    }

    // --- containerPath: which parent container this aggregated node lives in.
    // Nodes in the same container share a visible_path prefix; the container's
    // own path is one segment shallower than the node's effective path.
    //   - aggregated module group  : path = vpath[:depth-1]
    //   - individual module        : path = vpath[:-1]
    //   - individual op (has _op)  : path = vpath[:opIdx]  (its enclosing module)
    //   - io / get_attr            : null (floats outside containers)
    let containerPath = null;
    const first = members[0];
    const fvpath = first.visible_path || first.path || [];
    const fkind = groupKind.get(key);
    if (first.op !== 'placeholder' && first.op !== 'output' && first.op !== 'get_attr') {
      if (fkind === 'mod') {
        if (fvpath.length > depth) {
          containerPath = fvpath.slice(0, Math.max(0, depth - 1));
        } else {
          containerPath = fvpath.slice(0, Math.max(0, fvpath.length - 1));
        }
      } else if (fkind === 'op') {
        const opIdx = fvpath.indexOf('_op');
        containerPath = opIdx >= 0 ? fvpath.slice(0, opIdx) : [];
      }
    }

    aggNodes.push({
      id: key,
      label,
      kind: groupKind.get(key),
      shape,
      members: members.map(m => m.name),
      containerPath,
    });
  }

  // Edges between distinct groups, merging by (src,tgt) so parallel arrows
  // (e.g. split → matmul for q and k separately) become ONE edge with a
  // combined label like "[0→transpose, 1]". Avoids overlapping arrows that
  // can't be visually distinguished anyway.
  const edgesMerged = new Map();   // "sg\x00tg" -> { s, t, labels:[] }
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

  // Classify each node for rendering
  for (const n of aggNodes) {
    const orig = graph.nodes.find(g => g.name === n.id);
    n.subkind = classifySubkind(orig, n);
  }

  return { nodes: aggNodes, edges: aggEdges };
}

// === L1 special case: single block node + I/O ===
function buildL1View(spec, instance) {
  const inputNode = spec.graph.nodes.find(n => n.op === 'placeholder');
  const outputNode = spec.graph.nodes.find(n => n.op === 'output');
  const inputName = inputNode?.name || 'x';
  const outputName = outputNode?.name || 'output';
  const blockMembers = spec.graph.nodes
    .filter(n => n.op !== 'placeholder' && n.op !== 'output')
    .map(n => n.name);

  // L1 nodes never live inside a container — the "block" node IS the block.
  return {
    nodes: [
      { id: inputName, label: inputName, kind: 'io', subkind: 'io', shape: instance.input_shape, members: [inputName], containerPath: null },
      { id: 'block',  label: spec.class_name, kind: 'mod', subkind: 'module', shape: instance.output_shape, members: blockMembers, containerPath: null },
      { id: outputName, label: outputName, kind: 'io', subkind: 'io', shape: instance.output_shape, members: [outputName], containerPath: null },
    ],
    edges: [
      [inputName, 'block'],
      ['block', outputName],
    ],
  };
}

// === Parent containers ===
// At any level past L1, modules whose internals are visible deserve a labeled
// rectangle wrapping their children (matches the design in layout-l2.jsx for
// the architecture-wide view). Container bboxes are computed in depth-DESC
// order so each outer container's bbox is the union of its inner containers'
// bboxes (already padded) plus its own pad — guaranteeing clean nesting.
const CONTAINER_PAD_X      = 14;
const CONTAINER_PAD_TOP    = 24;   // room for the container's label
const CONTAINER_PAD_BOTTOM = 14;

function containerLabel(path, pathClasses, blockClass) {
  if (path.length === 0) return blockClass;
  const key = path.join('/');
  if (pathClasses && pathClasses[key]) return pathClasses[key];
  const last = path[path.length - 1];
  const m = last.match(/^(\d+)_(.+)$/);
  return m ? m[2] : last;
}

function arrayEq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function computeContainers(aggNodes, positions, pathClasses, blockClass) {
  // Collect every unique container path that appears, plus all of its
  // ancestor prefixes (so a nested chain like ['0_PSABlock','attn'] also
  // registers ['0_PSABlock'] and []).
  const pathSet = new Map();   // key string -> path array
  let anyContained = false;
  for (const n of aggNodes) {
    if (!n.containerPath) continue;
    anyContained = true;
    let p = n.containerPath.slice();
    pathSet.set(p.join('/'), p);
    while (p.length > 0) {
      p = p.slice(0, -1);
      pathSet.set(p.join('/'), p);
    }
  }
  if (!anyContained) return [];

  // Process deepest first so outer = union(inner_bboxes) + outer pad.
  const sorted = [...pathSet.values()].sort((a, b) => b.length - a.length);
  const bboxByKey = new Map();

  for (const p of sorted) {
    const key = p.join('/');
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    // Nodes whose containerPath is exactly this path
    for (const n of aggNodes) {
      if (!n.containerPath) continue;
      if (!arrayEq(n.containerPath, p)) continue;
      const pos = positions[n.id];
      if (!pos) continue;
      minX = Math.min(minX, pos.x);
      minY = Math.min(minY, pos.y);
      maxX = Math.max(maxX, pos.x + pos.w);
      maxY = Math.max(maxY, pos.y + pos.h);
    }
    // Already-computed child containers (one segment deeper, this as prefix)
    for (const [otherKey, ob] of bboxByKey) {
      const op = otherKey === '' ? [] : otherKey.split('/');
      if (op.length !== p.length + 1) continue;
      let isChild = true;
      for (let i = 0; i < p.length; i++) {
        if (op[i] !== p[i]) { isChild = false; break; }
      }
      if (!isChild) continue;
      minX = Math.min(minX, ob.x);
      minY = Math.min(minY, ob.y);
      maxX = Math.max(maxX, ob.x + ob.w);
      maxY = Math.max(maxY, ob.y + ob.h);
    }

    if (minX === Infinity) continue;
    bboxByKey.set(key, {
      x: minX - CONTAINER_PAD_X,
      y: minY - CONTAINER_PAD_TOP,
      w: (maxX - minX) + 2 * CONTAINER_PAD_X,
      h: (maxY - minY) + CONTAINER_PAD_TOP + CONTAINER_PAD_BOTTOM,
      path: p,
      depth: p.length,
      label: containerLabel(p, pathClasses, blockClass),
    });
  }

  // Render order: shallow-first so deeper containers paint on top.
  return [...bboxByKey.values()].sort((a, b) => a.depth - b.depth);
}

// Op subkind drives shape/size in render. Returns one of:
//   'module'   — call_module or aggregated module group (default rectangle)
//   'arith'    — operator.add / mul / etc. (circle)
//   'cat'      — torch.cat (potentially a staircase merge — sized later)
//   'split'    — Tensor.chunk / .split (fork)
//   'shape'    — view/permute/reshape (small dim)
//   'struct'   — getitem (small, will be collapsed if possible)
//   'attr'     — get_attr (small dim, distinct color)
//   'io'       — placeholder / output
//   'op'       — other call_function / call_method (default op rect)
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

// === Staircase detection ===
// A staircase is a `cat` (merge) whose predecessors form a chain of taps.
// e.g. SPPF: cat <- cv1, m, m_1, m_2  with cv1→m→m_1→m_2 internally.
// Returns { stairs: [{mergeId, taps: [id...]}], tapOffset: Map<id, number> }
function detectStaircases(nodes, edges) {
  const inMap = new Map(), outMap = new Map();
  for (const n of nodes) { inMap.set(n.id, []); outMap.set(n.id, []); }
  for (const e of edges) {
    const [s, t] = e;
    inMap.get(t)?.push(s);
    outMap.get(s)?.push(t);
  }
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  const stairs = [];
  const tapOffset = new Map();
  const tapInStaircase = new Set();
  const mergeIds = new Set();

  for (const n of nodes) {
    if (n.subkind !== 'cat') continue;
    const preds = inMap.get(n.id) || [];
    if (preds.length < 2) continue;
    // Find chain order among the preds.
    const predSet = new Set(preds);
    // The chain start: a pred whose incoming nodes (from predSet) are empty
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
    // Strict chain only. If the chain didn't span every predecessor, this is
    // NOT a staircase — leave the non-chaining preds to default routing (the
    // skip detector will catch them as rank-gap edges and route through the
    // corridor).
    if (ordered.length < preds.length) continue;
    if (ordered.length < 2) continue;

    // Strict chain also requires each non-start tap's ONLY predecessor to be
    // the previous tap. Without this incoming check, a residual-bypass pattern
    // (C3k2's chunk→add→cat, where `add` also takes the Bottleneck output) is
    // mistaken for a staircase: `add` gets a STAIR_FRAC offset while chunk→add
    // is still routed as a skip, leaving the skip dangling. A real staircase
    // tap (SPPF's maxpools) is fed solely by its predecessor in the chain.
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

// === Fan-in merge detection (generalization of staircase to non-chain) ===
//
// A fan-in merge is a `cat` whose predecessors did NOT form a strict chain
// (so it's not handled by detectStaircases) AND at least one predecessor is
// a "displaced tap" — non-adjacent (rank-gap ≥ 2) AND in a multi-node row at
// its rank. The displacement means packing has pushed that tap off the spine
// column, so the regular skip-line routing (which drops at source.x +
// SKIP_OFFSET) would land far from the merge's natural column.
//
// The classic case: C3k's inner cat, fed by `1_Bottleneck` (adjacent, end of
// bottleneck chain) AND `cv2` (rank-3 bypass that ended up packed right
// alongside cv1). We widen the cat to span both tap-centers and render each
// pred as a pure vertical drop onto the widened merge.
//
// CRITICAL discriminator: if NO tap is displaced, we leave the cat alone —
// existing skip routing handles long edges (e.g. C2PSA's split → cat where
// both source and target are alone in their rows, both on the spine column).
// === Fan-out source detection (symmetric to fan-in) ===
//
// A fan-out source is a node with ≥2 unique successors at adjacent rank
// (rank+1). The canonical case: Attention's `split` feeding `matmul-1` (for
// q+k) AND `pe` (for v→reshape→pe) — both at the same rank below split.
// Default packing would line up matmul-1 and pe side-by-side; without
// widening the source, the edge from split to the rightmost successor would
// be a long horizontal stretch.
//
// We widen the source rightward (post-placement fixup, since children at the
// later rank need to be placed first) to span the children's centers, and
// render each source→adjacent-successor edge as a pure vertical drop at the
// child's center-x. Non-adjacent successors of the same source (e.g. split
// → matmul-2 at rank+4) stay as skip edges — they need different routing.
function detectFanOutSources(nodes, edges, rankByName) {
  const outMap = new Map();
  nodes.forEach(n => outMap.set(n.id, new Set()));
  for (const [s, t] of edges) outMap.get(s)?.add(t);

  const sources = new Map();   // sourceId -> [adjacent successor ids]
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

  const merges = new Map();   // mergeId -> [tap ids, in pred order]
  const tapEdgeKeys = new Set();   // "src->dst" strings — to exclude from skip detection
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

// === Sizing per subkind (module/arith/cat/split/shape/struct/attr/op/io) ===
const NODE_W = 150;
const NODE_H = 60;
const ARITH_R = 18;   // circle radius for add/mul
const SMALL_W = 120;
const SMALL_H = 32;

function nodeSize(node, staircaseSpan) {
  if (node.subkind === 'arith') return { w: 2 * ARITH_R, h: 2 * ARITH_R };
  if (node.subkind === 'shape' || node.subkind === 'attr' || node.subkind === 'struct') {
    return { w: SMALL_W, h: SMALL_H };
  }
  if (node.subkind === 'cat' && staircaseSpan != null) {
    // Merge of a staircase — widen to span all the taps.
    return { w: staircaseSpan, h: NODE_H };
  }
  return { w: NODE_W, h: NODE_H };
}

// === Auto-layout: layered top-to-bottom, staircase-aware + skip corridor ===
//
// Spacing knobs — tune these to change padding / gaps globally:
//   ROW_GAP        vertical gap between consecutive ranks (rows)
//   COL_GAP        horizontal gap between same-rank peer nodes (packing)
//   MARGIN         outer margin around the whole diagram
//   SKIP_SHIFT     base right-shift for intermediates straddled by a skip
//   SKIP_OFFSET    inset of the innermost skip line from source's left edge
//   SKIP_LANE_W    additional offset per skip lane (nested skips)
//   FANIN_PAD      padding around fan-in / fan-out widened nodes
const CORRIDOR_W   = 80;   // (legacy) reserved left-side strip when skip edges exist
const SKIP_SHIFT   = 90;
const SKIP_OFFSET  = 20;
const SKIP_LANE_W  = 35;
const ROW_GAP_PX   = 70;
const COL_GAP_PX   = 50;   // ↑ bumped from 28 — gives same-rank peers room to breathe
const MARGIN_PX    = 40;

function autoLayout(nodes, edges, stairs) {
  const ROW_GAP = ROW_GAP_PX, COL_GAP = COL_GAP_PX, MARGIN = MARGIN_PX;

  if (nodes.length === 0) {
    return { positions: {}, totalW: 600, totalH: 200, skipEdges: new Set(), corridorX: 0 };
  }

  const inMap = new Map(), outMap = new Map();
  nodes.forEach(n => { inMap.set(n.id, []); outMap.set(n.id, []); });
  edges.forEach(([s, t]) => {
    inMap.get(t)?.push(s);
    outMap.get(s)?.push(t);
  });

  // Longest-path rank
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

  // Force output to the bottom
  const maxRank = Math.max(...Object.values(rank));
  nodes.forEach(n => {
    if ((n.subkind === 'io' || n.kind === 'io') && (n.id === 'output' || n.label === 'output')) {
      rank[n.id] = maxRank;
    }
  });

  // --- Staircase pre-pass: compute merge spans and node widths ---
  const tapOffset = stairs.tapOffset;       // Map<tapId, offset>
  const mergeSpanByMerge = new Map();        // mergeId -> width
  const mergeStaircase  = new Map();          // mergeId -> [tapId, ...]
  for (const s of stairs.stairs) {
    const N = s.taps.length;
    const span = NODE_W + (N - 1) * STAIR_FRAC * NODE_W;
    mergeSpanByMerge.set(s.mergeId, span);
    mergeStaircase.set(s.mergeId, s.taps);
  }

  // --- Group nodes by rank ---
  const rows = new Map();
  for (const n of nodes) {
    if (!rows.has(rank[n.id])) rows.set(rank[n.id], []);
    rows.get(rank[n.id]).push(n);
  }
  const rankKeys = [...rows.keys()].sort((a, b) => a - b);

  // --- Compute per-row widths ---
  const positions = {};
  const sizes = {};
  for (const n of nodes) {
    sizes[n.id] = nodeSize(n, mergeSpanByMerge.get(n.id));
  }

  // X anchor: we pick a "spine" x-coordinate. Non-staircase nodes (and tap[0])
  // sit at spineX; staircase taps with offset i sit at spineX + i*STAIR_FRAC*NODE_W;
  // staircase merge sits at spineX and is wide.
  // Compute maximum row width considering staircase overhang
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
      // Pack horizontally (multiple peer nodes on same row — rare in our graphs)
      w = row.reduce((acc, n) => acc + sizes[n.id].w + COL_GAP, -COL_GAP);
    }
    if (w > maxRowW) maxRowW = w;
  }

  // --- Detect fan-in merges (parallel-branch generalization of staircase).
  // Triggers only when at least one tap is displaced (non-adjacent + sharing
  // its rank with a peer, i.e. pushed off the spine by packing). When NO tap
  // is displaced, existing skip routing handles the long edges fine.
  const multiNodeRanks = new Set();
  for (const [r, nodesAtR] of rows) {
    if (nodesAtR.length > 1) multiNodeRanks.add(r);
  }
  const fanIns = detectFanInMerges(nodes, edges, stairs.mergeIds, rank, multiNodeRanks);
  const fanOuts = detectFanOutSources(nodes, edges, rank);

  // --- Detect skip edges: rank gap >= 2 AND not part of a staircase pattern
  // AND not a fan-in tap → merge edge AND not a fan-out source → tap edge.
  // (For fan-out, adjacent edges are taps by definition; we still need to
  //  exclude any non-adjacent skip case where the same pair would qualify.)
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

  // --- Skip lanes + per-rank shift.
  //
  // When multiple skips share a source (e.g. split → cat AND split → '+1' in
  // C2PSA), their vertical lines would coincide at source.x + SKIP_OFFSET.
  // We assign each skip a `lane` within its source-group — longest skip gets
  // lane 0 (innermost, closest to source's left edge), shorter skips fan
  // rightward at +lane * SKIP_LANE_W.
  //
  // Independently, each skip has a `nesting depth` = number of OTHER skips
  // strictly containing it. An intermediate rank's right-shift then becomes
  // SKIP_SHIFT + max_nesting_depth_crossing(r) * SKIP_LANE_W, so deeper-nested
  // skips' intermediates clear all the inner skip lines (including the inner
  // skip's OWN line, whose source may itself sit in the shifted column).
  const skipsList = [];
  for (const key of skipEdges) {
    const [s, t] = key.split('->');
    if (rank[s] == null || rank[t] == null) continue;
    skipsList.push({ src: s, tgt: t, srcRank: rank[s], tgtRank: rank[t], key });
  }
  const skipLane = new Map();   // edge key -> lane index within source-group
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
  const shiftByRank = new Map();   // rank -> px to shift strict intermediates
  for (const sk of skipsList) {
    const d = skipDepth.get(sk.key) || 0;
    const want = SKIP_SHIFT + d * SKIP_LANE_W;
    for (let r = sk.srcRank + 1; r < sk.tgtRank; r++) {
      shiftByRank.set(r, Math.max(shiftByRank.get(r) || 0, want));
    }
  }

  // --- Place nodes rank-by-rank, tracking ACTUAL x in `positionedX`.
  //
  // Three kinds of placement preference (in priority order):
  //   1. Staircase tap — keep its STAIR_FRAC offset (handled by tapOffset).
  //   2. Skip target — align to source.actualX + lane*LANE_W so the skip's
  //      vertical drop lands cleanly on the target. We use the source's
  //      ACTUAL placed x (not its nominal shift) — this is the fix for
  //      dangling arrows when the source ended up packed right in a multi-
  //      node row (e.g. C3k's inner cv2 sitting alongside cv1).
  //   3. Pure intermediate — clearance shift = spineX + shiftByRank[rank].
  //
  // For multi-node rows we sort by preferred x ascending and pack left-to-
  // right, advancing curX past each node. This gives the skip-branch peer
  // (whose preferred x is small / no shift) the leftmost slot, and the
  // main-path peer the next slot rightward — which is the natural reading.
  const skipsByTarget = new Map();
  for (const sk of skipsList) {
    if (!skipsByTarget.has(sk.tgt)) skipsByTarget.set(sk.tgt, []);
    skipsByTarget.get(sk.tgt).push(sk);
  }
  // Heuristic for ordering peers on the same rank: low outgoing-rank-gap
  // (main path, chains to next rank) goes LEFT; high outgoing-rank-gap
  // (skip-branch, jumps several ranks to a merge) goes RIGHT. This keeps the
  // chain column on the spine and lets the skip-branch swing out to the right.
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
    // Plain spine nodes are centered within the nominal NODE_W column (see
    // below) so chains of mixed-width nodes (modules / shape ops / arith)
    // line up center-to-center instead of left-edge-to-left-edge.
    const centeredSet = new Set();
    for (const n of row) {
      let want;
      if (fanIns.merges.has(n.id)) {
        // Fan-in merge: widen to span its taps' centers. Taps are at earlier
        // ranks so they're already placed.
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
          // Defensive — fall back to clearance shift.
          want = spineX + (shiftByRank.get(r) || 0);
        } else {
          want = minCx - FANIN_PAD;
          const w = Math.max(NODE_W, (maxCx - minCx) + 2 * FANIN_PAD);
          sizes[n.id] = { w, h: NODE_H };
        }
      } else if (tapOffset.has(n.id)) {
        // Staircase tap: include the intermediate-skip shift (if any). Without
        // this, a staircase tap at offset 0 sits at spineX while its non-tap
        // chain siblings (cv1_conv, cv1_bn) get pushed right by an unrelated
        // skip crossing — e.g. SPPF's hidden x→add residual makes ranks 1..10
        // intermediate, so the chain shifts but the tap doesn't, breaking the
        // Conv container column.
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
    // Sort: primary by preferred x ascending, secondary by fanOutGap ascending
    // (main-path peer leftmost, skip-branch peer rightmost when tied on x).
    const sortedRow = [...row].sort((a, b) => {
      const pa = preferredX.get(a.id);
      const pb = preferredX.get(b.id);
      if (pa !== pb) return pa - pb;
      return fanOutGap(a) - fanOutGap(b);
    });
    let curX = -Infinity;
    for (const n of sortedRow) {
      const sz = sizes[n.id];
      // Each peer occupies a uniform NODE_W-wide column slot. `slotLeft` is the
      // slot's left edge (preferred x, clamped past the previous peer's slot);
      // a narrow node is centered WITHIN its slot. curX advances by the nominal
      // column width — not the node's actual width — so a narrow node (e.g. an
      // arith circle) still reserves a full column and parallel branches stay
      // in consistent x positions row-to-row (no zigzag from width variance).
      let slotLeft = preferredX.get(n.id);
      if (slotLeft < curX) slotLeft = curX;
      let x = slotLeft;
      if (centeredSet.has(n.id) && sz.w < NODE_W) x += (NODE_W - sz.w) / 2;
      positions[n.id] = { x, y: r * (NODE_H + ROW_GAP) + MARGIN, ...sz };
      positionedX.set(n.id, x);
      curX = slotLeft + Math.max(sz.w, NODE_W) + COL_GAP;
    }
  }

  // --- Fan-out widening fixup: extend each fan-out source's bbox to span its
  // already-placed adjacent children's centers, so the per-child vertical
  // drop edges land cleanly inside the source's bottom edge. We only EXTEND
  // (never shrink, never move source left of its original x) — preserves
  // upstream alignments.
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
    skipLane,                          // edge key -> lane within source group
    fanInMerges: fanIns.merges,        // mergeId -> [tap ids]
    fanInTapEdges: fanIns.tapEdgeKeys,
    fanOutSources: fanOuts.sources,    // sourceId -> [adjacent successor ids]
    fanOutTapEdges: fanOuts.tapEdgeKeys,
    corridorX: spineX,                 // legacy field
  };
}

// === Right-angle bezier — for consecutive staircase taps ===
// Leaves p1 horizontally to the right, arrives at p2 vertically.
function rightAngleEdge(p1, p2, lead = SEQ_BEND) {
  const dy = p2.y - p1.y;
  const c1 = { x: p1.x + lead, y: p1.y };
  const c2 = { x: p2.x, y: p2.y + (dy >= 0 ? -lead : lead) };
  return `M ${p1.x} ${p1.y} C ${c1.x} ${c1.y} ${c2.x} ${c2.y} ${p2.x} ${p2.y}`;
}

// === Flat-bezier edge — vertical top-to-bottom ===
function edgePath(p1, p2, entry = 14, exit = 18) {
  const dy = p2.y - p1.y;
  const ady = Math.abs(dy);
  if (ady < 1) {
    const adx = Math.abs(p2.x - p1.x);
    if (adx < 1) return `M ${p1.x} ${p1.y}`;
    // Horizontal edge (same row) — handle with horizontal flat bezier
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

const KIND_COLORS = {
  io:  { fill: '#e2e8f0', border: '#94a3b8', text: '#1e293b' },
  mod: { fill: '#dcfce7', border: '#22c55e', text: '#14532d' },
  op:  { fill: '#fed7aa', border: '#fb923c', text: '#7c2d12' },
};

function formatShape(sh) {
  if (!sh) return '';
  if (Array.isArray(sh) && sh.length && typeof sh[0] === 'number') {
    return sh.join('×');
  }
  return '';
}

function SpecViewer() {
  const spec = window.YV_SPEC;
  if (!spec) {
    return <div style={{ padding: 24, color: '#dc2626' }}>
      spec-data.js failed to load — run <code>uv run python -m yolovex.block_spec</code> first.
    </div>;
  }

  // Available blocks: fx-OK AND not Detect (parked for separate design)
  const selectableInstances = spec.instances.filter(i => {
    const s = spec.specs[i.spec_id];
    return s.derivation_method === 'fx' && i.class_name !== 'Detect';
  });

  const [blockIdx, setBlockIdx] = useState(selectableInstances[0]?.idx ?? 0);
  // `level` is the user-facing fidelity level (1, 2, 3, ...). Internal aggregation
  // depth = level - 1. L1 is the single-node view; L2 first peels containers; etc.
  const [level, setLevel] = useState(2);

  const instance = spec.instances.find(i => i.idx === blockIdx);
  const blockSpec = instance && spec.specs[instance.spec_id];

  const { agg, layout, stairs, containers } = useMemo(() => {
    const empty = { agg: { nodes: [], edges: [] },
                    layout: { positions: {}, totalW: 600, totalH: 200, skipEdges: new Set(), corridorX: 0 },
                    stairs: { stairs: [], tapOffset: new Map(), tapInStaircase: new Set(), mergeIds: new Set() },
                    containers: [] };
    if (!blockSpec || blockSpec.derivation_method !== 'fx') return empty;

    let a;
    if (level === 1) {
      a = buildL1View(blockSpec, instance);
    } else {
      const depth = level - 1;
      const visibility = opVisibilityForLevel(level);
      const preprocessed = preprocessGraph(blockSpec.graph, visibility, instance.shapes_by_node || {});
      a = aggregateAtDepth(preprocessed, instance.shapes_by_node || {}, depth, blockSpec.path_classes);
    }
    const st = detectStaircases(a.nodes, a.edges);
    const l = autoLayout(a.nodes, a.edges, st);
    const c = computeContainers(a.nodes, l.positions, blockSpec.path_classes, blockSpec.class_name);
    return { agg: a, layout: l, stairs: st, containers: c };
  }, [blockIdx, level, blockSpec, instance]);

  // Available levels — only those where adding depth still changes the diagram.
  const levelInfo = useMemo(() => {
    if (!blockSpec || blockSpec.derivation_method !== 'fx') return [];
    const counts = [{ level: 1, nodes: 3 }];   // L1 is always 1 block + 2 io
    let prev = 3;
    for (let lv = 2; lv <= 6; lv++) {
      const visibility = opVisibilityForLevel(lv);
      const preprocessed = preprocessGraph(blockSpec.graph, visibility, instance.shapes_by_node || {});
      const a = aggregateAtDepth(preprocessed, instance.shapes_by_node || {}, lv - 1, blockSpec.path_classes);
      if (a.nodes.length === prev) break;
      counts.push({ level: lv, nodes: a.nodes.length });
      prev = a.nodes.length;
    }
    return counts;
  }, [blockSpec, instance]);

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{
        padding: '10px 16px',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        gap: 14,
        alignItems: 'center',
        background: 'white',
      }}>
        <strong style={{ fontSize: 14 }}>yolovex · block spec viewer</strong>
        <span style={{ color: '#cbd5e1' }}>|</span>
        <label>Block:
          <select value={blockIdx} onChange={e => { setBlockIdx(parseInt(e.target.value)); setLevel(2); }}>
            {selectableInstances.map(i => (
              <option key={i.idx} value={i.idx}>
                [{i.idx}] {i.class_name}
              </option>
            ))}
          </select>
        </label>
        <label>Level:
          <select value={level} onChange={e => setLevel(parseInt(e.target.value))}>
            {levelInfo.map(({ level: l, nodes }) => (
              <option key={l} value={l}>L{l} ({nodes} nodes)</option>
            ))}
          </select>
        </label>
        <span style={{ color: '#64748b', fontSize: 12, marginLeft: 'auto', fontFamily: 'ui-monospace, monospace' }}>
          {instance && (
            <>
              in: {formatShape(instance.input_shape)} → out: {formatShape(instance.output_shape)} · {instance.params.toLocaleString()} params · spec {instance.spec_id}
            </>
          )}
        </span>
      </header>
      <div style={{ flex: 1, overflow: 'auto', background: '#f8fafc' }}>
        <svg
          width={Math.max(layout.totalW, 800)}
          height={Math.max(layout.totalH, 400)}
          style={{ display: 'block' }}
        >
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 Z" fill="#94a3b8" />
            </marker>
            <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 Z" fill="#fb923c" />
            </marker>
          </defs>

          {/* Parent containers — drawn first so edges + nodes paint on top. */}
          {containers.map(c => {
            // Outermost (block-root) gets a subtler fill; nested containers get
            // a slightly tinted background so the hierarchy reads visually.
            const isRoot = c.depth === 0;
            const fill   = isRoot ? '#f8fafc' : '#eef2ff';
            const stroke = isRoot ? '#cbd5e1' : '#a5b4fc';
            const text   = isRoot ? '#475569' : '#3730a3';
            return (
              <g key={`container-${c.path.join('/') || 'root'}`}>
                <rect
                  x={c.x} y={c.y} width={c.w} height={c.h}
                  rx="10"
                  fill={fill} fillOpacity="0.55"
                  stroke={stroke} strokeWidth="1"
                  strokeDasharray="4 3"
                />
                <text
                  x={c.x + c.w - 12}
                  y={c.y + 15}
                  fontSize="11"
                  fontWeight="600"
                  fill={text}
                  fontFamily="ui-monospace, monospace"
                  letterSpacing="0.4"
                  textAnchor="end"
                >
                  {c.label}
                </text>
              </g>
            );
          })}

          {agg.edges.map((edge, i) => {
            const [a, b, label] = edge;
            const pa = layout.positions[a], pb = layout.positions[b];
            if (!pa || !pb) return null;
            const aNode = agg.nodes.find(n => n.id === a);
            const bNode = agg.nodes.find(n => n.id === b);
            const aIsTap = stairs.tapInStaircase.has(a);
            const bIsTap = stairs.tapInStaircase.has(b);
            const bIsMerge = stairs.mergeIds.has(b);
            const isSkip = layout.skipEdges?.has(`${a}->${b}`);
            const isFanInTap = layout.fanInTapEdges?.has(`${a}->${b}`);

            // Fan-in tap → widened merge: pure-vertical drop from tap.center
            // straight down to the merge's top edge at the same x. The merge
            // was widened during layout to span all tap-center x's, so the
            // landing point is always inside the merge.
            if (isFanInTap) {
              const x = pa.x + pa.w / 2;
              const y1 = pa.y + pa.h;
              const y2 = pb.y;
              return (
                <path key={i}
                  d={`M ${x} ${y1} L ${x} ${y2}`}
                  fill="none" stroke="#94a3b8" strokeWidth="1.5"
                  markerEnd="url(#arrow)" />
              );
            }

            // Fan-out source → adjacent successor: pure vertical drop at
            // successor.center.x. The source was widened in the layout fixup
            // to span all such successors so the line emanates from inside
            // its bottom edge.
            const isFanOutTap = layout.fanOutTapEdges?.has(`${a}->${b}`);
            if (isFanOutTap) {
              const x = pb.x + pb.w / 2;
              const y1 = pa.y + pa.h;
              const y2 = pb.y;
              return (
                <g key={i}>
                  <path d={`M ${x} ${y1} L ${x} ${y2}`}
                    fill="none" stroke="#94a3b8" strokeWidth="1.5"
                    markerEnd="url(#arrow)" />
                  {label != null && label !== '' && (
                    <text
                      x={x + 6}
                      y={(y1 + y2) / 2 + 4}
                      fontSize="11"
                      fontFamily="ui-monospace, monospace"
                      fill="#475569"
                    >
                      [{label}]
                    </text>
                  )}
                </g>
              );
            }

            // Skip edge: pure-vertical line dropping from source's bottom edge
            // (inset SKIP_OFFSET from the left corner, plus a per-lane offset
            // when this source has multiple outgoing skips) to target's top
            // edge at the same x. Intermediate nodes have been shifted right
            // by enough to give this line clearance, accounting for nesting.
            if (isSkip) {
              const lane = layout.skipLane?.get(`${a}->${b}`) ?? 0;
              const x = pa.x + SKIP_OFFSET + lane * SKIP_LANE_W;
              const y1 = pa.y + pa.h;
              const y2 = pb.y;
              return (
                <path key={i}
                  d={`M ${x} ${y1} L ${x} ${y2}`}
                  fill="none" stroke="#fb923c" strokeWidth="1.6"
                  markerEnd="url(#arrow-accent)"
                  opacity="0.9" />
              );
            }

            // tap → merge: pure-vertical line dropping from tap's bottom-mid to merge's top edge at the same x
            if (aIsTap && bIsMerge) {
              const x = pa.x + pa.w / 2;
              const y1 = pa.y + pa.h;
              const y2 = pb.y;
              return (
                <path key={i} d={`M ${x} ${y1} L ${x} ${y2}`}
                  fill="none" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
              );
            }

            // tap[i] → tap[i+1]: right-angle bezier
            if (aIsTap && bIsTap) {
              const off1 = stairs.tapOffset.get(a) ?? 0;
              const off2 = stairs.tapOffset.get(b) ?? 0;
              if (off2 > off1) {
                const p1 = { x: pa.x + pa.w, y: pa.y + pa.h / 2 };
                const p2 = { x: pb.x + pb.w / 2, y: pb.y };
                return (
                  <path key={i} d={rightAngleEdge(p1, p2)}
                    fill="none" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
                );
              }
            }

            // merge → downstream (non-tap): exit at the target's center x so the
            // arrow stays vertical and lines up with the next block's left edge.
            if (stairs.mergeIds.has(a) && !bIsTap) {
              const x = pb.x + pb.w / 2;
              const y1 = pa.y + pa.h;
              const y2 = pb.y;
              return (
                <path key={i} d={`M ${x} ${y1} L ${x} ${y2}`}
                  fill="none" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
              );
            }

            // arith ops are circles — connect to their center directly
            const aIsArith = aNode?.subkind === 'arith';
            const bIsArith = bNode?.subkind === 'arith';

            const aBottom = { x: pa.x + pa.w / 2, y: pa.y + pa.h };
            const bTop    = { x: pb.x + pb.w / 2, y: pb.y };
            const aRight  = { x: pa.x + pa.w, y: pa.y + pa.h / 2 };
            const bLeft   = { x: pb.x, y: pb.y + pb.h / 2 };
            let p1, p2;
            if (Math.abs(pa.y - pb.y) < 1) {
              if (pb.x > pa.x) { p1 = aRight; p2 = bLeft; }
              else             { p1 = { x: pa.x, y: pa.y + pa.h/2 }; p2 = { x: pb.x + pb.w, y: pb.y + pb.h/2 }; }
            } else if (pb.y > pa.y) {
              p1 = aBottom; p2 = bTop;
            } else {
              p1 = { x: pa.x + pa.w / 2, y: pa.y };
              p2 = { x: pb.x + pb.w / 2, y: pb.y + pb.h };
            }
            const path = edgePath(p1, p2);
            return (
              <g key={i}>
                <path d={path}
                  fill="none" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
                {label != null && label !== '' && (
                  <text
                    x={(p1.x + p2.x) / 2 + 6}
                    y={(p1.y + p2.y) / 2 + 4}
                    fontSize="11"
                    fontFamily="ui-monospace, monospace"
                    fill="#475569"
                  >
                    [{label}]
                  </text>
                )}
              </g>
            );
          })}

          {agg.nodes.map(n => {
            const p = layout.positions[n.id];
            if (!p) return null;
            const sk = n.subkind || n.kind;

            // ---- Arith (add/mul/...) : circle ----
            if (sk === 'arith') {
              const cx = p.x + p.w / 2, cy = p.y + p.h / 2;
              const orig = blockSpec.graph.nodes.find(g => g.name === n.id);
              const sym = orig ? ({ add: '+', mul: '×', sub: '−', truediv: '÷' }[opShortName(orig.target)] || '·') : '·';
              return (
                <g key={n.id}>
                  <circle cx={cx} cy={cy} r={p.w / 2}
                    fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.5" />
                  <text x={cx} y={cy + 6} fontSize="18" fontWeight="700" fill="#78350f"
                    textAnchor="middle">{sym}</text>
                </g>
              );
            }

            // ---- get_attr / shape ops / leftover struct : small dim node ----
            if (sk === 'attr' || sk === 'shape' || sk === 'struct') {
              const fill   = sk === 'attr'  ? '#ede9fe' : '#f1f5f9';
              const stroke = sk === 'attr'  ? '#a78bfa' : '#94a3b8';
              const text   = sk === 'attr'  ? '#4c1d95' : '#475569';
              return (
                <g key={n.id}>
                  <rect x={p.x} y={p.y} width={p.w} height={p.h} rx="4"
                    fill={fill} stroke={stroke} strokeWidth="1" strokeDasharray="3 2" />
                  <text x={p.x + p.w / 2} y={p.y + p.h / 2 + 4}
                    fontSize="11" fill={text} textAnchor="middle"
                    fontFamily="ui-monospace, monospace">
                    {n.label.split('\n')[0]}
                  </text>
                </g>
              );
            }

            // ---- cat / split / other op / module / io : rectangle ----
            const styleByKind = {
              cat:    { fill: '#fde68a', border: '#f59e0b', text: '#78350f' },  // amber band
              split:  { fill: '#dbeafe', border: '#3b82f6', text: '#1e3a8a' },
              op:     KIND_COLORS.op,
              module: KIND_COLORS.mod,
              mod:    KIND_COLORS.mod,
              io:     KIND_COLORS.io,
            };
            const c = styleByKind[sk] || KIND_COLORS.mod;
            const lines = n.label.split('\n');
            const sh = formatShape(n.shape);
            const lineH = 14;
            const startY = sh
              ? p.y + (p.h - lines.length * lineH - 14) / 2 + 11
              : p.y + (p.h - lines.length * lineH) / 2 + 11;
            return (
              <g key={n.id}>
                <rect
                  x={p.x} y={p.y} width={p.w} height={p.h}
                  rx="6" fill={c.fill} stroke={c.border} strokeWidth="1.5"
                />
                {lines.map((line, i) => (
                  <text
                    key={i}
                    x={p.x + p.w / 2}
                    y={startY + i * lineH}
                    fontSize={i === 0 ? "12" : "11"}
                    fontWeight={i === 0 ? "600" : "400"}
                    fill={c.text}
                    fontStyle={i > 0 ? "italic" : "normal"}
                    textAnchor="middle"
                  >
                    {line}
                  </text>
                ))}
                {sh && (
                  <text
                    x={p.x + p.w / 2}
                    y={p.y + p.h - 8}
                    fontSize="10"
                    fill={c.text}
                    opacity="0.65"
                    textAnchor="middle"
                    fontFamily="ui-monospace, monospace"
                  >
                    {sh}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<SpecViewer />);
