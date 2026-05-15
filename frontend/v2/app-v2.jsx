// yolovex v2 — app shell.
//
// Interaction model:
//   - bare left-click on a block / sub-node / inner container → open panel
//   - shift+click → expand / collapse
//   - hover → drive the top-right floating thumbnail
//   - click anywhere outside the panel → close it
//
// hover/selected payloads are objects: { idx, pathKey, fxKey?, subkind? } | null.
// pathKey === null means the L1 block's own output; otherwise fxKey is the
// fx-graph node name we look up in YV_ACT.nodes[idx].sub.

const { useState, useEffect, useCallback, useMemo, useRef } = React;

// =============================================================================
// Activation lookup
// =============================================================================

function lookupActivation(active) {
  if (!active || typeof window.YV_ACT === 'undefined') return null;
  const block = window.YV_ACT.nodes?.[String(active.idx)];
  if (!block) return null;
  if (active.pathKey == null) return block.output || null;
  if (!active.fxKey) return null;
  return block.sub?.[active.fxKey] || null;
}

function isDeferred(active) {
  return !!(active && window.YV_ACT?.meta?.skipped?.includes(active.idx));
}

// =============================================================================
// Learner-facing copy per type (ported from L2)
// =============================================================================

const TYPE_COPY = {
  Conv2d: {
    title: 'Conv2d — learnable filter bank',
    blurb: 'Slides a small kernel across the input and produces one output channel per filter. This is the ONLY layer in the Conv block that carries learnable weights — it does the actual feature extraction. Stride > 1 here also means spatial downsampling.',
  },
  BatchNorm2d: {
    title: 'BatchNorm — channel-wise normalization',
    blurb: 'Re-centers and re-scales each channel using running mean/variance learned during training. Keeps activations in a stable range so the next nonlinearity sees predictable inputs.',
  },
  SiLU: {
    title: 'SiLU — smooth nonlinearity',
    blurb: 'Computes x · sigmoid(x). For most positive activations SiLU is nearly x (so this thumbnail will look almost identical to BN output above), but it gently suppresses strong negative values.',
  },
  MaxPool2d: {
    title: 'MaxPool — local maximum',
    blurb: 'Inside SPPF, a 5×5 stride-1 max pool with padding 2. SPPF chains it three times so the n-th call has covered an effective receptive field of (4n+1)×(4n+1) — a "pyramid of receptive fields" without changing spatial size.',
  },
  Concat: {
    title: 'Concat — stack channels',
    blurb: 'Joins multiple tensors along the channel dimension. Output channel count = sum of inputs. Used inside C3k2/SPPF to merge parallel branches, and across the neck to fuse skip connections with upsampled features.',
  },
  Upsample: {
    title: 'Upsample — 2× nearest neighbor',
    blurb: 'Doubles spatial dimensions by repeating each pixel in a 2×2 block. No learnable params. The neck uses this to bring deep coarse features back up to higher-resolution scales for small-object detection.',
  },
  Bottleneck: {
    title: 'Bottleneck — residual mini-block',
    blurb: 'Two 3×3 convs sandwiching the channel count, with an additive residual skip. Lets each C3k2 add a couple of layers of nonlinear refinement on the parallel branch without inflating param count.',
  },
  C3k: {
    title: 'C3k — small CSP sub-block',
    blurb: 'A miniature CSP module (split → series of bottlenecks → concat → 1×1) used inside the deeper C3k2 layers.',
  },
  PSABlock: {
    title: 'PSABlock — position-sensitive self-attention',
    blurb: 'Runs self-attention (Q/K/V) followed by a small feed-forward. Lets the network propagate information across distant locations — useful at the deepest backbone scale where a single token already covers a big receptive field.',
  },
  Conv: {
    title: 'Conv block — Conv2d → BN → SiLU',
    blurb: 'The basic building block. A learnable Conv2d does feature extraction, BatchNorm normalizes the result, SiLU applies a smooth nonlinearity. Stride-2 versions also halve the spatial size.',
  },
  C3k2: {
    title: 'C3k2 block — CSP feature mixer',
    blurb: 'Splits the input into two halves, runs a Bottleneck (or nested C3k) on one half, then concats both halves back with the bottleneck output before a 1×1 projection. YOLO26\'s main feature-mixing unit at every backbone resolution.',
  },
  SPPF: {
    title: 'SPPF — spatial pyramid pooling (fast)',
    blurb: 'Stacks three 5×5 max-pools so a single forward gives features at 4 receptive-field scales, concats them, and projects with a 1×1. Cheap way to give the deepest backbone layer a "view" at multiple object sizes.',
  },
  C2PSA: {
    title: 'C2PSA — CSP + position-sensitive attention',
    blurb: 'Like C3k2 but with a self-attention block on the parallel branch. Adds long-range context on top of the local convolutional features at the deepest scale.',
  },
  Detect: {
    title: 'Detect head — anchor-free, NMS-free',
    blurb: 'Runs three parallel conv heads (one per scale) and emits boxes + class scores per anchor cell. Activations deferred — coming in a later pass.',
  },
};

function copyFor(typeOrSub) {
  return TYPE_COPY[typeOrSub] || { title: typeOrSub || 'node', blurb: '' };
}

// =============================================================================
// Upstream source resolution — for the IO strip
// =============================================================================

// For an L1 block: read upstream block ids from spec edges; -1 means input image.
function l1UpstreamSources(idx) {
  const edges = (window.YV_SPEC && window.YV_SPEC.edges) || [];
  const sources = edges.filter(e => e.dst === idx).map(e => e.src);
  if (sources.length === 0) return [{ isImage: true }];
  return sources.map(src => ({ srcIdx: src, isImage: src < 0 }));
}

// For a sub-node / container: find fx edges that CROSS the group's boundary
// (source outside the group, destination inside). `members` is the list of fx
// node names that belong to the selected group. Placeholders / get_attrs get
// inherited as the L1 block's own input sources.
//
// We also walk back through nodes that don't have a renderable activation:
//   - getitem (tuple/list index, hidden in the visual graph)
//   - chunk / split (return tuples, no 4-D tensor captured)
//   - any other fx node not present in YV_ACT.nodes[idx].sub
// So a cat fed by getitem(chunk(cv1)[0]) + getitem(chunk(cv1)[1]) + bottleneck
// shows TWO inputs (cv1's output, deduped, + the bottleneck), matching the
// visual graph rather than the raw 3-element list.
function subUpstreamSources(idx, members) {
  if (!members || !members.length) return [];
  const arch = window.YVV2.buildArch();
  const block = arch.find(a => a.idx === idx);
  const specId = block?.specId;
  const spec = specId && window.YV_SPEC?.specs?.[specId];
  if (!spec) return [];

  const memberSet = new Set(members);
  const nameToNode = new Map(spec.graph.nodes.map(n => [n.name, n]));
  const captured = window.YV_ACT?.nodes?.[String(idx)]?.sub || {};

  const incomingByDst = new Map();
  for (const [s, t] of spec.graph.edges) {
    if (!incomingByDst.has(t)) incomingByDst.set(t, []);
    incomingByDst.get(t).push(s);
  }

  // Walk back through nodes the visual graph hides — getitem unconditionally
  // (preprocessGraph in expand-v2.jsx line 42) plus anything we didn't capture
  // a 4-D activation for (tuple-returning chunk/split, etc.). getitem nodes
  // ARE captured (they're 4-D views), so the captured check alone would let
  // them slip through; the explicit name check matches the visual rule.
  function expand(name, visited) {
    if (visited.has(name)) return [];
    visited.add(name);
    const n = nameToNode.get(name);
    if (!n) return [];
    if (memberSet.has(name)) return [];
    if (n.op === 'placeholder' || n.op === 'get_attr') return [{ kind: 'L1' }];
    const last = String(n.target || '').split('.').pop();
    const hiddenVisually = (n.op === 'call_function' && last === 'getitem');
    if (hiddenVisually || !captured[name]) {
      const out = [];
      for (const s of (incomingByDst.get(name) || [])) {
        out.push(...expand(s, visited));
      }
      return out;
    }
    return [{ kind: 'real', name }];
  }

  const sources = [];
  const seen = new Set();
  for (const [s, t] of spec.graph.edges) {
    if (!memberSet.has(t)) continue;
    if (memberSet.has(s)) continue;
    for (const item of expand(s, new Set())) {
      const k = item.kind === 'L1' ? '<L1>' : item.name;
      if (seen.has(k)) continue;
      seen.add(k);
      sources.push(item);
    }
  }

  const out = [];
  let inheritedL1 = false;
  for (const item of sources) {
    if (item.kind === 'L1') {
      if (!inheritedL1) {
        l1UpstreamSources(idx).forEach(s => out.push(s));
        inheritedL1 = true;
      }
    } else {
      out.push({ srcIdx: idx, fxKey: item.name, isImage: false });
    }
  }
  return out;
}

function thumbnailForSource(s, inputImageUrl) {
  if (!s) return null;
  if (s.isImage) return inputImageUrl;
  const block = window.YV_ACT?.nodes?.[String(s.srcIdx)];
  if (!block) return null;
  if (s.fxKey) return block.sub?.[s.fxKey]?.mean || null;
  return block.output?.mean || null;
}

function shapeOfSource(s) {
  if (!s) return null;
  if (s.isImage) {
    const m = window.YV_ACT?.meta;
    if (m && m.image_w && m.image_h) return [1, 3, m.image_h, m.image_w];
    return null;
  }
  const block = window.YV_ACT?.nodes?.[String(s.srcIdx)];
  if (!block) return null;
  if (s.fxKey) return block.sub?.[s.fxKey]?.shape || null;
  return block.output?.shape || null;
}

function fmtShape(s) {
  if (!s) return '…';
  if (!Array.isArray(s)) return String(s);
  return `(${s.join(', ')})`;
}

// =============================================================================
// Floating overlay — persists last seen activation
// =============================================================================

function FlowOverlayV2({ active, lastActive }) {
  const meta = window.YV_ACT?.meta;
  const inputImageUrl = meta ? '../' + meta.image : '../assets/sammit_lighthouse.jpg';

  // Aspect ratio derived from the actual input image dims; never hardcoded.
  const aspect = (meta?.image_w && meta?.image_h)
    ? `${meta.image_w} / ${meta.image_h}`
    : '4 / 3';

  // Priority: live active → last shown active → input image.
  const display = active || lastActive;
  let src = inputImageUrl;
  let label = 'input image';
  let sub = 'before any layer runs';
  let stretchedActivation = false;

  if (display) {
    if (isDeferred(display)) {
      label = `[${display.idx}] activations deferred`;
      sub = 'detect head — coming later';
    } else {
      const act = lookupActivation(display);
      const block = window.YV_ACT?.nodes?.[String(display.idx)];
      const type = block?.type;
      const friendly = copyFor(type).title.split(' — ')[0] || type;
      if (act && act.mean) {
        src = act.mean;
        stretchedActivation = true;
        if (display.pathKey == null) {
          label = `[${display.idx}] ${friendly}`;
        } else {
          const last = display.pathKey.split('/').pop() || display.fxKey;
          label = `[${display.idx}] ${friendly} · ${last}`;
        }
        sub = act.shape ? `shape ${act.shape.join('×')}` : '';
      } else {
        label = `[${display.idx}] ${friendly || ''}`;
        sub = 'no captured activation';
      }
    }
  }

  return (
    <div className="flow-overlay">
      <div className="flow-overlay__frame" style={{ aspectRatio: aspect }}>
        <img
          src={src} alt=""
          style={{
            imageRendering: stretchedActivation ? 'pixelated' : 'auto',
            background: stretchedActivation ? '#0f172a' : 'transparent',
          }}
        />
      </div>
      <div className="flow-overlay__caption">{label}</div>
      <div className="flow-overlay__sub">{sub}</div>
    </div>
  );
}

// =============================================================================
// IO strip — shows N inputs → output mean
// =============================================================================

function IOStripV2({ active, output }) {
  const meta = window.YV_ACT?.meta;
  const inputImageUrl = meta ? '../' + meta.image : '../assets/sammit_lighthouse.jpg';

  const sources = active.pathKey == null
    ? l1UpstreamSources(active.idx)
    : subUpstreamSources(active.idx, active.members);

  const tile = { width: 96, height: 120 };
  const concat = { width: 80, height: 100 };
  const isMulti = sources.length > 1;
  const sizeStyle = isMulti ? concat : tile;

  const inputCaption = sources
    .map(s => fmtShape(shapeOfSource(s)))
    .join(' + ') || '…';

  return (
    <section className="panel-section">
      <h4 className="section-label">Input → Output</h4>
      <div className="io-strip">
        {sources.length === 0 ? (
          <div className="io-tile" style={{ ...tile, background: '#f1f5f9' }} />
        ) : (
          <div className="io-pair">
            {sources.map((s, i) => {
              const src = thumbnailForSource(s, inputImageUrl);
              return (
                <React.Fragment key={i}>
                  {i > 0 && <span className="plus">+</span>}
                  {src
                    ? <img src={src} alt="" className="io-tile" style={sizeStyle} />
                    : <div className="io-tile" style={{ ...sizeStyle, background: '#f1f5f9' }} />}
                </React.Fragment>
              );
            })}
          </div>
        )}
        <span className="arrow">→</span>
        {output?.mean
          ? <img src={output.mean} alt="" className="io-tile" style={tile} />
          : <div className="io-tile" style={{ ...tile, background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#94a3b8' }}>no output</div>
        }
      </div>
      <div className="shape-caption">
        {inputCaption} → {fmtShape(output?.shape)}
      </div>
    </section>
  );
}

// =============================================================================
// Detail panel
// =============================================================================

function DetailPanelV2({ selected, onClose, panelRef }) {
  const [pinnedCh, setPinnedCh] = useState(0);
  const [hoveredCh, setHoveredCh] = useState(null);

  // Reset pinned/hovered when the selection changes.
  useEffect(() => { setPinnedCh(0); setHoveredCh(null); }, [selected?.idx, selected?.fxKey, selected?.pathKey]);

  if (!selected) return <aside className="detail-panel" ref={panelRef} aria-hidden />;

  const block = window.YV_ACT?.nodes?.[String(selected.idx)];
  const arch = window.YVV2.buildArch();
  const archBlock = arch.find(a => a.idx === selected.idx);
  const ROLE_COLORS = window.YVV2.ROLE_COLORS;
  const role = archBlock?.role || 'Backbone';
  const roleColor = ROLE_COLORS[role] || '#64748b';

  // What "type" copy applies — for L1 use block.type; for sub-nodes try to
  // infer from the fx node's target_class via the spec.
  let copyKey = block?.type;
  let subTypeLabel = null;
  if (selected.pathKey != null && archBlock) {
    const spec = window.YV_SPEC?.specs?.[archBlock.specId];
    if (spec && selected.fxKey) {
      const node = spec.graph.nodes.find(n => n.name === selected.fxKey);
      if (node?.target_class) {
        copyKey = node.target_class;
        subTypeLabel = node.target_class;
      } else if (selected.subkind === 'container') {
        const last = selected.pathKey.split('/').pop() || '';
        const m = last.match(/^(\d+)_(.+)$/);
        subTypeLabel = m ? m[2] : (spec.path_classes?.[selected.pathKey] || last);
        copyKey = subTypeLabel;
      }
    }
  }
  const copy = copyFor(copyKey);

  const title = selected.pathKey == null
    ? `[${selected.idx}] ${copy.title}`
    : `[${selected.idx} · ${selected.pathKey}] ${copy.title}`;

  const deferred = isDeferred(selected);
  const act = deferred ? null : lookupActivation(selected);

  const visibleCh = act?.topK?.length || 0;
  const activeCh = hoveredCh != null ? hoveredCh : pinnedCh;
  const trueChIdx = (i) => act?.topIdx?.[i] ?? i;

  return (
    <aside className="detail-panel open" ref={panelRef}>
      <div className="panel-inner">
        <header className="panel-header">
          <div>
            <div className="panel-title">{title}</div>
            <div className="panel-desc">{copy.blurb}</div>
            {selected.pathKey != null && (
              <div className="panel-path-hint">
                Sub-module <code>{selected.pathKey}</code> of block {selected.idx}
                {block?.type ? <> ({block.type})</> : null}
                {subTypeLabel ? <> · <code>{subTypeLabel}</code></> : null}.
              </div>
            )}
            {selected.pathKey == null && archBlock?.desc && (
              <div className="panel-path-hint">{archBlock.desc}</div>
            )}
            <span className="role-pill" style={{ background: roleColor + '22', color: roleColor, marginTop: 10 }}>
              {role}
            </span>
          </div>
          <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
        </header>

        {deferred && (
          <section className="panel-section">
            <div className="panel-deferred">
              Activations for the detection head are deferred. The head is rendered structurally but its tensor outputs aren&apos;t captured in this pass — coming in a later sweep.
            </div>
          </section>
        )}

        {!deferred && !act && (
          <section className="panel-section">
            <div className="panel-empty">
              No 4-D tensor captured for this node. Container/op outputs without a (B,C,H,W) shape (e.g. concat over a list input, or shape-only ops) aren&apos;t rendered.
            </div>
          </section>
        )}

        {act && (
          <>
            <IOStripV2 active={selected} output={act} />

            {visibleCh > 0 && (
              <section className="panel-section">
                <h4 className="section-label">Channel stack</h4>
                <div className="channel-brochure">
                  <div className="brochure-preview">
                    <img
                      src={act.topK[activeCh]}
                      alt={`channel ${trueChIdx(activeCh)}`}
                      style={{
                        width: 170, height: 210,
                        objectFit: 'fill',     // stretch — match overlay
                        imageRendering: 'pixelated',
                        background: '#0f172a',
                        display: 'block', borderRadius: 4,
                        border: '2px solid var(--accent)',
                        boxShadow: '0 2px 6px rgba(15,23,42,.18)',
                      }}
                    />
                    <div className="brochure-meta">
                      <div className="meta-row">
                        <span className="meta-k">channel</span>
                        <span className="meta-v mono">{trueChIdx(activeCh)}</span>
                      </div>
                      <div className="meta-row">
                        <span className="meta-k">rank</span>
                        <span className="meta-v mono">#{activeCh + 1} of {act.totalChannels}</span>
                      </div>
                    </div>
                  </div>
                  <div className="brochure-grid" onMouseLeave={() => setHoveredCh(null)}>
                    {act.topK.map((b64, i) => (
                      <button
                        key={i}
                        className={`brochure-thumb ${pinnedCh === i ? 'pinned' : ''} ${hoveredCh === i ? 'hovered' : ''}`}
                        onMouseEnter={() => setHoveredCh(i)}
                        onClick={() => setPinnedCh(i)}
                        title={`channel ${trueChIdx(i)} (rank ${i + 1})`}
                      >
                        <img src={b64} alt="" />
                        <span className="thumb-idx">{trueChIdx(i)}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="stack-caption">
                  Showing {visibleCh} of {act.totalChannels} channels, ranked by mean |activation|.
                </div>
              </section>
            )}

            <section className="panel-section">
              <h4 className="section-label">Statistics</h4>
              <div className="stats-grid">
                <div><span className="k">shape</span><span className="v mono">{fmtShape(act.shape)}</span></div>
                <div><span className="k">activation min</span><span className="v mono">{act.stats.min}</span></div>
                <div><span className="k">activation max</span><span className="v mono">{act.stats.max}</span></div>
                <div><span className="k">activation mean</span><span className="v mono">{act.stats.mean}</span></div>
                <div><span className="k">activation std</span><span className="v mono">{act.stats.std}</span></div>
                <div><span className="k">top-5 channels</span><span className="v mono">{act.topIdx.slice(0, 5).join(', ')}</span></div>
              </div>
            </section>
          </>
        )}
      </div>
    </aside>
  );
}

// =============================================================================
// Shell
// =============================================================================

function AppV2() {
  const [hover, setHover] = useState(null);
  const [lastActive, setLastActive] = useState(null);  // sticky for the overlay
  const [selected, setSelected] = useState(null);
  const [expandedCount, setExpandedCount] = useState(0);
  const onExpandedCountChange = useCallback((n) => setExpandedCount(n), []);
  const panelRef = useRef(null);

  // Wrap setHover to persist the last non-null hover for the floating overlay,
  // so unhovering doesn't snap back to the input image.
  const onHover = useCallback((payload) => {
    setHover(payload);
    if (payload) setLastActive(payload);
  }, []);

  const onSelect = useCallback((payload) => {
    setSelected(cur => {
      if (!cur || !payload) return payload || null;
      if (cur.idx === payload.idx && cur.pathKey === payload.pathKey) return null;
      return payload;
    });
    // Pin the overlay to the selected node when nothing's currently hovered.
    if (payload) setLastActive(payload);
  }, []);

  // Click anywhere outside the panel closes it. We listen on mousedown at the
  // document so we catch background clicks even though the SVG container also
  // handles its own mousedown for pan-drag.
  useEffect(() => {
    if (!selected) return;
    const onDown = (e) => {
      const panel = panelRef.current;
      if (panel && panel.contains(e.target)) return;
      // Don't close when the click is on a node — that click will fire onSelect
      // and may pick a different node; the resulting state update handles it.
      if (e.target.closest && e.target.closest('[data-node]')) return;
      setSelected(null);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [selected]);

  // Escape to close
  useEffect(() => {
    if (!selected) return;
    const onKey = (e) => { if (e.key === 'Escape') setSelected(null); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [selected]);

  return (
    <div className="app">
      <header className="app-header">
        <strong>yolovex</strong>
        <span className="divider">/</span>
        <span className="subtitle">YOLO26 · architecture + activations</span>
        <span className="hint">
          {expandedCount > 0
            ? `${expandedCount} block${expandedCount === 1 ? '' : 's'} expanded · click for activations · shift+click to collapse`
            : 'click for activations · shift+click to expand · hover floats the mean activation'}
        </span>
      </header>
      <main className="app-main">
        <window.YVV2.GraphV2
          hover={hover}
          selected={selected}
          onHover={onHover}
          onSelect={onSelect}
          onExpandedCountChange={onExpandedCountChange}
        />
        <FlowOverlayV2 active={hover || selected} lastActive={lastActive} />
        <DetailPanelV2 selected={selected} onClose={() => setSelected(null)} panelRef={panelRef} />
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<AppV2 />);
