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
    blurb: 'Runs three parallel conv heads (one per scale) and emits boxes + class scores per anchor cell. Final detections come from a top-K filter on the one-to-one branch (NMS-free).',
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

  // Remember the last activation image we successfully rendered, so a step
  // that has no 4-D tensor captured (chunk/getitem/elementwise-add returning a
  // list, etc.) can fall back to "what we were just showing" instead of
  // snapping all the way back to the raw input image. This keeps the play-flow
  // illusion intact when traversing fx nodes that don't yield a renderable tensor.
  const lastImgRef = useRef({ src: null, label: null, sub: null });

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
  let detectFrame = null;

  let usingFallback = false;
  if (display) {
    const block = window.YV_ACT?.nodes?.[String(display.idx)];
    const type = block?.type;
    if (type === 'Detect' && block?.detect) {
      // Annotated final-detections frame — terminus of the flow.
      const survivors = (block.detect.boxes || []);
      detectFrame = { survivors, losers: [] };
      label = `[${display.idx}] final detections`;
      sub = `${survivors.length} survivor${survivors.length === 1 ? '' : 's'}`;
      lastImgRef.current = { src: null, label, sub };
    } else if (isDeferred(display)) {
      label = `[${display.idx}] activations deferred`;
      sub = 'detect head — coming later';
    } else {
      const act = lookupActivation(display);
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
        // Remember this frame for the next no-activation node.
        lastImgRef.current = { src, label, sub };
      } else if (lastImgRef.current.src) {
        // No 4-D tensor here — keep showing the previous step's activation
        // rather than snapping back to the raw input image.
        src = lastImgRef.current.src;
        stretchedActivation = true;
        const stepLabel = display.pathKey != null
          ? (display.pathKey.split('/').pop() || display.fxKey || '')
          : friendly;
        label = `[${display.idx}] ${stepLabel} · passthrough`;
        sub = 'no 4-D tensor — showing prior activation';
        usingFallback = true;
      } else {
        label = `[${display.idx}] ${friendly || ''}`;
        sub = 'no captured activation';
      }
    }
  }

  return (
    <div className={`flow-overlay ${usingFallback ? 'using-fallback' : ''}`}>
      <div className="flow-overlay__frame" style={{ aspectRatio: aspect }}>
        {detectFrame ? (
          <BoxOverlayImage
            imageUrl={inputImageUrl}
            survivors={detectFrame.survivors}
            losers={detectFrame.losers}
            fillContainer
          />
        ) : (
          <img
            src={src} alt=""
            style={{
              imageRendering: stretchedActivation ? 'pixelated' : 'auto',
              background: stretchedActivation ? '#0f172a' : 'transparent',
            }}
          />
        )}
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
// Detect panel — bbox overlay + threshold control + bar chart + per-class grid
// =============================================================================

const BOX_PALETTE = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7','#ec4899','#14b8a6','#f97316'];
const colorForCls = (clsId) => BOX_PALETTE[(clsId ?? 0) % BOX_PALETTE.length];

// Bbox overlay (ported from L2 app-l2.jsx:184).
function BoxOverlayImage({ imageUrl, survivors, losers, maxWidth = 300, fillContainer = false }) {
  const labelTag = (b) => (
    <div
      style={{
        position: 'absolute',
        left: `${b.x1 * 100}%`,
        top: b.y1 > 0.05 ? `calc(${b.y1 * 100}% - 14px)` : `${b.y1 * 100}%`,
        fontSize: 10, lineHeight: '14px', padding: '0 5px',
        background: colorForCls(b.cls_id), color: 'white',
        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
        fontWeight: 500, borderRadius: 2, whiteSpace: 'nowrap',
      }}
    >
      {b.cls_name} {b.conf.toFixed(2)}
    </div>
  );
  return (
    <div
      style={
        fillContainer
          ? { position: 'relative', width: '100%', height: '100%' }
          : { position: 'relative', display: 'inline-block', maxWidth: '100%' }
      }
    >
      <img
        src={imageUrl}
        alt=""
        style={
          fillContainer
            ? { display: 'block', width: '100%', height: '100%', objectFit: 'cover' }
            : { display: 'block', width: '100%', maxWidth, borderRadius: 6, border: '1px solid var(--line)' }
        }
      />
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {losers.map((b, i) => (
          <div key={`L${i}`} style={{
            position: 'absolute',
            left: `${b.x1 * 100}%`,
            top: `${b.y1 * 100}%`,
            width: `${Math.max(0.001, b.x2 - b.x1) * 100}%`,
            height: `${Math.max(0.001, b.y2 - b.y1) * 100}%`,
            border: `1.25px dashed ${colorForCls(b.cls_id)}`,
            borderRadius: 2, opacity: 0.5, boxSizing: 'border-box',
          }} />
        ))}
        {survivors.map((b, i) => (
          <React.Fragment key={`S${i}`}>
            <div style={{
              position: 'absolute',
              left: `${b.x1 * 100}%`,
              top: `${b.y1 * 100}%`,
              width: `${Math.max(0.001, b.x2 - b.x1) * 100}%`,
              height: `${Math.max(0.001, b.y2 - b.y1) * 100}%`,
              border: `2px solid ${colorForCls(b.cls_id)}`,
              borderRadius: 2, boxSizing: 'border-box',
            }} />
            {labelTag(b)}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// Per-class score grid (ported from L1 panel.jsx:419).
// classCount caps how many of the available classes are rendered.
function ScaleGridV2({ scales, classCount, imageUrl }) {
  const scaleNames = ['P3', 'P4', 'P5'];
  const sizeMap = { P3: 'small', P4: 'medium', P5: 'large' };
  const baseClasses = scales[scaleNames[0]]?.classes || [];
  const n = Math.max(0, Math.min(classCount, baseClasses.length));

  return (
    <div className="scale-grid">
      <div className="scale-grid-header first">class</div>
      {scaleNames.map((name) => (
        <div key={name} className="scale-grid-header">
          {name} <span className="sub">stride {scales[name]?.stride} · {sizeMap[name]}</span>
        </div>
      ))}

      <div className="scale-grid-imgrow">
        <div style={{ fontSize: 11, color: 'var(--ink-3)', padding: '0 4px' }}>image</div>
        {scaleNames.map((name) => (
          <img key={name} src={imageUrl} alt="" className="scale-ref" />
        ))}
      </div>

      {Array.from({ length: n }).map((_, ci) => {
        const cells = scaleNames.map(name => scales[name]?.classes?.[ci]).filter(Boolean);
        if (!cells.length) return null;
        const peaks = cells.map(c => c.peak);
        const peakScale = peaks.indexOf(Math.max(...peaks));
        const maxPeak = Math.max(...peaks);
        return (
          <React.Fragment key={ci}>
            <div className="scale-grid-class">
              <span className="cls-name">{cells[0].name}</span>
              <span className="cls-peak">peak {maxPeak.toFixed(2)}</span>
            </div>
            {cells.map((c, i) => {
              const isPeak = i === peakScale && c.peak >= 0.05;
              const faint = c.peak < 0.02;
              return (
                <div key={i} className={`scale-grid-cell ${isPeak ? 'is-peak' : ''} ${faint ? 'faint' : ''}`}>
                  <img src={c.png} alt="" />
                  <span className="cell-score">{c.peak.toFixed(2)}</span>
                </div>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function DetectPanelV2({ selected, block, archBlock, roleColor, role, onClose }) {
  const detect = block?.detect;
  const meta = window.YV_ACT?.meta;
  const inputImageUrl = meta ? '../' + meta.image : '../assets/sammit_lighthouse.jpg';

  const candidates = detect?.candidate_boxes && detect.candidate_boxes.length > 0
    ? detect.candidate_boxes
    : (detect?.boxes || []);
  const scales = detect?.scales || {};
  const availableClasses = scales.P3?.classes?.length || 0;

  const [confThreshold, setConfThreshold] = useState(0.25);
  const [classCount, setClassCount] = useState(Math.min(6, availableClasses || 6));

  const survivors = candidates.filter(b => b.conf >= confThreshold);
  const losers = candidates.filter(b => b.conf < confThreshold);

  const copy = copyFor('Detect');
  const title = `[${selected.idx}] ${copy.title}`;

  // Numeric input is decoupled from slider min so very low values can be typed.
  const onTypedThreshold = (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v) && v >= 0 && v <= 1) setConfThreshold(v);
  };

  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">{title}</div>
          <div className="panel-desc">{copy.blurb}</div>
          {archBlock?.desc && <div className="panel-path-hint">{archBlock.desc}</div>}
          <span className="role-pill" style={{ background: roleColor + '22', color: roleColor, marginTop: 10 }}>
            {role}
          </span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>

      {!detect && (
        <section className="panel-section">
          <div className="panel-empty">
            No detect payload captured. Re-run <code>yolovex build-assets-v2</code> to populate detections.
          </div>
        </section>
      )}

      {detect && (
        <>
          <section className="panel-section">
            <h4 className="section-label">Final detections</h4>
            <p className="micro-help">
              Solid boxes are above your threshold ({confThreshold.toFixed(3)}). Faded dashed boxes are runners-up the head also emitted but that fall below it. Slide or type a different threshold to watch which boxes graduate or get dropped.
            </p>
            <BoxOverlayImage
              imageUrl={inputImageUrl}
              survivors={survivors}
              losers={losers}
              maxWidth={420}
            />

            <div className="detect-conf-box">
              <div className="detect-conf-row">
                <strong>Confidence threshold</strong>
                <span className="num">{confThreshold.toFixed(3)}</span>
              </div>
              <div className="detect-conf-controls">
                <input
                  type="range"
                  min={0.005} max={1} step={0.005}
                  value={Math.min(1, Math.max(0.005, confThreshold))}
                  onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                />
                <input
                  type="number"
                  min={0} max={1} step="any"
                  value={confThreshold}
                  onChange={onTypedThreshold}
                  title="Type any value 0–1 (slider clamps to 0.005)"
                />
              </div>
              <div className="detect-conf-bounds">
                <span>0.005</span><span>1.000</span>
              </div>
              <div className="detect-conf-stats">
                <strong style={{ color: 'var(--ink)' }}>{survivors.length}</strong> survivors ·{' '}
                <span className="muted">{losers.length} runners-up</span> ·{' '}
                <span className="muted">{candidates.length} total candidates</span>
              </div>
            </div>

            {survivors.length > 0 ? (
              <div className="det-bars">
                {survivors.map((b, i) => (
                  <div key={i} className="det-bar">
                    <span className="dot" style={{ background: colorForCls(b.cls_id) }} />
                    <span className="name" title={b.cls_name}>{b.cls_name}</span>
                    <span className="track">
                      <span className="fill" style={{
                        width: `${Math.min(100, b.conf * 100)}%`,
                        background: colorForCls(b.cls_id),
                        opacity: 0.78,
                      }} />
                    </span>
                    <span className="conf">{b.conf.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="det-empty">No detections above the current threshold. Slide it down.</div>
            )}
          </section>

          {availableClasses > 0 && (
            <section className="panel-section">
              <h4 className="section-label">Per-class score heatmaps</h4>
              <p className="micro-help">
                Top {classCount} classes by peak score across all scales, arranged so each <strong>row</strong> is a class and each <strong>column</strong> is a pyramid scale. Easy way to see <em>which scale a class actually fires on</em> — small objects light up in P3, large ones in P5. The cell with the brightest peak for that class is highlighted.
              </p>
              <div className="detect-classes-control">
                <span>Classes shown:</span>
                <input
                  type="number"
                  min={1}
                  max={availableClasses}
                  step={1}
                  value={classCount}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (Number.isFinite(v)) setClassCount(Math.max(1, Math.min(availableClasses, v)));
                  }}
                />
                <span style={{ color: 'var(--ink-4)' }}>of {availableClasses} available</span>
              </div>
              <ScaleGridV2 scales={scales} classCount={classCount} imageUrl={inputImageUrl} />
            </section>
          )}

          <section className="panel-section">
            <h4 className="section-label">Structure</h4>
            <div className="stats-grid">
              <div><span className="k">classes</span><span className="v mono">{detect.nc ?? '—'}</span></div>
              <div><span className="k">strides</span><span className="v mono">{Array.isArray(detect.strides) ? detect.strides.join(', ') : '—'}</span></div>
              <div><span className="k">survivors (conf ≥ 0.05)</span><span className="v mono">{(detect.boxes || []).length}</span></div>
              <div><span className="k">total candidates</span><span className="v mono">{candidates.length}</span></div>
            </div>
          </section>
        </>
      )}
    </div>
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

  // Detect blocks (and any sub-click inside one — P3/P4/P5 etc.) all open the
  // same dedicated Detect panel: bbox overlay + threshold + bar chart + per-class grid.
  if (block?.type === 'Detect') {
    return (
      <aside className="detail-panel open" ref={panelRef}>
        <DetectPanelV2
          selected={selected}
          block={block}
          archBlock={archBlock}
          roleColor={roleColor}
          role={role}
          onClose={onClose}
        />
      </aside>
    );
  }

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

const FLOW_SPEEDS = { slow: 700, medium: 250, fast: 60 };

// =============================================================================
// Settings panel — live editors for layout / color / stroke / CSS tokens
// =============================================================================

const SETTINGS_GROUPS = [
  {
    label: 'Spacing & gaps',
    keys: ['ROW_GAP', 'COL_GAP', 'CONTAINER_GAP', 'NECK_Y_OFFSET_FOOT', 'NECK_Y_OFFSET_BODY', 'DETECT_GAP'],
  },
  {
    label: 'Node & container',
    keys: ['NODE_W', 'NODE_H', 'COL_TOP', 'CONTAINER_PAD', 'CONTAINER_PAD_T'],
  },
  {
    label: 'Edge tails',
    keys: ['H_ENTRY', 'H_EXIT', 'V_ENTRY', 'V_EXIT'],
  },
  {
    label: 'Edge stroke',
    keys: ['EDGE_STROKE_DEFAULT', 'EDGE_STROKE_FOCUSED'],
    step: 0.1,
  },
];

const SETTINGS_COLORS = [
  'ACCENT_COLOR', 'EDGE_COLOR_DEFAULT', 'EDGE_COLOR_DIMMED', 'EDGE_COLOR_FOCUSED',
];

// CSS variables on :root we let the user retint live.
const CSS_TOKENS = [
  { name: '--bg',       label: 'Page bg' },
  { name: '--bg-tint',  label: 'Section bg' },
  { name: '--ink',      label: 'Text' },
  { name: '--ink-2',    label: 'Text (2)' },
  { name: '--ink-3',    label: 'Text (3)' },
  { name: '--line',     label: 'Line' },
  { name: '--accent',   label: 'Accent (CSS)' },
];

function readCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function SettingsPanel({ rev, bump, onClose }) {
  const LS = window.YVV2.LAYOUT_SETTINGS;
  const DEF = window.YVV2.LAYOUT_SETTINGS_DEFAULTS;

  const setNum = (key, raw) => {
    const v = parseFloat(raw);
    if (Number.isFinite(v)) {
      LS[key] = v;
      bump();
    }
  };
  const setStr = (key, val) => {
    LS[key] = val;
    bump();
  };
  const setCssVar = (name, val) => {
    document.documentElement.style.setProperty(name, val);
    bump();
  };

  const reset = () => {
    Object.keys(DEF).forEach(k => { LS[k] = DEF[k]; });
    // Reset CSS variables (just remove inline overrides so :root takes over).
    CSS_TOKENS.forEach(t => document.documentElement.style.removeProperty(t.name));
    document.documentElement.style.removeProperty('--brochure-thumb-scale');
    document.documentElement.style.removeProperty('--scale-grid-cell-scale');
    bump();
  };

  return (
    <aside className="settings-panel">
      <header className="settings-header">
        <strong>Settings</strong>
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="settings-reset" onClick={reset} title="Reset all settings to defaults">reset</button>
          <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
        </div>
      </header>

      <div className="settings-body">
        {SETTINGS_GROUPS.map(group => (
          <div key={group.label} className="settings-group">
            <div className="settings-group-label">{group.label}</div>
            {group.keys.map(k => (
              <div key={k} className="settings-row">
                <label>{k}</label>
                <input
                  type="number"
                  step={group.step || 1}
                  value={LS[k]}
                  onChange={(e) => setNum(k, e.target.value)}
                />
              </div>
            ))}
          </div>
        ))}

        <div className="settings-group">
          <div className="settings-group-label">Colors (SVG)</div>
          {SETTINGS_COLORS.map(k => (
            <div key={k} className="settings-row">
              <label>{k}</label>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <input
                  type="color"
                  value={LS[k]}
                  onChange={(e) => setStr(k, e.target.value)}
                />
                <input
                  type="text"
                  value={LS[k]}
                  onChange={(e) => setStr(k, e.target.value)}
                  style={{ width: 84 }}
                />
              </div>
            </div>
          ))}
          <div className="settings-row">
            <label>CONTAINER_DASH</label>
            <input
              type="text"
              value={LS.CONTAINER_DASH}
              onChange={(e) => setStr('CONTAINER_DASH', e.target.value)}
              style={{ width: 84 }}
              title="SVG strokeDasharray (e.g. '4 4' or '6 3')"
            />
          </div>
        </div>

        <div className="settings-group">
          <div className="settings-group-label">CSS tokens</div>
          {CSS_TOKENS.map(t => {
            const val = readCssVar(t.name);
            const isColor = val.startsWith('#') || val.startsWith('rgb');
            return (
              <div key={t.name} className="settings-row">
                <label>{t.label} <code>{t.name}</code></label>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  {isColor && (
                    <input type="color" value={val.length === 7 ? val : '#000000'}
                      onChange={(e) => setCssVar(t.name, e.target.value)} />
                  )}
                  <input type="text" defaultValue={val}
                    onBlur={(e) => setCssVar(t.name, e.target.value)}
                    style={{ width: 96 }} />
                </div>
              </div>
            );
          })}
          <div className="settings-row">
            <label>Brochure thumb scale</label>
            <input
              type="number" step="0.05" min="0.3" max="1.2"
              defaultValue={readCssVar('--brochure-thumb-scale') || 0.7}
              onChange={(e) => setCssVar('--brochure-thumb-scale', e.target.value)}
            />
          </div>
          <div className="settings-row">
            <label>Scale-grid cell scale</label>
            <input
              type="number" step="0.05" min="0.3" max="1.2"
              defaultValue={readCssVar('--scale-grid-cell-scale') || 0.7}
              onChange={(e) => setCssVar('--scale-grid-cell-scale', e.target.value)}
            />
          </div>
        </div>
      </div>
    </aside>
  );
}

function AppV2() {
  const [hover, setHover] = useState(null);
  const [lastActive, setLastActive] = useState(null);  // sticky for the overlay
  const [selected, setSelected] = useState(null);
  const [expandedCount, setExpandedCount] = useState(0);
  const onExpandedCountChange = useCallback((n) => setExpandedCount(n), []);
  const panelRef = useRef(null);

  // Play-flow state — drives the floating overlay through every visible node.
  const [visibleOrder, setVisibleOrder] = useState([]);
  const [playing, setPlaying] = useState(null);   // current payload, or null
  const [speedKey, setSpeedKey] = useState('medium');
  const playTimerRef = useRef(null);
  const playStopRef = useRef(false);

  // Settings panel — rev counter forces GraphV2 useMemo to recompute when
  // any setting changes (the layout/graph code reads from window.YVV2.LAYOUT_SETTINGS
  // at call time, so we just need to invalidate the memo).
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsRev, setSettingsRev] = useState(0);
  const bumpSettings = useCallback(() => setSettingsRev(r => r + 1), []);

  // Theme (light / dark) — applied as data-theme on <html> so the CSS overrides
  // in yolovexv2.html flip every surface var. Also nudges a few SVG colors
  // (edge defaults) to a darker shade so they stay legible on the dark canvas.
  const [theme, setTheme] = useState(() => document.documentElement.getAttribute('data-theme') || 'light');
  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      const LS = window.YVV2.LAYOUT_SETTINGS;
      const DEF = window.YVV2.LAYOUT_SETTINGS_DEFAULTS;
      if (next === 'dark') {
        LS.EDGE_COLOR_DEFAULT = '#64748b';
        LS.EDGE_COLOR_DIMMED  = '#334155';
      } else {
        LS.EDGE_COLOR_DEFAULT = DEF.EDGE_COLOR_DEFAULT;
        LS.EDGE_COLOR_DIMMED  = DEF.EDGE_COLOR_DIMMED;
      }
      setSettingsRev(r => r + 1);
      return next;
    });
  }, []);

  const onVisibleOrderChange = useCallback((order) => setVisibleOrder(order), []);

  const stopPlay = useCallback(() => {
    playStopRef.current = true;
    if (playTimerRef.current) {
      clearTimeout(playTimerRef.current);
      playTimerRef.current = null;
    }
    setPlaying(null);
  }, []);

  const startPlay = useCallback(() => {
    if (!visibleOrder.length) return;
    playStopRef.current = false;
    const tickMs = FLOW_SPEEDS[speedKey] || FLOW_SPEEDS.medium;
    let i = 0;
    const step = () => {
      if (playStopRef.current) return;
      if (i >= visibleOrder.length) {
        // Park on the final frame (Detect → annotated image) — flow ends here
        // and the overlay persists until the user hovers another node or hits Play again.
        setPlaying(null);
        return;
      }
      const payload = visibleOrder[i];
      setPlaying(payload);
      setLastActive(payload);
      i += 1;
      playTimerRef.current = setTimeout(step, tickMs);
    };
    step();
  }, [visibleOrder, speedKey]);

  // Stop playback if the user hovers any block (gives them control back).
  useEffect(() => {
    if (playing && hover && (hover.idx !== playing.idx || hover.pathKey !== playing.pathKey)) {
      stopPlay();
    }
  }, [hover, playing, stopPlay]);

  useEffect(() => () => stopPlay(), [stopPlay]);

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
      // Clicks inside the Settings panel shouldn't dismiss the detail panel —
      // makes it possible to tune settings live while watching the panel react.
      if (e.target.closest && e.target.closest('.settings-panel')) return;
      // Same for the header (flow controls / settings toggle) and flow overlay.
      if (e.target.closest && (e.target.closest('.app-header') || e.target.closest('.flow-overlay'))) return;
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
        <div className="flow-controls">
          <button
            className="flow-btn"
            onClick={() => (playing ? stopPlay() : startPlay())}
            title={playing ? 'Stop flow' : 'Play flow — traverse all visible blocks'}
          >
            {playing ? '■ Stop' : '▶ Play flow'}
          </button>
          <span className="flow-speed">
            {Object.keys(FLOW_SPEEDS).map(k => (
              <button
                key={k}
                className={`flow-speed-btn ${speedKey === k ? 'active' : ''}`}
                onClick={() => setSpeedKey(k)}
              >{k}</button>
            ))}
          </span>
          <span className="flow-count">{visibleOrder.length} steps</span>
        </div>
        <button
          className="settings-toggle"
          onClick={() => setSettingsOpen(o => !o)}
          title="Open layout / color settings"
        >⚙ settings</button>
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
          playing={playing}
          onHover={onHover}
          onSelect={onSelect}
          onExpandedCountChange={onExpandedCountChange}
          onVisibleOrderChange={onVisibleOrderChange}
          settingsRev={settingsRev}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
        <FlowOverlayV2 active={playing || hover || selected} lastActive={lastActive} />
        <DetailPanelV2 selected={selected} onClose={() => setSelected(null)} panelRef={panelRef} />
        {settingsOpen && (
          <SettingsPanel rev={settingsRev} bump={bumpSettings} onClose={() => setSettingsOpen(false)} />
        )}
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<AppV2 />);
