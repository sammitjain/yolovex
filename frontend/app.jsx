// yolovex — app shell.
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

// Op-node kinds (subkind is classified in expand.jsx and rides on the payload).
const isSplitOp = (sel) => sel?.subkind === 'split';
const isShapeOp = (sel) => sel?.subkind === 'shape';

// Per-node instance shapes for one L1 block: { fxName: [..]|[[..]]|null }.
// A null (or non-array) shape marks a scalar / shape-argument node, not a tensor.
function instanceShapes(idx) {
  const inst = window.YV_SPEC?.instances?.find(i => i.idx === idx);
  return inst?.shapes_by_node || {};
}

// A split/chunk op's outputs are its (canvas-hidden) getitem children, each one
// already captured as a 4-D tensor in YV_ACT.sub. Returns them ordered by the
// getitem index, with each piece's captured activation (carrying its own shape).
function splitOpOutputs(selected) {
  const block = window.YV_ACT?.nodes?.[String(selected.idx)];
  const arch = window.YV.buildArch();
  const archBlock = arch.find(a => a.idx === selected.idx);
  const spec = archBlock && window.YV_SPEC?.specs?.[archBlock.specId];
  if (!block || !spec || !selected.fxKey) return [];
  const nameToNode = new Map(spec.graph.nodes.map(n => [n.name, n]));
  const outs = [];
  for (const [s, t] of spec.graph.edges) {
    if (s !== selected.fxKey) continue;
    const n = nameToNode.get(t);
    if (!n) continue;
    if (String(n.target || '').split('.').pop() !== 'getitem') continue;
    const gi = Array.isArray(n.args) ? n.args[1] : null;
    outs.push({ index: typeof gi === 'number' ? gi : outs.length, fxKey: t, act: block.sub?.[t] || null });
  }
  outs.sort((a, b) => a.index - b.index);
  return outs;
}

// Fit a tensor's true H×W (last two dims) into a max box, preserving aspect, so
// reshaped / non-image tensors aren't stretched to the image aspect. Used by
// both the brochure preview and the IO-strip tiles.
function fitBox(shape, maxW, maxH) {
  if (!Array.isArray(shape) || shape.length < 2) return { width: maxW, height: maxH };
  const H = shape[shape.length - 2], W = shape[shape.length - 1];
  if (!H || !W) return { width: maxW, height: maxH };
  let w = maxW, h = (maxW * H) / W;
  if (h > maxH) { h = maxH; w = (maxH * W) / H; }
  return { width: Math.round(w), height: Math.round(h) };
}

// Learner-facing verb for a shape op, by its fx node name (strip trailing _N).
function shapeOpVerb(fxKey) {
  const op = String(fxKey || '').replace(/_\d+$/, '');
  if (op === 'transpose' || op === 'permute') return 'axes reordered';
  if (op === 'flatten') return 'flattened';
  if (op === 'squeeze' || op === 'unsqueeze') return 'reshaped';
  return 'reshaped';
}

// =============================================================================
// Learner-facing copy per type. Content lives in content/blocks.js
// (window.YV_CONTENT); this only reads from it. title/blurb feed the panel
// header, the rest feed BlockContent.
// =============================================================================

// Functional ops carry no module class; route them to a content key by their
// fx `target`. `cat` reuses the real Concat copy; other ops fall through to a
// same-named scaffold entry in blocks.js (e.g. YV_CONTENT.add).
const OP_COPY_MAP = { cat: 'Concat' };

function copyFor(typeOrSub) {
  const c = window.YV_CONTENT && window.YV_CONTENT[typeOrSub];
  return { title: c?.title || typeOrSub || 'node', blurb: c?.blurb || '' };
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
  const arch = window.YV.buildArch();
  const block = arch.find(a => a.idx === idx);
  const specId = block?.specId;
  const spec = specId && window.YV_SPEC?.specs?.[specId];
  if (!spec) return [];

  const memberSet = new Set(members);
  const nameToNode = new Map(spec.graph.nodes.map(n => [n.name, n]));
  const captured = window.YV_ACT?.nodes?.[String(idx)]?.sub || {};
  const shapes = instanceShapes(idx);

  const incomingByDst = new Map();
  for (const [s, t] of spec.graph.edges) {
    if (!incomingByDst.has(t)) incomingByDst.set(t, []);
    incomingByDst.get(t).push(s);
  }

  // Walk back through nodes the visual graph hides — getitem unconditionally
  // (preprocessGraph in expand.jsx line 42) plus anything we didn't capture
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
    // Drop scalar / shape-argument nodes (e.g. the B,C,H,W extracted from
    // x.shape feeding a reshape) — they're tracked as fx edges but carry no
    // tensor data, so they must not count as inputs nor be chased to the L1
    // input. A known non-tensor shape (null / not a list) is the signal.
    if (Object.prototype.hasOwnProperty.call(shapes, name) && !Array.isArray(shapes[name])) {
      return [];
    }
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
  return s.join('×');
}

// =============================================================================
// Floating overlay — persists last seen activation
// =============================================================================

function FlowOverlay({ active, lastActive }) {
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
      // Shape ops capture a tensor, but it only relabels axes — showing it
      // stretched to image aspect would mislead. Force passthrough with a
      // learner-friendly caption instead (the image story is unchanged here).
      if (!isShapeOp(display) && act && act.mean) {
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
        if (isShapeOp(display)) {
          label = `[${display.idx}] ${stepLabel} · ${shapeOpVerb(display.fxKey)}`;
          sub = 'layout only — image unchanged';
        } else {
          label = `[${display.idx}] ${stepLabel} · passthrough`;
          sub = 'no 4-D tensor — showing prior activation';
        }
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

function IOStrip({ active, output }) {
  const meta = window.YV_ACT?.meta;
  const inputImageUrl = meta ? '../' + meta.image : '../assets/sammit_lighthouse.jpg';

  const sources = active.pathKey == null
    ? l1UpstreamSources(active.idx)
    : subUpstreamSources(active.idx, active.members);

  // Shape ops relabel axes without changing values — thumbnails would be
  // misleading (and identical-looking). Show the transform as in→out shapes.
  if (isShapeOp(active)) {
    const inShape = sources.length ? shapeOfSource(sources[0]) : null;
    return (
      <section className="panel-section">
        <h4 className="section-label">Shape transform</h4>
        <div className="shape-transform">
          <span className="shape-transform__dim mono">{fmtShape(inShape)}</span>
          <span className="shape-transform__arrow">→</span>
          <span className="shape-transform__dim mono">{fmtShape(output?.shape)}</span>
        </div>
        <div className="shape-caption">Same values, re-laid-out — no data changes here.</div>
      </section>
    );
  }

  const isMulti = sources.length > 1;
  // Each tile is sized from its tensor's own H×W so non-image tensors (the
  // split's 32×300 / 64×300 pieces, softmax's 300×300, …) render truthfully
  // instead of being squashed to the image aspect. Multi-input strips use a
  // slightly smaller box.
  const inMaxW = isMulti ? 80 : 96, inMaxH = isMulti ? 100 : 120;

  const inputCaption = sources
    .map(s => fmtShape(shapeOfSource(s)))
    .join(' + ') || '…';

  return (
    <section className="panel-section">
      <h4 className="section-label">Input → Output</h4>
      <div className="io-strip">
        {sources.length === 0 ? (
          <div className="io-tile" style={{ width: inMaxW, height: inMaxH, background: '#f1f5f9' }} />
        ) : (
          <div className="io-pair">
            {sources.map((s, i) => {
              const src = thumbnailForSource(s, inputImageUrl);
              const dims = fitBox(shapeOfSource(s), inMaxW, inMaxH);
              return (
                <React.Fragment key={i}>
                  {i > 0 && <span className="plus">+</span>}
                  {src
                    ? <img src={src} alt="" className="io-tile" style={dims} />
                    : <div className="io-tile" style={{ ...dims, background: '#f1f5f9' }} />}
                </React.Fragment>
              );
            })}
          </div>
        )}
        <span className="arrow">→</span>
        {output?.mean
          ? <img src={output.mean} alt="" className="io-tile" style={fitBox(output.shape, 96, 120)} />
          : <div className="io-tile" style={{ width: 96, height: 120, background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#94a3b8' }}>no output</div>
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
function ScaleGrid({ scales, classCount, imageUrl }) {
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

function DetectPanel({ selected, block, archBlock, roleColor, role, onClose }) {
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
            No detect payload captured. Re-run <code>yolovex build-assets</code> to populate detections.
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
              <ScaleGrid scales={scales} classCount={classCount} imageUrl={inputImageUrl} />
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

// Minimal markdown renderer for the curated explainer text in blocks.js.
// Supports paragraphs (blank-line separated), bullets (`- ` lines),
// `inline code` and **bold**. No fenced blocks needed — the curated text
// uses inline-only code spans. Safe (no dangerouslySetInnerHTML).
function MDInline({ text }) {
  // Split into runs of: bold (**...**) | code (`...`) | plain
  const parts = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push({ k: 't', v: text.slice(last, m.index) });
    const tok = m[0];
    if (tok.startsWith('**')) parts.push({ k: 'b', v: tok.slice(2, -2) });
    else                      parts.push({ k: 'c', v: tok.slice(1, -1) });
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push({ k: 't', v: text.slice(last) });
  return (
    <>
      {parts.map((p, i) => {
        if (p.k === 'b') return <strong key={i}>{p.v}</strong>;
        if (p.k === 'c') return <code key={i} className="md-code">{p.v}</code>;
        return <React.Fragment key={i}>{p.v}</React.Fragment>;
      })}
    </>
  );
}

function Markdown({ text }) {
  if (!text) return null;
  const blocks = String(text).split(/\n{2,}/);
  return (
    <>
      {blocks.map((block, i) => {
        const lines = block.split('\n');
        const isBulleted = lines.every(l => l.trim().startsWith('- ') || l.trim() === '');
        if (isBulleted) {
          return (
            <ul key={i} className="md-ul">
              {lines.filter(l => l.trim().startsWith('- ')).map((l, j) => (
                <li key={j}><MDInline text={l.replace(/^\s*-\s+/, '')} /></li>
              ))}
            </ul>
          );
        }
        // Paragraph — preserve single newlines as <br>
        return (
          <p key={i} className="md-p">
            {lines.map((l, j) => (
              <React.Fragment key={j}>
                {j > 0 && <br />}
                <MDInline text={l} />
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </>
  );
}

// Per-block explainer. Renders the merged content object resolved in
// DetailPanel (per-type entry from content/blocks.js + any position override).
// When no content exists for the key yet, shows a neutral placeholder rather
// than nothing, so the section is always present (activations render separately).
function BlockContent({ copyKey, content }) {
  if (!copyKey) return null;
  if (!content) {
    return (
      <section className="panel-section">
        <h4 className="section-label">About {copyKey}</h4>
        <p className="content-tagline">No curated explainer yet for <code>{copyKey}</code>.</p>
      </section>
    );
  }
  return (
    <section className="panel-section">
      <h4 className="section-label">About {copyKey}</h4>
      {content?.tagline && <p className="content-tagline"><MDInline text={content.tagline} /></p>}
      {content?.intuition && (
        <div className="content-block">
          <div className="content-sublabel">Intuition</div>
          <Markdown text={content.intuition} />
        </div>
      )}
      {content?.technical && (
        <div className="content-block">
          <div className="content-sublabel">How it works</div>
          <Markdown text={content.technical} />
        </div>
      )}
      {content?.shape && (
        <div className="content-block">
          <div className="content-sublabel">Shape</div>
          <Markdown text={content.shape} />
        </div>
      )}
      {content?.yolo26 && (
        <div className="content-block content-yolo26">
          <div className="content-sublabel">YOLO26 novelty</div>
          <Markdown text={content.yolo26} />
        </div>
      )}
      {content?.refs && content.refs.length > 0 && (
        <div className="content-refs">
          {content.refs.map((r, i) => (
            <a key={i} href={r.url} target="_blank" rel="noreferrer">{r.label}</a>
          ))}
        </div>
      )}
    </section>
  );
}

function DetailPanel({ selected, onClose, panelRef }) {
  const [pinnedCh, setPinnedCh] = useState(0);
  const [hoveredCh, setHoveredCh] = useState(null);
  // How many of the available top-K thumbs to render. The backend emits up to
  // 16 (TOP_K_CHANNELS); the user can lower this for a tighter view. The grid
  // is auto-fill, so the visible columns adapt to whatever K + thumb size
  // produces — last row may have fewer items.
  const [topK, setTopK] = useState(8);
  // Which output of a split/chunk op is shown (its pieces are separate tensors).
  const [activeOut, setActiveOut] = useState(0);

  // Reset pinned/hovered/output when the selection changes.
  useEffect(() => { setPinnedCh(0); setHoveredCh(null); setActiveOut(0); }, [selected?.idx, selected?.fxKey, selected?.pathKey]);

  if (!selected) return <aside className="detail-panel" ref={panelRef} aria-hidden />;

  const block = window.YV_ACT?.nodes?.[String(selected.idx)];
  const arch = window.YV.buildArch();
  const archBlock = arch.find(a => a.idx === selected.idx);
  const ROLE_COLORS = window.YV.ROLE_COLORS;
  const role = archBlock?.role || 'Backbone';
  const roleColor = ROLE_COLORS[role] || '#64748b';

  // Detect blocks (and any sub-click inside one — P3/P4/P5 etc.) all open the
  // same dedicated Detect panel: bbox overlay + threshold + bar chart + per-class grid.
  if (block?.type === 'Detect') {
    return (
      <aside className="detail-panel open" ref={panelRef}>
        <DetectPanel
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

  // What "type" copy applies. For L1 use block.type. For a sub-node, the
  // authoritative class is spec.path_classes[pathKey] — correct for collapsed
  // containers (cv1 → Conv) AND module leaves (cv1/act → SiLU), avoiding the
  // last-member fxKey pitfall. Functional ops (cat/add/…) aren't in
  // path_classes; route them by their fx `target` (cat → Concat; others fall
  // through to a same-named scaffold key).
  let copyKey = block?.type;
  let subTypeLabel = null;
  if (selected.pathKey != null && archBlock) {
    const spec = window.YV_SPEC?.specs?.[archBlock.specId];
    const cls = spec?.path_classes?.[selected.pathKey];
    if (cls) {
      copyKey = cls;
      subTypeLabel = cls;
    } else if (spec && selected.fxKey) {
      const node = spec.graph.nodes.find(n => n.name === selected.fxKey);
      if (node) {
        // fx targets can be qualified (e.g. `_VariableFunctionsClass.cat`);
        // take the last segment so `cat`/`add`/`chunk`/`getitem` map cleanly.
        const op = String(node.target || '').split('.').pop();
        copyKey = OP_COPY_MAP[op] || node.target_class || op;
        subTypeLabel = node.target_class || op;
      }
    }
  }

  // Position-specific overrides (e.g. C3k2's YOLO26 note only at block 22),
  // keyed by `idx` for L1 or `idx/pathKey` for a sub-node. Merged over the
  // per-type entry, override winning per field.
  const posKey = selected.pathKey == null
    ? String(selected.idx)
    : `${selected.idx}/${selected.pathKey}`;
  const baseContent = (window.YV_CONTENT && window.YV_CONTENT[copyKey]) || null;
  const override = (window.YV_CONTENT_OVERRIDES && window.YV_CONTENT_OVERRIDES[posKey]) || null;
  const mergedContent = (baseContent || override)
    ? { ...baseContent, ...override }
    : null;

  const copy = {
    title: mergedContent?.title || copyKey || 'node',
    blurb: mergedContent?.blurb || '',
  };

  const title = selected.pathKey == null
    ? `[${selected.idx}] ${copy.title}`
    : `[${selected.idx} · ${selected.pathKey}] ${copy.title}`;

  const deferred = isDeferred(selected);
  // A split/chunk op carries no single tensor; its pieces are the getitem
  // children. Show one brochure per piece, switchable via tabs.
  const splitOuts = !deferred && isSplitOp(selected) ? splitOpOutputs(selected) : [];
  const act = deferred
    ? null
    : (splitOuts.length ? splitOuts[Math.min(activeOut, splitOuts.length - 1)].act
       : lookupActivation(selected));

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

        {/* Per-class explainer is always shown when we have content for the
            key, regardless of whether activations are available — e.g. a
            shape-only op still has class-level context worth reading. */}
        <BlockContent copyKey={copyKey} content={mergedContent} />

        {act && (
          <>
            {splitOuts.length > 1 && (
              <div className="split-out-tabs">
                {splitOuts.map((o, i) => (
                  <button
                    key={o.fxKey}
                    className={`split-out-tab ${i === Math.min(activeOut, splitOuts.length - 1) ? 'active' : ''}`}
                    onClick={() => setActiveOut(i)}
                  >
                    out {o.index}
                    <span className="split-out-tab__shape mono">{fmtShape(o.act?.shape)}</span>
                  </button>
                ))}
              </div>
            )}
            <IOStrip active={selected} output={act} />

            {visibleCh > 0 && (
              <section className="panel-section">
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 12 }}>
                  <h4 className="section-label" style={{ margin: 0 }}>Channel stack</h4>
                  <div className="topk-control">
                    <span>Top</span>
                    <input
                      type="number"
                      min={1}
                      max={visibleCh}
                      step={1}
                      value={Math.min(topK, visibleCh)}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        if (Number.isFinite(v)) setTopK(Math.max(1, Math.min(visibleCh, v)));
                      }}
                    />
                    <span>of {visibleCh}</span>
                  </div>
                </div>
                <div className="channel-brochure">
                  <div className="brochure-preview">
                    <img
                      src={act.topK[Math.min(activeCh, visibleCh - 1)]}
                      alt={`channel ${trueChIdx(activeCh)}`}
                      style={{
                        ...fitBox(act.shape, 170, 210),
                        objectFit: 'fill',     // fill — box already matches tensor aspect
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
                    {act.topK.slice(0, Math.min(topK, visibleCh)).map((b64, i) => (
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
                  Showing top {Math.min(topK, visibleCh)} of {act.totalChannels} channels, ranked by mean |activation|.
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
  'GRAPH_BG', 'ACCENT_COLOR', 'EDGE_COLOR_DEFAULT', 'EDGE_COLOR_DIMMED', 'EDGE_COLOR_FOCUSED',
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
  const LS = window.YV.LAYOUT_SETTINGS;
  const DEF = window.YV.LAYOUT_SETTINGS_DEFAULTS;

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
    // Restore the active theme's block + role palette (drops user color edits).
    const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    const TP = window.YV.TYPE_PALETTES[theme];
    const RP = window.YV.ROLE_PALETTES[theme];
    LS.TYPE_PALETTE = Object.fromEntries(Object.entries(TP).map(([k, v]) => [k, { ...v }]));
    LS.ROLE_PALETTE = { ...RP };
    LS.ACCENT_COLOR = window.YV.ACCENTS[theme];
    // Same treatment for the inner sub-node palette + nesting tints +
    // expandable-dash — restore from the matching theme preset.
    const IP = window.YV.INNER_PALETTES[theme];
    LS.INNER_PALETTE = Object.fromEntries(Object.entries(IP).map(([k, v]) => [k, { ...v }]));
    LS.NESTING_TINTS = window.YV.NESTING_TINT_SETS[theme].map(o => ({ ...o }));
    LS.EXPANDABLE_DASH = '6 3';
    LS.GRAPH_BG = window.YV.GRAPH_BGS[theme];
    // Reset CSS variables (just remove inline overrides so :root takes over).
    CSS_TOKENS.forEach(t => document.documentElement.style.removeProperty(t.name));
    document.documentElement.style.removeProperty('--brochure-thumb-scale');
    document.documentElement.style.removeProperty('--scale-grid-cell-scale');
    document.documentElement.style.removeProperty('--accent');
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
          <div className="settings-group-label">Block palette</div>
          {Object.keys(LS.TYPE_PALETTE).map(type => {
            const entry = LS.TYPE_PALETTE[type];
            const setSlot = (slot, val) => {
              LS.TYPE_PALETTE = {
                ...LS.TYPE_PALETTE,
                [type]: { ...LS.TYPE_PALETTE[type], [slot]: val },
              };
              bump();
            };
            return (
              <div key={type} className="settings-palette-row">
                <span className="settings-palette-type">{type}</span>
                {['fill', 'border', 'text'].map(slot => (
                  <div key={slot} className="settings-palette-slot" title={`${type} ${slot}`}>
                    <input
                      type="color"
                      value={entry[slot]}
                      onChange={(e) => setSlot(slot, e.target.value)}
                    />
                    <span className="slot-label">{slot}</span>
                  </div>
                ))}
              </div>
            );
          })}
          <div className="settings-group-label" style={{ marginTop: 10 }}>Role colors</div>
          {Object.keys(LS.ROLE_PALETTE).map(role => (
            <div key={role} className="settings-row">
              <label>{role}</label>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <input
                  type="color"
                  value={LS.ROLE_PALETTE[role]}
                  onChange={(e) => {
                    LS.ROLE_PALETTE = { ...LS.ROLE_PALETTE, [role]: e.target.value };
                    bump();
                  }}
                />
                <input
                  type="text"
                  value={LS.ROLE_PALETTE[role]}
                  onChange={(e) => {
                    LS.ROLE_PALETTE = { ...LS.ROLE_PALETTE, [role]: e.target.value };
                    bump();
                  }}
                  style={{ width: 76 }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="settings-group">
          <div className="settings-group-label">Sub-node palette (inside expanded blocks)</div>
          <div className="settings-group-hint">Fallback colours for non-module fx ops (cat / split / add / io). Typed leaf modules (Conv2d, BN, SiLU…) colour from the Block palette above.</div>
          {Object.keys(LS.INNER_PALETTE || {}).map(kind => {
            const entry = LS.INNER_PALETTE[kind];
            const setSlot = (slot, val) => {
              LS.INNER_PALETTE = {
                ...LS.INNER_PALETTE,
                [kind]: { ...LS.INNER_PALETTE[kind], [slot]: val },
              };
              bump();
            };
            return (
              <div key={kind} className="settings-palette-row">
                <span className="settings-palette-type">{kind}</span>
                {['fill', 'border', 'text'].map(slot => (
                  <div key={slot} className="settings-palette-slot" title={`${kind} ${slot}`}>
                    <input
                      type="color"
                      value={entry[slot]}
                      onChange={(e) => setSlot(slot, e.target.value)}
                    />
                    <span className="slot-label">{slot}</span>
                  </div>
                ))}
              </div>
            );
          })}
        </div>

        <div className="settings-group">
          <div className="settings-group-label">Nesting tints (depth bands)</div>
          <div className="settings-group-hint">Depth shading inside an expanded block. Colour now comes from the block's family — only <strong>fill opacity</strong> (denser = deeper) and <strong>stroke width</strong> are applied; the colour pickers/dash are ignored at render.</div>
          {(LS.NESTING_TINTS || []).map((tint, i) => {
            const setField = (field, val) => {
              const next = (LS.NESTING_TINTS || []).map((t, j) => j === i ? { ...t, [field]: val } : t);
              LS.NESTING_TINTS = next;
              bump();
            };
            return (
              <div key={i} style={{ borderBottom: '1px dashed var(--line)', paddingBottom: 6, marginBottom: 6 }}>
                <div className="settings-row">
                  <label>depth {i + 1} fill / stroke</label>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    <input type="color" value={tint.fill} onChange={(e) => setField('fill', e.target.value)} title="fill" />
                    <input type="color" value={tint.stroke} onChange={(e) => setField('stroke', e.target.value)} title="stroke" />
                  </div>
                </div>
                <div className="settings-row">
                  <label>fill opacity</label>
                  <input
                    type="number" step="0.05" min="0" max="1"
                    value={tint.fOpacity}
                    onChange={(e) => setField('fOpacity', parseFloat(e.target.value))}
                  />
                </div>
                <div className="settings-row">
                  <label>stroke width</label>
                  <input
                    type="number" step="0.1" min="0" max="5"
                    value={tint.sw}
                    onChange={(e) => setField('sw', parseFloat(e.target.value))}
                  />
                </div>
                <div className="settings-row">
                  <label>dash</label>
                  <input
                    type="text"
                    value={tint.dash || ''}
                    onChange={(e) => setField('dash', e.target.value)}
                    style={{ width: 84 }}
                    title="SVG strokeDasharray"
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="settings-group">
          <div className="settings-group-label">Colors (SVG)</div>
          <div className="settings-group-hint">Canvas background (<strong>GRAPH_BG</strong>), accent + edge stroke colours. These are theme-specific — switching light/dark rebases them.</div>
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
          <div className="settings-group-hint">Page chrome colours (live CSS variables). The graph canvas background is under Colors (SVG) above.</div>
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
            <label>Brochure thumb min (px)</label>
            <input
              type="number" step="2" min="32" max="160"
              defaultValue={parseInt(readCssVar('--brochure-thumb-min'), 10) || 52}
              onChange={(e) => setCssVar('--brochure-thumb-min', `${e.target.value}px`)}
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

// ============================================================================
// Server mode: detect `yolovex serve` and add an image-upload flow.
// ============================================================================

const BUILD_STAGES = [
  { key: 'load_model',  label: 'Load model' },
  { key: 'preprocess',  label: 'Preprocess image' },
  { key: 'forward',     label: 'Forward pass' },
  { key: 'fx_capture',  label: 'Capture sub-activations' },
  { key: 'detect',      label: 'Detect head' },
  { key: 'write',       label: 'Write assets' },
];

function useServerMode() {
  // null = probing, true/false = result
  const [serverMode, setServerMode] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetch('/api/health', { cache: 'no-store' })
      .then(r => r.ok ? r.json() : null)
      .then(j => { if (!cancelled) setServerMode(!!(j && j.ok)); })
      .catch(() => { if (!cancelled) setServerMode(false); });
    return () => { cancelled = true; };
  }, []);
  return serverMode;
}

// Reload activations.js into window.YV_ACT without a page reload.
async function reloadActivations() {
  const res = await fetch('/activations.js?t=' + Date.now(), { cache: 'no-store' });
  if (!res.ok) throw new Error('failed to fetch updated activations');
  const text = await res.text();
  // The file is shaped `window.YV_ACT = {...};` — eval in global scope.
  (0, eval)(text);
}

function UploadButton({ disabled, onPick }) {
  const inputRef = useRef(null);
  return (
    <>
      <button
        className="settings-toggle upload-btn"
        onClick={() => inputRef.current && inputRef.current.click()}
        disabled={disabled}
        title={disabled ? 'A build is already running' : 'Upload a custom image and regenerate activations'}
      >📷 upload image</button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files && e.target.files[0];
          if (f) onPick(f);
          e.target.value = '';
        }}
      />
    </>
  );
}

function BuildProgressOverlay({ job, onClose, onRetry }) {
  // job: { id, file, previewUrl, stages: {key: 'pending'|'active'|'done'}, blockIdx, blockTotal, status, error, doneInfo }
  if (!job) return null;
  const elapsed = ((Date.now() - job.startedAt) / 1000).toFixed(1);
  return (
    <div className="upload-overlay-backdrop">
      <div className="upload-overlay">
        <header className="upload-overlay-header">
          <strong>
            {job.status === 'done'    ? '✓ Build complete' :
             job.status === 'error'   ? '✗ Build failed' :
             job.status === 'cancelled' ? 'Build cancelled' :
             'Building assets…'}
          </strong>
          <span className="upload-overlay-elapsed">{elapsed}s</span>
          <button className="upload-overlay-close" onClick={onClose}>×</button>
        </header>
        <div className="upload-overlay-body">
          <div className="upload-overlay-thumb">
            <img src={job.previewUrl} alt="" />
            <div className="upload-overlay-filename">{job.file?.name}</div>
          </div>
          <div className="upload-overlay-stages">
            {BUILD_STAGES.map(s => {
              const st = job.stages[s.key] || 'pending';
              const isFx = s.key === 'fx_capture';
              const label = isFx && job.blockTotal
                ? `${s.label} (${job.blockIdx} / ${job.blockTotal})`
                : s.label;
              return (
                <div key={s.key} className={`upload-overlay-stage ${st}`}>
                  <span className="upload-overlay-stage-marker">
                    {st === 'done' ? '✓' : st === 'active' ? '●' : '○'}
                  </span>
                  <span className="upload-overlay-stage-label">{label}</span>
                  {isFx && job.blockTotal > 0 && st !== 'pending' && (
                    <div className="upload-overlay-bar">
                      <div
                        className="upload-overlay-bar-fill"
                        style={{ width: `${(100 * job.blockIdx / job.blockTotal).toFixed(1)}%` }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
            {job.status === 'error' && (
              <div className="upload-overlay-error">{job.error}</div>
            )}
            {job.status === 'done' && job.doneInfo && (
              <div className="upload-overlay-done-summary">
                {job.doneInfo.n_blocks} blocks · {job.doneInfo.n_subs} sub-nodes
                {job.doneInfo.skipped && job.doneInfo.skipped.length > 0
                  ? ` · skipped [${job.doneInfo.skipped.join(', ')}]`
                  : ''}
              </div>
            )}
          </div>
        </div>
        <footer className="upload-overlay-footer">
          {(job.status === 'running' || job.status === 'queued') && (
            <button className="upload-overlay-cancel" onClick={onClose}>Cancel</button>
          )}
          {job.status === 'error' && (
            <button className="upload-overlay-retry" onClick={onRetry}>Retry</button>
          )}
          {(job.status === 'done' || job.status === 'error' || job.status === 'cancelled') && (
            <button className="upload-overlay-close-btn" onClick={onClose}>Close</button>
          )}
        </footer>
      </div>
    </div>
  );
}

function App() {
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

  // Settings panel — rev counter forces Graph useMemo to recompute when
  // any setting changes (the layout/graph code reads from window.YV.LAYOUT_SETTINGS
  // at call time, so we just need to invalidate the memo).
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsRev, setSettingsRev] = useState(0);
  const bumpSettings = useCallback(() => setSettingsRev(r => r + 1), []);

  // Theme (light / dark) — applied as data-theme on <html> so the CSS overrides
  // in index.html flip every surface var. Also nudges a few SVG colors
  // (edge defaults) to a darker shade so they stay legible on the dark canvas.
  const [theme, setTheme] = useState(() => document.documentElement.getAttribute('data-theme') || 'light');
  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      const LS = window.YV.LAYOUT_SETTINGS;
      // Block palette + role colors swap to the active theme's preset. User
      // edits in the Settings drawer overwrite individual entries afterwards;
      // toggling the theme again rebases on the new preset, dropping prior edits.
      const TP = window.YV.TYPE_PALETTES[next];
      const RP = window.YV.ROLE_PALETTES[next];
      LS.TYPE_PALETTE = Object.fromEntries(Object.entries(TP).map(([k, v]) => [k, { ...v }]));
      LS.ROLE_PALETTE = { ...RP };
      LS.ACCENT_COLOR = window.YV.ACCENTS[next];
      // Inner sub-node palette + nesting tints follow the theme too.
      const IP = window.YV.INNER_PALETTES[next];
      LS.INNER_PALETTE = Object.fromEntries(Object.entries(IP).map(([k, v]) => [k, { ...v }]));
      LS.NESTING_TINTS = window.YV.NESTING_TINT_SETS[next].map(o => ({ ...o }));
      // Canvas background is theme-specific — rebase from the preset.
      LS.GRAPH_BG = window.YV.GRAPH_BGS[next];
      if (next === 'dark') {
        // Calibrated to sit clearly above the dark #0c1118 graph canvas
        // without competing with the focused accent.
        LS.EDGE_COLOR_DEFAULT = '#2a4058';
        LS.EDGE_COLOR_DIMMED  = '#1a2c40';
        LS.EDGE_COLOR_FOCUSED = LS.ACCENT_COLOR;
      } else {
        const DEF = window.YV.LAYOUT_SETTINGS_DEFAULTS;
        LS.EDGE_COLOR_DEFAULT = DEF.EDGE_COLOR_DEFAULT;
        LS.EDGE_COLOR_DIMMED  = DEF.EDGE_COLOR_DIMMED;
        LS.EDGE_COLOR_FOCUSED = LS.ACCENT_COLOR;
      }
      // Drive the CSS --accent var too so non-SVG accents (panel border,
      // brochure pinned glow, etc.) follow the theme.
      document.documentElement.style.setProperty('--accent', LS.ACCENT_COLOR);
      setSettingsRev(r => r + 1);
      return next;
    });
  }, []);

  const onVisibleOrderChange = useCallback((order) => setVisibleOrder(order), []);

  // ---- Server mode (yolovex serve): image upload + live progress ----------
  const serverMode = useServerMode();
  const [uploadJob, setUploadJob] = useState(null);
  // dataRev bumps when activations are reloaded — used to force a remount of
  // the graph so every consumer re-reads window.YV_ACT from scratch.
  const [dataRev, setDataRev] = useState(0);
  const lastFileRef = useRef(null);
  const evtSrcRef = useRef(null);

  const closeUpload = useCallback(() => {
    if (evtSrcRef.current) { try { evtSrcRef.current.close(); } catch {} evtSrcRef.current = null; }
    if (uploadJob && (uploadJob.status === 'running' || uploadJob.status === 'queued')) {
      // Best-effort cancel
      fetch(`/api/jobs/${uploadJob.id}`, { method: 'DELETE' }).catch(() => {});
    }
    setUploadJob(null);
  }, [uploadJob]);

  const startUpload = useCallback(async (file) => {
    lastFileRef.current = file;
    const previewUrl = URL.createObjectURL(file);
    const startedAt = Date.now();
    setUploadJob({
      id: null, file, previewUrl, startedAt,
      stages: {}, blockIdx: 0, blockTotal: 0,
      status: 'queued', error: null, doneInfo: null,
    });

    const fd = new FormData();
    fd.append('file', file);
    let jobId;
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      if (r.status === 409) {
        setUploadJob(j => j && ({ ...j, status: 'error', error: 'Another build is already in progress.' }));
        return;
      }
      if (!r.ok) {
        const t = await r.text();
        setUploadJob(j => j && ({ ...j, status: 'error', error: `Upload failed: ${t || r.statusText}` }));
        return;
      }
      const body = await r.json();
      jobId = body.job_id;
      setUploadJob(j => j && ({ ...j, id: jobId, status: 'running' }));
    } catch (e) {
      setUploadJob(j => j && ({ ...j, status: 'error', error: 'Network error contacting server.' }));
      return;
    }

    // Subscribe to SSE
    const src = new EventSource(`/api/jobs/${jobId}/events`);
    evtSrcRef.current = src;
    src.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch { return; }
      setUploadJob(j => {
        if (!j) return j;
        const next = { ...j, stages: { ...j.stages } };
        if (ev.kind === 'stage') {
          // Mark the new stage active; previous active stages become done.
          BUILD_STAGES.forEach(s => {
            if (s.key === ev.stage) next.stages[s.key] = 'active';
            else if (next.stages[s.key] === 'active') next.stages[s.key] = 'done';
          });
        } else if (ev.kind === 'block') {
          next.blockIdx = (ev.idx || 0) + 1;
          next.blockTotal = ev.total || next.blockTotal;
          // Stay in fx_capture stage.
          if (next.stages.fx_capture !== 'done') next.stages.fx_capture = 'active';
        } else if (ev.kind === 'done') {
          BUILD_STAGES.forEach(s => { next.stages[s.key] = 'done'; });
          next.status = 'done';
          next.doneInfo = ev;
        } else if (ev.kind === 'error') {
          next.status = 'error';
          next.error = ev.message || 'Build failed';
        } else if (ev.kind === 'cancelled') {
          next.status = 'cancelled';
        }
        return next;
      });
      if (ev.kind === 'done') {
        // Reload activations + force a graph remount so every consumer re-reads.
        reloadActivations()
          .then(() => setDataRev(r => r + 1))
          .catch(e => {
            setUploadJob(j => j && ({ ...j, status: 'error', error: 'Reload failed: ' + e.message }));
          });
        src.close();
      } else if (ev.kind === 'error' || ev.kind === 'cancelled') {
        src.close();
      }
    };
    src.onerror = () => {
      // EventSource auto-retries; we only treat this as terminal if the job is
      // already past completion (in which case onmessage closed it).
    };
  }, []);

  const retryUpload = useCallback(() => {
    if (lastFileRef.current) startUpload(lastFileRef.current);
  }, [startUpload]);

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
        {serverMode && (
          <UploadButton
            disabled={!!uploadJob && (uploadJob.status === 'running' || uploadJob.status === 'queued')}
            onPick={startUpload}
          />
        )}
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
        <window.YV.Graph
          key={`graph-${dataRev}`}
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
        <FlowOverlay active={playing || hover || selected} lastActive={lastActive} />
        <DetailPanel selected={selected} onClose={() => setSelected(null)} panelRef={panelRef} />
        {settingsOpen && (
          <SettingsPanel rev={settingsRev} bump={bumpSettings} onClose={() => setSettingsOpen(false)} />
        )}
        <BuildProgressOverlay job={uploadJob} onClose={closeUpload} onRetry={retryUpload} />
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
