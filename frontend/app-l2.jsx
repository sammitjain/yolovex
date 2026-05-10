// L2 App — wires GraphL2 + DetailPanelL2 together.
// Forked from app.jsx because:
//   1. GraphL2 replaces Graph (different component).
//   2. selectedId is now a string (sub-node "4.cv1" or container "4"), not a numeric index.
//   3. DetailPanelL2 looks up blocks via YV_DATA rather than ARCH[idx].
//   4. PlayFlow traverses ARCH_L2 container order.
//   5. BN/SiLU/DetectHead special cases handled here, not in panel.jsx.

const { useState: useStateA2, useEffect: useEffectA2, useRef: useRefA2, useCallback: useCallbackA2 } = React;

// ---------- data helpers (mirrors panel.jsx) ----------

function dataForL2(id) {
  return window.YV_DATA?.blocks?.[id];
}

function hasRealDataL2(imageUrl) {
  return !!(window.YV_DATA && imageUrl && imageUrl === window.YV_DATA.meta.image);
}

// Resolve "what feeds this node" — returns an array of { id, isImage }
// describing every input thumbnail to show on the panel's I/O strip.
//
// Sub-nodes:
//   - For the .concat sub-node of a C3k2/SPPF, all sibling tensors fed into
//     the concatenation are inputs.
//   - For the FIRST sub-node of any other parent, inputs come from the
//     upstream parent block's `sources` (with src=-1 → the original image,
//     and src=N → the LAST sub-node of container N if it has subs, else
//     just container N's own activation).
//   - Otherwise, the input is the previous sub-node in the same parent.
//
// Containers (numeric id like "23"): inputs are the parent block's L1
// sources, exactly like L1's panel.
function inputSourcesL2(nodeId) {
  const arch = window.YV.getArchL2();
  const blocks = window.YV_DATA?.blocks || {};

  const lastSubOf = (parentIdx) => {
    const c = arch.find(a => a.idx === parentIdx);
    if (c && c.sub.length > 0) return c.sub[c.sub.length - 1].id;
    return String(parentIdx);
  };

  // Sub-node case
  if (typeof nodeId === 'string' && nodeId.includes('.')) {
    const parentIdx = parseInt(nodeId.split('.')[0], 10);
    const parent = arch.find(a => a.idx === parentIdx);
    if (!parent) return [];
    const subIdx = parent.sub.findIndex(s => s.id === nodeId);
    const sub = parent.sub[subIdx];
    if (!sub) return [];

    // .concat — siblings inside the same parent
    if (sub.path === 'concat') {
      if (parent.type === 'C3k2') {
        return [`${parentIdx}.cv1`, `${parentIdx}.m.0`].map(id => ({ id }));
      }
      if (parent.type === 'SPPF') {
        return [`${parentIdx}.cv1`, `${parentIdx}.m.0`, `${parentIdx}.m.1`, `${parentIdx}.m.2`]
          .map(id => ({ id }));
      }
    }

    // First sub-node — inherit parent's L1 sources
    if (subIdx === 0) {
      const parentBlock = blocks[String(parentIdx)] || {};
      const sources = parentBlock.sources || [];
      return sources.map(src => {
        if (src < 0) return { id: null, isImage: true };
        return { id: lastSubOf(src) };
      });
    }

    // Otherwise: previous sibling in the parent's sub list
    return [{ id: parent.sub[subIdx - 1].id }];
  }

  // Container click
  const blockData = blocks[String(nodeId)];
  if (!blockData) return [];
  const sources = blockData.sources || [];
  return sources.map(src => {
    if (src < 0) return { id: null, isImage: true };
    return { id: String(src) };
  });
}

function thumbnailSrcFor(srcEntry, imageUrl) {
  if (!srcEntry) return null;
  if (srcEntry.isImage) return imageUrl;
  return window.YV_DATA?.blocks?.[srcEntry.id]?.mean_thumbnail || null;
}

function shapeOfId(id) {
  return window.YV_DATA?.blocks?.[id]?.shape;
}

// ---- Learner-friendly copy keyed by sub-module / parent type ----
//
// `title` is shown next to the [idx] tag in the panel header; `blurb` is a
// 1-3 sentence learner-oriented explanation of what this layer is doing in
// the network. Kept centralized so we can iterate the wording without
// hunting through render code.
const TYPE_COPY = {
  // Sub-module leaves
  Conv2d: {
    title: 'Conv2d — learnable filter bank',
    blurb: 'Slides a small kernel across the input and produces one output channel per filter. This is the ONLY layer in the Conv block that carries learnable weights — it does the actual feature extraction. Stride > 1 here also means spatial downsampling.',
  },
  BatchNorm2d: {
    title: 'BatchNorm — channel-wise normalization',
    blurb: 'Re-centers and re-scales each channel using running mean/variance learned during training. Keeps activations in a stable range so the next layer (SiLU) sees predictable inputs. Visually subtle — it shifts the distribution, not the spatial pattern.',
  },
  SiLU: {
    title: 'SiLU — smooth nonlinearity',
    blurb: 'Computes x · sigmoid(x). For most positive activations SiLU is nearly x (so this thumbnail will look almost identical to BN output above), but it gently suppresses strong negative values. The smoothness makes training gradients well-behaved.',
  },
  MaxPool2d: {
    title: 'MaxPool — local maximum',
    blurb: 'Inside SPPF, a 5×5 stride-1 max pool with padding 2. SPPF chains it three times so that the n-th call has covered an effective receptive field of (4n+1)×(4n+1) — that\'s how you get a "pyramid of receptive fields" without changing spatial size.',
  },
  Concat: {
    title: 'Concat — stack channels',
    blurb: 'Joins multiple tensors along the channel dimension. Output channel count = sum of inputs. Used inside C3k2/SPPF to merge parallel branches, and across columns in the neck to fuse skip connections with upsampled features.',
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
    blurb: 'A miniature CSP module (split → series of bottlenecks → concat → 1×1) used inside the deeper C3k2 layers. Effectively a "C3 inside C3k2" — adds extra depth at coarser feature scales.',
  },
  Sequential: {
    title: 'Sequential — composed mini-block',
    blurb: 'A wrapped module list — for block 22, this is yolo26\'s own arrangement of bottleneck-style operations. Treated as one atomic unit at L2; we\'ll expand it at L3.',
  },
  PSABlock: {
    title: 'PSABlock — position-sensitive self-attention',
    blurb: 'Splits the feature map into spatial windows, runs self-attention (Q/K/V) inside each, then a small feed-forward. Lets the network propagate information across distant locations — useful at the deepest backbone scale where a single token already covers a big receptive field.',
  },
  // Parent containers — shown when the user clicks the container label band
  Conv: {
    title: 'Conv block — Conv2d → BN → SiLU',
    blurb: 'The basic building block. A learnable Conv2d does feature extraction, BatchNorm normalizes the result, SiLU applies a smooth nonlinearity. Stride-2 versions also halve the spatial size.',
  },
  C3k2: {
    title: 'C3k2 block — CSP feature mixer',
    blurb: 'Splits the input into two halves, runs a Bottleneck (or nested C3k) on one half, then concats both halves back with the bottleneck output before a 1×1 projection. This is yolo26\'s main feature-mixing unit at every backbone resolution.',
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
    blurb: 'Runs three parallel conv heads (one per scale) and emits boxes + class scores per anchor cell. yolo26 uses a one-to-one assignment so post-processing doesn\'t need NMS — it\'s already a top-K of the highest-scoring anchors.',
  },
  DetectHead: {
    title: 'Detect branch — one of P3 / P4 / P5',
    blurb: 'A single-scale prediction branch. Two parallel conv stacks emit (a) box-regression bins and (b) per-class scores at every grid cell of this scale. Strides 8/16/32 specialize the three branches for small/medium/large objects.',
  },
};

function copyForType(type) {
  return TYPE_COPY[type] || { title: type, blurb: '' };
}

// ---------- App ----------

// BoxOverlayImage — renders a base image with bbox rectangles + classic
// filled label tags (class name + confidence) drawn on top via HTML/CSS.
// Using divs (not SVG strokes) gives crisp text and predictable border
// widths regardless of how big or small the image is rendered.
function BoxOverlayImage({ imageUrl, survivors, losers, colorFor, maxWidth = 200, fillContainer = false }) {
  const labelTag = (b, opts) => {
    const c = colorFor(b.cls_id);
    return (
      <div
        style={{
          position: 'absolute',
          left: `${b.x1 * 100}%`,
          top: b.y1 > 0.05 ? `calc(${b.y1 * 100}% - 14px)` : `${b.y1 * 100}%`,
          fontSize: 10,
          lineHeight: '14px',
          padding: '0 5px',
          background: c,
          color: 'white',
          fontFamily: 'ui-monospace, SFMono-Regular, monospace',
          fontWeight: 500,
          borderRadius: 2,
          whiteSpace: 'nowrap',
          opacity: opts.faded ? 0.7 : 1,
        }}
      >
        {b.cls_name} {b.conf.toFixed(2)}
      </div>
    );
  };
  return (
    <div
      style={
        fillContainer
          // Match the plain <img objectFit:cover> exactly so the FlowOverlay
          // doesn't jitter when the active layer flips between thumbnail and
          // annotated-image modes.
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
            : { display: 'block', width: '100%', maxWidth, borderRadius: 6, border: '1px solid #e2e8f0' }
        }
      />
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {/* Losers behind */}
        {losers.map((b, i) => (
          <div
            key={`L${i}`}
            style={{
              position: 'absolute',
              left: `${b.x1 * 100}%`,
              top: `${b.y1 * 100}%`,
              width: `${Math.max(0.001, b.x2 - b.x1) * 100}%`,
              height: `${Math.max(0.001, b.y2 - b.y1) * 100}%`,
              border: `1.25px dashed ${colorFor(b.cls_id)}`,
              borderRadius: 2,
              opacity: 0.5,
              boxSizing: 'border-box',
            }}
          />
        ))}
        {/* Survivors on top */}
        {survivors.map((b, i) => (
          <React.Fragment key={`S${i}`}>
            <div
              style={{
                position: 'absolute',
                left: `${b.x1 * 100}%`,
                top: `${b.y1 * 100}%`,
                width: `${Math.max(0.001, b.x2 - b.x1) * 100}%`,
                height: `${Math.max(0.001, b.y2 - b.y1) * 100}%`,
                border: `2px solid ${colorFor(b.cls_id)}`,
                borderRadius: 2,
                boxSizing: 'border-box',
              }}
            />
            {labelTag(b, { faded: false })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// FlowOverlay — a fixed tile floating over the top-right of the graph area.
// It always shows ONE picture at a constant size: the mean activation
// thumbnail of the layer that is currently active (hovered, selected, or
// playing). For the Detect container, it shows the input image with the
// final detection bboxes drawn on top — so the play-flow ends on a "this
// is what came out of the network" frame.
function FlowOverlay({ activeId, imageUrl, playing }) {
  const arch = window.YV.getArchL2();
  const blocks = window.YV_DATA?.blocks || {};

  // Decide what to show. Defaults to the input image when nothing is active.
  let imgSrc = imageUrl;
  let label = 'input image';
  let subLabel = 'before any layer runs';
  let boxesToDraw = null;
  let strongest = null;

  if (activeId != null) {
    const data = blocks[String(activeId)];
    if (String(activeId) === '23') {
      // Detect container — annotated image
      imgSrc = imageUrl;
      const bx = data?.boxes || [];
      boxesToDraw = bx;
      label = '[23] Detect — final';
      subLabel = `${bx.length} detection${bx.length === 1 ? '' : 's'} above conf 0.25`;
      if (bx.length) strongest = bx[0];
    } else if (data && data.mean_thumbnail) {
      imgSrc = data.mean_thumbnail;
      const friendly = TYPE_COPY[data.type]?.title?.split(' — ')[0] || data.type;
      label = `[${activeId}] ${friendly}`;
      subLabel = Array.isArray(data.shape) ? `shape ${data.shape.join('×')}` : '';
    } else if (data && data.shape) {
      imgSrc = imageUrl;
      label = `[${activeId}] ${data.type}`;
      subLabel = 'no thumbnail (atomic op)';
    }
  }

  const colorFor = (clsId) => ['#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7','#ec4899','#14b8a6','#f97316'][clsId % 8];

  return (
    <div
      style={{
        position: 'absolute',
        top: 14,
        right: 14,
        width: 280,
        background: 'rgba(255,255,255,0.96)',
        backdropFilter: 'blur(6px)',
        borderRadius: 10,
        boxShadow: playing ? '0 6px 24px rgba(15,23,42,0.18)' : '0 3px 12px rgba(15,23,42,0.10)',
        padding: 10,
        pointerEvents: 'none',
        zIndex: 5,
        border: playing ? '1px solid #fb923c' : '1px solid #e2e8f0',
        transition: 'box-shadow 200ms, border-color 200ms',
      }}
    >
      <div style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '4 / 5',
        overflow: 'hidden',
        borderRadius: 6,
        background: '#f1f5f9',
      }}>
        {boxesToDraw ? (
          // Annotated image — use the shared BoxOverlayImage helper, sized to
          // fill the same aspect-ratio frame as the plain image, so swapping
          // between the two during play-flow doesn't introduce vertical jitter.
          <BoxOverlayImage
            imageUrl={imgSrc}
            survivors={boxesToDraw}
            losers={[]}
            colorFor={colorFor}
            fillContainer
          />
        ) : (
          <img
            src={imgSrc}
            alt=""
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              display: 'block',
              transition: 'opacity 250ms',
            }}
          />
        )}
      </div>
      <div style={{
        marginTop: 8,
        fontSize: 12.5,
        fontWeight: 600,
        color: '#0f172a',
        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>{label}</div>
      <div style={{
        fontSize: 10.5,
        color: '#64748b',
        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>{subLabel}</div>
    </div>
  );
}

// Per-tick durations for the flow animation. The slowest setting is what
// makes the flow overlay panel useful: the tile gets time to register on the
// eye before changing again.
const FLOW_SPEEDS = {
  fast:   60,    // legacy (~70 nodes in 4s)
  medium: 250,
  slow:   700,   // default — gives ~50s for a full pass
};

// ---------- Settings popover (gear in header) ----------
//
// Gear button in the header. Click to toggle a popover anchored below it
// containing app-wide controls. Today: bezier "lead" lengths for edge
// flattening. Future: more knobs land here too — that's why this is one
// unified Settings popover rather than one panel per concern.
function SettingsButton({ layoutParams, setLayoutParams }) {
  const [open, setOpen] = useStateA2(false);
  const wrapperRef = useRefA2(null);
  const defaults = window.YV.DEFAULT_LAYOUT_PARAMS || {};

  // Close on outside click.
  useEffectA2(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const update = (key, value) => setLayoutParams(p => ({ ...p, [key]: value }));
  const reset = () => setLayoutParams({ ...defaults });

  const Slider = ({ label, k, min, max }) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#475569' }}>
        <span>{label}</span>
        <span style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}>{layoutParams[k]}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={1}
        value={layoutParams[k]}
        onChange={(e) => update(k, parseInt(e.target.value, 10))}
        style={{ width: '100%' }}
      />
    </div>
  );

  return (
    <div ref={wrapperRef} style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        onClick={() => setOpen(o => !o)}
        title="Settings"
        aria-label="Settings"
        style={{
          width: 30, height: 30,
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          padding: 0,
          background: open ? '#e2e8f0' : 'white',
          border: '1px solid #cbd5e1',
          borderRadius: 4,
          cursor: 'pointer',
          color: '#475569',
          fontSize: 16,
          lineHeight: 1,
        }}
      >
        ⚙
      </button>
      {open && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 6px)',
          right: 0,
          zIndex: 20,
          width: 240,
          background: 'white',
          border: '1px solid #e2e8f0',
          borderRadius: 6,
          boxShadow: '0 4px 12px rgba(0,0,0,0.10)',
          padding: '12px 14px',
        }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: '#64748b',
            letterSpacing: '0.06em', textTransform: 'uppercase',
            marginBottom: 8,
          }}>
            Edge layout — horizontal
          </div>
          <Slider label="entry tail" k="horizontalEntry" min={5} max={50} />
          <Slider label="exit tail"  k="horizontalExit"  min={5} max={50} />
          <div style={{
            fontSize: 10, fontWeight: 700, color: '#64748b',
            letterSpacing: '0.06em', textTransform: 'uppercase',
            margin: '6px 0 8px',
          }}>
            Edge layout — vertical
          </div>
          <Slider label="entry tail" k="verticalEntry" min={5} max={50} />
          <Slider label="exit tail"  k="verticalExit"  min={5} max={50} />
          <button
            onClick={reset}
            style={{
              width: '100%',
              padding: '5px 8px',
              fontSize: 11,
              background: '#f1f5f9',
              border: '1px solid #cbd5e1',
              borderRadius: 4,
              cursor: 'pointer',
              color: '#475569',
            }}
          >
            reset to defaults
          </button>
        </div>
      )}
    </div>
  );
}

function AppL2() {
  const { GraphL2, getArchL2, ROLE_COLORS, ACCENT } = window.YV;
  const arch_l2 = getArchL2();

  const [selectedId, setSelectedId] = useStateA2(null);
  const [hoveredId, setHoveredId] = useStateA2(null);
  const [playingId, setPlayingId] = useStateA2(null);
  const [imageUrl, setImageUrl] = useStateA2('assets/sammit_lighthouse.jpg');
  const [pickerOpen, setPickerOpen] = useStateA2(false);
  const [flowSpeed, setFlowSpeed] = useStateA2('slow');
  const [layoutParams, setLayoutParams] = useStateA2(
    () => ({ ...(window.YV.DEFAULT_LAYOUT_PARAMS || {}) })
  );

  // What the flow-overlay tile and graph focus should track. Priority:
  // playing > hovered > selected > nothing.
  const activeId = playingId ?? hoveredId ?? selectedId;

  // Play flow — iterate over containers in order, then their sub-nodes,
  // and finally rest on the Detect container so the overlay shows the
  // annotated detections at the end.
  const playFlow = useCallbackA2(() => {
    if (playingId != null) return;
    const order = [];
    arch_l2.forEach(b => {
      // For non-atomic blocks, the parent container's "output" IS its last
      // sub-node's output — including the parent in the flow would just
      // double-show that frame. So we ONLY emit the parent for atomic blocks
      // (Concat, Upsample, etc.) which have no sub-nodes of their own.
      if (b.sub.length === 0) {
        order.push(String(b.idx));
      } else {
        b.sub.forEach(s => order.push(s.id));
      }
    });
    // Make sure the final frame is the Detect container so the overlay
    // settles on the annotated image with bboxes drawn.
    if (order[order.length - 1] !== '23') order.push('23');
    let i = 0;
    const tickMs = FLOW_SPEEDS[flowSpeed] ?? FLOW_SPEEDS.slow;
    const tick = () => {
      if (i >= order.length) {
        // Leave the final frame visible for ~1.5s before clearing.
        setTimeout(() => setPlayingId(null), 1500);
        return;
      }
      setPlayingId(order[i]);
      i++;
      setTimeout(tick, tickMs);
    };
    tick();
  }, [playingId, flowSpeed, arch_l2]);

  const onUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setPickerOpen(false);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="wordmark">
            yolovex
            <span style={{
              display: 'inline-block',
              marginLeft: 8,
              padding: '2px 7px',
              background: ACCENT,
              color: 'white',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.04em',
              verticalAlign: 'middle',
            }}>L2</span>
          </div>
          <div className="subtitle">sub-module level · YOLO26n</div>
        </div>
        <div className="header-right">
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <button className="play-btn" onClick={playFlow} disabled={playingId != null}>
              <span className="play-icon">▶</span> {playingId != null ? 'Playing…' : 'Play flow'}
            </button>
            <select
              value={flowSpeed}
              onChange={(e) => setFlowSpeed(e.target.value)}
              disabled={playingId != null}
              title="Flow speed"
              style={{
                fontSize: 12,
                padding: '4px 6px',
                borderRadius: 4,
                border: '1px solid #cbd5e1',
                background: 'white',
                color: '#334155',
                cursor: 'pointer',
              }}
            >
              <option value="slow">slow</option>
              <option value="medium">medium</option>
              <option value="fast">fast</option>
            </select>
          </div>
          <ImagePickerL2 open={pickerOpen} setOpen={setPickerOpen} onUpload={onUpload} setImageUrl={setImageUrl} currentUrl={imageUrl} />
          <SettingsButton layoutParams={layoutParams} setLayoutParams={setLayoutParams} />
          <div className="attribution">made by <span>Sammit</span></div>
        </div>
      </header>

      <main className="app-main">
        <div style={{ position: 'relative', flex: 1, minWidth: 0, display: 'flex' }}>
          <GraphL2
            selectedId={selectedId}
            hoveredId={hoveredId}
            onSelect={setSelectedId}
            onHover={setHoveredId}
            playingId={playingId}
            layoutParams={layoutParams}
          />
          <FlowOverlay activeId={activeId} imageUrl={imageUrl} playing={playingId != null} />
        </div>
        <DetailPanelL2
          nodeId={selectedId}
          onClose={() => setSelectedId(null)}
          imageUrl={imageUrl}
        />
      </main>

      <LegendL2 />
    </div>
  );
}

// ---------- Image picker (same as L1) ----------

function ImagePickerL2({ open, setOpen, onUpload, setImageUrl, currentUrl }) {
  const ref = useRefA2(null);
  useEffectA2(() => {
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    if (open) document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const thumbs = [{ url: 'assets/sammit_lighthouse.jpg', label: 'lighthouse' }];

  return (
    <div className="image-picker" ref={ref}>
      <button className="picker-btn" onClick={() => setOpen(o => !o)}>
        <img src={currentUrl} alt="" className="picker-thumb" />
        <span>image</span>
        <span className="caret">▾</span>
      </button>
      {open && (
        <div className="picker-panel">
          <div className="picker-label">Templates</div>
          <div className="picker-thumbs">
            {thumbs.map(t => (
              <button key={t.url} className="thumb-btn" onClick={() => { setImageUrl(t.url); setOpen(false); }}>
                <img src={t.url} alt={t.label} />
                <div className="thumb-label">{t.label}</div>
              </button>
            ))}
            {Array.from({ length: 4 }).map((_, i) => (
              <button key={i} className="thumb-btn empty" disabled>
                <div className="thumb-empty">+</div>
                <div className="thumb-label">empty</div>
              </button>
            ))}
          </div>
          <label className="upload-btn">
            Upload your own
            <input type="file" accept="image/*" onChange={onUpload} hidden />
          </label>
        </div>
      )}
    </div>
  );
}

// ---------- Legend (L2-aware colors) ----------

function LegendL2() {
  const { TYPE_COLORS_L2, ACCENT, ROLE_COLORS } = window.YV;
  const items = [
    ['Conv',    TYPE_COLORS_L2.Conv],
    ['C3k2',   TYPE_COLORS_L2.C3k2],
    ['SPPF',   TYPE_COLORS_L2.SPPF],
    ['C2PSA',  TYPE_COLORS_L2.C2PSA],
    ['Detect', TYPE_COLORS_L2.Detect],
    ['Conv2d', TYPE_COLORS_L2.Conv2d],
    ['BN',     TYPE_COLORS_L2.BatchNorm2d],
    ['SiLU',   TYPE_COLORS_L2.SiLU],
  ];
  return (
    <div className="legend">
      {items.map(([name, c]) => (
        <div key={name} className="legend-item">
          <span className="swatch" style={{ background: c.fill, borderColor: c.border }} />
          <span>{name}</span>
        </div>
      ))}
      <div className="legend-divider" />
      <div className="legend-item">
        <span className="swatch-line" style={{ background: ACCENT }} />
        <span>skip / detect</span>
      </div>
      <div className="legend-item">
        <span className="swatch-line" style={{ background: '#94a3b8' }} />
        <span>forward</span>
      </div>
    </div>
  );
}

// ---------- Detail panel for L2 ----------

function DetailPanelL2({ nodeId, onClose, imageUrl }) {
  const open = nodeId != null;

  useEffectA2(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const content = open ? renderPanelContent(nodeId, onClose, imageUrl) : null;

  return (
    <aside className={`detail-panel ${open ? 'open' : ''}`} aria-hidden={!open}>
      {content}
    </aside>
  );
}

function renderPanelContent(nodeId, onClose, imageUrl) {
  const blockData = dataForL2(nodeId);

  if (!blockData) {
    return <UnknownCard nodeId={nodeId} onClose={onClose} />;
  }

  // Detect parent container — consolidated card with final detections.
  if (blockData.type === 'Detect') {
    return <DetectContainerCard nodeId={nodeId} blockData={blockData} onClose={onClose} imageUrl={imageUrl} />;
  }

  // Per-scale Detect heads — class heatmaps for that scale.
  if (blockData.type === 'DetectHead') {
    return <DetectHeadCard nodeId={nodeId} blockData={blockData} onClose={onClose} imageUrl={imageUrl} />;
  }

  // Rare fallback: empty channel_pngs + a `note` (e.g. a backend that couldn't
  // capture a particular tensor). For yolo26n with the unfused-forward fix this
  // path no longer fires, but keep it for robustness.
  if (blockData.channel_pngs && blockData.channel_pngs.length === 0 && blockData.note) {
    return <BNFoldedCard nodeId={nodeId} blockData={blockData} onClose={onClose} />;
  }

  return <FeaturePanelL2 nodeId={nodeId} blockData={blockData} onClose={onClose} imageUrl={imageUrl} />;
}

// Minimal card for BN-folded sub-nodes
function BNFoldedCard({ nodeId, blockData, onClose }) {
  const { getTypeColor, ROLE_COLORS } = window.YV;
  const cs = getTypeColor(blockData.type);
  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[{nodeId}]</span>{' '}
            <span style={{ color: cs.text }}>{blockData.type}</span>
          </div>
          <div className="panel-desc">sub-module of block {blockData.parent}</div>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>
      <section className="panel-section">
        <div style={{
          padding: '14px 16px',
          background: '#fefce8',
          border: '1px solid #fde047',
          borderRadius: 8,
          fontSize: 13,
          color: '#713f12',
          lineHeight: 1.65,
        }}>
          <strong>Not available during inference:</strong>
          <br />
          {blockData.note}
        </div>
        <div className="shape-caption" style={{ marginTop: 10 }}>
          shape: ({(blockData.shape || []).join(', ')})
        </div>
      </section>
    </div>
  );
}

// DetectHead card — per-class heatmaps for one of the three scales (P3/P4/P5).
const DETECT_TOP_K = 5;   // contender classes shown per scale; bumped down from 6 to save horizontal space

function DetectHeadCard({ nodeId, blockData, onClose, imageUrl }) {
  const { getTypeColor, ROLE_COLORS } = window.YV;
  const cs = getTypeColor('Detect');
  const classes = (blockData.classes || []).slice(0, DETECT_TOP_K);
  const scale = blockData.scale || '';
  const sizeHint = scale === 'P3' ? 'small objects' : scale === 'P4' ? 'medium objects' : 'large objects';
  const stride = scale === 'P3' ? 8 : scale === 'P4' ? 16 : 32;

  // Detect parent's full data has the global box list; fish it out so we can
  // surface the boxes whose grid scale this is. We don't have explicit
  // anchor-of-origin per box, so we just show the same global box list with
  // a note — better than nothing, and lets the user cross-reference.
  const detectParent = window.YV_DATA?.blocks?.[String(blockData.parent)] || {};
  const boxes = (hasRealDataL2(imageUrl) ? detectParent.boxes : null) || [];

  // Sort classes by peak so the most-fired ones are first.
  const sorted = classes.slice().sort((a, b) => b.peak - a.peak);

  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[{nodeId}]</span>{' '}
            <span style={{ color: cs.text }}>{scale} branch · stride {stride}</span>
          </div>
          <div className="panel-desc">
            One of the three Detect branches. This one runs at <strong>stride {stride}</strong>, so it&apos;s primarily responsible for <strong>{sizeHint}</strong>. Each cell below shows where on the image that class fires hardest at this scale.
          </div>
          <span className="role-pill" style={{ background: ROLE_COLORS.Head + '22', color: ROLE_COLORS.Head }}>
            Head · {scale}
          </span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>

      <section className="panel-section">
        <h4 className="section-label">Class heatmaps</h4>
        <p className="micro-help">
          Top {sorted.length} classes by peak sigmoid score at {scale} resolution.
          A bright peak in one of these maps is what produced a final detection box.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          {sorted.map((cls, i) => (
            <div key={i} style={{ textAlign: 'center', width: 100 }}>
              <div className={`scale-grid-cell ${cls.peak >= 0.05 ? 'is-peak' : 'faint'}`}
                   style={{ width: 100, aspectRatio: '3 / 4', margin: '0 auto' }}>
                <img src={cls.png} alt={cls.name} style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }} />
                <span className="cell-score">{cls.peak.toFixed(2)}</span>
              </div>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 4, fontWeight: 600 }}>{cls.name}</div>
            </div>
          ))}
        </div>
      </section>

      {boxes.length > 0 && (
        <section className="panel-section">
          <h4 className="section-label">Final detections (all scales)</h4>
          <p className="micro-help">
            The full set of boxes the head emitted. We don&apos;t track which scale fired
            each one, but boxes for {sizeHint} are typically attributable to {scale}.
          </p>
          <ul style={{ padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
            {boxes.map((b, i) => (
              <li key={i} style={{
                fontSize: 12,
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                color: '#334155',
                display: 'flex',
                gap: 8,
              }}>
                <span style={{ fontWeight: 600, minWidth: 90 }}>{b.cls_name}</span>
                <span style={{ color: '#64748b' }}>{b.conf.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

// Generic feature panel for sub-nodes and parent containers
function FeaturePanelL2({ nodeId, blockData, onClose, imageUrl }) {
  const { getTypeColor, ROLE_COLORS, ACCENT } = window.YV;
  const cs = getTypeColor(blockData.type);
  const real = hasRealDataL2(imageUrl) ? blockData : null;
  const shape = blockData.shape;
  const totalChannels = real?.total_channels ?? (Array.isArray(shape) ? shape[1] : 0);
  const visibleCh = Math.min(16, totalChannels);

  // Determine role for display
  const parentType = blockData.parent_type || blockData.type;
  const role = blockData.role || getRoleForParent(blockData.parent);
  const roleColor = ROLE_COLORS[role] || '#64748b';

  const [pinnedCh, setPinnedCh] = useStateA2(0);
  const [hoveredCh, setHoveredCh] = useStateA2(null);
  const activeCh = hoveredCh != null ? hoveredCh : pinnedCh;
  const trueChIdx = (i) => real?.channel_indices?.[i] ?? i;

  const fmtShape = (s) => {
    if (!s) return '...';
    if (s === 'multi' || !Array.isArray(s)) return 'multi';
    return `(${s.join(', ')})`;
  };

  // Stats
  const stats = real?.stats ? {
    min: real.stats.min.toFixed(2),
    max: real.stats.max.toFixed(2),
    mean: real.stats.mean.toFixed(3),
    topCh: (real.channel_indices || []).slice(0, 5),
  } : null;

  // (We used to surface a "reuses parent" pill here. The data IS correct —
  // SiLU's output equals its parent Conv block's output exactly, verified
  // numerically — so the pill was just noise. Dropped.)

  const copy = copyForType(blockData.type);

  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[{nodeId}]</span>{' '}
            <span style={{ color: cs.text }}>{copy.title}</span>
          </div>
          <div className="panel-desc" style={{ lineHeight: 1.55 }}>
            {copy.blurb}
            {blockData.path != null && (
              <div style={{ marginTop: 6, fontSize: 11.5, color: '#64748b' }}>
                Sub-module <code style={{ fontFamily: 'monospace', fontSize: 11.5, background: '#f1f5f9', padding: '1px 5px', borderRadius: 3 }}>{blockData.path}</code> of block {blockData.parent} ({parentType}).
              </div>
            )}
            {blockData.path == null && blockData.desc && (
              <div style={{ marginTop: 6, fontSize: 11.5, color: '#64748b' }}>
                {blockData.desc}
              </div>
            )}
          </div>
          <span className="role-pill" style={{ background: roleColor + '22', color: roleColor }}>
            {role}
          </span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>

      <IOStripL2
        nodeId={nodeId}
        outputThumbnail={real?.mean_thumbnail}
        outputShape={shape}
        imageUrl={imageUrl}
        fmtShape={fmtShape}
      />


      {visibleCh > 0 && real && (
        <section className="panel-section">
          <h4 className="section-label">Channel stack</h4>
          <div className="channel-brochure">
            <div className="brochure-preview">
              <img
                src={real.channel_pngs[activeCh]}
                alt={`channel ${trueChIdx(activeCh)}`}
                style={{ width: 170, height: 210, objectFit: 'cover', display: 'block', borderRadius: 4, border: '2px solid var(--accent)', boxShadow: '0 2px 6px rgba(15,23,42,.18)' }}
              />
              <div className="brochure-meta">
                <div className="meta-row">
                  <span className="meta-k">channel</span>
                  <span className="meta-v mono">{trueChIdx(activeCh)}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-k">rank</span>
                  <span className="meta-v mono">#{activeCh + 1} of {totalChannels}</span>
                </div>
              </div>
            </div>
            <div className="brochure-grid" onMouseLeave={() => setHoveredCh(null)}>
              {Array.from({ length: visibleCh }).map((_, i) => (
                <button
                  key={i}
                  className={`brochure-thumb ${pinnedCh === i ? 'pinned' : ''} ${hoveredCh === i ? 'hovered' : ''}`}
                  onMouseEnter={() => setHoveredCh(i)}
                  onClick={() => setPinnedCh(i)}
                  title={`channel ${trueChIdx(i)}`}
                >
                  <img src={real.channel_pngs[i]} alt="" style={{ display: 'block', width: '100%', height: 'auto' }} />
                  <span className="thumb-idx">{trueChIdx(i)}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="stack-caption">
            Showing {visibleCh} of {totalChannels} channels, ranked by mean |activation|.
          </div>
        </section>
      )}

      {stats && (
        <section className="panel-section stats">
          <h4 className="section-label">Statistics</h4>
          <div className="stats-grid">
            <div><span className="k">shape</span><span className="v mono">{fmtShape(shape)}</span></div>
            <div><span className="k">activation min</span><span className="v mono">{stats.min}</span></div>
            <div><span className="k">activation max</span><span className="v mono">{stats.max}</span></div>
            <div><span className="k">activation mean</span><span className="v mono">{stats.mean}</span></div>
            <div><span className="k">top-5 channels</span><span className="v mono">{stats.topCh.join(', ')}</span></div>
          </div>
        </section>
      )}
    </div>
  );
}

// Reusable I/O strip — shows N input thumbnails and the output mean thumbnail.
function IOStripL2({ nodeId, outputThumbnail, outputShape, imageUrl, fmtShape }) {
  const inputs = inputSourcesL2(nodeId);
  const tileStyle = { width: 96, height: 120, objectFit: 'cover' };
  const concatStyle = { width: 80, height: 100, objectFit: 'cover' };
  const isMulti = inputs.length > 1;

  const inputCaption = inputs
    .map(s => s.isImage
      ? '(1, 3, image)'
      : (() => { const sh = shapeOfId(s.id); return sh ? fmtShape(sh) : '...'; })())
    .join(' + ');

  return (
    <section className="panel-section">
      <h4 className="section-label">Input → Output</h4>
      <div className="io-strip">
        {isMulti ? (
          <div className="io-pair">
            {inputs.map((s, i) => {
              const src = thumbnailSrcFor(s, imageUrl);
              return (
                <React.Fragment key={i}>
                  {i > 0 && <span className="plus">+</span>}
                  {src
                    ? <img src={src} alt="" className="io-tile" style={concatStyle} />
                    : <div className="io-tile" style={{ ...concatStyle, background: '#f1f5f9' }} />}
                </React.Fragment>
              );
            })}
          </div>
        ) : inputs.length === 1 ? (
          (() => {
            const src = thumbnailSrcFor(inputs[0], imageUrl);
            return src
              ? <img src={src} alt="" className="io-tile" style={tileStyle} />
              : <div className="io-tile" style={{ ...tileStyle, background: '#f1f5f9' }} />;
          })()
        ) : (
          <div className="io-tile" style={{ ...tileStyle, background: '#f1f5f9' }} />
        )}
        <span className="arrow">→</span>
        {outputThumbnail
          ? <img src={outputThumbnail} alt="" className="io-tile" style={tileStyle} />
          : <div className="io-tile" style={{ ...tileStyle, background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#94a3b8' }}>no output</div>
        }
      </div>
      <div className="shape-caption">
        {inputCaption || '...'} {' → '} {fmtShape(outputShape)}
      </div>
    </section>
  );
}

// Detect container card — final detections + per-scale summary + internals.
function DetectContainerCard({ nodeId, blockData, onClose, imageUrl }) {
  const { getTypeColor, ROLE_COLORS, ACCENT } = window.YV;
  const cs = getTypeColor('Detect');
  const real = hasRealDataL2(imageUrl) ? blockData : null;
  const finalBoxes = real?.boxes || [];                     // ≥0.25 conf (Ultralytics default)
  const candidateBoxes = real?.candidate_boxes || finalBoxes; // ≥0.001 from a re-run with low conf
  const scales = real?.scales || {};
  const internals = blockData.internals || {};

  // User-controlled confidence threshold. Default to 0.25 (Ultralytics default
  // for predict()), let the user slide DOWN to see runners-up appear, or UP
  // to see only the high-confidence picks survive.
  const [confThreshold, setConfThreshold] = useStateA2(0.25);
  const survivors = candidateBoxes.filter(b => b.conf >= confThreshold);
  const losers = candidateBoxes.filter(b => b.conf < confThreshold);

  // Find top-scoring class per scale to give the user the "story" at a glance
  const topClassPerScale = {};
  ['P3', 'P4', 'P5'].forEach(sk => {
    const cls = (scales[sk]?.classes) || [];
    if (cls.length) {
      const top = cls.slice().sort((a, b) => b.peak - a.peak)[0];
      topClassPerScale[sk] = top;
    }
  });

  // Color by class id for the box overlay (cycle through a palette).
  const boxColors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ec4899', '#14b8a6', '#f97316'];
  const colorFor = (clsId) => boxColors[clsId % boxColors.length];

  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[{nodeId}]</span>{' '}
            <span style={{ color: cs.text }}>Detect head</span>
          </div>
          <div className="panel-desc">
            Multi-scale anchor-free head — three branches predict boxes + class scores at the P3 / P4 / P5 feature maps. Final detections come from a top-K filter on the one-to-one branch (NMS-free).
          </div>
          <span className="role-pill" style={{ background: ROLE_COLORS.Head + '22', color: ROLE_COLORS.Head }}>
            Head
          </span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>

      {/* Final detections — boxes overlaid on the input image, plus runners-up */}
      <section className="panel-section">
        <h4 className="section-label">Final detections</h4>
        <p className="micro-help">
          Solid boxes are above your threshold ({confThreshold.toFixed(2)}). Faded dashed boxes
          are runners-up the head also emitted but that fall below it. Slide the threshold
          to watch which boxes graduate or get dropped.
        </p>
        {imageUrl ? (
          <BoxOverlayImage
            imageUrl={imageUrl}
            survivors={survivors}
            losers={losers}
            colorFor={colorFor}
            maxWidth={200}
          />
        ) : (
          <div style={{ width: 200, height: 150, background: '#f1f5f9', borderRadius: 6 }} />
        )}

        {/* Confidence threshold slider */}
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
            <span style={{ fontWeight: 600, color: '#334155' }}>Confidence threshold</span>
            <span style={{ fontFamily: 'monospace', color: '#0f172a' }}>{confThreshold.toFixed(3)}</span>
          </div>
          <input
            type="range"
            min={0.005}
            max={1}
            step={0.005}
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            style={{ width: '100%', cursor: 'pointer' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#94a3b8', marginTop: 2, fontFamily: 'monospace' }}>
            <span>0.005</span><span>1.000</span>
          </div>
          <div style={{ fontSize: 11, color: '#475569', marginTop: 6 }}>
            <strong style={{ color: '#0f172a' }}>{survivors.length}</strong> survivors ·{' '}
            <span style={{ color: '#94a3b8' }}>{losers.length} runners-up</span>
            {' · '}
            <span style={{ color: '#94a3b8' }}>{candidateBoxes.length} total candidates</span>
          </div>
        </div>

        {survivors.length > 0 ? (
          <ul style={{ marginTop: 10, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
            {survivors.map((b, i) => (
              <li key={i} style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 12,
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                color: '#334155',
              }}>
                <span style={{
                  width: 10, height: 10, borderRadius: 2,
                  background: colorFor(b.cls_id),
                  flexShrink: 0,
                }} />
                <span style={{ fontWeight: 600, minWidth: 90 }}>{b.cls_name}</span>
                <span style={{ color: '#64748b' }}>{b.conf.toFixed(3)}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>
                  ({b.x1.toFixed(2)}, {b.y1.toFixed(2)})–({b.x2.toFixed(2)}, {b.y2.toFixed(2)})
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <div style={{ marginTop: 10, fontSize: 12, color: '#94a3b8', fontStyle: 'italic' }}>
            No detections above the current threshold. Slide it down.
          </div>
        )}

        {losers.length > 0 && (
          <details style={{ marginTop: 8 }}>
            <summary style={{ fontSize: 11, color: '#64748b', cursor: 'pointer' }}>
              Show {losers.length} runner{losers.length === 1 ? '' : 's'}-up
            </summary>
            <ul style={{ marginTop: 6, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 3 }}>
              {losers.slice(0, 20).map((b, i) => (
                <li key={i} style={{
                  display: 'flex',
                  gap: 8,
                  fontSize: 11,
                  fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                  color: '#94a3b8',
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: 2,
                    background: colorFor(b.cls_id),
                    flexShrink: 0,
                    marginTop: 4,
                    opacity: 0.55,
                  }} />
                  <span style={{ minWidth: 90 }}>{b.cls_name}</span>
                  <span>{b.conf.toFixed(3)}</span>
                </li>
              ))}
              {losers.length > 20 && <li style={{ fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>… and {losers.length - 20} more.</li>}
            </ul>
          </details>
        )}
      </section>

      {/* Per-scale summary — one row per scale */}
      <section className="panel-section">
        <h4 className="section-label">Per-scale story</h4>
        <p className="micro-help">
          Each row is one of the three detection branches. The grid shows where that scale's
          top-scoring class fires; smaller scales (P3) catch small objects, larger scales (P5)
          catch large ones.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {['P3', 'P4', 'P5'].map(sk => {
            const sc = scales[sk] || {};
            const top = topClassPerScale[sk];
            const sizeHint = sk === 'P3' ? 'small objects' : sk === 'P4' ? 'medium objects' : 'large objects';
            return (
              <div key={sk} style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: 8,
                background: '#f8fafc',
                borderRadius: 6,
                border: '1px solid #e2e8f0',
              }}>
                {top && top.png
                  ? <img src={top.png} alt="" style={{ width: 60, height: 75, objectFit: 'cover', borderRadius: 3 }} />
                  : <div style={{ width: 60, height: 75, background: '#f1f5f9', borderRadius: 3 }} />}
                <div style={{ flex: 1, fontSize: 12 }}>
                  <div style={{ fontWeight: 700, color: '#334155' }}>{sk} <span style={{ color: '#94a3b8', fontWeight: 400 }}>· stride {sc.stride} · {sizeHint}</span></div>
                  <div style={{ color: '#64748b', marginTop: 2 }}>
                    grid {sc.grid_w}×{sc.grid_h}
                    {top && <> · top: <strong>{top.name}</strong> ({top.peak.toFixed(2)})</>}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Compact structural metadata */}
      <section className="panel-section">
        <h4 className="section-label">Structure</h4>
        <div className="stats-grid">
          <div><span className="k">classes</span><span className="v mono">{internals.nc ?? '—'}</span></div>
          <div><span className="k">reg_max (DFL bins)</span><span className="v mono">{internals.reg_max ?? '—'}</span></div>
          <div><span className="k">strides</span><span className="v mono">{Array.isArray(internals.strides) ? internals.strides.join(', ') : '—'}</span></div>
          <div><span className="k">no (per-anchor outs)</span><span className="v mono">{internals.no ?? '—'}</span></div>
        </div>
      </section>
    </div>
  );
}

function UnknownCard({ nodeId, onClose }) {
  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[{nodeId}]</span>
          </div>
          <div className="panel-desc">No activation data available for this node.</div>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>
    </div>
  );
}

function getRoleForParent(parentIdx) {
  const arch = window.YV.ARCH;
  if (!arch || parentIdx == null) return 'Backbone';
  const b = arch[parentIdx];
  return b?.role || 'Backbone';
}

window.YV = window.YV || {};
window.YV.AppL2 = AppL2;

ReactDOM.createRoot(document.getElementById('root')).render(<AppL2 />);
