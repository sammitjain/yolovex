// Detail panel — slides in from the right when a block is clicked.
// For feature blocks: input → output strip + channel brochure + stats.
// For Detect block (idx 23): bbox preview + per-scale per-class heatmap rows.
//
// When YV_DATA is present and the current image matches its source image,
// renders REAL activations/predictions (base64 JPEGs from the Python build step).
// Otherwise falls back to procedural heatmaps so uploaded images degrade gracefully.

const { useState: useStateP, useRef: useRefP, useEffect: useEffectP, useMemo: useMemoP } = React;

// ---------- data helpers ----------

function dataFor(idx) {
  return window.YV_DATA?.blocks?.[idx];
}
function hasRealData(imageUrl) {
  return !!(window.YV_DATA && imageUrl && imageUrl === window.YV_DATA.meta.image);
}
function fmtShape(s) {
  if (!s) return '...';
  if (s === 'multi' || !Array.isArray(s)) return 'multi';
  return `(${s.join(', ')})`;
}

// JSX requires component names to be capitalized identifiers — alias the global.
function BlockInternalsView({ block, real }) {
  const Comp = window.YV?.BlockInternals;
  if (!Comp) return <div className="internals-caption">internals diagram unavailable</div>;
  return <Comp block={block} real={real} />;
}

// ---------- main panel ----------

function DetailPanel({ blockIdx, onClose, imageUrl }) {
  const { ARCH } = window.YV;
  const block = blockIdx != null ? ARCH[blockIdx] : null;
  const open = blockIdx != null;

  useEffectP(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return (
    <aside className={`detail-panel ${open ? 'open' : ''}`} aria-hidden={!open}>
      {block && (
        block.type === 'Detect'
          ? <DetectPanel block={block} onClose={onClose} imageUrl={imageUrl} />
          : <FeaturePanel block={block} onClose={onClose} imageUrl={imageUrl} />
      )}
    </aside>
  );
}

// ---------- feature block panel ----------

function FeaturePanel({ block, onClose, imageUrl }) {
  const { TYPE_COLORS, ROLE_COLORS } = window.YV;
  const cs = TYPE_COLORS[block.type];
  const isConcat = block.type === 'Concat';
  const real = hasRealData(imageUrl) ? dataFor(block.idx) : null;

  const shape = block.shape;
  const totalChannels = real?.total_channels ?? (Array.isArray(shape) ? shape[1] : 0);
  const visibleCh = Math.min(16, totalChannels);

  // Resolution heuristic for procedural fallback
  const resolution = Array.isArray(shape) ? (
    shape[2] >= 160 ? 0.4 : shape[2] >= 80 ? 0.7 : shape[2] >= 40 ? 1 : 1.4
  ) : 1;

  // Stats — real if available, otherwise stable pseudo-random
  const stats = useMemoP(() => {
    if (real?.stats) {
      return {
        min: real.stats.min.toFixed(2),
        max: real.stats.max.toFixed(2),
        mean: real.stats.mean.toFixed(3),
        topCh: (real.channel_indices || []).slice(0, 5),
      };
    }
    const r = (s) => ((Math.sin(block.idx * s) * 1000) % 1 + 1) % 1;
    return {
      min: -(2 + r(1.3) * 3).toFixed(2),
      max: (3 + r(2.1) * 4).toFixed(2),
      mean: (r(0.7) * 0.6 - 0.05).toFixed(3),
      topCh: Array.from({length: 5}, (_, i) => Math.floor(r(i + 1) * Math.max(1, totalChannels))),
    };
  }, [block.idx, real]);

  // Input shape caption — real if available, else previous-block fallback
  const inputCaption = real?.input_shapes
    ? real.input_shapes.map(fmtShape).join(' + ')
    : fmtShape(getInputShapeFallback(block));

  return (
    <div className="panel-inner">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[{block.idx}]</span>{' '}
            <span style={{ color: cs.text }}>{block.type}</span>
          </div>
          <div className="panel-desc">{block.desc}</div>
          <span className="role-pill" style={{ background: ROLE_COLORS[block.role] + '22', color: ROLE_COLORS[block.role] }}>
            {block.role}
          </span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>

      <section className="panel-section">
        <h4 className="section-label">Input → Output</h4>
        <IOStrip block={block} isConcat={isConcat} resolution={resolution} imageUrl={imageUrl} real={real} />
        <div className="shape-caption">
          {inputCaption} {' → '} {fmtShape(shape)}
        </div>
      </section>

      <section className="panel-section">
        <h4 className="section-label">Channel stack</h4>
        <ChannelStack
          block={block}
          totalChannels={totalChannels}
          visibleCh={visibleCh}
          resolution={resolution}
          real={real}
        />
        <div className="stack-caption">
          Showing {visibleCh} of {totalChannels} channels, ranked by mean |activation|.
          {totalChannels > visibleCh && <> <span className="dim">and {totalChannels - visibleCh} more.</span></>}
        </div>
      </section>

      <section className="panel-section">
        <h4 className="section-label">Inside the block</h4>
        <BlockInternalsView block={block} real={real} />
      </section>

      <section className="panel-section stats">
        <h4 className="section-label">
          Statistics
          {!real && <span style={{color:'#94a3b8',fontWeight:400,fontSize:10,textTransform:'none',letterSpacing:0,marginLeft:8}}>(illustrative — no real data for this image)</span>}
        </h4>
        <div className="stats-grid">
          <div><span className="k">shape</span><span className="v mono">{fmtShape(shape)}</span></div>
          <div><span className="k">params</span><span className="v mono">{block.params}</span></div>
          <div><span className="k">activation min</span><span className="v mono">{stats.min}</span></div>
          <div><span className="k">activation max</span><span className="v mono">{stats.max}</span></div>
          <div><span className="k">activation mean</span><span className="v mono">{stats.mean}</span></div>
          <div><span className="k">top-5 channels</span><span className="v mono">{stats.topCh.join(', ')}</span></div>
        </div>
      </section>
    </div>
  );
}

// Fallback when no real data — use the immediate predecessor's shape.
function getInputShapeFallback(block) {
  const { ARCH } = window.YV;
  if (block.idx === 0) return [1, 3, 640, 480];
  const prev = ARCH[block.idx - 1];
  return prev.shape === 'multi' ? [1, 3, 640, 480] : prev.shape;
}

// ---------- IO strip ----------

function IOStrip({ block, isConcat, resolution, imageUrl, real }) {
  const tileW = 96;
  const tileH = 120;
  if (real) {
    return <RealIOStrip block={block} isConcat={isConcat} imageUrl={imageUrl} tileW={tileW} tileH={tileH} />;
  }
  return <ProceduralIOStrip block={block} isConcat={isConcat} resolution={resolution} imageUrl={imageUrl} tileW={tileW} tileH={tileH} />;
}

function RealIOStrip({ block, isConcat, imageUrl, tileW, tileH }) {
  const real = dataFor(block.idx);
  const sources = real.sources || [];

  // Image source per input position: model-input (-1) → original image,
  // otherwise the source block's mean_thumbnail.
  const inputSrc = (idx) => {
    const src = sources[idx];
    if (src == null) return null;
    if (src < 0) return imageUrl;
    return dataFor(src)?.mean_thumbnail || null;
  };

  const tileStyle = { width: tileW, height: tileH, objectFit: 'cover' };
  const concatStyle = { width: 80, height: 100, objectFit: 'cover' };

  return (
    <div className="io-strip">
      {isConcat ? (
        <div className="io-pair">
          {sources.map((_, i) => (
            <React.Fragment key={i}>
              {i > 0 && <span className="plus">+</span>}
              <img src={inputSrc(i)} alt="" className="io-tile" style={concatStyle} />
            </React.Fragment>
          ))}
        </div>
      ) : (
        <img src={inputSrc(0)} alt="" className="io-tile" style={tileStyle} />
      )}
      <span className="arrow">→</span>
      <img src={real.mean_thumbnail} alt="" className="io-tile" style={tileStyle} />
    </div>
  );
}

function ProceduralIOStrip({ block, isConcat, resolution, imageUrl, tileW, tileH }) {
  const inA = useRefP(null);
  const inB = useRefP(null);
  const out = useRefP(null);
  useEffectP(() => {
    const { makeHeatmap } = window.YV;
    if (block.idx === 0) {
      // input is the actual image — leave canvas alone
    } else if (inA.current) {
      makeHeatmap(inA.current, { seed: block.idx * 7 + 1, resolution: resolution * 0.9, edges: block.type === 'Conv', accent: 0.6 });
    }
    if (isConcat && inB.current) {
      makeHeatmap(inB.current, { seed: block.idx * 7 + 200, resolution: resolution * 0.9, edges: false, accent: 0.5 });
    }
    if (out.current) {
      makeHeatmap(out.current, { seed: block.idx * 13 + 5, resolution, edges: block.type === 'Conv', accent: 0.7 });
    }
  }, [block.idx]);

  return (
    <div className="io-strip">
      {isConcat ? (
        <div className="io-pair">
          <canvas ref={inA} width="80" height="100" className="io-tile" />
          <span className="plus">+</span>
          <canvas ref={inB} width="80" height="100" className="io-tile" />
        </div>
      ) : (
        block.idx === 0 ? (
          <img src={imageUrl} alt="" className="io-tile" style={{ objectFit: 'cover', width: tileW, height: tileH }} />
        ) : (
          <canvas ref={inA} width={tileW} height={tileH} className="io-tile" />
        )
      )}
      <span className="arrow">→</span>
      <canvas ref={out} width={tileW} height={tileH} className="io-tile" />
    </div>
  );
}

// ---------- channel stack (the "brochure") ----------

function ChannelStack({ block, totalChannels, visibleCh, resolution, real }) {
  const [hovered, setHovered] = useStateP(null);
  const [pinned, setPinned] = useStateP(0);
  const previewRef = useRefP(null);
  const thumbRefs = useRefP([]);

  const active = hovered != null ? hovered : pinned;
  const trueChannelIdx = (i) => real?.channel_indices?.[i] ?? i;

  // Procedural rendering for fallback path
  useEffectP(() => {
    if (real) return;
    const { makeHeatmap } = window.YV;
    thumbRefs.current.forEach((c, i) => {
      if (c) makeHeatmap(c, {
        seed: block.idx * 31 + i * 17, resolution,
        edges: block.type === 'Conv' && i < 4,
        accent: 0.4 + (i / Math.max(1, visibleCh)) * 0.5,
      });
    });
  }, [block.idx, visibleCh, real]);

  useEffectP(() => {
    if (real) return;
    const { makeHeatmap } = window.YV;
    if (previewRef.current && active != null) {
      makeHeatmap(previewRef.current, {
        seed: block.idx * 31 + active * 17, resolution,
        edges: block.type === 'Conv' && active < 4,
        accent: 0.4 + (active / Math.max(1, visibleCh)) * 0.5,
      });
    }
  }, [active, block.idx, real]);

  return (
    <div className="channel-brochure">
      <div className="brochure-preview">
        {real ? (
          <img
            src={real.channel_pngs[active]}
            alt={`channel ${trueChannelIdx(active)}`}
            style={{ width: 170, height: 210, objectFit: 'cover', display: 'block', borderRadius: 4, border: '2px solid var(--accent)', boxShadow: '0 2px 6px rgba(15,23,42,.18)' }}
          />
        ) : (
          <canvas ref={previewRef} width={170} height={210} />
        )}
        <div className="brochure-meta">
          <div className="meta-row">
            <span className="meta-k">channel</span>
            <span className="meta-v mono">{trueChannelIdx(active)}</span>
          </div>
          <div className="meta-row">
            <span className="meta-k">{hovered != null ? 'previewing' : 'pinned'}</span>
            <span className="meta-v">{hovered != null ? <em>hover</em> : <em>click to change</em>}</span>
          </div>
          <div className="meta-row">
            <span className="meta-k">rank</span>
            <span className="meta-v mono">#{active + 1} of {totalChannels}</span>
          </div>
        </div>
      </div>

      <div className="brochure-grid" onMouseLeave={() => setHovered(null)}>
        {Array.from({ length: visibleCh }).map((_, i) => {
          const isPinned = pinned === i;
          const isHover = hovered === i;
          return (
            <button
              key={i}
              className={`brochure-thumb ${isPinned ? 'pinned' : ''} ${isHover ? 'hovered' : ''}`}
              onMouseEnter={() => setHovered(i)}
              onClick={() => setPinned(i)}
              title={`channel ${trueChannelIdx(i)}`}
            >
              {real ? (
                <img src={real.channel_pngs[i]} alt="" style={{ display: 'block', width: '100%', height: 'auto' }} />
              ) : (
                <canvas
                  ref={(el) => { thumbRefs.current[i] = el; }}
                  width={48}
                  height={60}
                />
              )}
              <span className="thumb-idx">{trueChannelIdx(i)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---------- detect panel ----------

function DetectPanel({ block, onClose, imageUrl }) {
  const { ROLE_COLORS } = window.YV;
  const real = hasRealData(imageUrl) ? dataFor(23) : null;

  return (
    <div className="panel-inner detect-panel">
      <header className="panel-header">
        <div>
          <div className="panel-title">
            <span className="panel-idx">[23]</span>{' '}
            <span style={{ color: '#14532d' }}>Detect</span>
          </div>
          <div className="panel-desc">Multi-scale anchor-free detection head. Decodes class scores and bounding boxes from the three pyramid scales — end-to-end with no NMS.</div>
          <span className="role-pill" style={{ background: ROLE_COLORS.Head + '22', color: ROLE_COLORS.Head }}>
            Head
          </span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
      </header>

      <section className="panel-section">
        <h4 className="section-label">Predictions</h4>
        <BBoxPreview imageUrl={imageUrl} boxes={real?.boxes} />
        <PredsList boxes={real?.boxes} />
      </section>

      <section className="panel-section">
        <h4 className="section-label">Per-class score heatmaps</h4>
        <p className="micro-help">
          Top {(real?.scales?.P3?.classes?.length) || 6} classes by peak score across all scales,
          arranged so each <strong>row</strong> is a class and each <strong>column</strong>{' '}
          is a pyramid scale. Easy way to see <em>which scale a class actually fires on</em> —
          small objects light up in P3, large ones in P5. The cell with the brightest peak
          for that class is highlighted.
          {!real && <em style={{color:'#94a3b8',marginLeft:6}}>Showing illustrative stubs — no real predictions for this image.</em>}
        </p>
        {real
          ? <ScaleGrid scales={real.scales} imageUrl={imageUrl} />
          : <ProceduralScaleGrid imageUrl={imageUrl} />}
      </section>

      <section className="panel-section">
        <h4 className="section-label">Inside the head</h4>
        <BlockInternalsView block={block} real={real} />
      </section>
    </div>
  );
}

// Compact list of detection chips below the bbox preview
function PredsList({ boxes }) {
  if (!boxes || boxes.length === 0) {
    return <div className="preds-list"><span style={{color:'#94a3b8',fontSize:11}}>(none)</span></div>;
  }
  return (
    <div className="preds-list">
      {boxes.map((b, i) => (
        <span key={i} className={`pred-chip ${b.conf < 0.5 ? 'weak' : ''}`}>
          <span className="dot" />
          {b.cls_name}
          <span className="conf">{b.conf.toFixed(2)}</span>
        </span>
      ))}
    </div>
  );
}

// Pivoted grid: rows = classes, columns = P3/P4/P5
function ScaleGrid({ scales, imageUrl }) {
  const scaleNames = ['P3', 'P4', 'P5'];
  const sizeMap = { P3: 'small', P4: 'medium', P5: 'large' };
  const classes = scales[scaleNames[0]].classes; // same class order across scales

  return (
    <div className="scale-grid">
      {/* Header row */}
      <div className="scale-grid-header first">class</div>
      {scaleNames.map((name) => (
        <div key={name} className="scale-grid-header">
          {name} <span className="sub">stride {scales[name].stride} · {sizeMap[name]}</span>
        </div>
      ))}

      {/* Image reference row */}
      <div className="scale-grid-imgrow">
        <div style={{fontSize:11,color:'#64748b',padding:'0 4px'}}>image</div>
        {scaleNames.map((name) => (
          <img key={name} src={imageUrl} alt="" className="scale-ref" />
        ))}
      </div>

      {/* One row per class */}
      {classes.map((_, ci) => {
        const cells = scaleNames.map(n => scales[n].classes[ci]);
        const peakScale = cells.reduce((best, c, i) => c.peak > cells[best].peak ? i : best, 0);
        const top = cells[ci === peakScale ? ci : 0]; // for label
        return (
          <React.Fragment key={ci}>
            <div className="scale-grid-class">
              <span className="cls-name">{cells[0].name}</span>
              <span className="cls-peak">peak {Math.max(...cells.map(c => c.peak)).toFixed(2)}</span>
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

// Procedural fallback for uploaded images (preserves layout but stubs the data)
function ProceduralScaleGrid({ imageUrl }) {
  const scaleNames = ['P3', 'P4', 'P5'];
  const sizeMap = { P3: 'small', P4: 'medium', P5: 'large' };
  const strides = { P3: 8, P4: 16, P5: 32 };
  const classes = [
    { name: 'person',  pk: { P3: 0.95, P4: 0.85, P5: 0.55 } },
    { name: 'tie',     pk: { P3: 0.30, P4: 0.27, P5: 0.18 } },
    { name: 'backpack',pk: { P3: 0.18, P4: 0.16, P5: 0.10 } },
    { name: 'handbag', pk: { P3: 0.13, P4: 0.12, P5: 0.07 } },
    { name: 'vase',    pk: { P3: 0.10, P4: 0.09, P5: 0.05 } },
    { name: 'bottle',  pk: { P3: 0.07, P4: 0.06, P5: 0.04 } },
  ];

  return (
    <div className="scale-grid">
      <div className="scale-grid-header first">class</div>
      {scaleNames.map((name) => (
        <div key={name} className="scale-grid-header">
          {name} <span className="sub">stride {strides[name]} · {sizeMap[name]}</span>
        </div>
      ))}
      <div className="scale-grid-imgrow">
        <div style={{fontSize:11,color:'#64748b',padding:'0 4px'}}>image</div>
        {scaleNames.map((name) => (
          <img key={name} src={imageUrl} alt="" className="scale-ref" />
        ))}
      </div>
      {classes.map((c, ci) => {
        const peaks = scaleNames.map(n => c.pk[n]);
        const peakScale = peaks.indexOf(Math.max(...peaks));
        return (
          <React.Fragment key={ci}>
            <div className="scale-grid-class">
              <span className="cls-name">{c.name}</span>
              <span className="cls-peak">peak {Math.max(...peaks).toFixed(2)}</span>
            </div>
            {scaleNames.map((n, i) => {
              const isPeak = i === peakScale && peaks[i] >= 0.05;
              const faint = peaks[i] < 0.05;
              return (
                <ProcCell key={n} score={peaks[i]} isPeak={isPeak} faint={faint}
                          gridW={n === 'P3' ? 16 : n === 'P4' ? 10 : 5}
                          gridH={n === 'P3' ? 22 : n === 'P4' ? 14 : 7}
                          seed={ci * 31 + i * 7 + 1} />
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function ProcCell({ score, isPeak, faint, gridW, gridH, seed }) {
  const ref = useRefP(null);
  useEffectP(() => {
    const { makeClassHeatmap } = window.YV;
    if (ref.current) {
      makeClassHeatmap(ref.current, {
        seed, gridW, gridH,
        peakX: 0.55 + (seed % 7) * 0.02, peakY: 0.6,
        peakStrength: score, noise: 0.18,
      });
    }
  }, [seed]);
  return (
    <div className={`scale-grid-cell ${isPeak ? 'is-peak' : ''} ${faint ? 'faint' : ''}`}>
      <canvas ref={ref} width="80" height="106" style={{display:'block',width:'100%',height:'100%'}} />
      <span className="cell-score">{score.toFixed(2)}</span>
    </div>
  );
}

function BBoxPreview({ imageUrl, boxes }) {
  // viewBox dimensions are arbitrary; box coords are normalized [0,1] then scaled here.
  const VB_W = 1000;
  const VB_H = 1000;
  const shown = (boxes || []).filter(b => b.conf >= 0.2);
  return (
    <div className="bbox-preview">
      <img src={imageUrl} alt="" />
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} preserveAspectRatio="none" className="bbox-overlay">
        {shown.length > 0 ? shown.map((b, i) => {
          const x = b.x1 * VB_W;
          const y = b.y1 * VB_H;
          const w = (b.x2 - b.x1) * VB_W;
          const h = (b.y2 - b.y1) * VB_H;
          const isStrong = b.conf >= 0.5;
          const stroke = '#22c55e';
          return (
            <g key={i}>
              <rect
                x={x} y={y} width={w} height={h}
                fill="none" stroke={stroke}
                strokeWidth={isStrong ? 4 : 3}
                strokeDasharray={isStrong ? null : '6 4'}
                opacity={isStrong ? 1 : 0.75}
                rx="2"
              />
              <g transform={`translate(${x}, ${Math.max(0, y - 22)})`}>
                <rect width={130} height={22} rx="3" fill={stroke} opacity={isStrong ? 1 : 0.8} />
                <text x="6" y="16" fontSize="13" fill="#fff" fontFamily="ui-monospace, monospace" fontWeight="600">
                  {b.cls_name} {b.conf.toFixed(2)}
                </text>
              </g>
            </g>
          );
        }) : (
          // Procedural fallback boxes (for uploaded images)
          <>
            <rect x="305" y="450" width="345" height="540" fill="none" stroke="#22c55e" strokeWidth="4" rx="2" />
            <g transform="translate(305, 432)">
              <rect width="98" height="22" rx="3" fill="#22c55e" />
              <text x="6" y="16" fontSize="13" fill="#fff" fontFamily="ui-monospace, monospace" fontWeight="600">person 0.93</text>
            </g>
            <rect x="395" y="600" width="80" height="120" fill="none" stroke="#22c55e" strokeWidth="3" strokeDasharray="6 4" rx="2" opacity="0.75" />
            <g transform="translate(395, 582)">
              <rect width="68" height="20" rx="3" fill="#22c55e" opacity="0.8" />
              <text x="5" y="14" fontSize="11" fill="#fff" fontFamily="ui-monospace, monospace" fontWeight="600">tie 0.28</text>
            </g>
          </>
        )}
      </svg>
    </div>
  );
}

window.YV = window.YV || {};
window.YV.DetailPanel = DetailPanel;
