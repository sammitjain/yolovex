// Main app — header, image picker, play-flow controller, composes Graph + DetailPanel.

const { useState: useStateA, useEffect: useEffectA, useRef: useRefA, useCallback: useCallbackA } = React;

function App() {
  const { Graph, DetailPanel, ARCH } = window.YV;
  const [selectedIdx, setSelectedIdx] = useStateA(null);
  const [hoveredIdx, setHoveredIdx] = useStateA(null);
  const [playingIdx, setPlayingIdx] = useStateA(null);
  const [imageUrl, setImageUrl] = useStateA('assets/sammit_lighthouse.jpg');
  const [pickerOpen, setPickerOpen] = useStateA(false);

  const playFlow = useCallbackA(() => {
    if (playingIdx != null) return;
    const order = ARCH.map(b => b.idx);
    let i = 0;
    const tick = () => {
      if (i >= order.length) { setPlayingIdx(null); return; }
      setPlayingIdx(order[i]);
      i++;
      setTimeout(tick, 4000 / order.length);
    };
    tick();
  }, [playingIdx]);

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
          <div className="wordmark">yolovex</div>
          <div className="subtitle">interactive YOLO26 explainer</div>
        </div>
        <div className="header-right">
          <button className="play-btn" onClick={playFlow} disabled={playingIdx != null}>
            <span className="play-icon">▶</span> {playingIdx != null ? 'Playing...' : 'Play flow'}
          </button>
          <ImagePicker open={pickerOpen} setOpen={setPickerOpen} onUpload={onUpload} setImageUrl={setImageUrl} currentUrl={imageUrl} />
          <div className="attribution">made by <span>Sammit</span></div>
        </div>
      </header>

      <main className="app-main">
        <Graph
          selectedIdx={selectedIdx}
          hoveredIdx={hoveredIdx}
          onSelect={setSelectedIdx}
          onHover={setHoveredIdx}
          playingIdx={playingIdx}
        />
        <DetailPanel
          blockIdx={selectedIdx}
          onClose={() => setSelectedIdx(null)}
          imageUrl={imageUrl}
        />
      </main>

      <Legend />
    </div>
  );
}

function ImagePicker({ open, setOpen, onUpload, setImageUrl, currentUrl }) {
  const ref = useRefA(null);
  useEffectA(() => {
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    if (open) document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  // Only one populated thumbnail per brief
  const thumbs = [
    { url: 'assets/sammit_lighthouse.jpg', label: 'lighthouse' },
  ];

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

function Legend() {
  const { TYPE_COLORS, ACCENT } = window.YV;
  const items = [
    ['Conv', TYPE_COLORS.Conv],
    ['C3k2', TYPE_COLORS.C3k2],
    ['Upsample', TYPE_COLORS.Upsample],
    ['Concat', TYPE_COLORS.Concat],
    ['SPPF', TYPE_COLORS.SPPF],
    ['C2PSA', TYPE_COLORS.C2PSA],
    ['Detect', TYPE_COLORS.Detect],
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
        <span>skip connection</span>
      </div>
      <div className="legend-item">
        <span className="swatch-line" style={{ background: '#94a3b8' }} />
        <span>forward</span>
      </div>
    </div>
  );
}

window.YV = window.YV || {};
window.YV.App = App;

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
