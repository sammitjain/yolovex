// yolovex v2 — app shell. Header + the architecture graph. Selection state is
// held here so future steps (in-place expansion, explanation panels) can hook
// into it without restructuring.

const { useState, useCallback } = React;

function AppV2() {
  const [hoveredId, setHoveredId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [expandedCount, setExpandedCount] = useState(0);
  const onExpandedCountChange = useCallback((n) => setExpandedCount(n), []);

  return (
    <div className="app">
      <header className="app-header">
        <strong>yolovex</strong>
        <span className="divider">/</span>
        <span className="subtitle">YOLO26 architecture · L1</span>
        <span className="hint">
          {expandedCount > 0
            ? `${expandedCount} block${expandedCount === 1 ? '' : 's'} expanded · click to collapse`
            : 'click a block to expand it · hover to trace connections · scroll to zoom · drag to pan'}
        </span>
      </header>
      <main className="app-main">
        <window.YVV2.GraphV2
          hoveredIdx={hoveredId}
          selectedIdx={selectedId}
          onHover={setHoveredId}
          onSelect={(idx) => setSelectedId(cur => (cur === idx ? null : idx))}
          onExpandedCountChange={onExpandedCountChange}
        />
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<AppV2 />);
