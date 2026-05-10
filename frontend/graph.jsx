// Graph component — renders the architecture as SVG, handles pan/zoom, hover, click, play-flow.

const { useState, useRef, useEffect, useMemo, useCallback } = React;

function Graph({ selectedIdx, hoveredIdx, onSelect, onHover, playingIdx }) {
  const { ARCH, EDGES, TYPE_COLORS, ROLE_COLORS, ACCENT, layoutGraph, edgePath, NODE_W, NODE_H } = window.YV;
  const { nodes, totalW, totalH } = useMemo(() => layoutGraph(ARCH), []);
  const containerRef = useRef(null);

  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 });
  const [containerSize, setContainerSize] = useState({ w: 1000, h: 720 });
  const [didFit, setDidFit] = useState(false);
  const [drawnIn, setDrawnIn] = useState(true);
  const draggingRef = useRef(null);

  // Container size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setContainerSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setContainerSize({ w: r.width, h: r.height });
    return () => ro.disconnect();
  }, []);

  // Fit to screen
  const fit = useCallback(() => {
    const padX = 32;
    // Fit horizontally; let vertical overflow into pan/scroll. Cap zoom at 1.
    const k = Math.min((containerSize.w - padX * 2) / totalW, 1);
    const x = (containerSize.w - totalW * k) / 2;
    // Anchor near top with a small offset so users see the start of the graph immediately.
    const y = 24;
    setTransform({ x, y, k });
  }, [containerSize, totalW, totalH]);

  useEffect(() => {
    if (!didFit && containerSize.w > 100) {
      fit();
      setDidFit(true);
    }
  }, [containerSize, didFit, fit]);

  // Entry animation runs on mount via CSS only

  // Wheel zoom
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setTransform(t => {
      const factor = Math.exp(-e.deltaY * 0.0015);
      const newK = Math.max(0.3, Math.min(2.5, t.k * factor));
      // zoom around cursor
      const nx = mx - (mx - t.x) * (newK / t.k);
      const ny = my - (my - t.y) * (newK / t.k);
      return { x: nx, y: ny, k: newK };
    });
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [onWheel]);

  // Drag pan
  const onMouseDown = (e) => {
    if (e.target.closest('[data-node]')) return;
    draggingRef.current = { startX: e.clientX, startY: e.clientY, t: transform };
  };
  useEffect(() => {
    const onMove = (e) => {
      if (!draggingRef.current) return;
      const d = draggingRef.current;
      setTransform({
        x: d.t.x + (e.clientX - d.startX),
        y: d.t.y + (e.clientY - d.startY),
        k: d.t.k,
      });
    };
    const onUp = () => { draggingRef.current = null; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, []);

  // Edges connected to hovered/selected node
  const focusIdx = hoveredIdx ?? selectedIdx;
  const isEdgeFocused = (e) => focusIdx != null && (e[0] === focusIdx || e[1] === focusIdx);

  // Column bounding boxes
  const cols = useMemo(() => {
    const groups = { Backbone: [], Neck: [], Head: [] };
    ARCH.forEach(b => groups[b.role].push(b));
    const bbox = (idxs) => {
      const ns = idxs.map(b => nodes[b.idx]);
      const x = Math.min(...ns.map(n => n.x)) - 18;
      const y = Math.min(...ns.map(n => n.y)) - 28;
      const x2 = Math.max(...ns.map(n => n.x + n.w)) + 18;
      const y2 = Math.max(...ns.map(n => n.y + n.h)) + 18;
      return { x, y, w: x2 - x, h: y2 - y };
    };
    return {
      Backbone: bbox(groups.Backbone),
      Neck: bbox([...groups.Neck]),
      Head: bbox(groups.Head),
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="graph-container"
      onMouseDown={onMouseDown}
      style={{ cursor: draggingRef.current ? 'grabbing' : 'grab' }}
    >
      <svg
        width={containerSize.w}
        height={containerSize.h}
        style={{ display: 'block' }}
      >
        <defs>
          <marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
          </marker>
          <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={ACCENT} />
          </marker>
          <marker id="arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#cbd5e1" opacity="0.4" />
          </marker>
          <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="6" floodOpacity="0.12" />
          </filter>
          <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="0" stdDeviation="8" floodColor={ACCENT} floodOpacity="0.6" />
          </filter>
        </defs>
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.k})`}>
          {/* Column bounding boxes */}
          {Object.entries(cols).map(([role, b]) => (
            <g key={role}>
              <rect
                x={b.x} y={b.y} width={b.w} height={b.h}
                fill="none"
                stroke={ROLE_COLORS[role]}
                strokeWidth="1.2"
                strokeDasharray="4 4"
                rx="14"
                opacity="0.55"
              />
              <text
                x={b.x + 12}
                y={b.y - 8}
                fontSize="12"
                fontWeight="600"
                fill={ROLE_COLORS[role]}
                style={{ letterSpacing: '0.06em', textTransform: 'uppercase' }}
              >
                {role}
              </text>
            </g>
          ))}

          {/* Edges */}
          {EDGES.map((e, i) => {
            const [from, to, kind] = e;
            const path = edgePath(from, to, kind, nodes, ARCH);
            const focused = isEdgeFocused(e);
            const isSkip = kind === 'skip' || kind.startsWith('detect-');
            const stroke = focused ? ACCENT : (isSkip ? ACCENT : '#94a3b8');
            const opacity = focusIdx != null && !focused ? 0.18 : (isSkip ? 0.85 : 0.55);
            const sw = focused ? 2.6 : (isSkip ? 1.8 : 1);
            const marker = focused ? 'url(#arrow-accent)' : (focusIdx != null ? 'url(#arrow-dim)' : (isSkip ? 'url(#arrow-accent)' : 'url(#arrow-gray)'));

            return (
              <g key={i}>
                <path
                  d={path}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={sw}
                  opacity={opacity}
                  markerEnd={marker}
                  style={{
                    transition: 'stroke 200ms, opacity 200ms, stroke-width 200ms',
                    strokeDasharray: drawnIn ? 'none' : '600',
                    strokeDashoffset: drawnIn ? 0 : 600,
                    transitionProperty: 'stroke, opacity, stroke-width, stroke-dashoffset',
                    transitionDuration: '600ms',
                    transitionDelay: drawnIn ? `${300 + i * 8}ms` : '0ms',
                  }}
                />
                {/* Tooltip hit area for skip edges */}
                {isSkip && (
                  <SkipTooltip path={path} from={from} to={to} arch={ARCH} />
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {ARCH.map((b, i) => {
            const n = nodes[b.idx];
            return (
              <Node
                key={b.idx}
                block={b}
                node={n}
                hovered={hoveredIdx === b.idx}
                selected={selectedIdx === b.idx}
                playing={playingIdx === b.idx}
                dimmed={focusIdx != null && focusIdx !== b.idx && !connectedTo(focusIdx, b.idx)}
                onHover={onHover}
                onSelect={onSelect}
                drawnIn={drawnIn}
                colorScheme={TYPE_COLORS[b.type]}
                delay={50 + i * 18}
              />
            );
          })}

          {/* Input image marker (left of backbone) */}
          <g transform={`translate(${COL_X_FIRST() - 110}, ${nodes[0].y - 4})`}>
            <rect width="80" height="60" rx="6" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1" />
            <text x="40" y="34" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="ui-monospace, monospace">input</text>
            <text x="40" y="46" fontSize="9" fill="#94a3b8" textAnchor="middle" fontFamily="ui-monospace, monospace">(1,3,640,480)</text>
          </g>
          <path
            d={`M ${COL_X_FIRST() - 30} ${nodes[0].y + 26} L ${nodes[0].x} ${nodes[0].y + 26}`}
            stroke="#94a3b8"
            strokeWidth="1"
            opacity="0.5"
            markerEnd="url(#arrow-gray)"
            fill="none"
          />
        </g>
      </svg>

      {/* Zoom controls */}
      <div className="zoom-controls">
        <button onClick={() => setTransform(t => ({ ...t, k: Math.min(2.5, t.k * 1.2) }))} title="Zoom in">+</button>
        <button onClick={() => setTransform(t => ({ ...t, k: Math.max(0.3, t.k / 1.2) }))} title="Zoom out">−</button>
        <button onClick={fit} title="Fit to screen" className="fit-btn">⤢</button>
      </div>
    </div>
  );
}

function COL_X_FIRST() { return 120; }

function connectedTo(idx, other) {
  const { EDGES } = window.YV;
  return EDGES.some(e => (e[0] === idx && e[1] === other) || (e[1] === idx && e[0] === other));
}

function Node({ block, node, hovered, selected, playing, dimmed, onHover, onSelect, drawnIn, colorScheme, delay }) {
  const { ACCENT } = window.YV;
  const transform = `translate(${node.x}, ${node.y})`;
  const lift = (hovered || playing) ? 1.02 : 1;
  const opacity = drawnIn ? (dimmed ? 0.35 : 1) : 0;

  const isDetect = block.type === 'Detect';

  return (
    <g
      data-node={block.idx}
      transform={transform}
      style={{
        cursor: 'pointer',
        opacity,
        transition: `opacity 400ms cubic-bezier(.4,0,.2,1)`,
      }}
      className="node-fade-in"
      onMouseEnter={() => onHover(block.idx)}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => { e.stopPropagation(); onSelect(block.idx); }}
    >
      <g
        style={{
          transformOrigin: `${node.w / 2}px ${node.h / 2}px`,
          transform: `scale(${lift})`,
          transition: 'transform 180ms cubic-bezier(.4,0,.2,1)',
          filter: selected || playing ? 'url(#node-glow)' : (hovered ? 'url(#node-shadow)' : 'none'),
        }}
      >
        {isDetect ? (
          <DetectNode node={node} colorScheme={colorScheme} highlight={selected || hovered || playing} />
        ) : (
          <>
            <rect
              width={node.w}
              height={node.h}
              rx="8"
              fill={colorScheme.fill}
              stroke={selected || playing ? ACCENT : colorScheme.border}
              strokeWidth={selected || playing ? 2 : 1}
            />
            <text x="12" y="22" fontSize="11" fontWeight="500" fill={colorScheme.text} fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.7">
              [{block.idx}]
            </text>
            <text x="38" y="22" fontSize="13" fontWeight="600" fill={colorScheme.text}>
              {block.type}
            </text>
            <text x="12" y="44" fontSize="10.5" fill={colorScheme.text} fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.85">
              ({block.shape.slice(0, 4).join(', ')})
            </text>
            {block.skip && (
              <circle cx={node.w - 10} cy={10} r={3.5} fill={ACCENT} opacity="0.85">
                <title>This block is a skip-connection source</title>
              </circle>
            )}
          </>
        )}
      </g>
    </g>
  );
}

function DetectNode({ node, colorScheme, highlight }) {
  const { ACCENT } = window.YV;
  const subH = node.h / 3;
  const labels = ['P3', 'P4', 'P5'];
  const sizes = ['small', 'medium', 'large'];
  return (
    <g>
      <rect
        width={node.w}
        height={node.h}
        rx="8"
        fill="#ffffff"
        stroke={highlight ? ACCENT : colorScheme.border}
        strokeWidth={highlight ? 2 : 1}
      />
      <text x="12" y="-6" fontSize="11" fontWeight="500" fill={colorScheme.text} fontFamily="ui-monospace, SFMono-Regular, monospace">
        [23] Detect
      </text>
      {labels.map((lbl, i) => (
        <g key={lbl} transform={`translate(0, ${i * subH})`}>
          <rect
            x="6"
            y="10"
            width={node.w - 12}
            height={subH - 20}
            rx="5"
            fill={colorScheme.fill}
            stroke={colorScheme.border}
            strokeWidth="0.8"
          />
          <text x="16" y={subH / 2 + 4} fontSize="13" fontWeight="600" fill={colorScheme.text}>
            {lbl}
          </text>
          <text x={node.w - 16} y={subH / 2 + 4} fontSize="10" fill={colorScheme.text} opacity="0.7" textAnchor="end">
            {sizes[i]}
          </text>
        </g>
      ))}
    </g>
  );
}

function SkipTooltip({ path, from, to, arch }) {
  const [hover, setHover] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const fromBlock = arch[from];
  const toBlock = arch[to];
  const shape = Array.isArray(fromBlock.shape)
    ? `(${fromBlock.shape.join(', ')})`
    : 'multi';
  return (
    <g>
      <path
        d={path}
        fill="none"
        stroke="transparent"
        strokeWidth="14"
        style={{ cursor: 'help' }}
        onMouseMove={(e) => {
          const svg = e.currentTarget.ownerSVGElement;
          const pt = svg.createSVGPoint();
          pt.x = e.clientX; pt.y = e.clientY;
          const ctm = e.currentTarget.getScreenCTM().inverse();
          const lp = pt.matrixTransform(ctm);
          setPos({ x: lp.x, y: lp.y });
          setHover(true);
        }}
        onMouseLeave={() => setHover(false)}
      />
      {hover && (
        <g transform={`translate(${pos.x + 10}, ${pos.y - 30})`} pointerEvents="none">
          <rect x="0" y="0" width="180" height="42" rx="6" fill="#1f2328" />
          <text x="10" y="16" fontSize="11" fill="#fff" fontFamily="ui-monospace, monospace">
            [{from}] → [{to}]
          </text>
          <text x="10" y="32" fontSize="10" fill="#cbd5e1" fontFamily="ui-monospace, monospace">
            {shape}
          </text>
        </g>
      )}
    </g>
  );
}

window.YV = window.YV || {};
window.YV.Graph = Graph;
