// L2 Graph — renders containers with sub-nodes, handles pan/zoom/hover/click.

const { useState: useStateG2, useRef: useRefG2, useEffect: useEffectG2, useMemo: useMemoG2, useCallback: useCallbackG2 } = React;

function GraphL2({ selectedId, hoveredId, onSelect, onHover, playingId, layoutParams }) {
  const {
    getArchL2, getEdgesInterL2, getEdgesIntraL2,
    layoutGraphL2, edgePathL2,
    ROLE_COLORS, ACCENT,
    getTypeColor, TYPE_COLORS_L2,
    COL_X_L2,
  } = window.YV;

  const arch_l2 = useMemoG2(() => getArchL2(), []);
  const edgesInter = useMemoG2(() => getEdgesInterL2(), []);
  const edgesIntra = useMemoG2(() => getEdgesIntraL2(), []);
  const { containers, subnodes, totalW, totalH } = useMemoG2(() => layoutGraphL2(arch_l2), []);

  const containerRef = useRefG2(null);
  const [transform, setTransform] = useStateG2({ x: 0, y: 0, k: 1 });
  const [containerSize, setContainerSize] = useStateG2({ w: 1000, h: 720 });
  const [didFit, setDidFit] = useStateG2(false);
  const [drawnIn, setDrawnIn] = useStateG2(true);
  const draggingRef = useRefG2(null);

  // Container size observer
  useEffectG2(() => {
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

  const fit = useCallbackG2(() => {
    const padX = 32;
    const k = Math.min((containerSize.w - padX * 2) / totalW, 1);
    const x = (containerSize.w - totalW * k) / 2;
    const y = 24;
    setTransform({ x, y, k });
  }, [containerSize, totalW]);

  useEffectG2(() => {
    if (!didFit && containerSize.w > 100) {
      fit();
      setDidFit(true);
    }
  }, [containerSize, didFit, fit]);

  // Wheel zoom
  const onWheel = useCallbackG2((e) => {
    e.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setTransform(t => {
      const factor = Math.exp(-e.deltaY * 0.0015);
      const newK = Math.max(0.3, Math.min(2.5, t.k * factor));
      const nx = mx - (mx - t.x) * (newK / t.k);
      const ny = my - (my - t.y) * (newK / t.k);
      return { x: nx, y: ny, k: newK };
    });
  }, []);

  useEffectG2(() => {
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
  useEffectG2(() => {
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

  // Focus logic: which id is "active" for dimming / edge highlighting
  const focusId = hoveredId ?? selectedId;

  // Edge focus check: is either endpoint matching focusId?
  function isEdgeFocused(edge) {
    if (focusId == null) return false;
    // Check direct match
    if (edge.from === focusId || edge.to === focusId) return true;
    // If focusId is a container idx string, check if edge endpoints belong to that container
    const fStr = String(focusId);
    const fromStr = String(edge.from);
    const toStr = String(edge.to);
    if (fromStr.split('.')[0] === fStr || toStr.split('.')[0] === fStr) return true;
    return false;
  }

  // Dimming: a node is dimmed if there's a focus and it's not connected
  function isNodeDimmed(id) {
    if (focusId == null) return false;
    if (String(id) === String(focusId)) return false;
    // Same container as focus?
    const idStr = String(id);
    const focusStr = String(focusId);
    if (idStr.includes('.') && idStr.split('.')[0] === focusStr) return false;
    if (focusStr.includes('.') && focusStr.split('.')[0] === idStr) return false;
    // Check edges
    const allEdges = [...edgesInter, ...edgesIntra];
    return !allEdges.some(e => {
      const f = String(e.from), t = String(e.to);
      return (f === focusStr || t === focusStr) && (f === idStr || t === idStr);
    });
  }

  // Column bounding boxes (reuse L1 role groups)
  const colBoxes = useMemoG2(() => {
    const groups = { Backbone: [], Neck: [], Head: [] };
    arch_l2.forEach(b => groups[b.role].push(b));
    const bbox = (blocks) => {
      const cs = blocks.map(b => containers[b.idx]).filter(Boolean);
      if (!cs.length) return null;
      const x = Math.min(...cs.map(c => c.x)) - 18;
      const y = Math.min(...cs.map(c => c.y)) - 28;
      const x2 = Math.max(...cs.map(c => c.x + c.w)) + 18;
      const y2 = Math.max(...cs.map(c => c.y + c.h)) + 18;
      return { x, y, w: x2 - x, h: y2 - y };
    };
    return {
      Backbone: bbox(groups.Backbone),
      Neck: bbox(groups.Neck),
      Head: bbox(groups.Head),
    };
  }, []);

  const allEdges = [...edgesInter, ...edgesIntra];

  return (
    <div
      ref={containerRef}
      className="graph-container"
      onMouseDown={onMouseDown}
      style={{ cursor: 'grab' }}
    >
      <svg width={containerSize.w} height={containerSize.h} style={{ display: 'block' }}>
        <defs>
          {/* Arrow heads — concave (notched-tail) triangles drawn in user
             space so they keep a consistent size regardless of stroke width
             and connect seamlessly to the curves they cap. The notch at the
             back (the 4th vertex) is what gives the classic draw.io style:
             the arrow tail meets the curve at two points instead of one,
             which hides any tiny tangent mismatch. */}
          <marker id="arrow-gray-l2" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 12 6 L 0 12 L 3 6 z" fill="#94a3b8" />
          </marker>
          <marker id="arrow-accent-l2" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 12 6 L 0 12 L 3 6 z" fill={ACCENT} />
          </marker>
          <marker id="arrow-dim-l2" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 12 6 L 0 12 L 3 6 z" fill="#cbd5e1" opacity="0.5" />
          </marker>
          <marker id="arrow-intra-l2" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 12 6 L 0 12 L 3 6 z" fill="#94a3b8" />
          </marker>
          <marker id="arrow-fanin-l2" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 12 6 L 0 12 L 3 6 z" fill="#7c3aed" />
          </marker>
          <filter id="node-shadow-l2" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="6" floodOpacity="0.12" />
          </filter>
          <filter id="node-glow-l2" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="0" stdDeviation="8" floodColor={ACCENT} floodOpacity="0.6" />
          </filter>
        </defs>
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.k})`}>

          {/* Column bounding boxes */}
          {Object.entries(colBoxes).map(([role, b]) => b && (
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
                x={b.x + 12} y={b.y - 8}
                fontSize="12" fontWeight="600"
                fill={ROLE_COLORS[role]}
                style={{ letterSpacing: '0.06em', textTransform: 'uppercase' }}
              >
                {role}
              </text>
            </g>
          ))}

          {/* Edges */}
          {allEdges.map((edge, i) => {
            const path = edgePathL2(edge, containers, subnodes, layoutParams);
            if (!path) return null;
            const focused = isEdgeFocused(edge);
            const isSkip = edge.kind === 'skip' || edge.kind.startsWith('detect-');
            const isIntra = edge.kind === 'intra';
            const isFanin = edge.kind === 'fanin';
            const stroke = focused ? ACCENT : (isSkip ? ACCENT : (isFanin ? '#7c3aed' : '#94a3b8'));
            const opacity = focusId != null && !focused
              ? (isIntra || isFanin ? 0.1 : 0.15)
              : (isIntra ? 0.35 : (isFanin ? 0.55 : (isSkip ? 0.85 : 0.55)));
            const sw = focused ? 2.2 : (isIntra ? 1 : (isFanin ? 1.2 : (isSkip ? 1.6 : 1)));
            const dash = isFanin ? '4 3' : undefined;
            const marker = isIntra
              ? 'url(#arrow-intra-l2)'
              : isFanin
                ? 'url(#arrow-fanin-l2)'
                : (focused ? 'url(#arrow-accent-l2)' : (focusId != null ? 'url(#arrow-dim-l2)' : (isSkip ? 'url(#arrow-accent-l2)' : 'url(#arrow-gray-l2)')));

            return (
              <path
                key={`e-${i}`}
                d={path}
                fill="none"
                stroke={stroke}
                strokeWidth={sw}
                strokeDasharray={dash}
                opacity={opacity}
                markerEnd={marker}
                style={{
                  transition: 'stroke 200ms, opacity 200ms, stroke-width 200ms',
                }}
              />
            );
          })}

          {/* Container rects + sub-nodes */}
          {arch_l2.map((b, ci) => {
            const c = containers[b.idx];
            if (!c) return null;
            const dimmed = isNodeDimmed(String(b.idx));
            const isSelected = String(selectedId) === String(b.idx);
            const isPlaying = String(playingId) === String(b.idx);
            const typeColor = getTypeColor(b.type);

            return (
              <g key={`c-${b.idx}`}>
                {/* Container rect */}
                <g
                  data-node={String(b.idx)}
                  style={{ cursor: 'pointer', opacity: dimmed ? 0.3 : 1, transition: 'opacity 300ms' }}
                  onClick={(e) => { e.stopPropagation(); onSelect(String(b.idx)); }}
                  onMouseEnter={() => onHover(String(b.idx))}
                  onMouseLeave={() => onHover(null)}
                >
                  <rect
                    x={c.x} y={c.y} width={c.w} height={c.h}
                    rx="8"
                    fill={typeColor.fill}
                    fillOpacity="0.35"
                    stroke={isSelected || isPlaying ? ACCENT : typeColor.border}
                    strokeWidth={isSelected || isPlaying ? 2 : 1.2}
                    style={{
                      filter: isSelected || isPlaying ? 'url(#node-glow-l2)' : 'none',
                    }}
                  />
                  {/* Label band — parent idx + type, plus output shape if known */}
                  <text
                    x={c.x + 8}
                    y={c.y + 18}
                    fontSize="11"
                    fontWeight="600"
                    fill={typeColor.text}
                    fontFamily="ui-monospace, SFMono-Regular, monospace"
                  >
                    {c.label}
                  </text>
                  {(() => {
                    const pd = window.YV_DATA?.blocks?.[String(b.idx)];
                    const sh = pd?.shape;
                    if (!Array.isArray(sh)) return null;
                    return (
                      <text
                        x={c.x + c.w - 8}
                        y={c.y + 18}
                        fontSize="9"
                        fill={typeColor.text}
                        textAnchor="end"
                        fontFamily="ui-monospace, SFMono-Regular, monospace"
                        opacity="0.7"
                      >
                        {sh.join('×')}
                      </text>
                    );
                  })()}
                </g>

                {/* Sub-nodes */}
                {b.sub.map((s, si) => {
                  const sn = subnodes[s.id];
                  if (!sn) return null;
                  const snDimmed = isNodeDimmed(s.id);
                  const snSelected = selectedId === s.id;
                  const snPlaying = playingId === s.id;
                  const snHovered = hoveredId === s.id;
                  const snColor = getTypeColor(sn.type);

                  return (
                    <SubNode
                      key={s.id}
                      id={s.id}
                      node={sn}
                      label={s.label}
                      typeStr={sn.type}
                      colorScheme={snColor}
                      dimmed={snDimmed}
                      selected={snSelected}
                      hovered={snHovered}
                      playing={snPlaying}
                      onSelect={onSelect}
                      onHover={onHover}
                    />
                  );
                })}

                {/* Atomic container: render a single sub-rect if no sub-nodes */}
                {b.sub.length === 0 && (
                  <g
                    data-node={String(b.idx)}
                    style={{ cursor: 'pointer', opacity: dimmed ? 0.3 : 1, transition: 'opacity 300ms' }}
                    onClick={(e) => { e.stopPropagation(); onSelect(String(b.idx)); }}
                    onMouseEnter={() => onHover(String(b.idx))}
                    onMouseLeave={() => onHover(null)}
                  >
                    <text
                      x={c.x + c.w / 2}
                      y={c.y + c.h / 2 + 5}
                      fontSize="12"
                      fontWeight="600"
                      fill={typeColor.text}
                      textAnchor="middle"
                    >
                      {b.type}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Input image marker — rendered OUTSIDE every column bounding box.
             It sits to the left of the backbone's first block. */}
          {containers[0] && (
            <g data-input-marker="true">
              <g transform={`translate(${COL_X_L2[0] - 110}, ${containers[0].y - 4})`}>
                <rect width="80" height="60" rx="6" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1" />
                <text x="40" y="34" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="ui-monospace, monospace">input</text>
                <text x="40" y="46" fontSize="9" fill="#94a3b8" textAnchor="middle" fontFamily="ui-monospace, monospace">(1,3,640,480)</text>
              </g>
              <path
                d={`M ${COL_X_L2[0] - 30} ${containers[0].y + 30} L ${containers[0].x} ${containers[0].y + 30}`}
                stroke="#94a3b8" strokeWidth="1" opacity="0.5"
                markerEnd="url(#arrow-gray-l2)" fill="none"
              />
            </g>
          )}

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

function SubNode({ id, node, label, typeStr, colorScheme, dimmed, selected, hovered, playing, onSelect, onHover }) {
  const { ACCENT } = window.YV;
  const lift = (hovered || playing) ? 1.03 : 1;
  const opacity = dimmed ? 0.3 : 1;
  const compact = !!node.compact;

  // Format shape compactly: [1, 16, 320, 240] -> "1×16×320×240"
  // In compact mode (narrow tiles in fan-in rows), drop the leading "1×" so
  // the channel × spatial dims stay readable.
  const fullShape = Array.isArray(node.shape) ? node.shape.join('×') : (node.shape === 'multi' ? 'multi' : '');
  const compactShape = Array.isArray(node.shape) && node.shape.length === 4
    ? `${node.shape[1]}×${node.shape[2]}×${node.shape[3]}`
    : fullShape;
  const shapeStr = compact ? compactShape : fullShape;

  return (
    <g
      data-node={id}
      transform={`translate(${node.x}, ${node.y})`}
      style={{ cursor: 'pointer', opacity, transition: 'opacity 300ms' }}
      onClick={(e) => { e.stopPropagation(); onSelect(id); }}
      onMouseEnter={() => onHover(id)}
      onMouseLeave={() => onHover(null)}
    >
      <g
        style={{
          transformOrigin: `${node.w / 2}px ${node.h / 2}px`,
          transform: `scale(${lift})`,
          transition: 'transform 160ms cubic-bezier(.4,0,.2,1)',
          filter: selected || playing ? 'url(#node-glow-l2)' : (hovered ? 'url(#node-shadow-l2)' : 'none'),
        }}
      >
        <rect
          width={node.w}
          height={node.h}
          rx="5"
          fill={colorScheme.fill}
          stroke={selected || playing ? ACCENT : colorScheme.border}
          strokeWidth={selected || playing ? 2 : 1}
        />
        {/* Primary label = friendly type/path name */}
        <text
          x={node.w / 2}
          y={shapeStr ? node.h / 2 - 3 : node.h / 2 + 4}
          fontSize="11"
          fontWeight="600"
          fill={colorScheme.text}
          textAnchor="middle"
          fontFamily="ui-monospace, SFMono-Regular, monospace"
        >
          {label}
        </text>
        {/* Secondary line = output tensor shape (the L1-style annotation) */}
        {shapeStr && (
          <text
            x={node.w / 2}
            y={node.h / 2 + 11}
            fontSize="9"
            fill={colorScheme.text}
            textAnchor="middle"
            fontFamily="ui-monospace, SFMono-Regular, monospace"
            opacity="0.7"
          >
            {shapeStr}
          </text>
        )}
      </g>
    </g>
  );
}

window.YV = window.YV || {};
window.YV.GraphL2 = GraphL2;
