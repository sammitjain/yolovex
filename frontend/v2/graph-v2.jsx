// yolovex v2 — SVG graph: role containers, edges, block nodes. Pan / zoom /
// hover interactivity ported from the original frontend/graph.jsx. Clicking a
// block toggles in-place expansion (expand-v2.jsx builds the internal graph;
// layout-v2.jsx slides neighbours aside to make room).

const { useState, useRef, useEffect, useMemo, useCallback } = React;

function formatShape(sh) {
  if (Array.isArray(sh) && sh.length && typeof sh[0] === 'number') {
    return `(${sh.join(', ')})`;
  }
  return null;
}

function GraphV2({ selected, hover, playing, onSelect, onHover, onExpandedCountChange, onVisibleOrderChange, settingsRev = 0, theme = 'light', onToggleTheme }) {
  // L1-block id derived from the unified hover/select payload; the styling
  // logic below (edge dimming, lift, selected glow) is L1-only, so sub-node
  // hovers still focus their parent block.
  const hoveredIdx = hover ? hover.idx : null;
  const selectedIdx = selected ? selected.idx : null;
  const playingIdx = playing ? playing.idx : null;
  const playingFx = (playing && playing.pathKey != null) ? (playing.fxKey || null) : null;
  const V = window.YVV2;
  const arch = useMemo(() => V.buildArch(), []);
  const rawEdges = useMemo(() => V.buildEdges(), []);

  // Expansion state — any number of blocks can be expanded in place at once.
  // Within each expanded block, sub-nodes can be RECURSIVELY expanded too;
  // `subExpandedMap` holds the set of sub-path keys peeled open per block.
  const [expandedSet, setExpandedSet] = useState(() => new Set());
  const [subExpandedMap, setSubExpandedMap] = useState(() => new Map());

  const toggleExpand = useCallback((idx) => {
    setExpandedSet(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
    // Collapsing the top-level block also forgets its inner expansions.
    setSubExpandedMap(prev => {
      if (!prev.has(idx)) return prev;
      const next = new Map(prev);
      next.delete(idx);
      return next;
    });
  }, []);

  const toggleSubExpand = useCallback((blockIdx, pathKey) => {
    setSubExpandedMap(prev => {
      const next = new Map(prev);
      const set = new Set(next.get(blockIdx) || []);
      if (set.has(pathKey)) set.delete(pathKey); else set.add(pathKey);
      if (set.size === 0) next.delete(blockIdx);
      else next.set(blockIdx, set);
      return next;
    });
  }, []);

  const collapseAll = useCallback(() => {
    setExpandedSet(new Set());
    setSubExpandedMap(new Map());
  }, []);

  // Build the laid-out internal sub-graph for each expanded block. col-1 blocks
  // (FPN-up) flip so their internal flow runs bottom-to-top like the column.
  const expansionMap = useMemo(() => {
    const m = {};
    for (const idx of expandedSet) {
      const b = arch[idx];
      if (!b) continue;
      const ex = V.buildExpansion(idx, {
        flip: b.col === 1,
        expansions: subExpandedMap.get(idx),
      });
      if (ex) m[idx] = ex;
    }
    return m;
  }, [expandedSet, subExpandedMap, arch, V]);

  // Blocks that genuinely expanded (Detect / non-fx specs yield no expansion
  // even if clicked, so count the realised regions, not the raw click set).
  const expandedCount = Object.keys(expansionMap).length;

  const { nodes, totalW, totalH } = useMemo(() => V.layoutGraph(arch, expansionMap), [arch, expansionMap, settingsRev]);
  const edgeMeta = useMemo(() => V.buildEdgeMeta(rawEdges, arch), [rawEdges, arch]);
  const containers = useMemo(() => V.computeContainers(arch, nodes), [arch, nodes, settingsRev]);
  const skipSources = useMemo(() => {
    const s = new Set();
    rawEdges.forEach(e => { if (e.is_skip) s.add(e.src); });
    return s;
  }, [rawEdges]);

  // Surface the expanded count to the parent (for the header hint).
  useEffect(() => {
    if (onExpandedCountChange) onExpandedCountChange(expandedCount);
  }, [expandedCount, onExpandedCountChange]);

  // Surface the visible play-flow order to the parent — dataflow order over
  // exactly the blocks/sub-nodes that are currently rendered on the canvas.
  // Each entry is a payload identical to what onHover/onSelect dispatch.
  const visibleOrder = useMemo(() => {
    const order = [];
    for (const b of arch) {
      const ex = expansionMap[b.idx];
      if (ex && ex.subNodes && ex.subNodes.length) {
        for (const sn of ex.subNodes) {
          const members = sn.members || [sn.id];
          const fxKey = members[members.length - 1];
          const firstFxKey = members[0];
          order.push({
            idx: b.idx,
            pathKey: sn.pathKey,
            fxKey,
            firstFxKey,
            members,
            subkind: sn.subkind || 'sub',
          });
        }
      } else {
        order.push({ idx: b.idx, pathKey: null });
      }
    }
    return order;
  }, [arch, expansionMap]);

  useEffect(() => {
    if (onVisibleOrderChange) onVisibleOrderChange(visibleOrder);
  }, [visibleOrder, onVisibleOrderChange]);

  const { TYPE_COLORS, ROLE_COLORS } = V;
  const LS = window.YVV2.LAYOUT_SETTINGS;
  const ACCENT = LS.ACCENT_COLOR;
  const EDGE_DEFAULT = LS.EDGE_COLOR_DEFAULT;
  const EDGE_DIMMED = LS.EDGE_COLOR_DIMMED;
  const EDGE_FOCUSED = LS.EDGE_COLOR_FOCUSED;
  const SW_DEFAULT = LS.EDGE_STROKE_DEFAULT;
  const SW_FOCUSED = LS.EDGE_STROKE_FOCUSED;
  const CONTAINER_DASH = LS.CONTAINER_DASH;

  const containerRef = useRef(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 });
  const [containerSize, setContainerSize] = useState({ w: 1000, h: 720 });
  const [didFit, setDidFit] = useState(false);
  const draggingRef = useRef(null);

  // Track container size.
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

  // Fit to screen — fit horizontally, cap zoom at 1, anchor near top.
  const fit = useCallback(() => {
    const padX = 32;
    const k = Math.min((containerSize.w - padX * 2) / totalW, 1);
    const x = (containerSize.w - totalW * k) / 2;
    setTransform({ x, y: 24, k });
  }, [containerSize, totalW]);

  useEffect(() => {
    if (!didFit && containerSize.w > 100) {
      fit();
      setDidFit(true);
    }
  }, [containerSize, didFit, fit]);

  // Wheel zoom, centred on the cursor.
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setTransform(t => {
      const factor = Math.exp(-e.deltaY * 0.0015);
      const newK = Math.max(0.3, Math.min(2.5, t.k * factor));
      return {
        x: mx - (mx - t.x) * (newK / t.k),
        y: my - (my - t.y) * (newK / t.k),
        k: newK,
      };
    });
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [onWheel]);

  // Drag pan (on background only).
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
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  // playing takes priority for focus (so edges to/from the active step pop),
  // then hover, then selection.
  const focusIdx = playingIdx ?? hoveredIdx ?? selectedIdx;
  const isEdgeFocused = (e) => focusIdx != null && (e.src === focusIdx || e.dst === focusIdx);
  const connectedTo = (idx, other) =>
    edgeMeta.some(e => (e.src === idx && e.dst === other) || (e.dst === idx && e.src === other));

  return (
    <div
      ref={containerRef}
      className="graph-container"
      onMouseDown={onMouseDown}
      style={{ cursor: draggingRef.current ? 'grabbing' : 'grab' }}
    >
      <svg width={containerSize.w} height={containerSize.h} style={{ display: 'block' }}>
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
          {/* Role containers — Backbone/Head rects, Neck mirrored-L polygon. */}
          {containers.map(c => (
            <g key={c.role}>
              {c.kind === 'rect' ? (
                <rect
                  x={c.x} y={c.y} width={c.w} height={c.h}
                  fill="none" stroke={ROLE_COLORS[c.role]} strokeWidth="1.2"
                  strokeDasharray={CONTAINER_DASH} rx="14" opacity="0.55"
                />
              ) : (
                <path
                  d={c.path}
                  fill="none" stroke={ROLE_COLORS[c.role]} strokeWidth="1.2"
                  strokeDasharray={CONTAINER_DASH} opacity="0.55"
                />
              )}
              <text
                x={c.labelX} y={c.labelY}
                fontSize="12" fontWeight="600" fill={ROLE_COLORS[c.role]}
                style={{ letterSpacing: '0.06em', textTransform: 'uppercase' }}
              >
                {c.role}
              </text>
            </g>
          ))}

          {/* Edges — one L1 edge may render as multiple paths when the source
              or target is expanded with multi-exit / multi-entry internal
              nodes (e.g. SPPF's placeholder feeds both cv1 and the residual). */}
          {edgeMeta.flatMap((e, i) => {
            const paths = V.edgePaths(e, nodes, arch);
            const focused = isEdgeFocused(e);
            const isAccent = e.kind === 'skip' || e.kind === 'detect';
            const stroke = focused ? EDGE_FOCUSED : (isAccent ? ACCENT : EDGE_DEFAULT);
            const opacity = focusIdx != null && !focused ? 0.18 : (isAccent ? 0.85 : 0.55);
            const sw = focused ? SW_FOCUSED : (isAccent ? SW_DEFAULT * 1.25 : SW_DEFAULT);
            const marker = focused
              ? 'url(#arrow-accent)'
              : (focusIdx != null ? 'url(#arrow-dim)' : (isAccent ? 'url(#arrow-accent)' : 'url(#arrow-gray)'));
            return paths.map((d, j) => (
              <path
                key={`${i}-${j}`}
                d={d}
                fill="none"
                stroke={stroke}
                strokeWidth={sw}
                opacity={opacity}
                markerEnd={marker}
                style={{ transition: 'stroke 200ms, opacity 200ms, stroke-width 200ms' }}
              />
            ));
          })}

          {/* Nodes */}
          {arch.map(b => (
            <NodeV2
              key={b.idx}
              block={b}
              node={nodes[b.idx]}
              hovered={hoveredIdx === b.idx}
              selected={selectedIdx === b.idx}
              playing={playingIdx === b.idx}
              playingFx={playingIdx === b.idx ? playingFx : null}
              dimmed={focusIdx != null && focusIdx !== b.idx && !connectedTo(focusIdx, b.idx)}
              isSkipSource={skipSources.has(b.idx)}
              colorScheme={TYPE_COLORS[b.type] || TYPE_COLORS.Conv}
              onHover={onHover}
              onSelect={onSelect}
              onToggleExpand={toggleExpand}
              onToggleSubExpand={toggleSubExpand}
              accent={ACCENT}
            />
          ))}

          {/* Input image marker (left of the backbone). */}
          <g transform={`translate(${nodes[0].x - 110}, ${nodes[0].y - 4})`}>
            <rect width="80" height="60" rx="6" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1" />
            <text x="40" y="34" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="ui-monospace, monospace">input</text>
            <text x="40" y="46" fontSize="9" fill="#94a3b8" textAnchor="middle" fontFamily="ui-monospace, monospace">(1,3,640,480)</text>
          </g>
          <path
            d={`M ${nodes[0].x - 30} ${nodes[0].y + 26} L ${nodes[0].x} ${nodes[0].y + 26}`}
            stroke="#94a3b8" strokeWidth="1" opacity="0.5" fill="none"
            markerEnd="url(#arrow-gray)"
          />
        </g>
      </svg>

      {/* Zoom controls */}
      <div className="zoom-controls">
        <button onClick={() => setTransform(t => ({ ...t, k: Math.min(2.5, t.k * 1.2) }))} title="Zoom in">+</button>
        <button onClick={() => setTransform(t => ({ ...t, k: Math.max(0.3, t.k / 1.2) }))} title="Zoom out">−</button>
        <button onClick={fit} title="Fit to screen" className="fit-btn">⤢</button>
        {expandedCount > 0 && (
          <button onClick={collapseAll} title="Collapse all expanded blocks" className="fit-btn">⊟</button>
        )}
        {onToggleTheme && (
          <button onClick={onToggleTheme} className="theme-btn"
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        )}
      </div>
    </div>
  );
}

function NodeV2({ block, node, hovered, selected, playing, playingFx, dimmed, isSkipSource, colorScheme, onHover, onSelect, onToggleExpand, onToggleSubExpand, accent }) {
  const lift = (playing && !node.expanded) ? 1.05 : (hovered ? 1.02 : 1);
  const opacity = dimmed ? 0.35 : 1;
  const isDetect = block.type === 'Detect';
  const shape = formatShape(block.outputShape);
  const expanded = !!node.expanded;

  return (
    <g
      data-node={block.idx}
      transform={`translate(${node.x}, ${node.y})`}
      style={{ cursor: 'pointer', opacity, transition: 'opacity 200ms' }}
      onMouseEnter={() => onHover({ idx: block.idx, pathKey: null })}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => {
        // Shift+click expands/collapses; bare left-click opens the activation
        // panel. Sub-nodes / containers stopPropagation, so bare clicks on the
        // region background still hit this handler.
        e.stopPropagation();
        if (e.shiftKey) onToggleExpand(block.idx);
        else onSelect({ idx: block.idx, pathKey: null });
      }}
    >
      {expanded ? (
        <ExpandedNodeV2
          block={block}
          node={node}
          colorScheme={colorScheme}
          accent={accent}
          playingFx={playingFx}
          onHover={onHover}
          onSelect={onSelect}
          onToggleSubExpand={onToggleSubExpand}
        />
      ) : (
        <g
          style={{
            transformOrigin: `${node.w / 2}px ${node.h / 2}px`,
            transform: `scale(${lift})`,
            transition: 'transform 180ms cubic-bezier(.4,0,.2,1)',
            filter: (playing || selected) ? 'url(#node-glow)' : (hovered ? 'url(#node-shadow)' : 'none'),
          }}
        >
          {isDetect ? (
            <DetectNodeV2 node={node} colorScheme={colorScheme} highlight={selected || hovered} accent={accent} />
          ) : (
            <>
              <rect
                width={node.w} height={node.h} rx="8"
                fill={colorScheme.fill}
                stroke={(selected || playing) ? accent : colorScheme.border}
                strokeWidth={(selected || playing) ? 2 : 1}
              />
              <text x="12" y="22" fontSize="11" fontWeight="500" fill={colorScheme.text}
                fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.7">
                [{block.idx}]
              </text>
              <text x="38" y="22" fontSize="13" fontWeight="600" fill={colorScheme.text}>
                {block.type}
              </text>
              {shape && (
                <text x="12" y="44" fontSize="10.5" fill={colorScheme.text}
                  fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.85">
                  {shape}
                </text>
              )}
              {isSkipSource && (
                <circle cx={node.w - 10} cy={10} r={3.5} fill={accent} opacity="0.85">
                  <title>This block is a skip-connection source</title>
                </circle>
              )}
            </>
          )}
        </g>
      )}
    </g>
  );
}

// Expanded block — the region box (block label up top) with its internal
// component sub-graph rendered inside. All sub-node / sub-edge coords are
// LOCAL to the region, so they sit inside the node's translate(node.x,node.y).
// Collect fx node names whose visible_path begins with `containerPath`, in
// graph order. The container's "output" is the last such node; the "input
// boundary" is the first one. Used by hover / click on an inner container
// rect so it reports the container's OWN activation — i.e. the tensor that
// would be produced if you hadn't peeled it open.
function fxMembersInContainer(specId, containerPath) {
  if (!specId || !containerPath || !containerPath.length) return [];
  const spec = window.YV_SPEC?.specs?.[specId];
  if (!spec) return [];
  const out = [];
  for (const n of spec.graph.nodes) {
    if (n.op === 'placeholder' || n.op === 'output' || n.op === 'get_attr') continue;
    const vp = n.visible_path || n.path || [];
    if (vp.length < containerPath.length) continue;
    let match = true;
    for (let i = 0; i < containerPath.length; i++) {
      if (vp[i] !== containerPath[i]) { match = false; break; }
    }
    if (match) out.push(n.name);
  }
  return out;
}

function ExpandedNodeV2({ block, node, colorScheme, accent, playingFx, onHover, onSelect, onToggleSubExpand }) {
  const { SUB_KIND_COLORS, subFormatShape } = window.YVV2;
  const region = node.region;
  return (
    <g>
      {/* Region container */}
      <rect
        width={node.w} height={node.h} rx="10"
        fill={colorScheme.fill} fillOpacity="0.35"
        stroke={accent} strokeWidth="1.5" strokeDasharray="5 3"
      />
      <text x="12" y="19" fontSize="11" fontWeight="500" fill={colorScheme.text}
        fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.7">
        [{block.idx}]
      </text>
      <text x="38" y="19" fontSize="12.5" fontWeight="600" fill={colorScheme.text}>
        {block.type}
      </text>

      {/* Inner containers — drawn FIRST so edges + nodes paint on top.
          Hovering / clicking the rect surfaces the container's OWN activation
          (the output tensor of its last fx node — same as before this group
          was peeled open). The ▴ label is a dedicated collapse affordance. */}
      {(region.innerContainers || []).map(ic => {
        const members = fxMembersInContainer(block.specId, ic.path);
        const fxKey = members.length ? members[members.length - 1] : null;
        const firstFxKey = members.length ? members[0] : null;
        const payload = { idx: block.idx, pathKey: ic.pathKey, fxKey, firstFxKey, members, subkind: 'container' };
        const onEnter = (e) => { e.stopPropagation(); onHover && onHover(payload); };
        const onLeave = (e) => { e.stopPropagation(); onHover && onHover(null); };
        const onClickRect = (e) => {
          e.stopPropagation();
          if (e.shiftKey) onToggleSubExpand(block.idx, ic.pathKey);
          else onSelect && onSelect(payload);
        };
        return (
          <g key={`ic-${ic.pathKey}`}>
            <rect
              x={ic.x} y={ic.y} width={ic.w} height={ic.h} rx="8"
              fill="#eef2ff" fillOpacity="0.45"
              stroke="#a5b4fc" strokeWidth="1" strokeDasharray="4 3"
              style={{ cursor: 'pointer' }}
              onMouseEnter={onEnter}
              onMouseLeave={onLeave}
              onClick={onClickRect}
            />
            <text
              x={ic.x + ic.w - 10} y={ic.y + 14}
              fontSize="10.5" fontWeight="600" fill="#3730a3"
              fontFamily="ui-monospace, monospace" textAnchor="end"
              style={{ cursor: 'pointer' }}
              onClick={(e) => { e.stopPropagation(); onToggleSubExpand(block.idx, ic.pathKey); }}
            >
              {ic.label} ▴
            </text>
          </g>
        );
      })}

      {/* Internal sub-edges */}
      {region.subEdges.map((e, i) => (
        <g key={`se-${i}`}>
          <path
            d={e.path}
            fill="none"
            stroke={e.accent ? accent : '#94a3b8'}
            strokeWidth={e.accent ? 1.6 : 1.4}
            markerEnd={e.accent ? 'url(#arrow-accent)' : 'url(#arrow-gray)'}
          />
          {e.label && e.labelPos && (
            <text x={e.labelPos.x} y={e.labelPos.y} fontSize="10"
              fontFamily="ui-monospace, monospace" fill="#475569">
              [{e.label}]
            </text>
          )}
        </g>
      ))}

      {/* Internal sub-nodes */}
      {region.subNodes.map(sn => {
        // Sub-node is the play-flow's current step if its last fx member or
        // its pathKey matches the playing payload.
        const members = sn.members || [sn.id];
        const isPlaying = !!playingFx && (
          playingFx === members[members.length - 1] ||
          playingFx === members[0] ||
          members.includes(playingFx)
        );
        return (
          <SubNodeV2
            key={sn.id}
            sn={sn}
            blockIdx={block.idx}
            playing={isPlaying}
            accent={accent}
            onHover={onHover}
            onSelect={onSelect}
            onToggleSubExpand={onToggleSubExpand}
            SUB_KIND_COLORS={SUB_KIND_COLORS}
            subFormatShape={subFormatShape}
          />
        );
      })}
    </g>
  );
}

function SubNodeV2({ sn, blockIdx, playing, accent, onHover, onSelect, onToggleSubExpand, SUB_KIND_COLORS, subFormatShape }) {
  const sk = sn.subkind;
  const expandable = sn.expandable;
  // Lookup key for activations: the last fx-graph member of this group. For a
  // fully-revealed leaf, members[0] === sn.id. For an aggregated group the
  // last member's output IS the group's output tensor (matches the shape
  // already shown on the node).
  const members = sn.members || [sn.id];
  const fxKey = members[members.length - 1];
  const firstFxKey = members[0];
  const hoverPayload = { idx: blockIdx, pathKey: sn.pathKey, fxKey, firstFxKey, members, subkind: sk };
  const onEnter = onHover ? (e) => { e.stopPropagation(); onHover(hoverPayload); } : undefined;
  const onLeave = onHover ? (e) => { e.stopPropagation(); onHover(null); } : undefined;
  const handleClick = (e) => {
    e.stopPropagation();
    if (e.shiftKey && expandable) {
      onToggleSubExpand(blockIdx, sn.pathKey);
    } else if (onSelect) {
      onSelect(hoverPayload);
    }
  };
  const cursor = 'pointer';

  if (sk === 'arith') {
    const cx = sn.x + sn.w / 2, cy = sn.y + sn.h / 2;
    return (
      <g style={{ cursor }} onMouseEnter={onEnter} onMouseLeave={onLeave} onClick={handleClick}>
        <circle cx={cx} cy={cy} r={sn.w / 2} fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.5" />
        <text x={cx} y={cy + 5} fontSize="15" fontWeight="700" fill="#78350f" textAnchor="middle">
          {({ add: '+', mul: '×', sub: '−', truediv: '÷' }[sn.label] || sn.label.replace(/^fn:/, '') || '·')}
        </text>
      </g>
    );
  }

  if (sk === 'attr' || sk === 'shape' || sk === 'struct') {
    const fill   = sk === 'attr' ? '#ede9fe' : '#f1f5f9';
    const stroke = sk === 'attr' ? '#a78bfa' : '#94a3b8';
    const text   = sk === 'attr' ? '#4c1d95' : '#475569';
    return (
      <g style={{ cursor }} onMouseEnter={onEnter} onMouseLeave={onLeave} onClick={handleClick}>
        <rect x={sn.x} y={sn.y} width={sn.w} height={sn.h} rx="4"
          fill={fill} stroke={stroke} strokeWidth="1" strokeDasharray="3 2" />
        <text x={sn.x + sn.w / 2} y={sn.y + sn.h / 2 + 4} fontSize="10.5" fill={text}
          textAnchor="middle" fontFamily="ui-monospace, monospace">
          {sn.label}
        </text>
      </g>
    );
  }

  const c = SUB_KIND_COLORS[sk] || SUB_KIND_COLORS.mod;
  const sh = subFormatShape(sn.shape);
  return (
    <g style={{ cursor, filter: playing ? 'url(#node-glow)' : undefined }} onMouseEnter={onEnter} onMouseLeave={onLeave} onClick={handleClick}>
      <rect x={sn.x} y={sn.y} width={sn.w} height={sn.h} rx="6"
        fill={c.fill} stroke={playing ? accent : c.border} strokeWidth={playing ? 2.5 : (expandable ? 2 : 1.5)}
        strokeDasharray={expandable ? '6 3' : undefined} />
      <text x={sn.x + sn.w / 2} y={sn.y + sn.h / 2 + (sh ? -2 : 4)} fontSize="12" fontWeight="600"
        fill={c.text} textAnchor="middle">
        {sn.label}
      </text>
      {sh && (
        <text x={sn.x + sn.w / 2} y={sn.y + sn.h - 8} fontSize="9.5" fill={c.text}
          opacity="0.65" textAnchor="middle" fontFamily="ui-monospace, monospace">
          {sh}
        </text>
      )}
      {/* Expandable indicator — small chevron in the top-right corner. Hint that shift+click peels open. */}
      {expandable && (
        <text x={sn.x + sn.w - 8} y={sn.y + 12} fontSize="11" fontWeight="700"
          fill={c.text} opacity="0.75" textAnchor="end">
          ▾
        </text>
      )}
    </g>
  );
}

// Detect head — three separate, normal-sized boxes (P3 / P4 / P5), not one
// stretched node. Positions come from node.detect[i].relY (set in layout-v2).
function DetectNodeV2({ node, colorScheme, highlight, accent }) {
  const NH = window.YVV2.NODE_H;
  return (
    <g>
      <text x="0" y="-12" fontSize="11" fontWeight="500" fill={colorScheme.text}
        fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.8">
        [23] Detect
      </text>
      {(node.detect || []).map(box => (
        <g key={box.scale} transform={`translate(0, ${box.relY})`}>
          <rect
            width={node.w} height={NH} rx="8"
            fill={colorScheme.fill}
            stroke={highlight ? accent : colorScheme.border}
            strokeWidth={highlight ? 2 : 1}
          />
          <text x="14" y="23" fontSize="13" fontWeight="600" fill={colorScheme.text}>
            {box.label}
          </text>
          <text x="14" y="42" fontSize="10.5" fill={colorScheme.text}
            fontFamily="ui-monospace, SFMono-Regular, monospace" opacity="0.8">
            {box.size} objects
          </text>
        </g>
      ))}
    </g>
  );
}

window.YVV2 = window.YVV2 || {};
window.YVV2.GraphV2 = GraphV2;
