// Inside-the-block diagrams. One component per block type, rendering an SVG
// of the internal data-flow with the block's actual parameters from data.js.
//
// Layout: VERTICAL (top to bottom). Each SVG has a per-diagram `maxW` so
// simpler flows don't expand to fill the panel — the diagram stays roughly the
// same visual size whatever the panel width.

const Internals = (() => {

// Common SVG primitives ------------------------------------------------------

const ARROW_DEFS = (
  <defs>
    <marker id="ar-int" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
    </marker>
    <marker id="ar-int-acc" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#fb923c" />
    </marker>
  </defs>
);

function Box({ x, y, w, h, fill, stroke, label, sub, mono = false, text = '#1f2328' }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="6" fill={fill} stroke={stroke} strokeWidth="1.2" />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -3 : 4)} textAnchor="middle"
            fontSize={mono ? 11 : 12.5} fontWeight="600" fill={text}
            fontFamily={mono ? "'IBM Plex Mono', monospace" : "inherit"}>
        {label}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle"
              fontSize="10" fill="#64748b" fontFamily="'IBM Plex Mono', monospace">
          {sub}
        </text>
      )}
    </g>
  );
}

function Edge({ d, accent = false, dashed = false }) {
  return (
    <path d={d} fill="none" stroke={accent ? '#fb923c' : '#94a3b8'} strokeWidth="1.5"
          strokeDasharray={dashed ? '4 3' : null}
          markerEnd={accent ? 'url(#ar-int-acc)' : 'url(#ar-int)'} />
  );
}

function Annot({ x, y, text, anchor = 'start', size = 10 }) {
  return (
    <text x={x} y={y} fontSize={size} fill="#64748b" textAnchor={anchor}
          fontFamily="'IBM Plex Mono', monospace">
      {text}
    </text>
  );
}

// Per-diagram width cap (CSS max-width on the SVG element).
function svgStyle(maxW) {
  return { maxWidth: maxW, width: '100%', height: 'auto', display: 'block', margin: '0 auto' };
}

// Theme colors
const C = {
  conv: { fill: '#dcfce7', stroke: '#86efac', text: '#14532d' },
  norm: { fill: '#e0f2fe', stroke: '#7dd3fc', text: '#0c4a6e' },
  act:  { fill: '#fef3c7', stroke: '#fcd34d', text: '#78350f' },
  pool: { fill: '#fce7f3', stroke: '#f9a8d4', text: '#831843' },
  attn: { fill: '#ede9fe', stroke: '#c4b5fd', text: '#4c1d95' },
  io:   { fill: '#f1f5f9', stroke: '#cbd5e1', text: '#334155' },
  merge:{ fill: '#fff', stroke: '#cbd5e1', text: '#334155' },
  dfl:  { fill: '#ffe4e6', stroke: '#fda4af', text: '#881337' },
  final:{ fill: '#bbf7d0', stroke: '#22c55e', text: '#14532d' },
};

// =============================================================================
// Conv: input → Conv2d → BatchNorm → SiLU → output
// =============================================================================

function ConvInternals({ block, real }) {
  const i = real?.internals || {};
  const k = i.kernel_size?.[0] ?? '?';
  const s = i.stride?.[0] ?? '?';
  const cIn = i.in_channels ?? '?';
  const cOut = i.out_channels ?? '?';
  const downsamples = s === 2;

  return (
    <div className="block-internals">
      <svg viewBox="0 0 320 470" style={svgStyle(280)}>
        {ARROW_DEFS}
        <Box x={70} y={15}  w={180} h={48} {...C.io}   label="input"      sub={`c=${cIn}`} />
        <Box x={70} y={100} w={180} h={56} {...C.conv} label={`Conv2d ${k}×${k}`} sub={`stride ${s} · ${cIn}→${cOut}`} />
        <Box x={70} y={200} w={180} h={48} {...C.norm} label="BatchNorm"  sub="rescale per channel" />
        <Box x={70} y={290} w={180} h={48} {...C.act}  label="SiLU"       sub="x · σ(x)" />
        <Box x={70} y={390} w={180} h={48} {...C.io}   label="output"     sub={`c=${cOut}`} />

        <Edge d="M 160  63 L 160 100" />
        <Edge d="M 160 156 L 160 200" />
        <Edge d="M 160 248 L 160 290" />
        <Edge d="M 160 338 L 160 390" />
      </svg>
      <div className="internals-caption">
        <p>
          A <strong>convolution</strong> slides a small <code>{k}×{k}</code> window
          across the feature map. At each position it multiplies the window's
          values by learned weights and sums them, producing one output value.
          Repeating this for every output channel gives the next feature map.
        </p>
        <p>
          With <strong>stride={s}</strong>, the window steps {s} pixel{s===1?'':'s'} at a
          time. {downsamples
            ? <>Stride 2 means the output is <em>half</em> the height and width of the input — a downsampling step. </>
            : <>Stride 1 keeps spatial dimensions the same. </>}
          This block goes from <code>{cIn}</code> input channels to <code>{cOut}</code> output channels.
        </p>
        <p>
          <strong>BatchNorm</strong> rescales each output channel so its values sit
          in a stable range (zero mean, unit variance across the batch). It
          mostly exists to make training faster and more reliable.
        </p>
        <p>
          <strong>SiLU</strong> (also called Swish) is the <strong>activation function</strong>:{' '}
          <code>x · σ(x)</code> — a smoother cousin of ReLU. It passes positive
          values through and gently dampens negative ones. Without an activation
          function, stacking convolutions would just be one big linear transform;
          SiLU adds the nonlinearity that lets the network learn complex patterns.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// C3k2: split → two parallel streams → concat → fuse
// =============================================================================

function C3k2Internals({ block, real }) {
  const i = real?.internals || {};
  const n = i.n ?? '?';
  const sub = i.sub_block || 'Bottleneck';
  const cIn = i.in_channels ?? '?';
  const cOut = i.out_channels ?? '?';

  return (
    <div className="block-internals">
      <svg viewBox="0 0 360 560" style={svgStyle(330)}>
        {ARROW_DEFS}
        <Box x={90}  y={15}  w={180} h={48} {...C.io}   label="input"          sub={`c=${cIn}`} />
        <Box x={90}  y={100} w={180} h={56} {...C.conv} label="1×1 conv"       sub="split into 2 streams" />

        <Box x={20}  y={210} w={130} h={56} {...C.io}   label="passthrough"    sub="(no transform)" />
        <Box x={210} y={210} w={130} h={56} {...C.attn} label={`${sub} × ${n}`} sub="residual mix" />

        <Box x={90}  y={310} w={180} h={48} {...C.merge} label="concat" sub="along channel axis" />
        <Box x={90}  y={395} w={180} h={56} {...C.conv}  label="1×1 conv" sub={`fuse → c=${cOut}`} />
        <Box x={90}  y={495} w={180} h={48} {...C.io}    label="output" sub={`c=${cOut}`} />

        <Edge d="M 180  63 L 180 100" />
        <Edge d="M 180 156 C 180 180,  85 180,  85 210" />
        <Edge d="M 180 156 C 180 180, 275 180, 275 210" />
        <Edge d="M  85 266 C  85 295, 180 295, 180 310" />
        <Edge d="M 275 266 C 275 295, 180 295, 180 310" />
        <Edge d="M 180 358 L 180 395" />
        <Edge d="M 180 451 L 180 495" />
      </svg>
      <div className="internals-caption">
        <p>
          <strong>C3k2</strong> stands for "Cross-Stage Partial bottleneck, k=2."
          It's the main feature-mixing unit in YOLO's backbone and neck, and it
          works in three steps:
        </p>
        <ol>
          <li>
            <strong>Split.</strong> A <code>1×1 convolution</code> (a per-pixel
            channel mixer — like a small fully-connected layer applied
            independently at every spatial location) takes the input's{' '}
            <code>{cIn}</code> channels and routes them into two streams. One
            half will be heavily processed; the other will pass through unchanged.
          </li>
          <li>
            <strong>Process.</strong> One stream goes through {n}{' '}
            <code>{sub}</code> block{n === 1 ? '' : 's'}. Each one is a small
            residual unit (typically two convolutions plus a shortcut connection)
            that learns a refined version of the features at this depth.
          </li>
          <li>
            <strong>Merge.</strong> The unchanged half and the processed half are
            <strong> concatenated</strong> — stacked side-by-side along the
            channel axis. A final <code>1×1 conv</code> mixes them together,
            producing <code>{cOut}</code> output channels.
          </li>
        </ol>
        <p>
          The unchanged "skip half" matters because it preserves the original
          signal alongside the transformed one — the next block can decide how
          much of each to use. This pattern (originally introduced as CSPNet)
          makes the network easier to train and faster to run than processing
          every channel through every sub-block.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// SPPF: cascaded maxpools, 4 taps converge into concat
// =============================================================================

function SPPFInternals({ block, real }) {
  const i = real?.internals || {};
  const k = i.pool_kernel ?? 5;
  const cIn = i.in_channels ?? '?';
  const cOut = i.out_channels ?? '?';

  return (
    <div className="block-internals">
      <svg viewBox="0 0 380 700" style={svgStyle(340)}>
        {ARROW_DEFS}
        <Box x={20}  y={15}  w={200} h={48} {...C.io}   label="input" sub={`c=${cIn}`} />
        <Box x={20}  y={95}  w={200} h={56} {...C.conv} label="1×1 conv" sub="halve channels" />
        <Box x={20}  y={185} w={200} h={56} {...C.pool} label={`maxpool ${k}×${k}`} />
        <Box x={20}  y={275} w={200} h={56} {...C.pool} label={`maxpool ${k}×${k}`} />
        <Box x={20}  y={365} w={200} h={56} {...C.pool} label={`maxpool ${k}×${k}`} />

        <Edge d="M 120  63 L 120  95" />
        <Edge d="M 120 151 L 120 185" />
        <Edge d="M 120 241 L 120 275" />
        <Edge d="M 120 331 L 120 365" />

        <Annot x={230} y={128} text="x₁" />
        <Annot x={230} y={218} text="x₂" />
        <Annot x={230} y={308} text="x₃" />
        <Annot x={230} y={398} text="x₄" />

        <Edge d="M 220 123 C 320 123, 320 465, 160 465" accent={true} />
        <Edge d="M 220 213 C 330 213, 330 465, 180 465" accent={true} />
        <Edge d="M 220 303 C 340 303, 340 465, 200 465" accent={true} />
        <Edge d="M 220 393 C 350 393, 350 465, 220 465" accent={true} />

        <Box x={90}  y={465} w={200} h={48} {...C.merge} label="concat" sub="x₁ + x₂ + x₃ + x₄" />
        <Box x={90}  y={540} w={200} h={56} {...C.conv}  label="1×1 conv" sub={`fuse → c=${cOut}`} />
        <Box x={90}  y={635} w={200} h={48} {...C.io}    label="output" sub={`c=${cOut}`} />

        <Edge d="M 190 513 L 190 540" />
        <Edge d="M 190 596 L 190 635" />
      </svg>
      <div className="internals-caption">
        <p>
          <strong>SPPF</strong> stands for "Spatial Pyramid Pooling — Fast." It
          enriches features with multi-scale spatial context at very little cost.
        </p>
        <p>
          A <strong>max-pool</strong> with a <code>{k}×{k}</code> window slides
          across the feature map and keeps the largest value in each window.
          This means each output cell "sees" a {k}×{k} neighborhood of the input
          — a small region of spatial context.
        </p>
        <p>
          The trick: by stacking three max-pools in sequence, each subsequent
          pool effectively sees an even <em>wider</em> region. Pool 1 sees{' '}
          {k}×{k}, pool 2 sees about {2*k - 1}×{2*k - 1}, pool 3 sees about{' '}
          {3*k - 2}×{3*k - 2} — without any extra parameters.
        </p>
        <p>
          We <strong>tap</strong> the output at each stage and concatenate all
          four streams: the original 1×1-conv output (<code>x₁</code>) plus the
          three progressively-pooled versions (<code>x₂</code>, <code>x₃</code>,{' '}
          <code>x₄</code>). The next layer gets a stack of features at four
          different spatial extents — sharp local detail and broad context, side
          by side.
        </p>
        <p>
          Same multi-scale information as the older SPP (which ran parallel
          pools at different sizes) but ~2× faster, because it reuses
          computation across the cascade.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// C2PSA: same shape as C3k2 but the active stream is PSABlock × n
// =============================================================================

function C2PSAInternals({ block, real }) {
  const i = real?.internals || {};
  const n = i.n ?? '?';
  const sub = i.sub_block || 'PSABlock';
  const cIn = i.in_channels ?? '?';
  const cOut = i.out_channels ?? '?';

  return (
    <div className="block-internals">
      <svg viewBox="0 0 360 560" style={svgStyle(330)}>
        {ARROW_DEFS}
        <Box x={90}  y={15}  w={180} h={48} {...C.io}    label="input"            sub={`c=${cIn}`} />
        <Box x={90}  y={100} w={180} h={56} {...C.conv}  label="1×1 conv"         sub="split into 2 streams" />

        <Box x={20}  y={210} w={130} h={56} {...C.io}    label="passthrough"      sub="(no transform)" />
        <Box x={210} y={210} w={130} h={56} {...C.attn}  label={`${sub} × ${n}`}  sub="self-attention" />

        <Box x={90}  y={310} w={180} h={48} {...C.merge} label="concat"           sub="along channel axis" />
        <Box x={90}  y={395} w={180} h={56} {...C.conv}  label="1×1 conv"         sub={`fuse → c=${cOut}`} />
        <Box x={90}  y={495} w={180} h={48} {...C.io}    label="output"           sub={`c=${cOut}`} />

        <Edge d="M 180  63 L 180 100" />
        <Edge d="M 180 156 C 180 180,  85 180,  85 210" />
        <Edge d="M 180 156 C 180 180, 275 180, 275 210" />
        <Edge d="M  85 266 C  85 295, 180 295, 180 310" />
        <Edge d="M 275 266 C 275 295, 180 295, 180 310" />
        <Edge d="M 180 358 L 180 395" />
        <Edge d="M 180 451 L 180 495" />
      </svg>
      <div className="internals-caption">
        <p>
          <strong>C2PSA</strong> stands for "Cross-stage Partial Self-Attention."
          It uses the same split-and-merge structure as C3k2, but the active
          stream applies <strong>self-attention</strong> instead of regular
          convolutions.
        </p>
        <p>
          What's self-attention? Every spatial cell in the feature map can
          directly query every other cell and pull in information from wherever
          it's relevant — not just from a small neighborhood like a regular
          convolution. This lets distant parts of the image talk to each other
          ("the lighthouse top relates to the lighthouse base"), which a stack
          of small convolutions can only achieve indirectly through many layers.
        </p>
        <p>
          The "<strong>position-sensitive</strong>" part means the attention
          also takes into account <em>where</em> each cell sits in the image,
          not just <em>what's</em> there. The passthrough half preserves any
          local detail the attention pathway might smooth over.
        </p>
        <p>
          YOLO26 places this block deep in the backbone, after the spatial
          resolution has shrunk to 20×15. By that depth there are few enough
          cells that attention is computationally affordable. Earlier in the
          network, where feature maps are 320×240 or 160×120, attention would
          be far too expensive.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Concat: stack 2 (or more) inputs along the channel axis
// =============================================================================

function ConcatInternals({ block, real }) {
  const inputShapes = real?.input_shapes || [];
  const outputShape = block.shape;
  const inputCs = inputShapes.map(s => Array.isArray(s) ? s[1] : '?');
  const outC = Array.isArray(outputShape) ? outputShape[1] : '?';
  const N = Math.max(2, inputShapes.length);
  const colW = 320 / N;

  return (
    <div className="block-internals">
      <svg viewBox="0 0 360 360" style={svgStyle(320)}>
        {ARROW_DEFS}
        {inputShapes.map((s, idx) => {
          const x = 20 + idx * colW;
          const cx = x + colW / 2;
          const cs = Array.isArray(s) ? s : null;
          return (
            <g key={idx}>
              <Box x={x + 4} y={15} w={colW - 8} h={70} {...C.io}
                   label={`input ${String.fromCharCode(65 + idx)}`}
                   sub={cs ? `c=${cs[1]} · ${cs[2]}×${cs[3]}` : ''} />
              <Edge d={`M ${cx} 85 C ${cx} 130, 180 150, 180 175`} />
            </g>
          );
        })}

        <Box x={70} y={175} w={220} h={56} {...C.merge}
             label="stack along axis=1"
             sub={`(${inputCs.join(' + ')}) → ${outC}`} />
        <Box x={70} y={275} w={220} h={48} {...C.io}
             label="output" sub={`c=${outC} · same H×W`} />

        <Edge d="M 180 231 L 180 275" />
      </svg>
      <div className="internals-caption">
        <p>
          <strong>Concat</strong> simply stacks two (or more) feature maps along
          the channel axis — same height and width, different sets of channels
          glued side-by-side. If input A has <code>{inputCs[0] ?? '256'}</code>{' '}
          channels and input B has <code>{inputCs[1] ?? '128'}</code> channels,
          the output has <code>{inputCs.join(' + ')} = {outC}</code> channels.
          Same H×W in, same H×W out.
        </p>
        <p>
          No learning happens here, but the choice of <em>which</em> two streams
          to merge is what makes YOLO's neck work. In this block, one input
          came from earlier in the network (carrying sharper spatial detail)
          and the other came from deeper layers (carrying more abstract
          features). The convolution that comes <em>after</em> Concat is what
          actually learns to combine them.
        </p>
        <p>
          This is the mechanism behind every <strong>skip connection</strong>{' '}
          in the architecture — re-introducing earlier signals into the deeper
          data path so the network can use both fine details and high-level
          abstractions when making its predictions.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Upsample: nearest-neighbor 2×
// =============================================================================

function UpsampleInternals({ block, real }) {
  const i = real?.internals || {};
  const sf = i.scale_factor ?? 2;
  const mode = i.mode || 'nearest';
  const inputShape = real?.input_shapes?.[0];
  const outShape = block.shape;
  const inHW = Array.isArray(inputShape) ? `${inputShape[2]}×${inputShape[3]}` : 'H×W';
  const outHW = Array.isArray(outShape) ? `${outShape[2]}×${outShape[3]}` : `${sf}H×${sf}W`;
  const c = Array.isArray(inputShape) ? inputShape[1] : '?';

  return (
    <div className="block-internals">
      <svg viewBox="0 0 320 280" style={svgStyle(280)}>
        {ARROW_DEFS}
        <Box x={70}  y={15}  w={180} h={48} {...C.io}   label="input"  sub={`c=${c} · ${inHW}`} />
        <Box x={70}  y={100} w={180} h={56} {...C.pool} label={`upsample ${mode} ${sf}×`} sub="zero parameters" />
        <Box x={70}  y={200} w={180} h={48} {...C.io}   label="output" sub={`c=${c} · ${outHW}`} />

        <Edge d="M 160  63 L 160 100" />
        <Edge d="M 160 156 L 160 200" />
      </svg>
      <div className="internals-caption">
        <p>
          Doubles the height and width of the feature map by repeating each
          cell {sf}× in each direction. This uses the <strong>nearest-neighbor</strong>{' '}
          rule: each output pixel just copies its closest input pixel — no
          smoothing, no learnable parameters.
        </p>
        <p>
          <strong>Why we need this:</strong> the deep layers of the backbone
          produce small, abstract feature maps (e.g. 20×15) where each cell
          summarizes a large region of the original image. To detect smaller
          objects, we need fine spatial resolution again — so the neck
          "upsamples" these deep features back up (20×15 → 40×30, then →
          80×60), then mixes them with shallow backbone features at matching
          resolution via the next Concat.
        </p>
        <p>
          The result: the head receives feature maps that are simultaneously{' '}
          <em>deep</em> (semantically rich) and <em>spatially precise</em> —
          this is the FPN (Feature Pyramid Network) trick that made multi-scale
          object detection practical.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Detect: 3 distinct conv-head pairs (one per scale) → DFL → end-to-end top-K
// =============================================================================

function DetectInternals({ block, real }) {
  const i = real?.internals || {};
  const nc = i.nc ?? '?';
  const reg_max = i.reg_max ?? '?';
  const no = i.no ?? '?';
  const strides = i.strides || [];

  const cols = [
    { name: 'P3', x: 20,  fill: '#dbeafe', stride: strides[0] },
    { name: 'P4', x: 170, fill: '#bfdbfe', stride: strides[1] },
    { name: 'P5', x: 320, fill: '#93c5fd', stride: strides[2] },
  ];

  return (
    <div className="block-internals">
      <svg viewBox="0 0 460 720" style={svgStyle(420)}>
        {ARROW_DEFS}

        {cols.map((c, idx) => (
          <Box key={c.name} x={c.x} y={15} w={120} h={56}
               fill={c.fill} stroke="#3b82f6" text="#1e3a8a"
               label={c.name}
               sub={c.stride ? `stride ${c.stride}` : ''} />
        ))}

        {cols.map((col, idx) => (
          <g key={col.name + 'h'}>
            <Box x={col.x - 10} y={110} w={140} h={66} {...C.conv}
                 label="conv heads"
                 sub={`cv2 (4·rm) + cv3 (${nc})`} />
            <Edge d={`M ${col.x + 60} 71 L ${col.x + 60} 110`} />
          </g>
        ))}

        <Annot x={230} y={205} text={`each cell → ${no} channels (4·reg_max + ${nc} classes)`} anchor="middle" />

        {cols.map((col, idx) => (
          <Edge key={col.name + 'a'} d={`M ${col.x + 60} 176 C ${col.x + 60} 220, 230 220, 230 240`} />
        ))}

        <Box x={120} y={240} w={220} h={56} {...C.norm}
             label="per-anchor predictions"
             sub="≈ 6,300 anchors total" />

        <Edge d="M 230 296 L 230 340" />

        <Box x={120} y={340} w={220} h={56} {...C.dfl}
             label="DFL decode"
             sub={`reg_max=${reg_max} → boxes`} />

        <Edge d="M 230 396 L 230 440" />

        <Box x={100} y={440} w={260} h={66} {...C.attn}
             label="one2one top-K"
             sub="end-to-end · NMS-free" />

        <Edge d="M 230 506 L 230 545" accent={true} />

        <Box x={120} y={545} w={220} h={56} {...C.final}
             label="final detections"
             sub="(x1, y1, x2, y2, conf, cls)" />

        <Annot x={230} y={650} text="(training also uses a separate one2many branch — omitted here)" anchor="middle" size={10} />
      </svg>
      <div className="internals-caption">
        <p>
          The <strong>head</strong> is what turns feature maps into actual
          object detections. It works on three input feature maps simultaneously
          — one for spotting small objects (<strong>P3</strong>, fine 80×60
          grid), one for medium objects (<strong>P4</strong>, 40×30 grid), and
          one for large objects (<strong>P5</strong>, coarse 20×15 grid). Each
          grid cell is called an <strong>"anchor"</strong> — a candidate object
          position.
        </p>
        <p>
          For <em>each scale</em>, two small convolution heads run side by side
          at every anchor:
        </p>
        <ul>
          <li>
            <code>cv2</code> predicts <strong>where the box edges are</strong> —
            4 numbers per cell, telling the decoder how far left / up / right /
            down the box extends from this cell's center.
          </li>
          <li>
            <code>cv3</code> predicts <strong>class scores</strong> — {nc}{' '}
            numbers, one per COCO category ("is this a person? a tie? a bench?
            …").
          </li>
        </ul>
        <p>
          Across all three scales there are roughly 6,300 cells, each producing
          one candidate box plus class scores.
        </p>
        <p>
          "<strong>DFL decode</strong>" stands for{' '}
          <em>Distribution Focal Loss</em>. It converts the four box numbers
          into actual <code>(x1, y1, x2, y2)</code> pixel coordinates. Instead
          of predicting a box edge as a single number, DFL predicts it as a
          probability distribution and takes the expected value — turns out to
          be more accurate, especially for ambiguous edges.
        </p>
        <p>
          Traditionally, YOLO would then run <strong>NMS</strong> (Non-Maximum
          Suppression) — a post-processing step that throws away boxes that
          overlap too much with a more confident neighbor.{' '}
          <strong>YOLO26 is NMS-free.</strong> The "<code>one2one</code>" branch
          is trained to produce only one prediction per object, so we can take
          the top-K most confident detections directly. This makes inference
          faster and easier to deploy. (The trick is borrowed from YOLOv10's
          design.)
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Dispatch
// =============================================================================

function BlockInternals({ block, real }) {
  const T = block.type;
  if (T === 'Conv')     return <ConvInternals     block={block} real={real} />;
  if (T === 'C3k2' || T === 'C3' || T === 'C2f')
                        return <C3k2Internals     block={block} real={real} />;
  if (T === 'SPPF')     return <SPPFInternals     block={block} real={real} />;
  if (T === 'C2PSA')    return <C2PSAInternals    block={block} real={real} />;
  if (T === 'Concat')   return <ConcatInternals   block={block} real={real} />;
  if (T === 'Upsample') return <UpsampleInternals block={block} real={real} />;
  if (T === 'Detect')   return <DetectInternals   block={block} real={real} />;
  return null;
}

return { BlockInternals };
})();

window.YV = window.YV || {};
window.YV.BlockInternals = Internals.BlockInternals;
