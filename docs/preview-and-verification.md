# Preview & verification workflow

How to run the explorer in a live preview and verify UI changes by inspecting
the real DOM (not screenshots). This is the agent's reference so the steps below
aren't rediscovered each session. Prefer the live preview over screenshots (see
CLAUDE.md); let the user drive subjective feedback.

## Start the preview server

- Launch config name: **`yolovex`** (`.claude/launch.json` → `uv run yolovex
  serve`, port **8765**). Start it with the preview `preview_start` tool, name
  `yolovex`.
- **If it won't start** — `preview_start` reports *port in use* or *server not
  found* while a stale process lingers on 8765:

  ```sh
  lsof -ti :8765        # list PID(s) holding the port
  kill <pids>           # free it
  ```

  then `preview_start` again. (This happens after edits restart/orphan the FastAPI
  process; the preview tracker loses the handle but the port stays bound.)

## Which page

- Primary page is currently **`/index-elk.html`** (the ELK layout). `/` still
  serves the old `index.html` until the ELK-as-primary migration lands
  (see ROADMAP). Navigate in an eval: `location.href = '/index-elk.html'`.
- Data is bundled: `window.YV_SPEC` (per-block fx graphs + per-instance
  `shapes_by_node`), `window.YV_ACT` (captured activations: `.nodes[idx].output`
  and `.nodes[idx].sub[fxName]`, plus `.meta` with `image_w/image_h`),
  `window.YV_CONTENT` (side-panel copy).

## Reading the ELK layout (DOM shape)

- **L1 blocks**: `<g data-node="<idx>">` (idx 0–24).
- **Sub-nodes**: SVG groups labelled by their `<text>` content — e.g.
  `.split()`, `PSABlock`, `Attention`, `.softmax()`, `fn:matmul`, `.reshape()`,
  `Conv`. There is no `data-*` id on sub-nodes; locate them by label text.
- **Side panel**: `.detail-panel`. Inside it: `.panel-title`,
  `.split-out-tab` (split-op output tabs), `.io-tile` (IO-strip thumbnails),
  `.shape-caption`, `.shape-transform` (shape-op card), `.brochure-preview img`
  (main channel preview), `.brochure-thumb` + `.thumb-idx` (channel grid).

## Driving the canvas

- **Expand / collapse**: shift+click a node. **Select (open panel)**: plain
  click. Dispatch a native event on the node's `<text>` — it bubbles to React's
  delegated root handler:

  ```js
  const fire = (el, shift) => el && el.dispatchEvent(
    new MouseEvent('click', { bubbles: true, cancelable: true, view: window, shiftKey: !!shift }));
  ```

- **Drill to the QKV split** (a good test target — `out 0/1/2` = q/k/v, with
  non-image shapes): expand C2PSA (`[data-node="10"]`, shift) → expand
  `PSABlock` (shift) → expand `Attention` (shift) → click the inner `.split()`.

## State caveat — use a single eval

Canvas/React state does **not** reliably survive across separate `preview_eval`
calls (especially after a navigation/reload). Do the whole
expand→expand→select→inspect sequence inside **one** IIFE, with
`await sleep(~700ms)` between steps so ELK can relayout:

```js
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const fire = (el, shift) => el && el.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true,view:window,shiftKey:!!shift}));
  const txt = re => [...document.querySelectorAll('text')].filter(t => re.test(t.textContent||''));
  fire(document.querySelector('[data-node="10"]'), true); await sleep(800);
  fire(txt(/PSABlock/)[0], true);                          await sleep(800);
  fire(txt(/Attention/)[0], true);                         await sleep(800);
  // pick the split whose panel shows 3 output tabs (the QKV one):
  for (const s of txt(/\.split\(\)/)) { fire(s); await sleep(400);
    if (document.querySelectorAll('.split-out-tab').length === 3) break; }
  return [...document.querySelectorAll('.detail-panel .io-tile')].map(e => ({w:e.style.width, h:e.style.height}));
})()
```

## Inspect via DOM, not screenshots

Read computed/inline styles directly:
- IO tiles: `el.style.width` / `el.style.height` on `.detail-panel .io-tile`
  (sized by `fitBox` from the tensor's true H×W).
- Brochure preview: `.brochure-preview img` `style.width/height`.
- Brochure thumbs: `getComputedStyle(thumb).minHeight` (≥ 26px so the
  `.thumb-idx` label stays visible on wide-short maps).
- Tabs / captions: `.split-out-tab` text, `.shape-caption` text.

## Verify logic without the UI

The pure helpers in `app.jsx` are **global** — call them in `preview_eval`
against the real `YV_ACT`/`YV_SPEC` to check logic without driving the canvas:

- `fitBox(shape, maxW, maxH)` — aspect-preserving box for a tensor shape.
- `splitOpOutputs({idx, fxKey, subkind:'split', ...})` — a split's getitem
  outputs with shapes.
- `subUpstreamSources(idx, [fxName])` — upstream tensor inputs (scalars dropped).
- `instanceShapes(idx)` — `{ fxName: [..]|[[..]]|null }`; `null` ⇒ scalar/shape
  node, not a tensor.

Example — confirm the reshape "2 inputs" bug stays fixed:
`subUpstreamSources(10, ['reshape'])` should return exactly one source.
