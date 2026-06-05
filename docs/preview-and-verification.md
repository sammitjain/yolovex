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

## Stale `.jsx` after an edit (read this — it will bite you)

The dev server sends **no `cache-control`** for the `.jsx` files, and Babel
fetches them through the browser HTTP cache. After you edit a source file,
`location.reload()` (and even `?v=…` on the *page* URL, which only busts the
HTML, not its sub-resources) will often re-run the **old** compiled code — and
inconsistently, one file fresh while a sibling is stale. Symptom: your change is
in the file (`curl localhost:8765/app.jsx | grep …` confirms it) but the live
page still behaves the old way.

Force the browser to refresh each `.jsx` into cache, then navigate:

```js
(async () => {
  const files = ['app.jsx','arch.jsx','layout.jsx','graph-sem.jsx','expand-elk.jsx','graph-elk.jsx'];
  for (const f of files) { try { await fetch('/'+f, { cache: 'reload' }); } catch (e) {} }
  if (window.caches) { const ks = await caches.keys(); await Promise.all(ks.map(k => caches.delete(k))); }
  location.href = '/?v=' + Date.now();
})()
```

Then **confirm the new code is live** before trusting any result, e.g.
`someFn.toString().includes('<a string only in your edit>')`. (A `serve.py`
no-cache header for `.jsx` would retire this dance — not yet done.)

## Which page

- The ELK layout is the **primary** page — `/` serves it (`frontend/index.html`).
  There is no longer a separate old renderer. Navigate in an eval:
  `location.href = '/'`.
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
  Walks back through tensor-less nodes (tuple `chunk`/`split`, uncaptured ops)
  but **stops at a captured `getitem`**, so a split-branch consumer reports the
  actual slice, not the pre-split parent.
- `instanceShapes(idx)` — `{ fxName: [..]|[[..]]|null }`; `null` ⇒ scalar/shape
  node, not a tensor.
- `isImageShaped(shape, meta)` — true iff the last-two-dims aspect is
  proportional to `meta` image_w/h (the play-flow uses it to decide stretch vs
  passthrough). E.g. `[1,128,20,15]` → true, `[1,2,300,300]` → false.

Examples:
- Reshape "2 inputs" bug stays fixed: `subUpstreamSources(10, ['reshape'])`
  returns exactly one source.
- Split-slice dims (attention V·attnᵀ): `subUpstreamSources(10, ['matmul_1'])`
  → two sources whose shapes are `[1,2,64,300]` and `[1,2,300,300]` (NOT
  `[1,2,128,300]`); look each up via `YV_ACT.nodes['10'].sub[fxKey].shape`.
- Module-level IO inputs filter internal QKV slices: for a PSABlock-level
  member set inside C2PSA[10] (includes `split_1` but not `getitem_6/7/8`),
  `subUpstreamSources` returns **exactly one** source — `getitem_1`, the real
  module input. The same holds for the Attention member set. The internal
  q/k/v slices are filtered by an "internal getitem of an internal split"
  rule: their tuple parent (`split_1`) is itself in the member set, so they
  are visually-internal and don't surface as module inputs. The
  singleton-selection case above is preserved (parent `split_1` is *not* in
  `['matmul_1']`).

## Inspecting ELK output without clicking

`window.YV.buildExpansionELK(idx, { expansions: [...] })` is **async** (ELK
layouts off-thread) — `await` it; treating the returned Promise as the result
gives `{}` (its `Object.keys` are empty) and silently wrong conclusions. Pass
the path-key set you want peeled open (`''` is depth-1), then read the laid-out
`.subNodes` / `.subEdges` directly — e.g. to confirm a field reaches the
renderer:

```js
(async () => {
  const ex = await window.YV.buildExpansionELK(10, { expansions: ['','0_PSABlock','0_PSABlock/attn'] });
  return ex.subNodes.filter(n => n.subkind === 'arith').map(n => ({ id: n.id, scalarOperand: n.scalarOperand }));
})()
```
