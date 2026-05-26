# yolovex — Codex agent guide

Interactive YOLO26 architecture explainer. Python lives under `src/yolovex/`
for model loading, fx tracing, activation capture, CLI commands, and FastAPI
`serve`; the frontend is a no-build-step React/SVG explorer under `frontend/`
using Babel-in-browser and `window.YV*` globals.

## Read these first

- [`CONTEXT.md`](./CONTEXT.md) is the canonical project language. Use its terms
  exactly: Block, Node, Sub-node, Leaf, Box, Region, Role, Role frame, Flow,
  Layer, Port, Long edge, Merge, Split, Tail, Token, Theme, Palette, Type card,
  Intuition, Theory, Interpretation, Activation view, and Attention map.
  "Container" is retired except when quoting historical docs.
- [`ROADMAP.md`](./ROADMAP.md) is the source of upcoming work. ADRs record
  settled or proposed decisions; do not mine them as a to-do list.
- [`docs/adr/`](./docs/adr/) captures architectural decisions:
  - `0001`: ELK is adopted for in-Region layout.
  - `0002`: CSS custom properties should become the design-token source of
    truth. Status is proposed, not fully implemented.
  - `0003`: the side panel is an interpretation aid, not a reference.
  - `0004`: public hosting direction is a Hugging Face Docker Space.
  - `0005`: superseded by `0006`; do not build against it.
  - `0006`: ELK owns in-Region placement and routing; tune ELK inputs, not its
    output geometry.
  - `0007`: render ELK spline routes with flattened port tails.
- [`docs/preview-and-verification.md`](./docs/preview-and-verification.md) has
  useful UI verification targets, but it is written around Claude preview
  tools. In Codex, use the local server plus the Browser plugin instead.
- [`docs/DEVNOTES.archive.md`](./docs/DEVNOTES.archive.md) is historical only
  and describes an older frontend that no longer exists.

## Current shape

- The current ELK path is `frontend/index-elk.html`,
  `frontend/graph-elk.jsx`, `frontend/expand-elk.jsx`, and
  `frontend/vendor/elk.bundled.js`.
- The older root page still uses `frontend/index.html` and `frontend/graph.jsx`
  until the roadmap item "Make the ELK layout the primary page" lands.
- Shared frontend modules include `frontend/arch.jsx`,
  `frontend/layout.jsx`, `frontend/expand.jsx`, `frontend/app.jsx`,
  `frontend/spec-data.js`, `frontend/activations.js`, and
  `frontend/content/blocks.js`.
- Generated payload globals are `window.YV_SPEC` and `window.YV_ACT`; app code
  attaches helpers and components to `window.YV`.

## Run and verify

- Sync dependencies with `uv sync`.
- Build bundled activations with `uv run yolovex build-assets`.
- Run the interactive server with `uv run yolovex serve` and inspect
  `http://127.0.0.1:8765/index-elk.html` for ELK work.
- Static inspection also works by opening `frontend/index-elk.html`.
- For UI changes, prefer live DOM inspection over screenshots. Check the real
  app state, computed styles, and `window.YV` helpers where possible.

## Roadmap focus

The most immediate roadmap items are:

- Complete the side-panel content audit in `frontend/content/blocks.js`.
- Hide scalar arithmetic op nodes without hiding tensor arithmetic.
- Passthrough non-image-shaped outputs in the play-flow.
- Make the ELK layout the primary page and archive/remove old-renderer code.
- Prototype and then integrate the attention visualization.

When a roadmap item creates a hard-to-reverse decision, add or update an ADR and
link the work from `ROADMAP.md`. Keep `CONTEXT.md` for shared vocabulary only.

## Codex / Claude isolation

- Treat this file as Codex-owned. Do not edit `CLAUDE.md` or create `.claude/`
  files unless the user explicitly asks.
- If Codex-specific notes grow beyond this file, put them under `docs/codex/`
  rather than mixing them into Claude-owned files.
- Before implementation work, create or switch to a `codex/...` branch from this
  worktree. Keep Claude's branches, preview config, and scratch files separate.
- Check `git status --short --branch` before edits and before finishing. Assume
  unrelated changes may belong to the user or another agent; work around them
  rather than reverting them.
