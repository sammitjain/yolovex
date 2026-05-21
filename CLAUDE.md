# yolovex — agent guide

Interactive YOLO26 architecture explainer (CNN-Explainer-style). Python (fx
trace + activation capture, CLI, FastAPI `serve`) under `src/yolovex/`; a
no-build-step React SVG explorer under `frontend/`.

## Read these first

- **[`CONTEXT.md`](./CONTEXT.md)** — the authoritative shared vocabulary for the
  diagram/layout domain (Block, Node, Sub-node, Leaf, Box, Region, Role, the
  Sugiyama layout terms, Token/Theme/Palette, side-panel content terms). Use
  these words exactly; they were chosen to retire overloaded ones (e.g.
  "container" is retired).
- **[`docs/adr/`](./docs/adr/)** — architectural decisions and their rationale:
  - `0001` — ELK for in-Region layout (direction; spike in progress).
  - `0002` — CSS-first single source of truth for design tokens.
  - `0003` — side panel is an interpretation aid, not a reference.
  - `0004` — public hosting on a Hugging Face Docker Space.
  - `0005` — Region coordinate + routing post-process (the left-align spec;
    **next thing to implement** — read this before touching `expand-elk.jsx`).

These were produced via the `grill-with-docs` convention: when a decision or a
term crystallises, update `CONTEXT.md` (glossary only) or add an ADR — don't let
docs drift from code.

## Notes

- `docs/DEVNOTES.archive.md` is historical and does NOT describe current code.
- Frontend has no build step (Babel-in-browser, `window.YV` globals); the real
  requirement is free static-hostable output, not "never use tooling" (see 0001).
- Prefer the live preview over screenshots when verifying UI; let the user drive
  feedback unless they invite screenshot reading.
