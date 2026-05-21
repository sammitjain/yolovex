# CSS custom properties are the single source of truth for design tokens

---
Status: proposed — direction agreed; not yet implemented.
---

## Context

Color lived in three homes with hand-synced duplication: CSS custom properties
in `index.html` (chrome) and two JS palette systems (`arch.jsx`
`TYPE_PALETTES`/`ROLE_PALETTES`/`ACCENTS`/`GRAPH_BGS`, and `expand.jsx`
`SUB_KIND_COLORS`). Some values were defined twice in two languages (`--accent`
≙ `ACCENTS.light`; `--graph-bg` ≙ `GRAPH_BGS.light`) and theme-switching ran
through two parallel paths (the `data-theme` CSS swap AND the JS palette mirror
into `LAYOUT_SETTINGS`). There was no single named concept of a token with one
home — the fragmentation behind the "styling feels scattered" perception.

## Decision

**CSS custom properties are the one source of truth for every themeable style
value** — chrome *and* the graph node palettes (e.g. `--conv-fill`,
`--conv-border`, `--role-neck`, `--subkind-cat`). The SVG renderer sets
`fill` / `stroke` via `var(--…)` (valid as CSS *properties* on SVG elements), so
nodes re-theme automatically with no re-render. There is **one** theme
mechanism: the `data-theme` attribute, with `:root` (light) and
`[data-theme="dark"]` (dark) blocks. Both variants are retained.

The Settings drawer edits tokens via `style.setProperty` instead of mutating
`LAYOUT_SETTINGS` palette objects. The JS palettes retire as a *source*; if a
consumer needs a numeric color value (e.g. computing nesting-depth opacity,
canvas thumbnails) it reads the token once via `getComputedStyle`.

## Consequences

- Kills cross-language duplication and the second theme path.
- Orthogonal to the ELK migration (ELK does layout, never color) and survives
  every build path (no-build, Vite, or React Flow all consume CSS vars).
- A handful of JS color consumers switch to a one-line `getComputedStyle` read.
- Token naming becomes the only discipline to maintain (one name per value).
