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
- **[`ROADMAP.md`](./ROADMAP.md)** — the running list of intended/next work.
  ADRs record decisions *already made*; the roadmap is where *upcoming* work
  lives. Check it for "what's next" — not the ADRs.
- **[`docs/adr/`](./docs/adr/)** — architectural decisions and their rationale:
  - `0001` — ELK for in-Region layout (accepted; ELK is the layout engine).
  - `0002` — CSS-first single source of truth for design tokens.
  - `0003` — side panel is an interpretation aid, not a reference.
  - `0004` — public hosting on a Hugging Face Docker Space.
  - `0005` — left-align coordinate/routing post-process (SUPERSEDED by 0006 —
    record of a path not taken; do not build against it).
  - `0006` — ELK owns node placement and in-Region edge routing (no post-process).
  - `0007` — edge rendering: ELK spline routing with flattened port tails.

These were produced via the `grill-with-docs` convention: when a decision or a
term crystallises, update `CONTEXT.md` (glossary only) or add an ADR; when a
*work item* surfaces, add it to `ROADMAP.md`. Don't let docs drift from code,
and don't use ADRs as a to-do list.

## Working alongside Codex (branching)

Claude and Codex work the same repo on independent problems with occasional
shared dependencies. The convention (mirrored in `AGENTS.md`):

- **Trunk-based.** `main` is the single integration point and stays green.
  Prefer small, frequently-merged slices over long-lived forks.
- **Branch per topic, namespaced by agent:** `claude/<topic>` (Codex uses
  `codex/<topic>`). The prefix shows ownership at a glance and makes orphan
  cleanup trivial (`git branch --merged main | grep claude/`).
- **Delete on merge.** Once a branch lands in `main`, delete the branch *and*
  remove its worktree (`git worktree remove …`). Leftover branches/worktrees
  are how the repo accumulated stale `claude/*` and `codex/*` cruft before.
- **Overlap → integrate small and often.** If a shared dependency is needed
  first, land *it* to `main` as its own tiny slice, then both agents build on
  it. Don't sit on two long-lived branches editing the same files; whoever
  lands first wins and the other rebases on `main`.
- **Worktree storage note.** Worktrees share one `.git` (history isn't
  duplicated) and a working copy is cheap (~14 MB). The real cost is running
  `uv sync` *inside* a worktree — that grows a per-worktree 1.3 GB `.venv`.
  Prefer reusing the main `.venv`; if you must sync in a worktree, remove the
  worktree when done.

## Notes

- Frontend has no build step (Babel-in-browser, `window.YV` globals); the real
  requirement is free static-hostable output, not "never use tooling" (see 0001).
- Prefer the live preview over screenshots when verifying UI; let the user drive
  feedback unless they invite screenshot reading.
- **Debugging an ELK layout failure** (`buildExpansionELK` catches and returns
  `null`, so the block silently collapses): don't reconstruct the graph by hand
  and guess — it's bigger/subtler than you think. Capture the *real* ELK input:
  temporarily `window.__elkFail = JSON.stringify(root)` right before
  `_elk.layout(root)`, have the user trigger the failing expand and `copy(window.__elkFail)`,
  then replay that exact JSON headlessly with `node` against
  `frontend/vendor/elk.bundled.js` (it's a UMD bundle — `require()` works) and
  bisect by toggling one option/edge at a time. This pinned the
  `considerModelOrder`-on-boundary-crossed-container NPE in minutes after hand-built
  repros failed (see ADR-0006).

- **Inspecting ELK's *output* (edge routing / placement), not just its input:**
  don't hand-build a graph and guess what ELK returned. Drive the running Space
  with the Claude preview tools and, in `preview_eval`, wrap
  `window.YV._elk.layout` to capture the laid-out `root`, then call
  `window.YV.buildExpansionELK(idx, {expansions:[...]})` and walk the captured
  tree. This reads the real per-edge sections (start/bend/end points) for any
  expansion state — how ADR-0007's flattened-tail decisions were verified
  (e.g. which edges were single-cubic-but-angled vs lane-routed).
