# The side panel is an interpretation aid, not a reference

---
Status: proposed — direction agreed; layer 3 (Interpretation) not yet built.
---

## Context

The side panel runs on two content axes — Type-level (`YV_CONTENT`/`YV_FORWARD`,
keyed by class: tagline, intuition, technical, shape, yolo26, forward, refs) and
Instance-level (`YV_ACT`: IO strip, channel brochure, raw min/max/mean/std). It
hands the learner the activation images plus raw statistics and leaves all
*reading* of them to the user. Separately, some `technical` fields reproduce
textbook PyTorch definitions — the thing we explicitly do not want to be.

## Decision

Model panel content as **three explicit layers**, governed by one principle:
**the panel is an interpretation aid, not a reference.**

1. **Type card** — what this *kind* of block is. Lead with the YOLO role and the
   friendly `intuition` (the asset; keep its voice). Trim `technical` to what is
   *notable / non-obvious in YOLO*; **link** the textbook definition via `refs`
   rather than reproducing it.
2. **Activation view** — the images + stats, as today (unchanged).
3. **Interpretation** (new) — *what to look for* in this activation and whether
   anything is notable. **Hand-authored**, keyed per-block-position with a
   per-type fallback (the same type at different depths reads very differently).
   Deliberately modest; never overclaim semantic meaning we cannot substantiate.
   Must be **image-robust** — phrased as tendencies / what-to-look-for, since
   users upload arbitrary images; never assert content that only holds for the
   bundled sample.

## Considered / deferred

- **Auto per-instance interpretation heuristics** (e.g. flag "mostly inactive,
  low std" from existing stats) — deferred. Interpretation is hand-authored
  first; heuristics are a possible later augmentation, kept honest if added.

## Consequences

- `intuition` is preserved as the substance that keeps the panel from feeling
  thin; `technical` shrinks toward YOLO-relevance + links.
- A new content channel is needed for Interpretation (keying TBD: per-type vs
  per-block-position).
- Reversible by editing content; recorded so the "don't reproduce PyTorch docs"
  boundary isn't quietly undone by future panel bloat.
