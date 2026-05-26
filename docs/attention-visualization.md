# Attention Visualization Notes

This note explains the prototype at `frontend/attention-prototype.html` in
learner-facing terms. It is deliberately about reading the visualization, not
about production implementation.

## What The Map Shows

The C2PSA block contains a small self-attention module at the deepest feature
scale. At this point the image has been compressed into a coarse attention grid:
for the bundled sample, the grid is 20 rows by 15 columns.

When you pick one grid cell, that cell is the **query**. The heatmap shows one
row of the post-softmax attention matrix: which other grid cells the query uses
when it mixes information from the value tensor. Bright areas are the cells that
contribute more strongly to the selected query.

Read it as: "this patch is borrowing context from these patches."

## What It Does Not Show

- It is not an object mask.
- It is not a detector confidence heatmap.
- It is not a complete explanation of the final bounding boxes.
- It does not include the depthwise position-bias output (`pe`); it shows the
  content-similarity attention weights before values are mixed and projected.

The grid is coarse. A single attention cell corresponds to a fairly large patch
of the letterboxed input image, so use it for broad relationships rather than
pixel-precise boundaries.

## Controls

- **Mean / Head 0 / Head 1**: each attention head can learn a different pattern.
  Mean is a good first read, but the heads are worth comparing.
- **Per query**: stretches each selected query's map to its own min/max. This is
  best for seeing shape and relative structure.
- **Global**: uses one scale across the selected head. This is better for asking
  whether a bright-looking map is actually strong compared with other queries.
- **Overlay**: blends the heatmap with the image.
- **Speed**: plays through query cells in raster order.

## Sample Image Observations

On the bundled lighthouse/person sample, the two heads behave differently:

- **Head 0 is sharper and more local.** The strongest key is the query cell
  itself for about 40% of query positions. That is normal: attention often keeps
  a strong self/nearby component while still allowing longer-range context.
- **Head 1 is broader and more contextual.** It has higher entropy and rarely
  peaks on the query cell. Several distant query cells peak around the central
  person/lighthouse region, which is a hint that this head is pulling from a
  shared high-level context instead of just the local patch.
- **The mean view blends those stories.** It is useful for orientation, but it
  can hide the difference between a sharp local head and a diffuse context head.

The visual pattern to look for is not just "where is it bright?" but "does this
query mostly look at itself, nearby texture, repeated structure, or a distant
semantic region?" Try the same query under all three head modes and then switch
between per-query and global normalization.

## Running The Prototype

Static sample:

```bash
uv run python -m http.server 8766 --directory frontend
```

Then open `http://127.0.0.1:8766/attention-prototype.html`.

To choose another image inside the visualizer, use the prototype server:

```bash
uv run python scripts/serve_attention_prototype.py
```

Open the same URL and click **Choose image**. Generation can take several
seconds because it runs the model and captures the attention tensor.
