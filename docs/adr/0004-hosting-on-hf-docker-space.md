# Public hosting: a Hugging Face Docker Space running `yolovex serve`

---
Status: proposed — direction agreed; not yet deployed.
---

## Context

The tool should be publicly usable for free, and — importantly — let users
**upload their own image and run inference live**, even if slow (tens of seconds
is acceptable). That makes it server-backed, not pure-static. `serve.py` is
already a single-origin FastAPI + uvicorn app that serves the static frontend
plus `/api/upload` and an SSE progress stream, and enforces one active job at a
time. Nothing comparable was previously documented — this is greenfield.

## Decision

Host on a **Hugging Face Docker Space running `yolovex serve`**:

- **Docker, not Gradio** — the frontend is a custom React SPA already served by
  `serve.py`; Gradio's component model would fight it.
- **Weights baked at build** — download `yolo26n.pt` during `docker build` so a
  cold container doesn't fetch on first request.
- **Precomputed activations bundled** — the committed `activations.js` for the
  sample image makes the page instantly useful and serves as the fallback when
  no upload has happened. Live upload overwrites in-session. (Precomputed ships
  regardless of hosting — see ADR-0001's static-output requirement.)
- **Free CPU tier**, slow inference accepted; the existing one-job-at-a-time
  queue matches a single free worker. **Paid tier deferred** until user feedback
  justifies cost.

## Considered options

- **Pure static (no live upload)** — rejected as the *primary* experience: live
  upload is a stated product goal. Precomputed-static remains the fallback layer.
- **Gradio Space** — rejected: wrong fit for a custom React SPA.
- **Other free server hosts (Render, Fly free tiers)** — not chosen: HF Spaces
  is purpose-built for ML demos and was the recommended path for PyTorch.

## Consequences / known risks

- HF free Spaces sleep on inactivity → cold start reloads the model (slow first
  hit; acceptable).
- `serve.py` writes `activations.js` + uploads into frontend/assets dirs; the
  Space filesystem is constrained/ephemeral, so those writes must be redirected
  to a writable path (`/data` or `/tmp`) and served from there. This is the most
  likely first-deploy breakage.
- No upload persistence across restarts (not required).
