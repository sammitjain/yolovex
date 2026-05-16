"""Local web server for the v2 explorer with live image upload.

Serves `frontend/v2/yolovexv2.html` plus its sibling assets, and exposes a
small API for uploading a custom image, watching the build progress over SSE,
and refreshing the activation payload without a hard reload.

The build runs in a single background worker thread; only one job at a time.
A second concurrent upload returns 409 — the UI is expected to disable the
upload button while a job is in flight, this is just a defense.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .build_assets_v2 import BuildCancelled, build, write_activations_js
from .model import DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root = current working directory).
# `yolovex serve` is expected to be run from the repo root, same as the other
# CLI commands.
# ---------------------------------------------------------------------------
ROOT = Path.cwd()
FRONTEND_DIR = ROOT / "frontend"
V2_DIR = FRONTEND_DIR / "v2"
ASSETS_DIR = ROOT / "assets"
UPLOAD_DIR = ASSETS_DIR / "uploads"
ACTIVATIONS_JS = FRONTEND_DIR / "activations-v2.js"
SPEC_JS = FRONTEND_DIR / "spec-data.js"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Job state — guarded by a lock; SSE queue per job; one active job at a time.
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, job_id: str, image_path: Path):
        self.id = job_id
        self.image_path = image_path
        self.events: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.started_at = time.time()


_state_lock = threading.Lock()
_active_job: Job | None = None
_recent_jobs: dict[str, Job] = {}  # keep a few recent for late SSE subscribers


def _set_active(job: Job | None) -> None:
    global _active_job
    with _state_lock:
        _active_job = job


def _get_active() -> Job | None:
    with _state_lock:
        return _active_job


def _record_job(job: Job) -> None:
    with _state_lock:
        _recent_jobs[job.id] = job
        # cap retention to ~5 jobs
        while len(_recent_jobs) > 5:
            _recent_jobs.pop(next(iter(_recent_jobs)))


def _find_job(job_id: str) -> Job | None:
    with _state_lock:
        return _recent_jobs.get(job_id)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
def _worker(job: Job, weights: str, imgsz: int) -> None:
    def progress(event: dict) -> None:
        job.events.put(event)

    def cancel_check() -> bool:
        return job.cancelled.is_set()

    # Pass the image as a project-root-relative path so meta.image renders
    # correctly in the frontend (`'../' + meta.image` resolves against /v2/).
    try:
        rel_image = job.image_path.relative_to(ROOT)
    except ValueError:
        rel_image = job.image_path

    try:
        data = build(
            rel_image,
            weights=weights,
            imgsz=imgsz,
            progress=progress,
            cancel_check=cancel_check,
        )
        progress({"kind": "stage", "stage": "write"})
        # Ensure meta.image is a forward-slashed, root-relative path — the
        # frontend prepends '../' to it.
        try:
            data["meta"]["image"] = str(Path(data["meta"]["image"]).as_posix())
        except Exception:
            pass
        write_activations_js(data, ACTIVATIONS_JS)
        size = ACTIVATIONS_JS.stat().st_size
        n_blocks = len(data["nodes"])
        n_subs = sum(len(b.get("sub", {})) for b in data["nodes"].values())
        progress({
            "kind": "done",
            "n_blocks": n_blocks,
            "n_subs": n_subs,
            "skipped": data["meta"]["skipped"],
            "bytes": size,
            "image": data["meta"]["image"],
            "image_w": data["meta"]["image_w"],
            "image_h": data["meta"]["image_h"],
        })
    except BuildCancelled:
        job.events.put({"kind": "cancelled"})
    except Exception as e:
        job.events.put({"kind": "error", "message": f"{e.__class__.__name__}: {e}"})
    finally:
        job.done.set()
        _set_active(None)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def create_app(weights: str, imgsz: int) -> FastAPI:
    app = FastAPI(title="yolovex", docs_url=None, redoc_url=None)

    # Static files. The frontend HTML uses relative paths like `../spec-data.js`
    # and `../activations-v2.js` — when served from `/v2/yolovexv2.html`,
    # those resolve to `/spec-data.js` and `/activations-v2.js`, which we mount
    # explicitly so we can disable caching on `/activations-v2.js`.
    app.mount("/v2", StaticFiles(directory=str(V2_DIR)), name="v2")
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

    @app.get("/")
    def index():
        # Redirect so relative script paths inside the HTML (../spec-data.js,
        # arch-v2.jsx, etc.) resolve correctly against /v2/ as the base.
        return RedirectResponse(url="/v2/yolovexv2.html", status_code=307)

    @app.get("/spec-data.js")
    def spec_js():
        return FileResponse(SPEC_JS, media_type="application/javascript")

    @app.get("/activations-v2.js")
    def activations_js():
        # No caching — this file is rewritten on each upload.
        return FileResponse(
            ACTIVATIONS_JS,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "weights": weights,
            "imgsz": imgsz,
            "busy": _get_active() is not None,
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        if _get_active() is not None:
            raise HTTPException(status_code=409, detail="another build is in progress")

        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported extension {suffix!r}; allowed: {sorted(ALLOWED_EXTS)}",
            )

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(data).hexdigest()[:12]
        target = UPLOAD_DIR / f"{digest}{suffix}"
        target.write_bytes(data)

        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, target)
        _record_job(job)
        _set_active(job)

        rel_path = target.relative_to(ROOT)
        t = threading.Thread(
            target=_worker, args=(job, weights, imgsz), daemon=True, name=f"yv-build-{job_id}",
        )
        t.start()
        return {"job_id": job_id, "image": str(rel_path)}

    @app.delete("/api/jobs/{job_id}")
    def cancel(job_id: str):
        job = _find_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        job.cancelled.set()
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str):
        job = _find_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")

        async def stream():
            # Reply quickly with a comment so the client confirms connection.
            yield ": connected\n\n"
            while True:
                try:
                    ev = await asyncio.get_event_loop().run_in_executor(
                        None, job.events.get, True, 1.0,
                    )
                except queue.Empty:
                    if job.done.is_set() and job.events.empty():
                        return
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("kind") in ("done", "error", "cancelled"):
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


def run(host: str = "127.0.0.1", port: int = 8765, weights: str = DEFAULT_WEIGHTS, imgsz: int = 640) -> None:
    if not V2_DIR.exists():
        raise SystemExit(f"frontend/v2 not found at {V2_DIR} — run `yolovex serve` from the repo root")
    if not ACTIVATIONS_JS.exists():
        print(
            f"warning: {ACTIVATIONS_JS} doesn't exist yet — run `uv run yolovex build-assets-v2` "
            f"first, or upload an image once the server is up."
        )
    app = create_app(weights=weights, imgsz=imgsz)
    print(f"yolovex serve → http://{host}:{port}/  (weights={weights}, imgsz={imgsz})")
    uvicorn.run(app, host=host, port=port, log_level="info")
