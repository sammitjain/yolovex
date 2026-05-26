"""Prototype-only server for trying attention maps on uploaded images.

This intentionally stays outside `yolovex serve` while the attention UI is a
throwaway prototype. It serves `frontend/` and adds one upload endpoint that
returns the same payload shape as `scripts/export_attention_json.py`.

Usage:
  uv run python scripts/serve_attention_prototype.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

sys.path.insert(0, str(ROOT / "scripts"))

from export_attention_json import build_payload  # noqa: E402
from yolovex.model import DEFAULT_WEIGHTS  # noqa: E402


def create_app(weights: str, imgsz: int, jpeg_quality: int) -> FastAPI:
    app = FastAPI(title="yolovex attention prototype", docs_url=None, redoc_url=None)

    @app.get("/attention-prototype-data.js")
    def prototype_data():
        return FileResponse(
            FRONTEND_DIR / "attention-prototype-data.js",
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/attention-prototype/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported extension {suffix!r}; allowed: {sorted(ALLOWED_EXTS)}",
            )

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")

        digest = hashlib.sha1(data).hexdigest()[:12]
        scratch = Path(tempfile.mkdtemp(prefix="yolovex-attention-"))
        target = scratch / f"{digest}{suffix}"
        target.write_bytes(data)

        payload = build_payload(
            target,
            imgsz=imgsz,
            weights=weights,
            jpeg_quality=jpeg_quality,
            source_label=file.filename or target.name,
        )
        return payload

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--jpeg-quality", type=int, default=84)
    args = parser.parse_args()

    app = create_app(args.weights, args.imgsz, args.jpeg_quality)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
