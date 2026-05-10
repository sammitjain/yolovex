#!/usr/bin/env bash
set -euo pipefail
SCRATCH="${YOLOVEX_SCRATCH:-/tmp/yolovex-integration}"
rm -rf "$SCRATCH"
git clone https://github.com/sammitjain/yolovex.git "$SCRATCH"
cd "$SCRATCH"
uv sync
uv run yolovex predict   # exercises model load + head viz; downloads yolo26n.pt
