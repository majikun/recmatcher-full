#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# 端口可改，默认 9777
uvicorn server:app --host 127.0.0.1 --port 9777