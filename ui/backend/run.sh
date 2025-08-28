#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
uvicorn recmatcher_ui.app.main:app --host 127.0.0.1 --port 8787 --reload
