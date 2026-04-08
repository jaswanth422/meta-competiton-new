#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:7860}"

echo "[1/3] Checking /health"
curl -fsS "${BASE_URL}/health" >/dev/null

echo "[2/3] Checking /reset"
curl -fsS -X POST "${BASE_URL}/reset" -H "content-type: application/json" -d '{}' >/dev/null

echo "[3/3] Running baseline inference"
python inference.py >/dev/null

echo "Validation passed."
