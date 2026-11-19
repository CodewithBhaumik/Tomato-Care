#!/usr/bin/env bash
# Final round after training (scripts/21 arms in experiments/queue_final.txt), in order; stops at the first error.
#   36  select / calibrate / threshold / unusual-image gate on out-of-fold data  -> final_models.json, deploy/*.pt
#   38  test photos scored ONCE + stress + out-of-dataset; deployment.json then serves the two final pipelines
#   34  every example image through the HTTP API (after)        35  per-condition behaviour audit (after)
#   39  docs/FINAL_REPORT.md      27  platform.json (apps)      29  docs/RESULTS.md      26  weights manifest
# Usage: bash scripts/run_final_round.sh [release-copy-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONPATH="src;."
OUT="${1:-}"
step() { echo "=== $(date +%H:%M:%S) $*"; }
step 36; "$PY" scripts/36_final_pipeline.py
step 38; "$PY" scripts/38_final_test.py
step 34; CUDA_VISIBLE_DEVICES="" "$PY" scripts/34_audit_examples.py
step 35; "$PY" scripts/35_audit_behaviour.py cuda
step 39; CUDA_VISIBLE_DEVICES="" TOMATO_RUNTIME=torch "$PY" scripts/39_final_report.py
step 27; "$PY" scripts/27_export_platform_data.py
step 29; "$PY" scripts/29_results_report.py
step 26; if [ -n "$OUT" ]; then "$PY" scripts/26_package_release.py --copy-to "$OUT"; else "$PY" scripts/26_package_release.py; fi
step done
