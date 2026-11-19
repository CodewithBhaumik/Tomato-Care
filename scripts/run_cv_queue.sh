#!/usr/bin/env bash
# Sequential GPU queue for the upgrade round's cross-validation runs (resumable: finished folds are skipped).
# Usage: bash scripts/run_cv_queue.sh <task> <recipe> [<recipe> ...]
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
task="$1"; shift
for recipe in "$@"; do
  echo "=== $(date +%H:%M:%S) $task $recipe"
  "$PY" scripts/21_train_cv.py --recipe "$recipe" --task "$task" || echo "!!! $recipe failed"
done
echo "=== $(date +%H:%M:%S) queue done"
