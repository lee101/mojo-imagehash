#!/usr/bin/env bash
# Acquire the shared machine benchmark lock, then run this repo's immutable
# bench.py, then the stage splitter, while holding it. Output goes beside the
# repo in /tmp so nothing but bench/analysis.md is added to the tree.
set -uo pipefail
REPO=/nvme0n1-disk/code/mojo-imagehash
cd "$REPO" || exit 1
exec 9>>/tmp/mojo-bench.lock
if ! flock -w 10800 9; then
  echo "LOCK_TIMEOUT $(date -u +%FT%TZ)" | tee /tmp/mh_analysis_measure.out
  exit 2
fi
{
  echo "LOCK_ACQUIRED $(date -u +%FT%TZ)"
  echo "=== canonical: pixi run bench (timeout 600) ==="
  timeout 600 pixi run bench
  echo "BENCH_EXIT=$?"
  echo "=== stage split ==="
  timeout 1800 pixi run python -u bench/_analysis_stage.py
  echo "STAGE_EXIT=$?"
  echo "=== per-thread cpu time ==="
  timeout 1800 pixi run python -u bench/_analysis_cpu.py
  echo "CPU_EXIT=$?"
  echo "LOCK_RELEASED $(date -u +%FT%TZ)"
} > /tmp/mh_analysis_measure.out 2>&1
exit 0
