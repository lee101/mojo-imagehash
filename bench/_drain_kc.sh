#!/usr/bin/env bash
# One-shot: resolve the stale karateclub bench waiters by running their exact
# command once (BENCH_NO_REENTRY makes run.sh execute bench.py in the active
# default env, whose absolute interpreter path the waiters already have), then
# confirm no holder remains. Read-only w.r.t. our repo.
set -uo pipefail
cd /nvme0n1-disk/code/mojo-karateclub || exit 1
echo "== running karateclub bench body once $(date -u +%H:%M:%S) =="
BENCH_NO_REENTRY=1 timeout 900 bash bench/run.sh 2>&1 | tail -n 25
echo "== done $(date -u +%H:%M:%S) =="
for p in $(pgrep -x flock); do
  a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  echo "holder: $p $a"
done
