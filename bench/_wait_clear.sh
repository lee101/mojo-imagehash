#!/usr/bin/env bash
# Wait (read-only, never acquires the lock) until no other process holds
# /tmp/mojo-bench.lock, then print a timestamp. Used to queue this session's
# measurements behind the foreign waiter without joining its flock convoy.
lock=/tmp/mojo-bench.lock
while true; do
  holders=""
  for p in $(pgrep -x flock); do
    if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "flock $lock"; then
      holders="$holders $p"
    fi
  done
  busy=""
  for p in 4187564 4191945 20880 20601 20758 4189301 4190384 4190813; do
    if ps -p "$p" >/dev/null 2>&1; then busy="$busy $p"; fi
  done
  if [ -z "$holders$busy" ]; then
    echo "CLEAR $(date -u +%H:%M:%S)"
    exit 0
  fi
  sleep 3
done
