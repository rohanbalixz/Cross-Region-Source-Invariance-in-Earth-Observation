#!/usr/bin/env bash
# Resumable harder-task acquisition: ~50 S2+WorldCover patches per city.
# Skips cities already done; continues past per-city failures (e.g. MPC
# outages). Safe to re-run until all cities are present.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
CITIES=$($PY -c "from scripts.acquire.regions import CITIES; print(' '.join(c.name for c in CITIES))")
ok=0; fail=0
for c in $CITIES; do
  if $PY -m scripts.acquire.worldcover_s2 --city "$c" --patches 50 >> logs/hardtask_acq.log 2>&1; then
    ok=$((ok+1))
  else
    echo "[$(date +%H:%M:%S)] FAILED $c (will retry on next run)" >> logs/hardtask_acq.log; fail=$((fail+1))
  fi
done
echo "[$(date +%H:%M:%S)] ACQUIRE PASS DONE ok=$ok fail=$fail total_with_patches=$(find data/hardtask -name patches.npz | wc -l | tr -d ' ')" >> logs/hardtask_acq.log
