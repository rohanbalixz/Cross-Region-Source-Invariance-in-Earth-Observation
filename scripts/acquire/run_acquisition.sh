#!/usr/bin/env bash
# End-to-end raw-data acquisition driver for the cross-region benchmark.
#
# Idempotent: every step skips data that already exists, so previously
# acquired cities are never re-fetched. Robust: a single OSM download failure
# (e.g. a Geofabrik path that differs for one country) does not abort the run.
# Each step logs with timestamps under ${LOG_DIR}.
#
# Usage:  bash scripts/acquire/run_acquisition.sh [region ...]
#         (no args -> the four scale-up regions: mena sea eeca oceania)
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
REGIONS=("${@:-mena sea eeca oceania}")
# shellcheck disable=SC2206
REGIONS=(${REGIONS[*]})

ts()   { date "+%H:%M:%S"; }
step() { echo "[$(ts)] === $* ==="; }

EPOCHS="--epoch 1975 --epoch 1980 --epoch 1985 --epoch 1990 --epoch 1995 --epoch 2000 --epoch 2005 --epoch 2010 --epoch 2015"
LAYERS="--layer BUILT_S --layer BUILT_V --layer POP"

step "GHSL settlement layers (${REGIONS[*]})"
for R in "${REGIONS[@]}"; do
  echo "[$(ts)] ghsl ${R}"
  $PY -m scripts.acquire.ghsl --region "$R" $LAYERS $EPOCHS > "$LOG_DIR/ghsl_${R}.log" 2>&1
  echo "[$(ts)] ghsl ${R} exit=$?  zips=$(find data/raw/ghsl -path "*${R}*" -name '*.zip' | wc -l | tr -d ' ')"
done

step "Decompress GHSL archives in place (and remove zips to save disk)"
find data/raw/ghsl -name '*.zip' -execdir unzip -o -q {} \; 2>/dev/null
find data/raw/ghsl -name '*.zip' -delete 2>/dev/null
echo "[$(ts)] decompress done; tifs=$(find data/raw/ghsl -name '*.tif' | wc -l | tr -d ' ')"

step "Copernicus DEM (all registered cities; skips existing)"
$PY -m scripts.acquire.srtm > "$LOG_DIR/srtm.log" 2>&1
echo "[$(ts)] srtm exit=$?  city_tifs=$(ls data/raw/srtm/*.tif 2>/dev/null | wc -l | tr -d ' ')"

step "OpenStreetMap road graphs (all registered cities; skips existing)"
$PY -m scripts.acquire.osm_pbf > "$LOG_DIR/osm.log" 2>&1
echo "[$(ts)] osm exit=$?  graphmls=$(ls data/raw/osm/*.graphml 2>/dev/null | wc -l | tr -d ' ')"

step "ACQUISITION COMPLETE"
echo "[$(ts)] GHSL_tifs=$(find data/raw/ghsl -name '*.tif' | wc -l | tr -d ' ')  DEM=$(ls data/raw/srtm/*.tif 2>/dev/null | wc -l | tr -d ' ')  OSM=$(ls data/raw/osm/*.graphml 2>/dev/null | wc -l | tr -d ' ')"
