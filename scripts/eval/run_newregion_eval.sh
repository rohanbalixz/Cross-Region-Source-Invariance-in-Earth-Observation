#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
LOG=${LOG:-./logs}
# wait for preprocess to reach 44/44
while [ "$(find data/processed/weur data/processed/latam data/processed/camcar data/processed/canada -name builtup_2015.tif 2>/dev/null|wc -l|tr -d ' ')" -lt 44 ]; do
  pgrep -f build_city_tiffs >/dev/null || break
  sleep 30
done
N=$(find data/processed/{weur,latam,camcar,canada} -name builtup_2015.tif 2>/dev/null|wc -l|tr -d ' ')
echo "[$(date +%H:%M)] preprocess at $N/44; freeing GHSL" >> $LOG/eval.log
[ "$N" -ge 44 ] && { rm -rf data/raw/ghsl; echo "[$(date +%H:%M)] GHSL freed disk=$(df -h .|tail -1|awk '{print $4}')" >> $LOG/eval.log; }
# eval CONUS-trained models on each new city
for c in london paris berlin madrid rome amsterdam vienna barcelona milan munich lisbon sao_paulo rio_de_janeiro buenos_aires brasilia montevideo asuncion belo_horizonte curitiba porto_alegre salvador_br recife mexico_city guatemala_city havana san_salvador panama_city tegucigalpa managua santo_domingo san_jose_cr kingston port_au_prince toronto montreal vancouver calgary ottawa edmonton winnipeg quebec_city hamilton_ca halifax victoria; do
  $PY -m scripts.eval.cross_region_eval --city "$c" >> $LOG/eval.log 2>&1 || echo "EVAL FAIL $c" >> $LOG/eval.log
done
echo "[$(date +%H:%M)] EVAL DONE: $(find data/processed/{weur,latam,camcar,canada} -name eval_metrics.json 2>/dev/null|wc -l|tr -d ' ')/44 new eval_metrics" >> $LOG/eval.log
