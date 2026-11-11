#!/bin/bash
# Author: tbindas
# What it does: runs the checks of probes/tests/ and prints PASSED / FAILED per check.
#   PY=.venv/bin/python GPU=0 RUNS=/mnt/ssd1/data/cartplus/runs/<new dir> STAGES="data params smoke" bash probes/tests/run_all.sh
# Stages: data (CPU) loader and metric code against the water_loss store and code; params (CPU) parameter table;
# smoke (GPU) two 50-iteration runs of CaRT+ on the first 50 gauges with seed 111111, then the checks: finite loss,
# no skipped step, same loss in both runs, water balance, test-period scores, metric code on the run.
# RUNS must be a path that does not exist (nothing under /mnt/ssd1 is overwritten).
# Experimental purpose: one command that shows the CaRT+ scaffolding on the dHBV MERIT setup is correct before a full run.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}; GPU=${GPU:-0}; K=${K:-50}; ITERS=${ITERS:-50}
STAGES=${STAGES:-"data params smoke"}
export PYTHONDONTWRITEBYTECODE=1
has() { [[ " $STAGES " == *" $1 "* ]]; }
fail=0
run() { echo "## $*"; "$@" 2>&1 | grep -v "^SUMMARY\|_POSIX_C_SOURCE\|^In file included\|^ *[0-9]* | \|^      | \|note: this is the location\|^<command-line>\|^compilation terminated"; [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1; }
has data && run $PY probes/tests/check_merit_data.py
has params && run $PY probes/tests/check_params.py
if has smoke; then
  : "${RUNS:?set RUNS to a new directory for the smoke runs}"
  [ -e "$RUNS" ] && { echo "$RUNS exists; pass a new path"; exit 1; }
  A=$RUNS/smoke_K${K}_i${ITERS}_a; B=$RUNS/smoke_K${K}_i${ITERS}_b
  echo "## GPU memory before: $(nvidia-smi --id=$GPU --query-gpu=memory.used,memory.total --format=csv,noheader)"
  run $PY probes/train_merit.py --K $K --iters $ITERS --gpu $GPU --out $A
  run $PY probes/train_merit.py --K $K --iters $ITERS --gpu $GPU --out $B
  run $PY probes/tests/check_smoke.py --run $A --same $B
  run $PY probes/check_balance_merit.py --run $A --gpu $GPU
  run $PY probes/eval_merit.py --run $A
  run $PY probes/tests/check_merit_data.py --run $A
fi
echo "## done: $([ $fail -eq 0 ] && echo 'all checks passed' || echo 'some checks FAILED')"
exit $fail
