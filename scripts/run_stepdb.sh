#!/usr/bin/env bash
# Run PCL-lite's single + iterative agentic flow end-to-end against a
# benchcard CSV. Tested against StepDB-derived task YAMLs under
# StepDB/pcl_lite_tasks/ with the gpt-oss-120b config; pass a different
# CSV / model to target other suites.
#
# Usage:
#   scripts/run_stepdb.sh [--benchcard PATH] [--model NAME] [--tag TAG]
#                         [--samples N] [--max-tokens N] [--start-from STAGE]
#                         [--no-iter]
#
# Defaults match the gpt-oss-120b smoke run; see flags below.

set -euo pipefail

# --- defaults -----------------------------------------------------------
BENCHCARD="experiments/benchcard_stepdb_transformer_subparts.csv"
MODEL="gpt-oss-120b"
TAG=""                              # auto-derived from benchcard if unset
SAMPLES=4
TEMPERATURE=0.7
MAX_TOKENS=200000
NUM_GROUPS=1
OUT_ROOT="/workspace/pcl_run"
PYTHON="${PYTHON:-python}"          # honor pre-existing venv, default `python`
START_FROM="single"                 # "single" or "iter"
RUN_ITER=1

# --- arg parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchcard)   BENCHCARD="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --samples)     SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --num-groups)  NUM_GROUPS="$2"; shift 2 ;;
        --out-root)    OUT_ROOT="$2"; shift 2 ;;
        --python)      PYTHON="$2"; shift 2 ;;
        --start-from)  START_FROM="$2"; shift 2 ;;     # "single" | "iter"
        --no-iter)     RUN_ITER=0; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Resolve PCL-lite project root from this script's location, regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STEPBASE="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$STEPBASE:${PYTHONPATH:-}"

# Resolve benchcard relative to project root if not absolute.
if [[ "$BENCHCARD" != /* ]]; then
    BENCHCARD="$STEPBASE/$BENCHCARD"
fi
[[ -f "$BENCHCARD" ]] || { echo "benchcard not found: $BENCHCARD" >&2; exit 1; }

# Default tag = benchcard stem (e.g. "benchcard_stepdb_smoke").
if [[ -z "$TAG" ]]; then
    TAG="$(basename "$BENCHCARD" .csv)"
fi

RUN_DIR="$OUT_ROOT/$TAG"
SINGLE_BASE="$RUN_DIR/single"
ITER_BASE="$RUN_DIR/iter"
SINGLE_CSV="$RUN_DIR/single_result.csv"
ITER_STAGE="$STEPBASE/experiments/$TAG"
mkdir -p "$RUN_DIR" "$ITER_STAGE"

echo "=== run_stepdb.sh ==="
echo "  STEPBASE   = $STEPBASE"
echo "  benchcard  = $BENCHCARD"
echo "  model      = $MODEL"
echo "  tag        = $TAG"
echo "  samples    = $SAMPLES   temperature=$TEMPERATURE   max_tokens=$MAX_TOKENS"
echo "  run dir    = $RUN_DIR"
echo "  iter stage = $ITER_STAGE"
echo

# --- stage 1: single pass (baseline) ------------------------------------
if [[ "$START_FROM" == "single" ]]; then
    echo "[1/2] single/main.py — baseline ($SAMPLES samples/task)"
    "$PYTHON" "$STEPBASE/experiments/single/main.py" \
        --model_name "$MODEL" \
        --base_path "$SINGLE_BASE" \
        --input_csv "$BENCHCARD" \
        --output_csv "$SINGLE_CSV" \
        --example_path "$STEPBASE/prompts/proposer_base.yaml" \
        --num_samples "$SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS"
    echo "  -> $SINGLE_CSV"
    echo
fi

[[ -f "$SINGLE_CSV" ]] || { echo "baseline csv missing: $SINGLE_CSV (rerun without --start-from iter)" >&2; exit 1; }

# Stage the baseline result where iterative/main.py expects to find it.
SEED_CSV="$ITER_STAGE/result_${MODEL}_merged_0.csv"
cp "$SINGLE_CSV" "$SEED_CSV"

# --- stage 2: iterative refinement --------------------------------------
if [[ "$RUN_ITER" -eq 0 ]]; then
    echo "skipping iterative (--no-iter)"
    exit 0
fi

# Iterative needs at least one task with success>0 to seed the hard-pool.
SEED_PASSES="$("$PYTHON" - <<EOF
import pandas as pd
print((pd.read_csv("$SEED_CSV")["success"] > 0).sum())
EOF
)"
if [[ "$SEED_PASSES" -eq 0 ]]; then
    echo "WARNING: no tasks passed in the baseline — iterative loop has no hard-pool to draw from and will raise."
    echo "         Either bump --samples / --max-tokens or skip iterative with --no-iter."
    exit 1
fi

echo "[2/2] iterative/main.py — refining ($SEED_PASSES task(s) seed the hard-pool)"
"$PYTHON" "$STEPBASE/experiments/iterative/main.py" \
    --model_name "$MODEL" \
    --model_nickname "$MODEL" \
    --base_path "$ITER_BASE" \
    --base_date "$TAG" \
    --num_samples "$SAMPLES" \
    --num_groups "$NUM_GROUPS" \
    --start_iter 0 \
    --max_tokens "$MAX_TOKENS" \
    --ref_csv "$BENCHCARD"

# Report final cumulative result.
echo
echo "=== done ==="
ls "$ITER_STAGE"/result_"${MODEL}"_merged_*.csv | sort -V | tail -1 | xargs -I{} echo "final csv: {}"
