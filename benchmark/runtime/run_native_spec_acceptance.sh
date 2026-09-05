#!/usr/bin/env bash
# Native batch=4 MTP3 and DFlash acceptance: identical prompts, numerics and GPU.
set -euo pipefail
PYTHON=${1:-python}
OUT=${2:?Usage: bash benchmark/runtime/run_native_spec_acceptance.sh PYTHON FRESH_OUTPUT_DIR [MODEL] [DRAFT]}
MODEL=${3:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B}
DRAFT=${4:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SPEC_OMP_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
COMMON=(-m minisgl.runtime.benchmark --model "$MODEL"
  --workload "$ROOT/benchmark/runtime/workloads/chat-long4.jsonl"
  --batch-size 4 --max-context 4096 --target-numerics stable
  --verify-mode parallel --gdn-extend packed --cuda-graph
  --gpu-cache-mib 0 --host-cache-mib 0 --warmup 2 --repeat 5)
if [[ ${SHOW_TEXT:-0} == 1 ]]; then
  COMMON+=(--show-text)
fi
for name in target mtp3 dflash8; do
  if [[ -e "$OUT/$name.json" || -e "$OUT/$name.log" ]]; then
    echo "Refusing to overwrite $OUT/$name" >&2
    exit 1
  fi
done
for name in target mtp3 dflash8; do
  case "$name" in
    target) MODE=(--mode target);;
    mtp3) MODE=(--mode mtp --mtp-steps 3);;
    dflash8) MODE=(--mode fixed --draft "$DRAFT" --block-size 8);;
  esac
  echo "Running $name; log: $OUT/$name.log"
  "$PYTHON" "${COMMON[@]}" "${MODE[@]}" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
done
"$PYTHON" "$ROOT/benchmark/runtime/compare_native_spec.py" "$OUT/target.json" \
  "$OUT/mtp3.json" "$OUT/dflash8.json" --summary "$OUT/summary.json"
