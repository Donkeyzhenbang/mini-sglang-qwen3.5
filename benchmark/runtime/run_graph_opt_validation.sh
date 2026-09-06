#!/usr/bin/env bash
# Native graph regression: same raw prompts, all output token IDs retained.
set -euo pipefail
PY=${1:-python}
OUT=${2:?Usage: bash benchmark/runtime/run_graph_opt_validation.sh PYTHON FRESH_OUTPUT_DIR}
MODEL=${MODEL:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B}
DRAFT=${DRAFT:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -e "$OUT" ]]; then echo "Refusing existing output: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$ROOT/python" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(s) for s in Path('benchmark/runtime/workloads/chat-long4.jsonl').read_text().splitlines() if s.strip()]
out = Path(sys.argv[1])
for length in (256, 512):
    (out / f'inputs-{length}.jsonl').write_text(''.join(json.dumps(dict(r, max_new_tokens=length), ensure_ascii=False)+'\n' for r in rows))
mixed = [dict(rows[i % 4], max_new_tokens=(1,17,73,129)[i % 4]) for i in range(12)]
(out / 'inputs-ragged.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in mixed))
PY
COMMON=(-m minisgl.runtime.benchmark --model "$MODEL" --batch-size 4
  --max-context 4096 --target-numerics stable --verify-mode parallel
  --gdn-extend packed --cuda-graph --gpu-cache-mib 0 --host-cache-mib 0 --warmup 2)
run_case() {
  local label=$1
  shift
  echo "Running $label"
  "$PY" "${COMMON[@]}" "$@" --output "$OUT/$label.json" > "$OUT/$label.log" 2>&1
}
for length in 256 512; do
  INPUT=(--workload "$OUT/inputs-$length.jsonl" --repeat 5)
  run_case "target-$length" "${INPUT[@]}" --mode target
  run_case "mtp3-$length" "${INPUT[@]}" --mode mtp --mtp-steps 3
  run_case "dflash8-$length" "${INPUT[@]}" --mode fixed --draft "$DRAFT" --block-size 8
  "$PY" benchmark/runtime/compare_native_spec.py "$OUT/target-$length.json" \
    "$OUT/mtp3-$length.json" "$OUT/dflash8-$length.json" --summary "$OUT/summary-$length.json"
done
for block in 4 16; do
  run_case "dflash$block-256" --workload "$OUT/inputs-256.jsonl" --repeat 5 \
    --mode fixed --draft "$DRAFT" --block-size "$block"
done
"$PY" benchmark/runtime/compare_native_spec.py "$OUT/target-256.json" \
  "$OUT/dflash4-256.json" "$OUT/dflash16-256.json" --summary "$OUT/summary-blocks.json"
for mode in target mtp1 mtp3 dflash8; do
  case "$mode" in
    target) MODE=(--mode target);;
    mtp1) MODE=(--mode mtp --mtp-steps 1);;
    mtp3) MODE=(--mode mtp --mtp-steps 3);;
    dflash8) MODE=(--mode fixed --draft "$DRAFT" --block-size 8);;
  esac
  run_case "$mode-ragged" --workload "$OUT/inputs-ragged.jsonl" --repeat 1 \
    --continuous-batching "${MODE[@]}"
done
"$PY" benchmark/runtime/compare_native_spec.py "$OUT/target-ragged.json" \
  "$OUT/mtp1-ragged.json" "$OUT/mtp3-ragged.json" "$OUT/dflash8-ragged.json" \
  --summary "$OUT/summary-ragged.json"
