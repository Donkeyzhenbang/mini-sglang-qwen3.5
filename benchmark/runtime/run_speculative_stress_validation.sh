#!/usr/bin/env bash
set -euo pipefail
# Accuracy/capacity stress, not a speedup gate. B16 uses a 2048 context cap:
# the conservative DFlash admission estimate rejects B16/C4096 on this 24GB GPU.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
OUT=${1:?fresh output directory required}
if [[ -e "$OUT" ]]; then echo "Refusing existing output directory" >&2; exit 1; fi
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
MODEL=${MODEL:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B}
DRAFT=${DRAFT:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash}
export PYTHONPATH="$PWD/python" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python - "$OUT" "$MODEL" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
out, model = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(x) for x in Path('benchmark/runtime/workloads/graph-regression8.jsonl').read_text().splitlines() if x.strip()]
stress = [dict(rows[i % len(rows)], max_new_tokens=64) for i in range(16)]
(out/'batch16.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in stress))
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
text = ('A cache stores confirmed history. Rejected draft tokens must not change the accepted recurrent state. ' * 400)
ids = tokenizer.encode(text, add_special_tokens=False)[:4352]
assert len(ids) == 4352
(out/'long4352.jsonl').write_text(json.dumps(dict(input_ids=ids, max_new_tokens=64))+'\n')
PY
COMMON=(-m minisgl.runtime.benchmark --model "$MODEL" --target-numerics stable
 --gdn-extend packed --verify-mode parallel --cuda-graph --gpu-cache-mib 0 --host-cache-mib 0
 --warmup 1 --repeat 1)
for case in batch8 batch16 long4352; do
 case "$case" in
   batch8) B=8; C=4096; W="$OUT/batch16.jsonl";;
   batch16) B=16; C=2048; W="$OUT/batch16.jsonl";;
   long4352) B=1; C=8192; W="$OUT/long4352.jsonl";;
 esac
 for mode in target mtp3 dflash8; do
   case "$mode" in
     target) MODE=(--mode target);;
     mtp3) MODE=(--mode mtp --mtp-steps 3);;
     dflash8) MODE=(--mode fixed --draft "$DRAFT" --block-size 8);;
   esac
   echo "Running $case $mode"
   python "${COMMON[@]}" --batch-size "$B" --max-context "$C" --workload "$W" \
      "${MODE[@]}" --output "$OUT/$case-$mode.json" > "$OUT/$case-$mode.log" 2>&1
 done
 python benchmark/runtime/compare_native_spec.py "$OUT/$case-target.json" \
   "$OUT/$case-mtp3.json" "$OUT/$case-dflash8.json" --minimum-speedup 0 \
   --summary "$OUT/summary-$case.json"
done
