#!/usr/bin/env bash
# Same explicit token IDs, batch four, warmup two, five waves, no prefix reuse.
set -euo pipefail
SGLPY=${1:?Usage: bash benchmark/runtime/run_framework_comparison.sh SGLANG_PYTHON FRESH_OUTPUT [MINI_PYTHON]}
OUT=${2:?Provide a fresh output directory}
PY=${3:-/root/miniconda3/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B}
DRAFT=${DRAFT:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -e "$OUT" ]]; then echo "Refusing existing output directory: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$ROOT"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH="$ROOT/python"
# The output must reside on an executable filesystem, not a noexec /dev/shm.
mkdir -p "$OUT/tmp" "$OUT/triton" "$OUT/torchinductor"
export TMPDIR="$OUT/tmp" TRITON_CACHE_DIR="$OUT/triton" TORCHINDUCTOR_CACHE_DIR="$OUT/torchinductor"
"$PY" - "$MODEL" "$OUT" <<'PY'
import json,sys
from pathlib import Path
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(sys.argv[1])
rows=[json.loads(s) for s in Path('benchmark/runtime/workloads/chat-long4.jsonl').read_text().splitlines() if s.strip()]
assert len(rows)==4
for r in rows: r['input_ids']=r.get('input_ids') or tok.encode(r['prompt'])
out=Path(sys.argv[2])
for length in (256,512):
 for r in rows: r['max_new_tokens']=length
 (out/f'inputs-{length}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
PY
for mode in target mtp; do
 "$SGLPY" benchmark/runtime/bench_sglang_mtp.py --model "$MODEL" --workload "$OUT/inputs-256.jsonl" --mode "$mode" --steps 3 --lengths 256 512 --batches 4 --warmup 2 --repeats 5 --context-length 4096 --output "$OUT/sglang-$mode.json" > "$OUT/sglang-$mode.log" 2>&1
done
for length in 256 512; do
 for mode in target mtp3 dflash8; do
  case "$mode" in
   target) MODE=(--mode target);;
   mtp3) MODE=(--mode mtp --mtp-steps 3);;
   dflash8) MODE=(--mode fixed --block-size 8 --draft "$DRAFT");;
  esac
  "$PY" -m minisgl.runtime.benchmark --model "$MODEL" --workload "$OUT/inputs-$length.jsonl" --batch-size 4 --max-context 4096 --target-numerics stable --verify-mode parallel --gdn-extend packed --cuda-graph --gpu-cache-mib 0 --host-cache-mib 0 --warmup 2 --repeat 5 "${MODE[@]}" --output "$OUT/mini-$mode-$length.json" > "$OUT/mini-$mode-$length.log" 2>&1
 done
 "$PY" benchmark/runtime/compare_native_spec.py "$OUT/mini-target-$length.json" "$OUT/mini-mtp3-$length.json" "$OUT/mini-dflash8-$length.json" --summary "$OUT/native-summary-$length.json"
done
"$PY" benchmark/runtime/compare_frameworks.py "$OUT"
