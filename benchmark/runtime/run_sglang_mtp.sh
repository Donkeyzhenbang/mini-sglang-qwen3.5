#!/usr/bin/env bash
# Reuse an isolated SGLang environment; do not install packages here.
set -euo pipefail
SGLPY=${1:?Usage: bash benchmark/runtime/run_sglang_mtp.sh PYTHON OUTPUT_DIR [MODEL]}
OUT=${2:?Provide a fresh output directory}
MODEL=${3:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
mkdir -p "$OUT/tmp" "$OUT/torchinductor" "$OUT/triton"
# /dev/shm is mounted noexec on the experiment host. JIT shared libraries
# must be on an executable filesystem even if download caches use /dev/shm.
export TMPDIR="$OUT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$OUT/torchinductor"
export TRITON_CACHE_DIR="$OUT/triton"
export TOKENIZERS_PARALLELISM=false
COMMON=("$ROOT/benchmark/runtime/bench_sglang_mtp.py"
  --model "$MODEL" --workload "$ROOT/benchmark/runtime/workloads/chat-long4.jsonl"
  --lengths 256 512 --batches 1 4 --repeats 3)
run_case() {
  local name=$1
  shift
  if [[ -e "$OUT/$name.json" || -e "$OUT/$name.log" ]]; then
    echo "Refusing to overwrite $OUT/$name" >&2
    exit 1
  fi
  echo "RUN $name"
  "$SGLPY" "${COMMON[@]}" "$@" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
}
run_case target --mode target
run_case mtp1 --mode mtp --steps 1
run_case mtp3 --mode mtp --steps 3
echo "Finished. Raw answers, tokens, counters and timings: $OUT"
