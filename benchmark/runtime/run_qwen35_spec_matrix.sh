#!/usr/bin/env bash
# Controlled MiniSGLang Qwen3.5 target/DFlash matrix. Run only on a GPU host.
set -euo pipefail
PYTHON=${1:?Usage: bash benchmark/runtime/run_qwen35_spec_matrix.sh PYTHON OUTPUT_DIR [MODEL] [DRAFT]}
OUT=${2:?Provide a fresh output directory}
MODEL=${3:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B}
DRAFT=${4:-/root/autodl-tmp/models/hybrid-runtime/Qwen3.5-4B-DFlash}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKLOAD="$ROOT/benchmark/runtime/workloads/chat-long4.jsonl"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

COMMON=(-m minisgl.runtime.benchmark
  --model "$MODEL" --draft "$DRAFT" --workload "$WORKLOAD"
  --batch-size 4 --verify-mode parallel --gdn-extend packed
  --max-context 2048 --gpu-budget-gib 24
  --gpu-cache-mib 0 --host-cache-mib 0 --warmup 1 --repeat 3)
run_case() {
  local name=$1
  shift
  if [[ -e "$OUT/$name.json" || -e "$OUT/$name.log" ]]; then
    echo "Refusing to overwrite $OUT/$name" >&2
    exit 1
  fi
  echo "RUN $name"
  "$PYTHON" "${COMMON[@]}" "$@" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
}

# Stable arithmetic is the correctness reference used by the native runtime.
run_case target-stable-graph --mode target --target-numerics stable --cuda-graph
run_case dflash-b4-stable-graph --mode fixed --block-size 4 --target-numerics stable --cuda-graph
run_case dflash-b8-stable-graph --mode fixed --block-size 8 --target-numerics stable --cuda-graph
run_case adaptive-stable-graph --mode adaptive --block-size 16 --target-numerics stable --cuda-graph

# Fast BF16 library kernels diagnose whether strict batch invariance is the cost center.
run_case target-fast-graph --mode target --target-numerics fast --cuda-graph
run_case dflash-b4-fast-graph --mode fixed --block-size 4 --target-numerics fast --cuda-graph
run_case dflash-b8-fast-graph --mode fixed --block-size 8 --target-numerics fast --cuda-graph

# Ablations isolate context-KV fusion and target decode CUDA Graph contribution.
run_case dflash-b4-stable-unfused-context --mode fixed --block-size 4 --target-numerics stable --cuda-graph --no-draft-context-kv-fusion
run_case target-stable-eager --mode target --target-numerics stable
run_case dflash-b4-stable-eager --mode fixed --block-size 4 --target-numerics stable

echo "Finished. JSON contains raw prompts, outputs, counters and timings: $OUT"
