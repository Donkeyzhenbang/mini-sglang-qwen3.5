#!/usr/bin/env bash
# Reuse the experiment host's Torch/CUDA stack. This is not a general SGLang
# installation: only text BF16 Qwen3.5 target/MTP inference is validated.
set -euo pipefail
DEST=${1:?Usage: bash benchmark/runtime/setup_sglang_mtp_env.sh NEW_ENV [BASE_PYTHON]}
BASE=${2:-/root/miniconda3/bin/python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$DEST" ]]; then
  echo "Refusing to modify existing environment: $DEST" >&2
  exit 1
fi
"$BASE" - <<'PY'
import importlib.metadata as m
import sys
import torch
assert sys.version_info[:2] == (3, 12), "Expected the tested Python 3.12 base"
assert torch.__version__.split('+')[0] == '2.9.1', "Expected Torch 2.9.1"
assert torch.version.cuda == '12.8', "Expected the CUDA 12.8 Torch build"
assert m.version('sgl-kernel') == '0.3.21', "Expected sgl-kernel 0.3.21"
assert m.version('transformers') == '4.57.3', "Expected Transformers 4.57.3"
assert torch.cuda.is_available(), "GPU is required"
PY
uv venv --system-site-packages --python "$BASE" "$DEST"
# --no-deps is deliberate: dependency resolution would pull a second Torch
# and CUDA stack instead of reusing system site-packages.
uv pip install --python "$DEST/bin/python" --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r "$SCRIPT_DIR/sglang-mtp-reuse.txt"
"$DEST/bin/python" "$SCRIPT_DIR/patch_sglang_059_mrope.py"
"$DEST/bin/python" - <<'PY'
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
print('SGLang text target/MTP imports passed; run the GPU benchmark next.')
PY
