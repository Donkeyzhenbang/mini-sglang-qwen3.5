"""Independent Qwen3.5 greedy oracle; run in a separate Transformers environment.

Do not upgrade the mini-SGLang environment in place. The reference environment
must expose transformers.Qwen3_5ForConditionalGeneration. This script measures
whole generation wall time only, not TTFT or per-token timing.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("GPU_UNAVAILABLE: reference inference was not run")
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if cls is None:
        raise SystemExit(
            "Use an isolated Transformers environment with Qwen3.5 support; do not replace the runtime environment."
        )
    model = cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    from transformers.models.qwen3_5 import modeling_qwen3_5

    operators = {
        name: (
            f"{op.__module__}.{op.__name__}"
            if (op := getattr(modeling_qwen3_5, name, None))
            else None
        )
        for name in (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
        )
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    data = Path(args.workload).read_bytes()
    results = []
    with torch.inference_mode():
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids = row.get("input_ids")
            if ids is None:
                ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            x = torch.tensor([ids], device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            y = model.generate(
                x,
                do_sample=False,
                max_new_tokens=row.get("max_new_tokens", 64),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            results.append(
                dict(
                    token_ids=y[0, len(ids) :].tolist(),
                    total_ms=(time.perf_counter() - start) * 1000,
                )
            )
    output = dict(
        measured=True,
        mode="hf_reference",
        transformers=transformers.__version__,
        gdn_operators=operators,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        workload_sha256=hashlib.sha256(data).hexdigest(),
        target_config_sha256=hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()
        ).hexdigest(),
        requests=results,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
