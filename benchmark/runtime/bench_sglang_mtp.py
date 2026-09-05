"""Isolated SGLang target/MTP comparison; does not import MiniSGLang.

Run each mode in a fresh process using the same environment. Timing includes
prefill, decode and Engine IPC, but excludes model loading and graph capture.
Fixed length ignores EOS deliberately; answers may be truncated or continue
past EOS. All raw token IDs and speculative counters are retained.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("target", "mtp"), required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fp32-lm-head", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Use a fresh output path")
    if min(args.batches + args.lengths + [args.repeats, args.steps]) < 1:
        parser.error("Sizes must be positive")
    # Spec-v2 is opt-in in SGLang 0.5.9. Do not silently let an inherited
    # environment change the scheduling policy of a controlled comparison.
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "true" if args.overlap else "false"
    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    rows = [
        json.loads(line) for line in Path(args.workload).read_text().splitlines() if line.strip()
    ]
    if max(args.batches) > len(rows):
        parser.error("Batch size exceeds workload size")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = [
        r.get("input_ids")
        or tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in rows
    ]
    kwargs = dict(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        context_length=args.context_length,
        max_total_tokens=8192,
        max_running_requests=8,
        mem_fraction_static=0.75,
        attention_backend=args.attention_backend,
        disable_radix_cache=True,
        disable_overlap_schedule=not args.overlap,
        disable_cuda_graph=args.eager,
        cuda_graph_bs=[1, 2, 4, 8],
        chunked_prefill_size=1024,
        random_seed=42,
        skip_tokenizer_init=True,
        enable_multimodal=False,
        log_level="info",
        enable_deterministic_inference=args.deterministic,
        enable_fp32_lm_head=args.fp32_lm_head,
    )
    if args.mode == "mtp":
        kwargs.update(
            speculative_algorithm="NEXTN",
            speculative_draft_model_path=args.model,
            speculative_num_steps=args.steps,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=args.steps + 1,
        )
    result = dict(
        arguments=vars(args),
        engine_arguments=kwargs,
        gpu=torch.cuda.get_device_name(),
        versions={
            p: importlib.metadata.version(p)
            for p in ["torch", "sglang", "sgl-kernel", "transformers", "flashinfer-python"]
        },
        timing="Engine.generate wall time including prefill and IPC; warmup excluded",
        fixed_length_ignore_eos=True,
        input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        benchmark_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        cases=[],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = sgl.Engine(**kwargs)
    try:
        for batch in args.batches:
            prompts = ids[:batch]
            for length in args.lengths:
                params = dict(temperature=0.0, max_new_tokens=length, ignore_eos=True)
                # Warm exactly this batch and output length before measuring.
                for _ in range(args.warmup):
                    engine.generate(input_ids=prompts, sampling_params=params)
                samples = []
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    answers = engine.generate(input_ids=prompts, sampling_params=params)
                    elapsed = time.perf_counter() - start
                    if not isinstance(answers, list):
                        answers = [answers]
                    assert len(answers) == batch
                    accepted = drafted = verifies = completions = 0
                    records = []
                    for index, answer in enumerate(answers):
                        meta = answer["meta_info"]
                        tokens = answer.get("output_ids")
                        if tokens is None:
                            raise RuntimeError("Engine did not return output token IDs")
                        assert len(tokens) == length, (len(tokens), length)
                        assert meta["completion_tokens"] == length, meta
                        accepted += meta.get("spec_accept_token_num", 0)
                        drafted += meta.get("spec_draft_token_num", 0)
                        verifies += meta.get("spec_verify_ct", 0)
                        completions += meta["completion_tokens"]
                        records.append(
                            dict(
                                prompt=rows[index].get("prompt", tokenizer.decode(prompts[index])),
                                input_ids=prompts[index],
                                output_ids=tokens,
                                text=tokenizer.decode(tokens, skip_special_tokens=True),
                                meta_info=meta,
                            )
                        )
                    samples.append(
                        dict(
                            repeat=repeat,
                            wall_seconds=elapsed,
                            output_tokens_per_second=completions / elapsed,
                            accepted_draft_tokens=accepted,
                            drafted_tokens=drafted,
                            verify_rounds=verifies,
                            acceptance_rate=accepted / drafted if drafted else None,
                            progress_tokens_per_verify=completions / verifies if verifies else None,
                            requests=records,
                        )
                    )
                case = dict(
                    batch=batch,
                    length=length,
                    samples=samples,
                    median_output_tokens_per_second=statistics.median(
                        s["output_tokens_per_second"] for s in samples
                    ),
                )
                result["cases"].append(case)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(json.dumps({k: v for k, v in case.items() if k != "samples"}), flush=True)
        result["server_info"] = engine.get_server_info()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
