"""Compare native benchmark wall throughput and exact greedy token outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def metrics(data):
    waves = data["waves"]
    seconds = sum(w["decode_ms"] for w in waves) / 1000
    total_seconds = sum(w["total_ms"] for w in waves) / 1000
    if seconds <= 0 or total_seconds <= 0:
        raise ValueError("Benchmark must contain positive measured wave wall times")
    rates = [w["decoded_tokens"] / w["decode_ms"] * 1000 for w in waves if w["decode_ms"] > 0]
    output_tokens = sum(len(r["token_ids"]) for r in data["requests"])
    if output_tokens != sum(w["output_tokens"] for w in waves):
        raise ValueError("Request and wave output token accounting disagree")
    return dict(
        requests=len(data["requests"]),
        output_tokens=output_tokens,
        decode_tok_s=sum(w["decoded_tokens"] for w in waves) / seconds,
        e2e_tok_s=output_tokens / total_seconds,
        wave_decode_min=min(rates),
        wave_decode_median=statistics.median(rates),
        wave_decode_max=max(rates),
        acceptance_rate=data["speculation"]["acceptance_rate"],
        peak_allocated_gib=data["peak_allocated_bytes"] / 2**30,
        peak_reserved_gib=data["peak_reserved_bytes"] / 2**30,
    )


def compare(reference, candidate):
    for key in ("workload_sha256", "target_config_sha256", "gpu", "torch"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Incompatible result metadata: {key}")
    for key in (
        "model",
        "batch_size",
        "target_numerics",
        "cuda_graph",
        "max_context",
        "gpu_cache_mib",
        "host_cache_mib",
        "repeat",
        "chat_template",
        "seed",
    ):
        if reference["arguments"].get(key) != candidate["arguments"].get(key):
            raise ValueError(f"Incompatible benchmark argument: {key}")
    if len(reference["requests"]) != len(candidate["requests"]):
        raise ValueError("Different measured request counts")
    mismatches = []
    for i, (expected, actual) in enumerate(zip(reference["requests"], candidate["requests"])):
        if expected["prompt_token_ids"] != actual["prompt_token_ids"]:
            raise ValueError(f"Different input tokens for request {i}")
        a, b = expected["token_ids"], actual["token_ids"]
        if a != b:
            position = next(
                (j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
            )
            mismatches.append(
                dict(
                    request=i,
                    first_difference=position,
                    expected_length=len(a),
                    actual_length=len(b),
                )
            )
    base, result = metrics(reference), metrics(candidate)
    result.update(
        exact_token_match=not mismatches,
        mismatches=mismatches,
        decode_speedup=result["decode_tok_s"] / base["decode_tok_s"],
        e2e_speedup=result["e2e_tok_s"] / base["e2e_tok_s"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()
    target = json.loads(args.target.read_text())
    if target["mode"] != "target":
        parser.error("Reference must be a target-only run")
    report = dict(target=str(args.target), target_metrics=metrics(target), candidates={})
    passed = True
    for path in args.candidates:
        result = compare(target, json.loads(path.read_text()))
        report["candidates"][path.stem] = result
        passed &= result["exact_token_match"] and result["decode_speedup"] >= args.minimum_speedup
    report["passed"] = passed
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.summary:
        args.summary.write_text(serialized + "\n")
    print(serialized)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
