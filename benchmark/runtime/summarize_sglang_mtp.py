"""Summarize measured SGLang runs and audit token agreement, without hiding differences."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(target, candidates):
    baseline = {(c["batch"], c["length"]): c for c in target["cases"]}
    rows = []
    for label, run in [("target", target), *candidates]:
        # Scheduling may deliberately differ in a separate overlap experiment.
        for key in ("model_path", "dtype", "context_length", "disable_radix_cache"):
            if run["engine_arguments"][key] != target["engine_arguments"][key]:
                raise ValueError(f"Incomparable {key}: {label}")
        if run["versions"] != target["versions"] or run["gpu"] != target["gpu"]:
            raise ValueError(f"Incomparable environment: {label}")
        run_cases = {(c["batch"], c["length"]): c for c in run["cases"]}
        for case in run["cases"]:
            batch, length = case["batch"], case["length"]
            ref = baseline[batch, length]
            reference = ref["samples"][0]["requests"]
            accepted = drafted = verifies = total = equal = 0
            repeat_equal = repeat_total = 0
            mismatches = []
            rates = []
            for sample in case["samples"]:
                requests = sample["requests"]
                if len(requests) != batch or len(reference) != batch:
                    raise ValueError("Incomplete batch")
                for index, (request, expected) in enumerate(zip(requests, reference)):
                    actual, want = request["output_ids"], expected["output_ids"]
                    if sample["repeat"]:
                        repeat_total += 1
                        repeat_equal += (
                            actual
                            == case["samples"][0]["requests"][index]["output_ids"]
                        )
                    if request["input_ids"] != expected["input_ids"]:
                        raise ValueError("Prompt token mismatch")
                    if len(actual) != length or len(want) != length:
                        raise ValueError("Output length mismatch")
                    if request["meta_info"].get("cached_tokens", 0):
                        raise ValueError("Unexpected prefix cache hit")
                    same = actual == want
                    equal += same
                    total += 1
                    if not same:
                        pos = next(
                            i for i, pair in enumerate(zip(actual, want)) if pair[0] != pair[1]
                        )
                        mismatches.append(
                            dict(
                                repeat=sample["repeat"],
                                request=index,
                                first_token=pos,
                                target_token=want[pos],
                                candidate_token=actual[pos],
                            )
                        )
                accepted += sample["accepted_draft_tokens"]
                drafted += sample["drafted_tokens"]
                verifies += sample["verify_rounds"]
                rates.append(batch * length / sample["wall_seconds"])
            median = statistics.median(rates)
            shape_reference = run_cases.get((1, length))
            shape_match = shape_first_difference = None
            if batch != 1 and shape_reference is not None:
                one = shape_reference["samples"][0]["requests"][0]["output_ids"]
                many = case["samples"][0]["requests"][0]["output_ids"]
                shape_match = one == many
                if not shape_match:
                    shape_first_difference = next(
                        i for i, pair in enumerate(zip(one, many)) if pair[0] != pair[1]
                    )
            rows.append(
                dict(
                    mode=label,
                    batch=batch,
                    output_length=length,
                    median_output_tokens_per_second=median,
                    min_output_tokens_per_second=min(rates),
                    max_output_tokens_per_second=max(rates),
                    speedup_vs_target=median / ref["median_output_tokens_per_second"],
                    accepted_draft_tokens=accepted,
                    drafted_tokens=drafted,
                    acceptance_rate=accepted / drafted if drafted else None,
                    verify_rounds=verifies,
                    completion_tokens_per_verify=batch * length * len(rates) / verifies
                    if verifies
                    else None,
                    exact_token_matches=equal,
                    compared_outputs=total,
                    repeat_stable_matches=repeat_equal,
                    repeat_stable_compared_outputs=repeat_total,
                    batch1_first_request_match=shape_match,
                    batch1_first_difference=shape_first_difference,
                    mismatches=mismatches,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    for pair in args.candidate:
        label, path = pair.split("=", 1)
        candidates.append((label, json.loads(Path(path).read_text())))
    rows = summarize(json.loads(Path(args.target).read_text()), candidates)
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("mode batch length tok/s speedup accept exact")
    for row in rows:
        rate = row["acceptance_rate"]
        acceptance = "-" if rate is None else f"{rate:.1%}"
        print(
            f"{row['mode']} {row['batch']} {row['output_length']} "
            f"{row['median_output_tokens_per_second']:.2f} {row['speedup_vs_target']:.3f}x "
            f"{acceptance} {row['exact_token_matches']}/{row['compared_outputs']}"
        )


if __name__ == "__main__":
    main()
