"""Compare measured runs, refusing mismatched workloads or token outputs."""

import argparse
import json
from pathlib import Path


def summarize(data):
    requests = data["requests"]
    output_tokens = sum(len(r["token_ids"]) for r in requests)
    decoded = sum(max(0, len(r["token_ids"]) - 1) for r in requests)
    decode_ms = sum(r["decode_ms"] for r in requests)
    total_ms = decode_ms + sum(r["ttft_ms"] for r in requests)
    if data.get("waves"):
        decode_ms = sum(w["decode_ms"] for w in data["waves"])
        total_ms = sum(w["total_ms"] for w in data["waves"])
    rounds = [x for r in requests for x in r["rounds"]]
    return dict(
        mode=data["mode"],
        execution=data.get("execution"),
        throughput_time_basis="wave wall time" if data.get("waves") else "serial request time",
        requests=len(requests),
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens * 1000 / max(total_ms, 1e-9),
        decode_tokens_per_second=decoded * 1000 / max(decode_ms, 1e-9),
        mean_ttft_ms=sum(r["ttft_ms"] for r in requests) / max(len(requests), 1),
        aggregate_tpot_ms=decode_ms / max(decoded, 1),
        mean_progress_per_round=sum(x["progress"] for x in rounds) / max(len(rounds), 1),
        draft_ms=sum(x["draft_ms"] for x in rounds),
        verify_ms=sum(x["verify_ms"] for x in rounds),
        restore_ms=sum(x["restore_ms"] for x in rounds),
        peak_allocated_bytes=data["peak_allocated_bytes"],
        cache=data["cache"],
    )


def check_tokens(runs):
    baseline = runs[0]
    if not all(r.get("measured") is True for r in runs):
        raise ValueError("Only measured GPU runs may be compared")
    for run in runs[1:]:
        for key in ("workload_sha256", "target_config_sha256", "gpu"):
            if not baseline.get(key) or baseline[key] != run.get(key):
                raise ValueError(f"Runs differ in {key}")
        if [r["token_ids"] for r in baseline["requests"]] != [
            r["token_ids"] for r in run["requests"]
        ]:
            raise ValueError("Greedy token parity failed; performance comparison is invalid")
    return True


def compare(runs):
    check_tokens(runs)
    return [summarize(run) for run in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument(
        "--tokens-only", action="store_true", help="Also accepts HF reference output"
    )
    args = parser.parse_args()
    runs = [json.loads(Path(p).read_text()) for p in args.runs]
    print(
        json.dumps(
            {"token_parity": check_tokens(runs)} if args.tokens_only else compare(runs), indent=2
        )
    )


if __name__ == "__main__":
    main()
