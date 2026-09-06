"""Compare the pinned SGLang PR target, MTP and DFlash variants."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def difference(a, b):
    return None if a == b else next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    names = ("pr-target", "pr-mtp", "pr-dflash", "pr-dflash-compatible", "pr-dflash-rope-only")
    data = {n: json.loads((args.directory / (n + ".json")).read_text()) for n in names}
    report = {"source": "SGLang unmerged PR #19952 at 5106aa523aee7a77471fdb50e0cb1e8b21da9427", "date": "2026-09-06", "cases": {}, "artifacts": {}}
    for length in (256, 512):
        cases = {}
        target_records = None
        for name, d in data.items():
            case = next(c for c in d["cases"] if c["batch"] == 4 and c["length"] == length)
            samples = case["samples"]
            assert len(samples) == 5 and d["arguments"]["warmup"] == 2
            records = [r for s in samples for r in s["requests"]]
            assert len(records) == 20 and all(len(r["output_ids"]) == length for r in records)
            assert all(r["meta_info"].get("cached_tokens", 0) == 0 for r in records)
            for key in ("context_length", "disable_radix_cache", "disable_overlap_schedule", "disable_cuda_graph", "disable_piecewise_cuda_graph", "dtype", "model_path", "attention_backend", "max_running_requests", "max_total_tokens", "enable_deterministic_inference", "enable_fp32_lm_head"):
                assert d["engine_arguments"][key] == data["pr-target"]["engine_arguments"][key], key
            assert d["versions"] == data["pr-target"]["versions"] and d["gpu"] == data["pr-target"]["gpu"]
            if target_records is None:
                target_records = records
            assert [r["input_ids"] for r in records] == [r["input_ids"] for r in target_records]
            delta = [difference(a["output_ids"], b["output_ids"]) for a, b in zip(target_records, records)]
            repeated = [difference(a["output_ids"], b["output_ids"]) for a, b in zip(records[:4] * 4, records[4:])]
            rates = [s["output_tokens_per_second"] for s in samples]
            drafted = sum(s["drafted_tokens"] for s in samples)
            accepted = sum(s["accepted_draft_tokens"] for s in samples)
            cases[name] = {"e2e_tok_s": 20 * length / sum(s["wall_seconds"] for s in samples),
                "wave_min": min(rates), "wave_median": statistics.median(rates), "wave_max": max(rates),
                "acceptance_rate": accepted / drafted if drafted else None,
                "drafted_tokens": drafted, "accepted_tokens": accepted,
                "exact_requests_vs_target": delta.count(None), "requests": 20,
                "first_difference_one_based": delta, "repeat_identical": all(x is None for x in repeated),
                "source_revision": d["source_revision"], "versions": d["versions"],
                "imported_sglang_path": d["imported_sglang_path"]}
            cases[name]["speedup_vs_target"] = cases[name]["e2e_tok_s"] / cases["pr-target"]["e2e_tok_s"]
        cases["compatibility_effect"] = {"throughput_ratio": cases["pr-dflash-compatible"]["e2e_tok_s"] / cases["pr-dflash"]["e2e_tok_s"],
            "acceptance_delta": cases["pr-dflash-compatible"]["acceptance_rate"] - cases["pr-dflash"]["acceptance_rate"]}
        report["cases"][str(length)] = cases
    for name in names:
        path = args.directory / (name + ".json")
        report["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["audit_passed"] = True
    report["limitations"] = ["PR was closed without merging; not a current released SGLang result.", "Compatibility variant changes only the draft model's RoPE and attention mask configuration.", "Default BF16 cross-shape token equality is reported, not assumed; no task-quality score.", "Same four raw prompts, repeated five times; long-context window GPU boundary remains untested."]
    (args.directory / "pr-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
