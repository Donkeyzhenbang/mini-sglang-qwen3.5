"""Audit matched offline batches and compare wall throughput, not quality scores."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def first_difference(a, b):
    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def parity(a, b):
    if len(a) != len(b):
        raise ValueError("Unequal request counts")
    diffs = [first_difference(x, y) for x, y in zip(a, b)]
    return dict(
        exact_requests=sum(x is None for x in diffs),
        requests=len(diffs),
        first_difference_zero_based=diffs,
        first_difference_one_based=[None if x is None else x + 1 for x in diffs],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    root = args.directory
    sgl = {m: json.loads((root / f"sglang-{m}.json").read_text()) for m in ("target", "mtp")}
    report = dict(
        date="2026-09-06",
        timing="Output tokens / summed measured wall time, including prefill; SGLang also includes Engine IPC.",
        limitations=[
            "MiniSGLang stable and SGLang default BF16 use different target arithmetic.",
            "No task-quality benchmark or random-sampling distribution validation.",
            "Four unique raw prompts repeated five times; fixed lengths 256/512.",
            "SGLang DFlash unmeasured: installed 0.5.9 has no DFlash, downloaded main is incompatible with reused environment.",
        ],
        sglang_versions=sgl["target"]["versions"],
        lengths={},
        artifacts={},
    )
    for length in (256, 512):
        modes = {}
        sg_records = {}
        for mode, data in sgl.items():
            assert data["engine_arguments"]["disable_radix_cache"]
            assert not data["engine_arguments"]["disable_cuda_graph"]
            assert data["engine_arguments"]["disable_overlap_schedule"]
            assert data["engine_arguments"]["context_length"] == 4096
            assert data["arguments"]["warmup"] == 2
            case = next(c for c in data["cases"] if c["batch"] == 4 and c["length"] == length)
            samples = case["samples"]
            assert len(samples) == 5
            records = [r for s in samples for r in s["requests"]]
            assert len(records) == 20 and all(len(r["output_ids"]) == length for r in records)
            assert all(r["meta_info"].get("cached_tokens", 0) == 0 for r in records)
            rates = [s["output_tokens_per_second"] for s in samples]
            drafted = sum(s["drafted_tokens"] for s in samples)
            outputs = [r["output_ids"] for r in records]
            modes[f"sglang_{mode}"] = dict(
                e2e_tok_s=20 * length / sum(s["wall_seconds"] for s in samples),
                wave_e2e_min=min(rates),
                wave_e2e_median=statistics.median(rates),
                wave_e2e_max=max(rates),
                acceptance_rate=sum(s["accepted_draft_tokens"] for s in samples) / drafted
                if drafted
                else None,
                repeated_output_agreement=parity(outputs[:4] * 4, outputs[4:]),
            )
            sg_records[mode] = records
        mini_records = {}
        for mode in ("target", "mtp3", "dflash8"):
            data = json.loads((root / f"mini-{mode}-{length}.json").read_text())
            assert data["arguments"]["target_numerics"] == "stable"
            assert data["arguments"]["warmup"] == 2 and data["arguments"]["repeat"] == 5
            assert data["arguments"]["max_context"] == 4096
            assert data["arguments"]["cuda_graph"] and data["arguments"]["batch_size"] == 4
            assert data["arguments"]["gpu_cache_mib"] == data["arguments"]["host_cache_mib"] == 0
            assert data["gpu"] == sgl["target"]["gpu"]
            assert data["torch"].split("+")[0] == sgl["target"]["versions"]["torch"]
            records = data["requests"]
            assert len(records) == 20 and all(len(r["token_ids"]) == length for r in records)
            for r, s in zip(records, sg_records["target"]):
                assert r["prompt_token_ids"] == s["input_ids"], "Input token mismatch"
            waves = data["waves"]
            assert len(waves) == 5
            assert sum(w["output_tokens"] for w in waves) == 20 * length
            rates = [w["output_tokens"] * 1000 / w["total_ms"] for w in waves]
            outputs = [r["token_ids"] for r in records]
            modes[f"mini_{mode}"] = dict(
                e2e_tok_s=20 * length * 1000 / sum(w["total_ms"] for w in waves),
                decode_tok_s=sum(w["decoded_tokens"] for w in waves)
                * 1000
                / sum(w["decode_ms"] for w in waves),
                wave_e2e_min=min(rates),
                wave_e2e_median=statistics.median(rates),
                wave_e2e_max=max(rates),
                acceptance_rate=data["speculation"]["acceptance_rate"],
                peak_allocated_gib=data["peak_allocated_bytes"] / 2**30,
                peak_reserved_gib=data["peak_reserved_bytes"] / 2**30,
                repeated_output_agreement=parity(outputs[:4] * 4, outputs[4:]),
            )
            mini_records[mode] = records
        for name, metrics in modes.items():
            baseline = "sglang_target" if name.startswith("sglang") else "mini_target"
            metrics["e2e_speedup_vs_own_target"] = (
                metrics["e2e_tok_s"] / modes[baseline]["e2e_tok_s"]
            )
        sg_ids = {m: [r["output_ids"] for r in rows] for m, rows in sg_records.items()}
        mi_ids = {m: [r["token_ids"] for r in rows] for m, rows in mini_records.items()}
        checks = dict(
            sglang_mtp_vs_target=parity(sg_ids["target"], sg_ids["mtp"]),
            mini_mtp_vs_target=parity(mi_ids["target"], mi_ids["mtp3"]),
            mini_dflash_vs_target=parity(mi_ids["target"], mi_ids["dflash8"]),
            mini_target_vs_sglang_target=parity(sg_ids["target"], mi_ids["target"]),
            mini_mtp_vs_sglang_mtp=parity(sg_ids["mtp"], mi_ids["mtp3"]),
        )
        assert checks["mini_mtp_vs_target"]["exact_requests"] == 20
        assert checks["mini_dflash_vs_target"]["exact_requests"] == 20
        for mode in ("target", "mtp"):
            assert [r["input_ids"] for r in sg_records[mode]] == [
                r["input_ids"] for r in sg_records["target"]
            ]
        report["lengths"][str(length)] = dict(
            modes=modes,
            token_checks=checks,
            mini_mtp_vs_sglang_mtp_ratio=modes["mini_mtp3"]["e2e_tok_s"]
            / modes["sglang_mtp"]["e2e_tok_s"],
            input_lengths=[len(r["input_ids"]) for r in sg_records["target"][:4]],
        )
    for path in sorted(root.glob("*.json")):
        if path.name != "framework-summary.json":
            report["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["audit_passed"] = True
    (root / "framework-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "artifacts"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
