"""Run isolated GPU processes and require exact greedy parity in each group.

Each JSON/log is retained, including failures. Model loading and graph capture
are outside the measured generation interval. The suite is sequential on one GPU.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from minisgl.runtime.analyze import check_tokens, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Do not mix a previous run's successful JSON with a new failed process.
    if (out / "suite.json").exists():
        parser.error("Use a fresh output directory; suite.json already exists")
    env = dict(os.environ, PYTHONPATH=str(root / "python"))
    long = root / "benchmark/runtime/workloads/chat-long4.jsonl"
    longer = out / "chat-512.jsonl"
    rows = [dict(json.loads(line), max_new_tokens=512) for line in long.read_text().splitlines()]
    longer.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    short = root / "benchmark/runtime/workloads/chat4.jsonl"
    common = [
        sys.executable,
        "-B",
        "-m",
        "minisgl.runtime.benchmark",
        "--model",
        args.model,
        "--draft",
        args.draft,
        "--max-context",
        "1024",
        "--gdn-extend",
        "packed",
        "--target-numerics",
        "stable",
        "--chat-template",
        "--show-text",
        "--block-size",
        "8",
        "--batch-size",
        "4",
    ]
    results, summaries = {}, {}

    def run(name, workload, mode, *extra):
        path = out / (name + ".json")
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing run: {path}")
        print(f"RUN {name}", flush=True)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                common
                + ["--workload", str(workload), "--mode", mode, "--output", str(path)]
                + list(extra),
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        data = json.loads(path.read_text())
        assert data["execution"]["target_numerics"] == "stable"
        results[name] = data
        summaries[name] = summarize(data)
        return name

    groups = []
    for label, workload in [("long256", long), ("long512", longer)]:
        names = [
            run(label + "-" + mode, workload, mode, "--cuda-graph")
            for mode in ("target", "fixed", "adaptive")
        ]
        if label == "long256":
            names.extend(
                [
                    run(
                        label + "-sequential",
                        workload,
                        "fixed",
                        "--cuda-graph",
                        "--verify-mode",
                        "sequential",
                    ),
                    run(label + "-eager", workload, "fixed"),
                    run(label + "-batch1", workload, "fixed", "--cuda-graph", "--batch-size", "1"),
                    run(
                        label + "-block16", workload, "fixed", "--cuda-graph", "--block-size", "16"
                    ),
                ]
            )
        check_tokens([results[name] for name in names])
        groups.append(dict(group=label, cases=names, exact_greedy_parity=True))
        print(f"PASS {label}", flush=True)

    names = [run("chat8-target", short, "target", "--cuda-graph", "--repeat", "2")]
    names.extend(
        [
            run(
                "chat8-batch8-target",
                short,
                "target",
                "--cuda-graph",
                "--repeat",
                "2",
                "--batch-size",
                "8",
            ),
            run(
                "chat8-batch8-fixed",
                short,
                "fixed",
                "--cuda-graph",
                "--repeat",
                "2",
                "--batch-size",
                "8",
            ),
            run(
                "chat8-continuous",
                short,
                "fixed",
                "--cuda-graph",
                "--repeat",
                "2",
                "--continuous-batching",
            ),
            run(
                "chat8-host-cache",
                short,
                "fixed",
                "--cuda-graph",
                "--repeat",
                "2",
                "--continuous-batching",
                "--gpu-cache-mib",
                "64",
                "--host-cache-mib",
                "1024",
            ),
        ]
    )
    check_tokens([results[name] for name in names])
    host = results["chat8-host-cache"]
    assert host["cache"]["hits"] > 0 and host["cache"]["offloads"] > 0
    assert any(
        event["tier"] == "cpu" for wave in host["waves"] for event in wave["request_cache_events"]
    )
    groups.append(dict(group="chat8", cases=names, exact_greedy_parity=True, cpu_cache_hit=True))
    (out / "suite.json").write_text(
        json.dumps(
            dict(
                measured=True,
                target_numerics="stable",
                groups=groups,
                summaries=summaries,
                measurement_note="Single runs for correctness coverage, not repeated performance estimates",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("PASS: all generation groups have exact greedy parity", flush=True)


if __name__ == "__main__":
    main()
