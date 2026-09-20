"""Profile one warmed speculative wave; timings are diagnostic, not benchmark results.

Example (two warmup waves):
  PYTHONPATH=python python benchmark/runtime/profile_speculative.py \
    --profile-dir /tmp/profile-new --profile-wave 3 -- \
    --model MODEL --draft DRAFT --mode fixed --block-size 8 --batch-size 4 \
    --workload WORKLOAD --warmup 2 --repeat 1 --cuda-graph --output RESULT.json
"""

import argparse
import cProfile
import functools
import json
import pstats
import runpy
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import minisgl.speculative.batch_loop as loop
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--profile-wave", default=3, type=int)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.profile_wave < 1:
        parser.error("--profile-wave must be positive")
    if not args.benchmark_args or args.benchmark_args[0] != "--":
        parser.error("put runtime benchmark arguments after --")
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    original = loop.generate_batch
    calls = 0
    captured = False

    def annotate(name, method):
        @functools.wraps(method)
        def wrapped(*a, **kw):
            with torch.profiler.record_function("speculative::" + name):
                return method(*a, **kw)

        return wrapped

    @functools.wraps(original)
    def profiled(*a, **kw):
        nonlocal calls, captured
        calls += 1
        if calls != args.profile_wave:
            return original(*a, **kw)
        executor = kw.get("executor", a[4] if len(a) > 4 else None)
        cpu = cProfile.Profile()
        with ExitStack() as stack:
            for name in (
                "prefill",
                "feasible_blocks",
                "propose",
                "checkpoint",
                "verify",
                "restore",
                "commit_verify_states",
            ):
                if hasattr(executor, name):
                    stack.enter_context(
                        patch.object(executor, name, annotate(name, getattr(executor, name)))
                    )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                cpu.enable()
                try:
                    result = original(*a, **kw)
                finally:
                    cpu.disable()
        cpu.dump_stats(str(args.profile_dir / "cpu.prof"))
        with (args.profile_dir / "cpu.txt").open("w") as out:
            pstats.Stats(cpu, stream=out).strip_dirs().sort_stats("cumtime").print_stats(80)
        (args.profile_dir / "kernels.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
        )
        prof.export_chrome_trace(str(args.profile_dir / "trace.json"))
        (args.profile_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "wave": calls,
                    "benchmark_args": args.benchmark_args[1:],
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "diagnostic_only": True,
                    "note": "Includes profiler overhead. Use separate unprofiled runs for performance gates.",
                },
                indent=2,
            )
        )
        # Keep timing provenance next to the wave, not only in the trace manifest.
        result[1]["profiled"] = True
        captured = True
        return result

    with (
        patch.object(loop, "generate_batch", profiled),
        patch.object(sys, "argv", ["minisgl.runtime.benchmark", *args.benchmark_args[1:]]),
    ):
        runpy.run_module("minisgl.runtime.benchmark", run_name="__main__")
    if not captured:
        raise RuntimeError(f"Requested wave {args.profile_wave}, but only {calls} ran")


if __name__ == "__main__":
    main()
