"""NVTX and cudaProfilerApi instrumentation for Nsight Systems (no Torch Profiler).

Run under nsys. For --scope wave use --capture-range=cudaProfilerApi and
--capture-range-end=stop; for --scope all omit capture-range. Benchmark arguments
go after --. Every observed wave is marked profiled to exclude throughput claims.
"""

import argparse
import functools
import json
import runpy
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import minisgl.speculative.batch_loop as loop
import torch
from minisgl.engine.engine import Engine
from minisgl.speculative.target import MiniSGLTarget


def annotate(name, method):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        with torch.cuda.nvtx.range(name):
            return method(*args, **kwargs)

    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("wave", "all"), default="wave")
    parser.add_argument("--capture-wave", type=int, default=3)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.capture_wave < 1:
        parser.error("--capture-wave must be positive")
    if not args.benchmark_args or args.benchmark_args[0] != "--":
        parser.error("put runtime benchmark arguments after --")
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    original = loop.generate_batch
    calls = 0
    captured = []

    @functools.wraps(original)
    def wave(*a, **kw):
        nonlocal calls
        calls += 1
        selected = args.scope == "all" or calls == args.capture_wave
        gated = args.scope == "wave" and selected
        if gated:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(f"inference/wave_{calls}"), ExitStack() as stack:
                executor = kw.get("executor", a[4] if len(a) > 4 else None)
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
                            patch.object(
                                executor,
                                name,
                                annotate("speculative/" + name, getattr(executor, name)),
                            )
                        )
                result = original(*a, **kw)
                if selected:
                    # Close the range only after its GPU work has completed.
                    torch.cuda.synchronize()
        finally:
            if gated:
                torch.cuda.profiler.stop()
        # Even unselected waves run inside an instrumented process; do not
        # accidentally compare their throughput with an unprofiled benchmark.
        result[1]["profiled"] = True
        if selected:
            captured.append(calls)
        return result

    with (
        patch.object(loop, "generate_batch", wave),
        patch.object(Engine, "__init__", annotate("runtime/engine_init", Engine.__init__)),
        patch.object(
            MiniSGLTarget,
            "synchronize",
            annotate("speculative/synchronize", MiniSGLTarget.synchronize),
        ),
        patch.object(sys, "argv", ["minisgl.runtime.benchmark", *args.benchmark_args[1:]]),
        torch.cuda.nvtx.range("runtime/main"),
    ):
        runpy.run_module("minisgl.runtime.benchmark", run_name="__main__")
    if not captured:
        raise RuntimeError(f"Requested wave {args.capture_wave}, but only {calls} ran")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "scope": args.scope,
                "captured_waves": captured,
                "total_waves": calls,
                "benchmark_args": args.benchmark_args[1:],
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "diagnostic_only": True,
                "note": "NVTX + Nsight Systems only. Use separate unprofiled runs for throughput.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
