"""Reproducible Qwen3.5 target, DFlash and native MTP experiments.

python -m minisgl.runtime.benchmark --help
Requires a real GPU; refuses to label CPU simulation as inference throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
from dataclasses import asdict, replace
from pathlib import Path


def checkpoint_bytes(folder, dtype_bytes=2, prefixes=None):
    """Inspect safetensors headers without loading tensors or executing model code."""
    total, keys = 0, set()
    for path in Path(folder).glob("*.safetensors"):
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if length > 16 << 20:
                raise ValueError("Unreasonably large safetensors header")
            header = json.loads(f.read(length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if prefixes and not name.startswith(tuple(prefixes)):
                continue
            if name in keys:
                raise ValueError(f"Duplicate weight: {name}")
            keys.add(name)
            total += math.prod(tensor["shape"]) * dtype_bytes
    if not keys:
        raise ValueError(f"No safetensors weights found in {folder}")
    return total


def validate_model_pair(target_config, draft_config, block_size):
    if draft_config.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("Only DFlash v1 draft models are supported")
    if (
        draft_config["hidden_size"] != target_config.hidden_size
        or draft_config["vocab_size"] != target_config.vocab_size
        or draft_config["num_target_layers"] != target_config.num_layers
    ):
        raise ValueError("Draft/target hidden size, vocabulary or depth mismatch")
    d = draft_config["dflash_config"]
    if not 1 <= block_size <= d["block_size"]:
        raise ValueError("Block size exceeds checkpoint training size")
    if not 0 <= d["mask_token_id"] < target_config.vocab_size:
        raise ValueError("Mask token is outside the target vocabulary")
    if any(i < 0 or i >= target_config.num_layers - 1 for i in d["target_layer_ids"]):
        raise ValueError("Unsupported target hidden-state tap")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Qwen3.5 target directory")
    p.add_argument(
        "--draft", help="Local DFlash v1 directory; no remote code is executed"
    )
    p.add_argument(
        "--mode",
        choices=["target", "fixed", "adaptive", "mtp"],
        default="target",
    )
    p.add_argument(
        "--mtp-steps",
        type=int,
        choices=[1, 3],
        default=3,
        help="Native Qwen3.5 MTP recurrent proposal steps",
    )
    p.add_argument(
        "--workload", required=True, help="JSONL with input_ids or prompt, max_new_tokens"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--block-size", type=int, choices=[1, 2, 4, 8, 16], default=16)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--gpu-budget-gib", type=float, default=24)
    p.add_argument("--gpu-cache-mib", type=int, default=0)
    p.add_argument("--host-cache-mib", type=int, default=0)
    p.add_argument("--cache-policy", choices=["cost", "lru"], default="cost")
    p.add_argument("--gdn-extend", choices=["recurrent", "packed"], default="recurrent")
    p.add_argument(
        "--target-numerics",
        choices=["stable", "fast"],
        default="stable",
        help=(
            "Stable uses fixed target reductions and FP32 logits; "
            "fast uses legacy BF16 library kernels"
        ),
    )
    p.add_argument(
        "--verify-mode",
        choices=["parallel", "sequential"],
        default="parallel",
        help=(
            "Sequential uses one-token forwards for diagnostics; "
            "fast numerics can vary with batch shape"
        ),
    )
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Opt into wave batching, including size 1 for eager/graph comparisons",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture target one-token decode only; draft/multi-token verify stay eager",
    )
    p.add_argument(
        "--draft-context-kv-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project all DFlash context K/V layers with one packed GEMM",
    )
    p.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Refill completed request slots between rounds; offline experimental scheduler",
    )
    p.add_argument(
        "--repeat", type=int, default=1, help="Repeat the entire workload, retaining cache"
    )
    p.add_argument(
        "--chat-template",
        action="store_true",
        help="Format prompt strings as user chats, with thinking disabled",
    )
    p.add_argument("--show-text", action="store_true", help="Print prompt, answer and cache status")
    args = p.parse_args()
    if args.mode in ("fixed", "adaptive") and not args.draft:
        p.error("--draft is required for DFlash modes")
    if args.mode == "mtp" and args.draft:
        p.error("--draft is not used by embedded Qwen3.5 MTP")
    if (
        min(args.max_context, args.block_size, args.gpu_budget_gib) <= 0
        or min(args.gpu_cache_mib, args.host_cache_mib, args.warmup) < 0
        or args.repeat < 1
        or (args.batch_size is not None and not 1 <= args.batch_size <= 16)
    ):
        p.error("Invalid budget, context or block size")
    import torch
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.runtime.adaptive import AdaptiveBlockController
    from minisgl.runtime.hybrid_cache import HybridPrefixCache
    from minisgl.runtime.speculation_stats import format_speculation, speculation_stats
    from minisgl.runtime.workload import describe_result, prepare_workload, print_result
    from minisgl.speculative.draft import DFlashDraft
    from minisgl.speculative.loop import generate
    from minisgl.speculative.mtp import Qwen3_5MTPDraft
    from minisgl.speculative.target import MiniSGLTarget
    from minisgl.utils import load_tokenizer

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU_UNAVAILABLE: CPU policy tests are available under tests/cpu; no benchmark was run."
        )
    torch.manual_seed(args.seed)
    batch_size = args.batch_size or 1
    use_batched = (
        args.batch_size is not None
        or args.cuda_graph
        or args.continuous_batching
        or args.target_numerics == "stable"
    )
    tokenizer = load_tokenizer(args.model)
    workload_bytes = Path(args.workload).read_bytes()
    rows, workload_hash = prepare_workload(
        workload_bytes,
        tokenizer,
        repeat=args.repeat,
        chat_template=args.chat_template,
    )
    if any(len(row["input_ids"]) + row["max_new_tokens"] > args.max_context for row in rows):
        p.error("Workload exceeds --max-context")
    draft_config = (
        json.loads((Path(args.draft) / "config.json").read_text())
        if args.mode in ("fixed", "adaptive")
        else None
    )
    taps = (
        tuple(draft_config["dflash_config"]["target_layer_ids"])
        if draft_config
        else ()
    )
    capture_final_hidden = args.mode == "mtp"
    external = checkpoint_bytes(args.draft) if draft_config else 0
    if capture_final_hidden:
        external += checkpoint_bytes(args.model, prefixes=("mtp.",))
    spec_block_size = args.mtp_steps + 1 if capture_final_hidden else args.block_size
    draft_fusion_buffer_bytes = 0
    if draft_config and args.draft_context_kv_fusion:
        draft_fusion_buffer_bytes = (
            draft_config["num_hidden_layers"]
            * 2
            * draft_config["num_key_value_heads"]
            * draft_config["head_dim"]
            * draft_config["hidden_size"]
            * 2
        )
        external += draft_fusion_buffer_bytes
    config = EngineConfig(
        model_path=args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=batch_size,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        page_size=256,
        max_seq_len_override=args.max_context,
        num_page_override=math.ceil(args.max_context / 256) * batch_size,
        prefix_state_budget_bytes=0,
        external_memory_bytes=external + (args.gpu_cache_mib << 20),
        gdn_extend_backend=args.gdn_extend,
        memory_budget_bytes=int(args.gpu_budget_gib * 2**30),
        runtime_workspace_bytes=1 << 30,
    )
    if config.model_config.model_type not in ("qwen3_5_text", "qwen3_5"):
        raise ValueError("Experimental runner supports dense Qwen3.5 text targets only")
    if getattr(config.hf_config, "quantization_config", None):
        raise ValueError(
            "INT4/AWQ/GPTQ target loading is not implemented; use a BF16 target"
        )
    c = config.model_config
    if capture_final_hidden:
        hf_text = getattr(config.hf_config, "text_config", config.hf_config)
        mtp_layers = (
            hf_text.get("mtp_num_hidden_layers", 0)
            if isinstance(hf_text, dict)
            else getattr(hf_text, "mtp_num_hidden_layers", 0)
        )
        if mtp_layers != 1:
            raise ValueError("Target checkpoint does not contain one-layer Qwen3.5 MTP")
    # Conservative activation/feature reservation, not a measured peak or hard
    # allocator cap. Larger contexts must pass admission before model execution.
    activation = (
        args.max_context
        * (
            4 * c.hidden_size
            + 3 * c.intermediate_size
            + 3
            * (len(taps) + int(capture_final_hidden))
            * c.hidden_size
        )
        * 2
    )
    draft_context = 0
    if draft_config:
        draft_context = (
            args.max_context
            * draft_config["num_hidden_layers"]
            * draft_config["num_key_value_heads"]
            * draft_config["head_dim"]
            * 4
        )
    elif capture_final_hidden:
        # Persistent MTP K/V plus a temporary recursive proposal chain.
        draft_context = (
            args.max_context
            * c.num_kv_heads
            * c.head_dim
            * 4
        )
    config = replace(
        config,
        runtime_workspace_bytes=max(
            1 << 30, (activation + 2 * draft_context) * batch_size + (256 << 20)
        ),
    )
    if draft_config:
        validate_model_pair(config.model_config, draft_config, args.block_size)
    engine = Engine(config)
    try:
        if draft_config:
            draft = DFlashDraft.from_directory(
                args.draft,
                engine.device,
                torch.bfloat16,
                fuse_context_kv=args.draft_context_kv_fusion,
            )
        elif capture_final_hidden:
            draft = Qwen3_5MTPDraft.from_directory(
                args.model,
                engine.device,
                torch.bfloat16,
                max_steps=args.mtp_steps,
                max_position=args.max_context,
            )
        else:
            draft = None
        cache = HybridPrefixCache(
            args.gpu_cache_mib << 20, args.host_cache_mib << 20, policy=args.cache_policy
        )
        cache_enabled = bool(args.gpu_cache_mib or args.host_cache_mib)
        if not cache_enabled:
            print(
                "Prefix cache disabled: both cache budgets are 0; misses are not counted.",
                flush=True,
            )
        executor = None
        if use_batched:
            from minisgl.speculative.batch import BatchedTargetExecutor
            from minisgl.speculative.batch_loop import generate_batch

            executor = BatchedTargetExecutor(
                engine,
                taps,
                batch_size,
                cuda_graph=args.cuda_graph,
                target_numerics=args.target_numerics,
                capture_final_hidden=capture_final_hidden,
            )
            executor.batching = "continuous offline refill" if args.continuous_batching else "wave"
        targets = [
            MiniSGLTarget(
                engine,
                capture_layer_ids=taps,
                capture_final_hidden=capture_final_hidden,
                cache=cache if cache_enabled else None,
                budget_bytes=int(args.gpu_budget_gib * 2**30),
                verify_mode=args.verify_mode,
                slot=slot,
                executor=executor,
            )
            for slot in range(batch_size)
        ]
        target = targets[0]
        drafts = [draft] + [draft.fork_context() if draft else None for _ in range(batch_size - 1)]
        controller = (
            AdaptiveBlockController(tuple(b for b in (1, 2, 4, 8, 16) if b <= args.block_size))
            if args.mode == "adaptive"
            else None
        )

        def run(row):
            ids = row.get("input_ids")
            if ids is None:
                ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            count = int(row.get("max_new_tokens", 64))
            if len(ids) + count > args.max_context:
                raise ValueError("Workload exceeds --max-context")
            blocks = (
                (1,)
                if args.mode == "target"
                else (
                    tuple(range(1, spec_block_size + 1))
                    if capture_final_hidden
                    else tuple(
                        b
                        for b in (1, 2, 4, 8, 16)
                        if b <= spec_block_size
                    )
                )
            )
            return generate(
                target,
                draft,
                ids,
                count,
                block_size=1 if args.mode == "target" else spec_block_size,
                adaptive=controller,
                eos_token_id=tokenizer.eos_token_id,
                feasible=lambda context: target.feasible_blocks(context, blocks),
            )

        def wave(group):
            n = min(len(group), batch_size)
            return generate_batch(
                targets[:n],
                drafts[:n],
                [r["input_ids"] for r in group],
                [r["max_new_tokens"] for r in group],
                executor,
                block_size=1 if args.mode == "target" else spec_block_size,
                adaptive=controller,
                eos_token_id=tokenizer.eos_token_id,
                sequential=args.verify_mode == "sequential",
            )

        for _ in range(args.warmup):
            wave(rows[:batch_size]) if executor else run(rows[0])
        cache.clear()
        cache.resize_gpu_budget(args.gpu_cache_mib << 20)
        # Warmup is excluded from controller history and reported counters.
        cache.stats = dict(
            hits=0, misses=0, offloads=0, evictions=0, recompute_choices=0, transfer_ms=0.0
        )
        if controller:
            controller.observations.clear()
            controller.rounds = 0
        for target in targets:
            target.memory_events.clear()
        target = targets[0]
        if executor:
            executor.reset_stats()
        torch.cuda.reset_peak_memory_stats(engine.device)
        results, waves = [], []
        group_size = len(rows) if args.continuous_batching else batch_size
        for start in range(0, len(rows), group_size):
            group = rows[start : start + group_size]
            if executor:
                generated, timing = wave(group)
                waves.append(timing)
            else:
                generated = [run(group[0])]
            for offset, (row, generated_row) in enumerate(zip(group, generated)):
                result = describe_result(
                    asdict(generated_row),
                    row,
                    tokenizer,
                    timing["request_cache_events"][offset]
                    if executor
                    else targets[offset].last_cache_event,
                )
                results.append(result)
                if args.show_text:
                    print_result(len(results) - 1, result)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision = None
            dirty = None
        output = dict(
            schema_version=1,
            measured=True,
            mode=args.mode,
            arguments=vars(args),
            workload_sha256=workload_hash,
            source_workload_sha256=hashlib.sha256(workload_bytes).hexdigest(),
            target_config_sha256=hashlib.sha256(
                (Path(args.model) / "config.json").read_bytes()
            ).hexdigest(),
            revision=revision,
            git_dirty=dirty,
            python=platform.python_version(),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(engine.device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(engine.device),
            cache=cache.stats,
            cache_enabled=cache_enabled,
            draft_fusion_buffer_bytes=draft_fusion_buffer_bytes,
            native_mtp_steps=args.mtp_steps if capture_final_hidden else None,
            speculative_block_size=(
                spec_block_size if args.mode != "target" else 1
            ),
            execution=executor.stats()
            if executor
            else dict(
                batch_size=1,
                batching="serial legacy",
                cuda_graph_enabled=False,
                graph_replays=0,
            ),
            waves=waves,
            memory_events=[
                dict(slot=t.slot, **event) for t in targets for event in t.memory_events
            ],
            requests=results,
            speculation=speculation_stats(r for result in results for r in result["rounds"]),
        )
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(format_speculation(output["speculation"]), flush=True)
        print(
            json.dumps(
                {
                    k: output[k]
                    for k in (
                        "mode",
                        "gpu",
                        "peak_allocated_bytes",
                        "cache_enabled",
                        "cache",
                        "execution",
                        "speculation",
                    )
                }
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
