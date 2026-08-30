"""Reproducible single-GPU target / fixed-DFlash / adaptive-DFlash experiments.

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
from dataclasses import asdict
from pathlib import Path


def checkpoint_bytes(folder, dtype_bytes=2):
    """Inspect safetensors headers without loading tensors or executing model code."""
    total, keys = 0, set()
    for path in Path(folder).glob('*.safetensors'):
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            if length > 16 << 20:
                raise ValueError('Unreasonably large safetensors header')
            header = json.loads(f.read(length))
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if name in keys:
                raise ValueError(f'Duplicate weight: {name}')
            keys.add(name)
            total += math.prod(tensor['shape']) * dtype_bytes
    if not keys:
        raise ValueError(f'No safetensors weights found in {folder}')
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local Qwen3.5 target directory')
    p.add_argument('--draft', help='Local DFlash v1 directory; no remote code is executed')
    p.add_argument('--mode', choices=['target', 'fixed', 'adaptive'], default='target')
    p.add_argument('--workload', required=True, help='JSONL with input_ids or prompt, max_new_tokens')
    p.add_argument('--output', required=True)
    p.add_argument('--block-size', type=int, choices=[1, 2, 4, 8, 16], default=16)
    p.add_argument('--max-context', type=int, default=4096)
    p.add_argument('--gpu-budget-gib', type=float, default=24)
    p.add_argument('--gpu-cache-mib', type=int, default=0)
    p.add_argument('--host-cache-mib', type=int, default=0)
    p.add_argument('--cache-policy', choices=['cost', 'lru'], default='cost')
    p.add_argument('--gdn-extend', choices=['recurrent', 'packed'], default='recurrent')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.mode != 'target' and not args.draft:
        p.error('--draft is required for speculative modes')
    if min(args.max_context, args.block_size, args.gpu_budget_gib) <= 0 or min(
            args.gpu_cache_mib, args.host_cache_mib, args.warmup) < 0:
        p.error('Invalid budget, context or block size')
    import torch
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.runtime.adaptive import AdaptiveBlockController
    from minisgl.runtime.hybrid_cache import HybridPrefixCache
    from minisgl.speculative.draft import DFlashDraft
    from minisgl.speculative.loop import generate
    from minisgl.speculative.target import MiniSGLTarget
    from minisgl.utils import load_tokenizer
    if not torch.cuda.is_available():
        raise SystemExit('GPU_UNAVAILABLE: CPU policy tests are available under tests/cpu; no benchmark was run.')
    torch.manual_seed(args.seed)
    draft_config = json.loads((Path(args.draft) / 'config.json').read_text()) if args.mode != 'target' else None
    taps = tuple(draft_config['dflash_config']['target_layer_ids']) if draft_config else ()
    external = checkpoint_bytes(args.draft) if draft_config else 0
    config = EngineConfig(model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend='fi', cuda_graph_max_bs=0, page_size=256,
        max_seq_len_override=args.max_context, num_page_override=math.ceil(args.max_context / 256),
        prefix_state_budget_bytes=0, external_memory_bytes=external + (args.gpu_cache_mib << 20),
        gdn_extend_backend=args.gdn_extend,
        memory_budget_bytes=int(args.gpu_budget_gib * 2**30), runtime_workspace_bytes=1 << 30)
    if config.model_config.model_type not in ('qwen3_5_text', 'qwen3_5'):
        raise ValueError('Experimental runner supports dense Qwen3.5 text targets only')
    if getattr(config.hf_config, 'quantization_config', None):
        raise ValueError('INT4/AWQ/GPTQ target loading is not implemented; use a BF16 target')
    if draft_config:
        if draft_config['hidden_size'] != config.model_config.hidden_size or draft_config['vocab_size'] != config.model_config.vocab_size:
            raise ValueError('Draft/target hidden size or vocabulary mismatch')
        if args.block_size > draft_config['dflash_config']['block_size']:
            raise ValueError('Block size exceeds checkpoint training size')
    engine = Engine(config)
    try:
        draft = DFlashDraft.from_directory(args.draft, engine.device, torch.bfloat16) if draft_config else None
        tokenizer = load_tokenizer(args.model)
        cache = HybridPrefixCache(args.gpu_cache_mib << 20, args.host_cache_mib << 20, policy=args.cache_policy)
        target = MiniSGLTarget(engine, capture_layer_ids=taps, cache=cache,
                              budget_bytes=int(args.gpu_budget_gib * 2**30))
        controller = AdaptiveBlockController(tuple(b for b in (1, 2, 4, 8, 16) if b <= args.block_size)) if args.mode == 'adaptive' else None
        workload_bytes = Path(args.workload).read_bytes()
        rows = [json.loads(s) for s in workload_bytes.decode('utf-8').splitlines() if s.strip()]
        if not rows:
            raise ValueError('Empty workload')
        def run(row):
            ids = row.get('input_ids')
            if ids is None:
                ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
            count = int(row.get('max_new_tokens', 64))
            if len(ids) + count > args.max_context:
                raise ValueError('Workload exceeds --max-context')
            blocks = (1,) if args.mode == 'target' else tuple(b for b in (1, 2, 4, 8, 16) if b <= args.block_size)
            return generate(target, draft, ids, count,
                block_size=1 if args.mode == 'target' else args.block_size,
                adaptive=controller, eos_token_id=tokenizer.eos_token_id,
                feasible=lambda context: target.feasible_blocks(context, blocks))
        for _ in range(args.warmup):
            run(rows[0])
        cache.clear()
        cache.resize_gpu_budget(args.gpu_cache_mib << 20)
        # Warmup is excluded from controller history and reported counters.
        cache.stats = dict(hits=0, misses=0, offloads=0, evictions=0, recompute_choices=0, transfer_ms=0.0)
        if controller:
            controller.observations.clear()
            controller.rounds = 0
        target.memory_events.clear()
        torch.cuda.reset_peak_memory_stats(engine.device)
        results = [asdict(run(row)) for row in rows]
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        output = dict(schema_version=1, measured=True, mode=args.mode, arguments=vars(args),
            workload_sha256=hashlib.sha256(workload_bytes).hexdigest(),
            target_config_sha256=hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest(),
            revision=revision, python=platform.python_version(), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(engine.device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(engine.device),
            cache=cache.stats, memory_events=target.memory_events, requests=results)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(json.dumps({k: output[k] for k in ('mode', 'gpu', 'peak_allocated_bytes', 'cache')}))
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
