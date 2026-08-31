"""Compare eager/graph logits, features and state at identical real-model histories."""

import argparse
import json
from pathlib import Path

import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.runtime.hybrid_cache import HybridPrefixCache
from minisgl.speculative.batch import BatchedTargetExecutor
from minisgl.speculative.batch_draft import propose_batch
from minisgl.speculative.draft import DFlashDraft
from minisgl.speculative.target import MiniSGLTarget
from minisgl.utils import load_tokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-numerics", choices=["stable", "fast"], default="stable")
    args = parser.parse_args()
    draft = json.loads((Path(args.draft) / "config.json").read_text())
    taps = tuple(draft["dflash_config"]["target_layer_ids"])
    config = EngineConfig(
        model_path=args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        page_size=256,
        max_seq_len_override=256,
        num_page_override=4,
        prefix_state_budget_bytes=0,
        gdn_extend_backend="packed",
        memory_budget_bytes=24 << 30,
        runtime_workspace_bytes=1 << 30,
    )
    engine = Engine(config)
    try:
        executor = BatchedTargetExecutor(
            engine, taps, 4, cuda_graph=True, target_numerics=args.target_numerics
        )
        torch.cuda.set_stream(engine.stream)
        cache = HybridPrefixCache(512 << 20, 1024 << 20)
        targets = [
            MiniSGLTarget(engine, capture_layer_ids=taps, slot=i, executor=executor, cache=cache)
            for i in range(4)
        ]
        tokenizer = load_tokenizer(args.model)
        prompts = [tokenizer.encode("Explain KV caching. " * (i + 1)) for i in range(4)]
        anchors = executor.prefill(targets, prompts)
        assert executor.prefill_batch_sizes == {4: 1}
        saved = [t.checkpoint() for t in targets]
        saved_features = [torch.cat(t.pending_features).clone() for t in targets]
        saved_kv = {
            (i, lid, kind): t._kv_view(lid, kind)[: t.length].clone()
            for i, t in enumerate(targets)
            for lid in engine.kv_cache._layer_mapping
            for kind in ("k", "v")
        }
        for tier in ("gpu", "cpu"):
            if tier == "cpu":
                cache.resize_gpu_budget(0)
                assert cache.stats["offloads"] == 4
            assert executor.prefill(targets, prompts) == anchors
            for i, target in enumerate(targets):
                assert target.last_cache_event["tier"] == tier
                actual = target.checkpoint()
                for lid, states in actual.states.items():
                    for a, b in zip(states, saved[i].states[lid]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(
                    torch.cat(target.pending_features), saved_features[i], rtol=0, atol=0
                )
            for (i, lid, kind), expected in saved_kv.items():
                torch.testing.assert_close(
                    targets[i]._kv_view(lid, kind)[: targets[i].length], expected, rtol=0, atol=0
                )
        # Native draft model: compare padded batch proposals at identical features.
        model = DFlashDraft.from_directory(args.draft, engine.device, engine.dtype)
        batch_drafts = [model.fork_context() for _ in targets]
        single_drafts = [model.fork_context() for _ in targets]
        rows = [
            (d, saved_features[i].unsqueeze(0), anchors[i], [8, 2, 4, 8][i], targets[i].length)
            for i, d in enumerate(batch_drafts)
        ]
        with torch.cuda.stream(engine.stream):
            expected = [
                d.propose(row[1], row[2], row[3], row[4], targets[0].embedding, targets[0].head)
                for d, row in zip(single_drafts, rows)
            ]
            actual = propose_batch(rows, targets[0].embedding, targets[0].head)
        bf16_matches = sum(
            a == b for ra, rb in zip(actual, expected) for a, b in zip(ra[1:], rb[1:])
        )
        bf16_proposed = sum(len(row) - 1 for row in actual)
        print(json.dumps(dict(bf16_batched=actual, bf16_serial=expected)), flush=True)
        # BF16 GEMM shapes may change draft argmax. Validate the same weights
        # in FP32 separately, rather than requiring BF16 draft-token identity.
        fp32_model = DFlashDraft.from_directory(args.draft, engine.device, torch.float32)
        fp32_batch = [fp32_model.fork_context() for _ in targets]
        fp32_single = [fp32_model.fork_context() for _ in targets]
        with torch.cuda.stream(engine.stream):
            embedding, head = targets[0].embedding.float(), targets[0].head.float()
            fp32_rows = [(d, row[1].float(), *row[2:]) for d, row in zip(fp32_batch, rows)]
            expected32 = [
                d.propose(row[1], row[2], row[3], row[4], embedding, head)
                for d, row in zip(fp32_single, fp32_rows)
            ]
            actual32 = propose_batch(fp32_rows, embedding, head)
            assert actual32 == expected32
            for a, b in zip(fp32_batch, fp32_single):
                for la, lb in zip(a.layers, b.layers):
                    for name in ("cached_k", "cached_v"):
                        torch.testing.assert_close(
                            getattr(la.self_attn, name),
                            getattr(lb.self_attn, name),
                            rtol=2e-4,
                            atol=2e-4,
                        )
        del fp32_model, fp32_batch, fp32_single, fp32_rows, embedding, head
        cases = []
        for slots in [(0, 1, 2, 3), (3, 0, 2), (2,), (1, 3, 0, 2)]:
            before = [t.checkpoint() for t in targets]
            pooled = executor.checkpoint([targets[i] for i in slots])
            items = [(targets[i], [100 + i]) for i in slots]
            executor.graph_enabled = False
            eager = executor.forward(items)
            after_eager = [t.checkpoint() for t in targets]
            kv_eager = {
                (i, lid, kind): targets[i]._kv_view(lid, kind)[targets[i].length - 1].clone()
                for i in slots
                for lid in engine.kv_cache._layer_mapping
                for kind in ("k", "v")
            }
            executor.restore(list(zip([targets[i] for i in slots], pooled)))
            for i, target in enumerate(targets):
                restored = target.checkpoint()
                assert restored.history == before[i].history
                for lid, states in restored.states.items():
                    for a, b in zip(states, before[i].states[lid]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            executor.graph_enabled = True
            graphed = executor.forward(items)
            for (a, fa), (b, fb) in zip(eager, graphed):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(fa, fb, rtol=0, atol=0)
            for i, target in enumerate(targets):
                actual = target.checkpoint()
                expected = after_eager[i] if i in slots else before[i]
                assert actual.history == expected.history
                for lid in actual.states:
                    for a, b in zip(actual.states[lid], expected.states[lid]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            for (i, lid, kind), expected in kv_eager.items():
                torch.testing.assert_close(
                    targets[i]._kv_view(lid, kind)[targets[i].length - 1],
                    expected,
                    rtol=0,
                    atol=0,
                )
            # Returned features must not alias the mutable graph output buffer.
            saved = [(a.clone(), b.clone()) for a, b in graphed]
            executor.forward(items)
            for (a, fa), (b, fb) in zip(graphed, saved):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(fa, fb, rtol=0, atol=0)
            cases.append(
                dict(
                    slots=slots,
                    exact_logits_features_kv_gdn=True,
                    inactive_slots_unchanged=True,
                    output_buffers_independent=True,
                )
            )
        output = dict(
            measured=True,
            gpu=torch.cuda.get_device_name(),
            cases=cases,
            execution=executor.stats(),
            batched_prefill=True,
            exact_gpu_and_host_prefix_restore=True,
            exact_pooled_checkpoint_restore=True,
            real_draft_fp32_ragged_proposal_and_kv_parity=True,
            bf16_draft_token_matches=bf16_matches,
            bf16_draft_tokens=bf16_proposed,
        )
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(json.dumps(output))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
