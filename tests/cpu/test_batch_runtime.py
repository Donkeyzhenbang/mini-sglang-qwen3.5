import json

import pytest
from minisgl.engine.graph import _determine_cuda_graph_bs
from minisgl.runtime.adaptive import AdaptiveBlockController
from minisgl.runtime.analyze import summarize
from minisgl.runtime.workload import prepare_workload
from minisgl.speculative.batch_loop import generate_batch
from minisgl.speculative.loop import generate
from test_speculative_loop import Draft, ToyTarget


class WaveTarget(ToyTarget):
    def feasible_blocks(self, context, blocks, batch_size=1):
        return [b for b in blocks if b <= 8]


class WaveExecutor:
    def __init__(self):
        self.batch_sizes = []

    def verify(self, items, *, sequential):
        self.batch_sizes.append(len(items))
        return [target.verify(tokens) for target, tokens in items]


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("sequential", [False, True])
def test_wave_ragged_rejection_eos_and_slot_reuse(adaptive, sequential):
    prompts, limits = [[1, 2, 3], [5], [2, 9], [4, 3]], [1, 11, 23, 17]
    targets = [WaveTarget(p) for p in ["perfect", "reject", "partial", "perfect"]]
    # Stop one sequence within a speculative block, while others keep decoding.
    eos = generate(ToyTarget(), None, prompts[2], 20, block_size=1).token_ids[3]
    expected = [
        generate(ToyTarget(), None, p, n, block_size=1, eos_token_id=eos)
        for p, n in zip(prompts, limits)
    ]
    executor = WaveExecutor()
    for _ in range(2):
        results, timing = generate_batch(
            targets,
            [Draft() for _ in targets],
            prompts,
            limits,
            executor,
            block_size=8,
            adaptive=AdaptiveBlockController() if adaptive else None,
            eos_token_id=eos,
            sequential=sequential,
        )
        assert [r.token_ids for r in results] == [r.token_ids for r in expected]
        for t, p, r in zip(targets, prompts, expected):
            assert t.tokens == p + r.token_ids[:-1]
        assert timing["output_tokens"] == sum(len(r.token_ids) for r in results)
    assert max(executor.batch_sizes) > 1
    assert targets[1].restores > 0


def test_graph_sizes_respect_small_maximum():
    for maximum, expected in [(0, []), (1, [1]), (2, [1, 2]), (3, [1, 2]), (4, [1, 2, 4])]:
        assert _determine_cuda_graph_bs(None, maximum, 24 << 30) == expected


def test_repeated_chat_hash_and_raw_workload_compatibility():
    import hashlib

    class Tokenizer:
        def encode(self, prompt, **kwargs):
            return [3, 4]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return [1, 3, 4, 2]

    raw = json.dumps(dict(prompt="hi", max_new_tokens=4)).encode()
    rows, digest = prepare_workload(raw, Tokenizer())
    assert digest == hashlib.sha256(raw).hexdigest()
    chats, chat_digest = prepare_workload(raw, Tokenizer(), repeat=2, chat_template=True)
    assert len(chats) == 2 and chats[0]["input_ids"] == [1, 3, 4, 2]
    assert chat_digest != digest and rows[0]["input_ids"] == [3, 4]


def test_batched_throughput_uses_wall_time_not_sum_of_request_latencies():
    data = dict(
        mode="target",
        requests=[dict(token_ids=[1, 2], ttft_ms=10, decode_ms=90, rounds=[]) for _ in range(4)],
        peak_allocated_bytes=0,
        cache={},
        waves=[dict(total_ms=100, decode_ms=80)],
    )
    result = summarize(data)
    assert result["output_tokens_per_second"] == 80
    assert result["decode_tokens_per_second"] == 50
