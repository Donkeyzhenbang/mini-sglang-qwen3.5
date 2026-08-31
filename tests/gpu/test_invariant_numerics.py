import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.parametrize("n,k", [(64, 2560), (12288, 2560), (2560, 9216), (137, 131)])
def test_linear_is_exact_across_row_permutation_and_chunking(n, k):
    from minisgl.kernel.triton.invariant import invariant_linear

    torch.manual_seed(42)
    x = torch.randn(35, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    expected = invariant_linear(x, w, fp32_output=True)
    # Exercise different batch sizes, tile boundaries and positions within tiles.
    order = torch.randperm(len(x), device="cuda")
    shuffled = x[order]
    actual = torch.cat(
        [invariant_linear(chunk, w, fp32_output=True) for chunk in shuffled.split([1, 4, 8, 16, 6])]
    )
    torch.testing.assert_close(actual, expected[order], rtol=0, atol=0)
    oracle = (x.double() @ w.double().t()).float()
    torch.testing.assert_close(expected, oracle, rtol=2e-5, atol=5e-4)
    rounded = invariant_linear(x, w)
    torch.testing.assert_close(rounded, expected.bfloat16(), rtol=0, atol=0)


def test_fp32_head_preserves_a_close_greedy_winner():
    from minisgl.kernel.triton.invariant import invariant_linear

    x = torch.tensor([[1.0, 1 / 256]], device="cuda", dtype=torch.bfloat16)
    w = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device="cuda", dtype=torch.bfloat16)
    logits = invariant_linear(x, w, fp32_output=True)
    assert logits.argmax().item() == 1
    assert logits.bfloat16().argmax().item() == 0  # BF16 rounds these logits to a tie.


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_paged_attention_is_causal_and_invariant_with_fp64_oracle(dim):
    from minisgl.kernel.triton.invariant import invariant_attention

    torch.manual_seed(123)
    count, hq, hk, capacity = 9, 4, 2, 160
    q = torch.randn(count, hq, dim, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(3 * capacity, hk, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    table = torch.randperm(3 * capacity, device="cuda").int().view(3, capacity)
    slots = torch.tensor([2, 0, 1, 2, 1, 0, 2, 1, 0], device="cuda", dtype=torch.int32)
    positions = torch.tensor(
        [0, 31, 32, 63, 64, 127, 128, 129, 151], device="cuda", dtype=torch.int32
    )
    actual = invariant_attention(q, k, v, table, slots, positions)
    expected = []
    for i in range(count):
        ids = table[slots[i].long(), : int(positions[i]) + 1].long()
        keys = k[ids].repeat_interleave(hq // hk, dim=1).double().transpose(0, 1)
        values = v[ids].repeat_interleave(hq // hk, dim=1).double().transpose(0, 1)
        scores = torch.einsum("hd,hnd->hn", q[i].double(), keys) / dim**0.5
        expected.append(torch.einsum("hn,hnd->hd", scores.softmax(-1), values))
    torch.testing.assert_close(actual.float(), torch.stack(expected).float(), rtol=5e-3, atol=1e-3)
    order = torch.tensor([6, 3, 0, 7, 1, 4, 8, 2, 5], device="cuda")
    regrouped = torch.cat(
        [
            invariant_attention(q[ix], k, v, table, slots[ix], positions[ix])
            for ix in order.split([1, 4, 2, 2])
        ]
    )
    torch.testing.assert_close(regrouped, actual[order], rtol=0, atol=0)
    # Unreachable future KV must not influence a query, even if it is NaN.
    poisoned_k, poisoned_v = k.clone(), v.clone()
    ids = table[2, 129:].long()
    poisoned_k[ids] = float("nan")
    poisoned_v[ids] = float("nan")
    repeated = invariant_attention(q, poisoned_k, poisoned_v, table, slots, positions)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


def test_attention_graph_replays_new_slots_and_context_lengths():
    from minisgl.kernel.triton.invariant import invariant_attention

    torch.manual_seed(318)
    q = torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(384, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    table = torch.randperm(384, device="cuda").int().view(4, 96)
    slots = torch.arange(4, device="cuda", dtype=torch.int32)
    positions = torch.zeros(4, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        invariant_attention(q, k, v, table, slots, positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invariant_attention(q, k, v, table, slots, positions)
    slots.copy_(torch.tensor([3, 0, 2, 1], device="cuda", dtype=torch.int32))
    positions.copy_(torch.tensor([95, 64, 32, 63], device="cuda", dtype=torch.int32))
    graph.replay()
    expected = invariant_attention(q, k, v, table, slots, positions)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
