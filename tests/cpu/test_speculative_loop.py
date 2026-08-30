import pytest
from minisgl.runtime.adaptive import AdaptiveBlockController
from minisgl.speculative.loop import generate, greedy_accept


class ToyTarget:
    """Independent deterministic autoregressive oracle, with destructive state updates."""

    def __init__(self, proposal="perfect"):
        self.proposal = proposal
        self.tokens = []
        self.restores = 0

    @property
    def length(self):
        return len(self.tokens)

    def decision(self):
        return (sum(self.tokens) * 3 + len(self.tokens)) % 31

    def synchronize(self):
        pass

    def prefill(self, prompt):
        self.tokens = list(prompt)
        return self.decision()

    def checkpoint(self):
        return list(self.tokens)

    def restore(self, snapshot):
        self.restores += 1
        self.tokens = list(snapshot)

    def propose(self, draft, anchor, block):
        state = list(self.tokens)
        result = [anchor]
        for i in range(block - 1):
            self.tokens.append(result[-1])
            next_token = self.decision()
            if self.proposal == "reject" or (self.proposal == "partial" and i == 1):
                next_token = (next_token + 1) % 31
            result.append(next_token)
        self.tokens = state
        return result

    def verify(self, tokens):
        predictions = []
        for token in tokens:
            self.tokens.append(token)
            predictions.append(self.decision())
        return predictions, list(tokens)

    def commit_features(self, features, length):
        assert len(features) == length


class Draft:
    def reset(self):
        pass


@pytest.mark.parametrize("proposal", ["perfect", "reject", "partial"])
@pytest.mark.parametrize("count", [0, 1, 2, 7, 33])
def test_speculative_matches_autoregressive(proposal, count):
    prompt = [1, 2, 3]
    baseline = generate(ToyTarget(), None, prompt, count, block_size=1)
    target = ToyTarget(proposal)
    speculative = generate(target, Draft(), prompt, count, block_size=8)
    assert speculative.token_ids == baseline.token_ids
    if count:
        assert target.tokens == prompt + baseline.token_ids[:-1]
    if proposal == "reject" and count > 2:
        assert target.restores > 0


@pytest.mark.parametrize("stop_position", [0, 1, 3, 9])
def test_eos_truncates_output_and_state(stop_position):
    prompt = [2, 9]
    baseline = generate(ToyTarget(), None, prompt, 40, block_size=1)
    eos = baseline.token_ids[stop_position]
    expected = baseline.token_ids[: baseline.token_ids.index(eos) + 1]
    target = ToyTarget()
    result = generate(target, Draft(), prompt, 40, block_size=8, eos_token_id=eos)
    assert result.token_ids == expected
    assert target.tokens == prompt + expected[:-1]


def test_adaptive_loop_and_low_memory_fallback():
    prompt = [1, 3, 5]
    baseline = generate(ToyTarget(), None, prompt, 40, block_size=1)
    result = generate(
        ToyTarget("partial"),
        Draft(),
        prompt,
        40,
        block_size=16,
        adaptive=AdaptiveBlockController(),
        feasible=lambda context: [1, 2, 4],
    )
    assert result.token_ids == baseline.token_ids
    assert all(r["block"] <= 4 for r in result.rounds)


def test_rejection_stops_at_first_mismatch():
    assert greedy_accept([9, 1, 2, 3], [1, 8, 3, 4]) == (1, 8)
    with pytest.raises(ValueError):
        greedy_accept([1, 2], [2])
