import pytest
from minisgl.runtime.speculation_stats import format_speculation, speculation_stats


def row(block, accepted, progress):
    return dict(
        block=block,
        accepted_draft=accepted,
        progress=progress,
        draft_ms=1,
        verify_ms=2,
        restore_ms=3,
    )


def test_weighted_acceptance_excludes_anchor_and_one_token_fallbacks():
    stats = speculation_stats([row(8, 7, 8), row(2, 0, 1), row(1, 0, 1)])
    assert stats["draft_tokens_proposed"] == 8  # 7 + 1; anchors are not proposals.
    assert stats["acceptance_rate"] == 7 / 8  # Not the mean of 100% and 0%.
    assert stats["fallback_rounds"] == 1 and stats["speculative_rounds"] == 2
    assert stats["full_accept_rounds"] == stats["zero_accept_rounds"] == 1
    assert stats["mean_output_tokens_per_round"] == 10 / 3
    assert stats["by_block"]["1"]["acceptance_rate"] is None
    assert "87.50%" in format_speculation(stats)


def test_eos_does_not_count_matching_but_unemitted_draft_tokens():
    stats = speculation_stats([row(8, 7, 3)])
    assert stats["draft_tokens_matched"] == 7
    assert stats["draft_tokens_emitted"] == 3
    assert stats["match_rate_before_eos"] == 1
    assert stats["acceptance_rate"] == 3 / 7
    assert stats["full_accept_rounds"] == 0


def test_no_proposals_is_not_zero_percent_acceptance():
    for rounds in [[], [row(1, 0, 1)]]:
        stats = speculation_stats(rounds)
        assert stats["acceptance_rate"] is None
        assert "inactive" in format_speculation(stats)


def test_invalid_round_is_rejected():
    with pytest.raises(ValueError):
        speculation_stats([row(8, 8, 8)])
