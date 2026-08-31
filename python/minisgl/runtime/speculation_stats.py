"""DFlash counters, separate from prefix-cache hits and insensitive to EOS padding."""

from __future__ import annotations


def _counts(rounds):
    proposed = sum(r["block"] - 1 for r in rounds)
    matched = sum(r["accepted_draft"] for r in rounds)
    # Existing logs record a matching prefix before EOS truncation. Tokens after
    # the first emitted EOS must not be claimed as useful accepted draft output.
    emitted = sum(min(r["accepted_draft"], r["progress"]) for r in rounds)
    speculative = [r for r in rounds if r["block"] > 1]
    return {
        "decode_rounds": len(rounds),
        "speculative_rounds": len(speculative),
        "fallback_rounds": len(rounds) - len(speculative),
        "draft_tokens_proposed": proposed,
        "draft_tokens_matched": matched,
        "draft_tokens_emitted": emitted,
        "acceptance_rate": emitted / proposed if proposed else None,
        "match_rate_before_eos": matched / proposed if proposed else None,
        "full_accept_rounds": sum(
            min(r["accepted_draft"], r["progress"]) == r["block"] - 1 for r in speculative
        ),
        "zero_accept_rounds": sum(r["accepted_draft"] == 0 for r in speculative),
        "mean_accepted_per_speculative_round": emitted / len(speculative) if speculative else None,
        "mean_output_tokens_per_round": (
            sum(r["progress"] for r in rounds) / len(rounds) if rounds else None
        ),
        "mean_output_tokens_per_speculative_round": (
            sum(r["progress"] for r in speculative) / len(speculative) if speculative else None
        ),
        "draft_ms": sum(r["draft_ms"] for r in rounds),
        "verify_ms": sum(r["verify_ms"] for r in rounds),
        "restore_ms": sum(r["restore_ms"] for r in rounds),
    }


def speculation_stats(rounds):
    rounds = list(rounds)
    for r in rounds:
        if not (
            r["block"] >= 1
            and 0 <= r["accepted_draft"] < r["block"]
            and 1 <= r["progress"] <= r["block"]
        ):
            raise ValueError("Invalid speculative round counters")
    result = _counts(rounds)
    result["by_block"] = {
        str(block): _counts([r for r in rounds if r["block"] == block])
        for block in sorted({r["block"] for r in rounds})
    }
    return result


def format_speculation(stats):
    rate = stats["acceptance_rate"]
    if rate is None:
        return "DFlash: inactive (no draft tokens proposed)"
    return (
        f"DFlash: acceptance={rate:.2%} "
        f"({stats['draft_tokens_emitted']}/{stats['draft_tokens_proposed']} draft tokens emitted); "
        f"progress={stats['mean_output_tokens_per_round']:.2f} tokens/round; "
        f"spec_rounds={stats['speculative_rounds']}; fallback_rounds={stats['fallback_rounds']}"
    )
