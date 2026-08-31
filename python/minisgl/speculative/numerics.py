"""Configure only this target model; never change draft/global PyTorch operators."""


def configure_target_numerics(engine, mode):
    if mode not in ("fast", "stable"):
        raise ValueError("Target numerics must be fast or stable")
    if mode == "fast":
        return
    import torch
    from minisgl.attention.gdn import HybridLinearBackend
    from minisgl.attention.invariant import InvariantAttentionBackend
    from minisgl.distributed import get_tp_info
    from minisgl.layers.base import BaseOP, OPList
    from minisgl.layers.linear import _LinearTPImpl

    if (
        engine.dtype != torch.bfloat16
        or get_tp_info().size != 1
        or not isinstance(engine.attn_backend, HybridLinearBackend)
    ):
        raise ValueError("Stable target numerics currently require BF16 hybrid attention and TP=1")
    seen = set()

    def visit(op):
        if id(op) in seen:
            return
        seen.add(id(op))
        if isinstance(op, _LinearTPImpl):
            op._batch_invariant = True
        children = op.op_list if isinstance(op, OPList) else vars(op).values()
        for child in children:
            if isinstance(child, BaseOP):
                visit(child)

    visit(engine.model.model)
    engine.attn_backend.full_backend = InvariantAttentionBackend()
