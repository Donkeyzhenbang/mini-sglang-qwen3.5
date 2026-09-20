"""Opt-in operator/shape NVTX labels. Diagnostic only; never synchronizes tensors.

Python labels execute during eager runs and graph construction, not replay.
Use a separate eager capture for unambiguous per-layer kernel attribution.
"""

import functools
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F


def shape(x):
    return "x".join(str(n) for n in x.shape)


def gemm_shape(x, weight):
    return f"M={x.numel() // x.shape[-1]},N={weight.shape[0]},K={weight.shape[1]}"


def annotate(method, label):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        with torch.cuda.nvtx.range(label(*args, **kwargs)):
            return method(*args, **kwargs)

    return wrapped


@contextmanager
def operator_annotations():
    import minisgl.engine.engine as engine
    import minisgl.kernel.triton.invariant as invariant
    from minisgl.layers.base import BaseOP, OPList
    from minisgl.layers.linear import _LinearTPImpl
    from minisgl.models.qwen3_5 import Qwen3_5GatedDeltaNet
    from minisgl.models.utils import GatedMLP, RopeAttn
    from minisgl.speculative.draft import DFlashDraft, DraftAttention, DraftLayer, DraftMLP

    def name_target(op, path):
        op._nsys_path = path
        children = enumerate(op.op_list) if isinstance(op, OPList) else list(vars(op).items())
        for key, child in children:
            if isinstance(child, BaseOP):
                name_target(child, f"{path}.{key}")

    original_create = engine.create_model

    def create_model(*args, **kwargs):
        model = original_create(*args, **kwargs)
        name_target(model, "target")
        return model

    original_draft_init = DFlashDraft.__init__

    def draft_init(self, *args, **kwargs):
        original_draft_init(self, *args, **kwargs)
        for name, module in self.named_modules():
            module._nsys_path = "draft" + ("." + name if name else "")

    def linear_label(self, x):
        backend = "triton_stable" if self._batch_invariant else "torch"
        path = getattr(self, "_nsys_path", type(self).__name__)
        return f"op/{path}/GEMM[{gemm_shape(x, self.weight)};{backend}]"

    with ExitStack() as stack:

        def wrap(obj, name, label):
            stack.enter_context(patch.object(obj, name, annotate(getattr(obj, name), label)))

        stack.enter_context(patch.object(engine, "create_model", create_model))
        stack.enter_context(patch.object(DFlashDraft, "__init__", draft_init))
        wrap(_LinearTPImpl, "_linear", linear_label)
        wrap(
            torch.nn.Linear,
            "forward",
            lambda self, x: (
                f"op/{getattr(self, '_nsys_path', 'nn.Linear')}/GEMM[{gemm_shape(x, self.weight)};torch]"
            ),
        )
        wrap(
            F,
            "linear",
            lambda input, weight, bias=None: f"op/GEMM.torch[{gemm_shape(input, weight)}]",
        )
        wrap(
            invariant,
            "invariant_linear",
            lambda x, weight, *a, **kw: f"op/GEMM.triton_stable[{gemm_shape(x, weight)}]",
        )
        wrap(
            F,
            "scaled_dot_product_attention",
            lambda query, key, value, *a, **kw: (
                f"op/SDPA.torch[Q={shape(query)};K={shape(key)};V={shape(value)}]"
            ),
        )
        for cls, kind in (
            (Qwen3_5GatedDeltaNet, "GDN"),
            (RopeAttn, "FullAttention"),
            (GatedMLP, "MLP"),
            (DraftLayer, "DraftLayer"),
            (DraftAttention, "DraftAttention"),
            (DraftMLP, "DraftMLP"),
        ):
            wrap(
                cls,
                "forward",
                lambda self, x, *a, _kind=kind, **kw: (
                    f"op/{getattr(self, '_nsys_path', type(self).__name__)}/{_kind}[X={shape(x)}]"
                ),
            )
        yield
