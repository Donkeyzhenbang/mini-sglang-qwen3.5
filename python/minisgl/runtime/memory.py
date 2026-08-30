"""Byte accounting shared by cache allocation and experimental scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridMemoryLayout:
    kv_bytes_per_token: int
    state_bytes_per_slot: int

    @classmethod
    def from_model(cls, config, dtype_bytes: int = 2, tp_size: int = 1):
        def heads(n):
            if n % tp_size == 0:
                return n // tp_size
            if tp_size % n == 0:
                return 1
            raise ValueError("Head count and tensor parallel size are incompatible")

        full = len(config.full_attention_layer_ids)
        linear = config.num_layers - full
        kv = 2 * full * heads(config.num_kv_heads) * config.head_dim * dtype_bytes
        state = 0
        if linear:
            hk, hv = heads(config.linear_num_key_heads), heads(config.linear_num_value_heads)
            k, v = config.linear_key_head_dim, config.linear_value_head_dim
            conv = (2 * hk * k + hv * v) * (config.linear_conv_kernel_dim - 1) * dtype_bytes
            state = linear * (hv * v * k * 4 + conv)
        return cls(kv, state)


def plan_kv_pages(
    *,
    available_bytes: int,
    page_size: int,
    layout: HybridMemoryLayout,
    slots: int,
    workspace_bytes: int,
    snapshot_bytes: int,
    override: int | None = None,
) -> int:
    if min(available_bytes, page_size, slots) <= 0 or min(workspace_bytes, snapshot_bytes) < 0:
        raise ValueError("Invalid memory budget")
    # Retained prefix states plus capture buffers from two overlapping batches.
    reserve = slots * layout.state_bytes_per_slot + workspace_bytes + 3 * snapshot_bytes
    per_page = page_size * layout.kv_bytes_per_token
    if per_page <= 0:
        raise ValueError("At least one full-attention layer is required")
    maximum = (available_bytes - reserve) // per_page - 1  # dummy KV page
    pages = maximum if override is None else override
    if pages < 2 or pages > maximum:
        raise ValueError(
            f"KV budget exhausted: requested={pages}, maximum={maximum}; "
            "reduce running requests, context, snapshots or workspace"
        )
    return pages
