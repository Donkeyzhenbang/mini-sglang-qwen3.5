"""Host metadata checks: no GPU synchronization or data-dependent graph work."""


def validate_graph_requests(items, slots, capacity):
    if not items or len(items) != len(slots):
        raise ValueError("Graph requires one slot per request")
    if any(not isinstance(s, int) or isinstance(s, bool) for s in slots):
        raise ValueError("Graph slots must be host integers")
    if any(s < 0 or s >= capacity for s in slots):
        raise ValueError("Graph slot out of range")
    if len(set(slots)) != len(slots) or len({id(r[0]) for r in items}) != len(items):
        raise ValueError("Graph requires distinct slots and request contexts")


def cache_import_required(source, destination):
    """Require an exact pool view or an independent, compatible eager cache.

    Relocation between slots of one pool is deliberately unsupported: copying
    requests one at a time can overwrite another request's source during swaps.
    Validate the whole batch before performing any eager-cache imports.
    """
    if source is None:
        if destination.shape[-2]:
            raise ValueError("Missing confirmed cache")
        return False
    if (
        source.shape != destination.shape
        or source.dtype != destination.dtype
        or source.device != destination.device
    ):
        raise ValueError("Confirmed cache shape, dtype or device mismatch")
    if source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
        if (
            source.storage_offset() != destination.storage_offset()
            or source.stride() != destination.stride()
        ):
            raise ValueError("Confirmed cache belongs to a different graph slot or view")
        return False
    return True


def can_capture_graph(graphs, key, free_bytes, limit=32, reserve_bytes=2 * 2**30):
    """Bound new shapes; retain existing graphs and their stable allocations."""
    return key in graphs or (len(graphs) < limit and free_bytes >= reserve_bytes)
