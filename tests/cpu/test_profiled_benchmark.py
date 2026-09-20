import importlib.util
from pathlib import Path

import pytest


def test_comparison_refuses_profiled_timings():
    path = Path(__file__).resolve().parents[2] / "benchmark/runtime/compare_native_spec.py"
    spec = importlib.util.spec_from_file_location("compare_native_spec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Profiled waves are diagnostic"):
        module.metrics({"waves": [{"profiled": True}]})
