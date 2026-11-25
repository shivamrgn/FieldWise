"""On-device benchmarking suite for FieldWise PS1."""

from typing import Any

__all__ = ["run_benchmark_suite"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import run_benchmarks

        return getattr(run_benchmarks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
