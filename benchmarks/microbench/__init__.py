"""
Micro-benchmarks for measuring specific performance overhead in flashlight.

These benchmarks are designed to isolate and measure:
- Layout conversion overhead (NCHW <-> NHWC)
- Method dispatch overhead (tensor.sum() vs ops.sum())
- Autograd tape construction overhead
"""

from .layout import LAYOUT_BENCHMARKS
from .method_dispatch import DISPATCH_BENCHMARKS
from .autograd_overhead import AUTOGRAD_BENCHMARKS
from .before_after import BeforeAfterComparator, ComparisonResult

MICROBENCH_BENCHMARKS = (
    LAYOUT_BENCHMARKS +
    DISPATCH_BENCHMARKS +
    AUTOGRAD_BENCHMARKS
)

__all__ = [
    "LAYOUT_BENCHMARKS",
    "DISPATCH_BENCHMARKS",
    "AUTOGRAD_BENCHMARKS",
    "MICROBENCH_BENCHMARKS",
    "BeforeAfterComparator",
    "ComparisonResult",
]
