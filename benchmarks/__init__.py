"""
mlx_compat Benchmarking Suite

Comprehensive performance benchmarking with PyTorch MPS comparison.

Usage:
    python -m benchmarks                    # Run all benchmarks
    python -m benchmarks --level operator   # Operator benchmarks only
    python -m benchmarks --compare-pytorch  # Include PyTorch comparison
    python -m benchmarks --help             # Show all options
"""

from benchmarks.core.config import BenchmarkConfig, BenchmarkResult, TimingStats
from benchmarks.core.runner import BenchmarkRunner, BenchmarkLevel
from benchmarks.core.timing import Timer

__version__ = "1.0.0"

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "TimingStats",
    "BenchmarkRunner",
    "BenchmarkLevel",
    "Timer",
]
