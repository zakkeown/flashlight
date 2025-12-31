"""Layer-level benchmarks."""

from benchmarks.layers.base import LayerBenchmark
from benchmarks.layers.linear import LINEAR_BENCHMARKS
from benchmarks.layers.conv import CONV_BENCHMARKS
from benchmarks.layers.normalization import NORMALIZATION_BENCHMARKS
from benchmarks.layers.attention import ATTENTION_BENCHMARKS

ALL_LAYER_BENCHMARKS = (
    LINEAR_BENCHMARKS +
    CONV_BENCHMARKS +
    NORMALIZATION_BENCHMARKS +
    ATTENTION_BENCHMARKS
)

__all__ = [
    "LayerBenchmark",
    "ALL_LAYER_BENCHMARKS",
    "LINEAR_BENCHMARKS",
    "CONV_BENCHMARKS",
    "NORMALIZATION_BENCHMARKS",
    "ATTENTION_BENCHMARKS",
]
