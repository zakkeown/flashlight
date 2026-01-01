"""Layer-level benchmarks."""

from benchmarks.layers.base import LayerBenchmark
from benchmarks.layers.linear import LINEAR_BENCHMARKS
from benchmarks.layers.conv import CONV_BENCHMARKS
from benchmarks.layers.normalization import NORMALIZATION_BENCHMARKS
from benchmarks.layers.attention import ATTENTION_BENCHMARKS
from benchmarks.layers.pooling import POOLING_BENCHMARKS
from benchmarks.layers.activation import ACTIVATION_BENCHMARKS
from benchmarks.layers.embedding import EMBEDDING_BENCHMARKS
from benchmarks.layers.rnn import RNN_BENCHMARKS
from benchmarks.layers.losses import LOSS_BENCHMARKS
from benchmarks.layers.padding import PADDING_BENCHMARKS

ALL_LAYER_BENCHMARKS = (
    LINEAR_BENCHMARKS +
    CONV_BENCHMARKS +
    NORMALIZATION_BENCHMARKS +
    ATTENTION_BENCHMARKS +
    POOLING_BENCHMARKS +
    ACTIVATION_BENCHMARKS +
    EMBEDDING_BENCHMARKS +
    RNN_BENCHMARKS +
    LOSS_BENCHMARKS +
    PADDING_BENCHMARKS
)

__all__ = [
    "LayerBenchmark",
    "ALL_LAYER_BENCHMARKS",
    "LINEAR_BENCHMARKS",
    "CONV_BENCHMARKS",
    "NORMALIZATION_BENCHMARKS",
    "ATTENTION_BENCHMARKS",
    "POOLING_BENCHMARKS",
    "ACTIVATION_BENCHMARKS",
    "EMBEDDING_BENCHMARKS",
    "RNN_BENCHMARKS",
    "LOSS_BENCHMARKS",
    "PADDING_BENCHMARKS",
]
