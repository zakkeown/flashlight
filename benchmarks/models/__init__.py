"""Model-level benchmarks."""

from benchmarks.models.base import ModelBenchmark
from benchmarks.models.mlp import MLP_BENCHMARKS
from benchmarks.models.cnn import CNN_BENCHMARKS
from benchmarks.models.resnet import RESNET_BENCHMARKS
from benchmarks.models.transformer import TRANSFORMER_BENCHMARKS

ALL_MODEL_BENCHMARKS = (
    MLP_BENCHMARKS +
    CNN_BENCHMARKS +
    RESNET_BENCHMARKS +
    TRANSFORMER_BENCHMARKS
)

__all__ = [
    "ModelBenchmark",
    "ALL_MODEL_BENCHMARKS",
    "MLP_BENCHMARKS",
    "CNN_BENCHMARKS",
    "RESNET_BENCHMARKS",
    "TRANSFORMER_BENCHMARKS",
]
