"""Operator-level benchmarks."""

from benchmarks.operators.base import OperatorBenchmark
from benchmarks.operators.arithmetic import ARITHMETIC_BENCHMARKS
from benchmarks.operators.activations import ACTIVATION_BENCHMARKS
from benchmarks.operators.reductions import REDUCTION_BENCHMARKS
from benchmarks.operators.convolution import CONVOLUTION_BENCHMARKS
from benchmarks.operators.pooling import POOLING_BENCHMARKS
from benchmarks.operators.linalg import LINALG_BENCHMARKS

ALL_OPERATOR_BENCHMARKS = (
    ARITHMETIC_BENCHMARKS +
    ACTIVATION_BENCHMARKS +
    REDUCTION_BENCHMARKS +
    CONVOLUTION_BENCHMARKS +
    POOLING_BENCHMARKS +
    LINALG_BENCHMARKS
)

__all__ = [
    "OperatorBenchmark",
    "ALL_OPERATOR_BENCHMARKS",
    "ARITHMETIC_BENCHMARKS",
    "ACTIVATION_BENCHMARKS",
    "REDUCTION_BENCHMARKS",
    "CONVOLUTION_BENCHMARKS",
    "POOLING_BENCHMARKS",
    "LINALG_BENCHMARKS",
]
