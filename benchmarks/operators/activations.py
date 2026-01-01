"""
Activation function benchmarks.

Benchmarks for:
- relu, gelu, sigmoid, tanh
- softmax, log_softmax
- silu/swish, leaky_relu, elu
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.operators.base import UnaryOperatorBenchmark, OperatorBenchmark


class ReluBenchmark(UnaryOperatorBenchmark):
    """Benchmark for ReLU activation."""

    op_name = "relu"

    def mlx_operation(self, x):
        import flashlight.nn.functional as F
        return F.relu(x)

    def pytorch_operation(self, x):
        import torch.nn.functional as F
        return F.relu(x)


class GeluBenchmark(UnaryOperatorBenchmark):
    """Benchmark for GELU activation."""

    op_name = "gelu"

    def mlx_operation(self, x):
        import flashlight.nn.functional as F
        return F.gelu(x)

    def pytorch_operation(self, x):
        import torch.nn.functional as F
        return F.gelu(x)


class SigmoidBenchmark(UnaryOperatorBenchmark):
    """Benchmark for sigmoid activation."""

    op_name = "sigmoid"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.sigmoid(x)

    def pytorch_operation(self, x):
        import torch
        return torch.sigmoid(x)


class TanhBenchmark(UnaryOperatorBenchmark):
    """Benchmark for tanh activation."""

    op_name = "tanh"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.tanh(x)

    def pytorch_operation(self, x):
        import torch
        return torch.tanh(x)


class SiluBenchmark(UnaryOperatorBenchmark):
    """Benchmark for SiLU/Swish activation."""

    op_name = "silu"

    def mlx_operation(self, x):
        import flashlight.nn.functional as F
        return F.silu(x)

    def pytorch_operation(self, x):
        import torch.nn.functional as F
        return F.silu(x)


class SoftmaxBenchmark(OperatorBenchmark):
    """Benchmark for softmax activation."""

    name = "softmax"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (32, 1000), "dim": -1},  # Classification logits
            {"shape": (32, 50, 512), "dim": -1},  # Attention scores
            {"shape": (64, 128, 128), "dim": -1},  # Larger attention
            {"shape": (16, 256, 256), "dim": -1},  # Vision attention
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["dim"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["dim"])

    def mlx_operation(self, x, dim):
        import flashlight.nn.functional as F
        return F.softmax(x, dim=dim)

    def pytorch_operation(self, x, dim):
        import torch.nn.functional as F
        return F.softmax(x, dim=dim)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "elem/sec"

        shape = config["shape"]
        elements = 1
        for dim in shape:
            elements *= dim

        elements_per_sec = elements / (time_ms / 1000.0)

        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        else:
            return elements_per_sec, "elem/sec"


class LogSoftmaxBenchmark(SoftmaxBenchmark):
    """Benchmark for log_softmax activation."""

    name = "log_softmax"

    def mlx_operation(self, x, dim):
        import flashlight.nn.functional as F
        return F.log_softmax(x, dim=dim)

    def pytorch_operation(self, x, dim):
        import torch.nn.functional as F
        return F.log_softmax(x, dim=dim)


class LeakyReluBenchmark(OperatorBenchmark):
    """Benchmark for Leaky ReLU activation."""

    name = "leaky_relu"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000, 1000), "negative_slope": 0.01},
            {"shape": (32, 256, 256), "negative_slope": 0.01},
            {"shape": (16, 128, 128, 128), "negative_slope": 0.2},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["negative_slope"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["negative_slope"])

    def mlx_operation(self, x, negative_slope):
        import flashlight.nn.functional as F
        return F.leaky_relu(x, negative_slope=negative_slope)

    def pytorch_operation(self, x, negative_slope):
        import torch.nn.functional as F
        return F.leaky_relu(x, negative_slope=negative_slope)


class EluBenchmark(OperatorBenchmark):
    """Benchmark for ELU activation."""

    name = "elu"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000, 1000), "alpha": 1.0},
            {"shape": (32, 256, 256), "alpha": 1.0},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["alpha"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["alpha"])

    def mlx_operation(self, x, alpha):
        import flashlight.nn.functional as F
        return F.elu(x, alpha=alpha)

    def pytorch_operation(self, x, alpha):
        import torch.nn.functional as F
        return F.elu(x, alpha=alpha)


# List of all activation benchmarks
ACTIVATION_BENCHMARKS = [
    ReluBenchmark,
    GeluBenchmark,
    SigmoidBenchmark,
    TanhBenchmark,
    SiluBenchmark,
    SoftmaxBenchmark,
    LogSoftmaxBenchmark,
    LeakyReluBenchmark,
    EluBenchmark,
]
