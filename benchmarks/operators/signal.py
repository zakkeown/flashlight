"""
Signal processing operator benchmarks.

Benchmarks for:
- Window functions (bartlett, blackman, cosine, etc.)
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark


class WindowBenchmark(OperatorBenchmark):
    """
    Base class for window function benchmarks.

    Window functions are 1D tensor operations that generate
    tapered windows for signal processing applications.
    """

    level = BenchmarkLevel.OPERATOR

    # Override in subclass
    window_name: str = "window"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.name = f"window_{self.window_name}"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        """Standard window sizes for benchmarking."""
        return [
            {"M": 64},
            {"M": 256},
            {"M": 1024},
            {"M": 4096},
            {"M": 16384},
            {"M": 65536},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        # Window functions don't need input tensors, just the size
        return (config["M"],)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        # Window functions don't need input tensors, just the size
        return (config["M"],)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate elements per second for window generation."""
        if time_ms <= 0:
            return 0.0, "elem/sec"

        M = config["M"]
        elements_per_sec = M / (time_ms / 1000.0)

        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        else:
            return elements_per_sec, "elem/sec"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        """Calculate bytes for bandwidth estimation."""
        M = config["M"]
        # Output only (float32)
        return M * 4


class BartlettBenchmark(WindowBenchmark):
    """Benchmark for Bartlett window."""

    window_name = "bartlett"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.bartlett(M)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.bartlett(M)


class BlackmanBenchmark(WindowBenchmark):
    """Benchmark for Blackman window."""

    window_name = "blackman"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.blackman(M)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.blackman(M)


class CosineBenchmark(WindowBenchmark):
    """Benchmark for Cosine window."""

    window_name = "cosine"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.cosine(M)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.cosine(M)


class ExponentialBenchmark(WindowBenchmark):
    """Benchmark for Exponential window."""

    window_name = "exponential"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.exponential(M, tau=1.0)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.exponential(M, tau=1.0)


class GaussianBenchmark(WindowBenchmark):
    """Benchmark for Gaussian window."""

    window_name = "gaussian"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.gaussian(M, std=7.0)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.gaussian(M, std=7.0)


class GeneralCosineBenchmark(WindowBenchmark):
    """Benchmark for General Cosine window (Blackman coefficients)."""

    window_name = "general_cosine"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        # Use Blackman coefficients as representative case
        return windows.general_cosine(M, a=[0.42, 0.5, 0.08])

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.general_cosine(M, a=[0.42, 0.5, 0.08])


class GeneralHammingBenchmark(WindowBenchmark):
    """Benchmark for General Hamming window."""

    window_name = "general_hamming"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.general_hamming(M, alpha=0.54)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.general_hamming(M, alpha=0.54)


class HammingBenchmark(WindowBenchmark):
    """Benchmark for Hamming window."""

    window_name = "hamming"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.hamming(M)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.hamming(M)


class HannBenchmark(WindowBenchmark):
    """Benchmark for Hann window."""

    window_name = "hann"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.hann(M)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.hann(M)


class KaiserBenchmark(WindowBenchmark):
    """Benchmark for Kaiser window."""

    window_name = "kaiser"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.kaiser(M, beta=12.0)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.kaiser(M, beta=12.0)


class NuttallBenchmark(WindowBenchmark):
    """Benchmark for Nuttall window."""

    window_name = "nuttall"

    def mlx_operation(self, M):
        from flashlight.signal import windows
        return windows.nuttall(M)

    def pytorch_operation(self, M):
        import torch.signal.windows as tw
        return tw.nuttall(M)


# List of all signal benchmarks
SIGNAL_BENCHMARKS = [
    BartlettBenchmark,
    BlackmanBenchmark,
    CosineBenchmark,
    ExponentialBenchmark,
    GaussianBenchmark,
    GeneralCosineBenchmark,
    GeneralHammingBenchmark,
    HammingBenchmark,
    HannBenchmark,
    KaiserBenchmark,
    NuttallBenchmark,
]
