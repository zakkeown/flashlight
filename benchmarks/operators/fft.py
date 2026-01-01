"""
FFT operation benchmarks.

Benchmarks for torch.fft functions:
- fft, ifft
- rfft, irfft
- fft2, ifft2
- fftn
- fftshift, ifftshift
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark


class FFT1DBenchmark(OperatorBenchmark):
    """Benchmark for 1D FFT operation."""

    name = "fft"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 256},
            {"size": 1024},
            {"size": 4096},
            {"size": 16384},
            {"size": 65536},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["size"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["size"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.fft(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.fft(x)


class IFFT1DBenchmark(OperatorBenchmark):
    """Benchmark for 1D inverse FFT operation."""

    name = "ifft"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 256},
            {"size": 1024},
            {"size": 4096},
            {"size": 16384},
            {"size": 65536},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        import mlx.core as mx

        # Complex input
        x_real = mlx_compat.randn(config["size"])
        x_imag = mlx_compat.randn(config["size"])
        x = mlx_compat.tensor(x_real._mlx_array + 1j * x_imag._mlx_array)
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["size"], dtype=torch.complex64, device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.ifft(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.ifft(x)


class RFFT1DBenchmark(OperatorBenchmark):
    """Benchmark for 1D real FFT operation."""

    name = "rfft"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 256},
            {"size": 1024},
            {"size": 4096},
            {"size": 16384},
            {"size": 65536},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["size"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["size"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.rfft(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.rfft(x)


class FFT2DBenchmark(OperatorBenchmark):
    """Benchmark for 2D FFT operation."""

    name = "fft2"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"height": 64, "width": 64},
            {"height": 128, "width": 128},
            {"height": 256, "width": 256},
            {"height": 512, "width": 512},
            {"height": 1024, "width": 1024},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["height"], config["width"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["height"], config["width"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.fft2(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.fft2(x)


class RFFT2DBenchmark(OperatorBenchmark):
    """Benchmark for 2D real FFT operation."""

    name = "rfft2"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"height": 64, "width": 64},
            {"height": 128, "width": 128},
            {"height": 256, "width": 256},
            {"height": 512, "width": 512},
            {"height": 1024, "width": 1024},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["height"], config["width"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["height"], config["width"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.rfft2(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.rfft2(x)


class FFTNDBenchmark(OperatorBenchmark):
    """Benchmark for N-dimensional FFT operation."""

    name = "fftn"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (32, 32, 32)},
            {"shape": (64, 64, 64)},
            {"shape": (16, 16, 16, 16)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(*config["shape"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(*config["shape"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.fftn(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.fftn(x)


class FFTShiftBenchmark(OperatorBenchmark):
    """Benchmark for fftshift operation."""

    name = "fftshift"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1024,)},
            {"shape": (256, 256)},
            {"shape": (64, 64, 64)},
            {"shape": (1024, 1024)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(*config["shape"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(*config["shape"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.fft.fftshift(x)

    def pytorch_operation(self, x):
        import torch

        return torch.fft.fftshift(x)


class FFTFreqBenchmark(OperatorBenchmark):
    """Benchmark for fftfreq operation."""

    name = "fftfreq"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"n": 256},
            {"n": 1024},
            {"n": 4096},
            {"n": 16384},
            {"n": 65536},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        return (config["n"],)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        return (config["n"], device)

    def mlx_operation(self, n):
        import mlx_compat

        return mlx_compat.fft.fftfreq(n)

    def pytorch_operation(self, n, device):
        import torch

        return torch.fft.fftfreq(n, device=device)


# Export all benchmark classes
BENCHMARKS = [
    FFT1DBenchmark,
    IFFT1DBenchmark,
    RFFT1DBenchmark,
    FFT2DBenchmark,
    RFFT2DBenchmark,
    FFTNDBenchmark,
    FFTShiftBenchmark,
    FFTFreqBenchmark,
]
