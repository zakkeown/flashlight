"""
Layout conversion overhead benchmarks.

Measures the cost of NCHW <-> NHWC conversions in spatial operations.
"""

from typing import List, Dict, Any, Tuple
from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BaseBenchmark


class TransposeOverheadBenchmark(BaseBenchmark):
    """Measure isolated transpose overhead for layout conversion."""

    name = "transpose_overhead"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        """Configurations for NCHW tensors of various sizes."""
        return [
            {"batch": 1, "channels": 64, "h": 224, "w": 224, "label": "1x64x224x224"},
            {"batch": 32, "channels": 64, "h": 56, "w": 56, "label": "32x64x56x56"},
            {"batch": 32, "channels": 256, "h": 28, "w": 28, "label": "32x256x28x28"},
            {"batch": 64, "channels": 512, "h": 14, "w": 14, "label": "64x512x14x14"},
            {"batch": 128, "channels": 64, "h": 7, "w": 7, "label": "128x64x7x7"},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(
            config["batch"], config["channels"], config["h"], config["w"]
        )
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(
            config["batch"], config["channels"], config["h"], config["w"],
            device=device
        )
        return (x,)

    def mlx_operation(self, x):
        """NCHW -> NHWC -> NCHW round trip."""
        import mlx.core as mx
        # Convert NCHW to NHWC
        nhwc = mx.transpose(x._mlx_array, [0, 2, 3, 1])
        # Convert back NHWC to NCHW
        nchw = mx.transpose(nhwc, [0, 3, 1, 2])
        mx.eval(nchw)
        return nchw

    def pytorch_operation(self, x):
        """NCHW -> NHWC -> NCHW round trip."""
        import torch
        # Use contiguous() to force memory copy like MLX transpose does
        nhwc = x.permute(0, 2, 3, 1).contiguous()
        nchw = nhwc.permute(0, 3, 1, 2).contiguous()
        return nchw

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        """Total bytes for 2 transpose operations."""
        elements = (
            config["batch"] * config["channels"] *
            config["h"] * config["w"]
        )
        # 2 reads + 2 writes (2 transposes)
        return elements * 4 * 4


class Conv2dLayoutOverheadBenchmark(BaseBenchmark):
    """Measure layout conversion overhead in conv2d operations."""

    name = "conv2d_layout_overhead"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 1, "in_ch": 64, "out_ch": 64, "h": 56, "w": 56, "kernel": 3},
            {"batch": 32, "in_ch": 64, "out_ch": 128, "h": 56, "w": 56, "kernel": 3},
            {"batch": 32, "in_ch": 256, "out_ch": 256, "h": 28, "w": 28, "kernel": 3},
            {"batch": 64, "in_ch": 512, "out_ch": 512, "h": 14, "w": 14, "kernel": 3},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(config["batch"], config["in_ch"], config["h"], config["w"])
        w = flashlight.randn(
            config["out_ch"], config["in_ch"],
            config["kernel"], config["kernel"]
        )
        return (x, w)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(
            config["batch"], config["in_ch"], config["h"], config["w"],
            device=device
        )
        w = torch.randn(
            config["out_ch"], config["in_ch"],
            config["kernel"], config["kernel"],
            device=device
        )
        return (x, w)

    def mlx_operation(self, x, w):
        import flashlight.nn.functional as F
        return F.conv2d(x, w, padding=1)

    def pytorch_operation(self, x, w):
        import torch.nn.functional as F
        return F.conv2d(x, w, padding=1)


class PoolingLayoutOverheadBenchmark(BaseBenchmark):
    """Measure layout conversion overhead in pooling operations."""

    name = "pool2d_layout_overhead"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 1, "channels": 64, "h": 112, "w": 112, "kernel": 2, "stride": 2},
            {"batch": 32, "channels": 64, "h": 56, "w": 56, "kernel": 2, "stride": 2},
            {"batch": 32, "channels": 256, "h": 28, "w": 28, "kernel": 2, "stride": 2},
            {"batch": 64, "channels": 512, "h": 14, "w": 14, "kernel": 2, "stride": 2},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(config["batch"], config["channels"], config["h"], config["w"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(
            config["batch"], config["channels"], config["h"], config["w"],
            device=device
        )
        return (x,)

    def mlx_operation(self, x):
        import flashlight.nn.functional as F
        return F.max_pool2d(x, kernel_size=2, stride=2)

    def pytorch_operation(self, x):
        import torch.nn.functional as F
        return F.max_pool2d(x, kernel_size=2, stride=2)


class ChainedSpatialOpsBenchmark(BaseBenchmark):
    """Measure overhead in typical CNN block: Conv -> BN -> ReLU -> Pool."""

    name = "chained_spatial_ops"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._mlx_conv = None
        self._mlx_bn = None
        self._torch_conv = None
        self._torch_bn = None

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 1, "in_ch": 64, "out_ch": 64, "h": 56, "w": 56},
            {"batch": 32, "in_ch": 64, "out_ch": 128, "h": 56, "w": 56},
            {"batch": 32, "in_ch": 128, "out_ch": 256, "h": 28, "w": 28},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        import flashlight.nn as nn

        x = flashlight.randn(config["batch"], config["in_ch"], config["h"], config["w"])

        # Create layers
        self._mlx_conv = nn.Conv2d(config["in_ch"], config["out_ch"], 3, padding=1)
        self._mlx_bn = nn.BatchNorm2d(config["out_ch"])
        self._mlx_bn.eval()

        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        import torch.nn as nn

        x = torch.randn(
            config["batch"], config["in_ch"], config["h"], config["w"],
            device=device
        )

        self._torch_conv = nn.Conv2d(
            config["in_ch"], config["out_ch"], 3, padding=1
        ).to(device)
        self._torch_bn = nn.BatchNorm2d(config["out_ch"]).to(device)
        self._torch_bn.eval()

        return (x,)

    def mlx_operation(self, x):
        import flashlight.nn.functional as F

        # Conv -> BN -> ReLU -> MaxPool
        x = self._mlx_conv(x)
        x = self._mlx_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def pytorch_operation(self, x):
        import torch.nn.functional as F

        x = self._torch_conv(x)
        x = self._torch_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


# List of benchmark classes for registration
LAYOUT_BENCHMARKS = [
    TransposeOverheadBenchmark,
    Conv2dLayoutOverheadBenchmark,
    PoolingLayoutOverheadBenchmark,
    ChainedSpatialOpsBenchmark,
]
