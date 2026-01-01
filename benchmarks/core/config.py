"""
Benchmark configuration and result dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import datetime


class BenchmarkLevel(Enum):
    """Benchmark granularity level."""
    OPERATOR = "operator"
    LAYER = "layer"
    MODEL = "model"
    ALL = "all"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # Execution settings
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    num_trials: int = 5
    min_time_seconds: float = 0.1

    # Statistical settings
    percentiles: List[int] = field(default_factory=lambda: [50, 90, 95, 99])

    # Memory settings
    track_memory: bool = True
    force_gc: bool = True

    # Comparison settings
    compare_pytorch: bool = True
    pytorch_device: str = "mps"  # "mps", "cpu", "cuda"

    # Input size configurations
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 128])
    sequence_lengths: List[int] = field(default_factory=lambda: [32, 128, 512])
    feature_sizes: List[int] = field(default_factory=lambda: [64, 256, 1024])
    image_sizes: List[int] = field(default_factory=lambda: [32, 64, 224])

    # Precision settings (no float64 in MLX)
    dtypes: List[str] = field(default_factory=lambda: ["float32"])

    # Output settings
    output_format: str = "console"  # "console", "json", "both"
    output_file: Optional[str] = None
    verbose: bool = False

    # Tolerance for numerical comparison
    rtol: float = 1e-5
    atol: float = 1e-6

    def validate(self) -> None:
        """Validate configuration settings."""
        if "float64" in self.dtypes:
            raise ValueError(
                "MLX does not support float64. "
                "Use float32, float16, or bfloat16."
            )

        valid_devices = ["mps", "cpu", "cuda"]
        if self.pytorch_device not in valid_devices:
            raise ValueError(
                f"Invalid pytorch_device: {self.pytorch_device}. "
                f"Must be one of {valid_devices}"
            )

        valid_formats = ["console", "json", "both"]
        if self.output_format not in valid_formats:
            raise ValueError(
                f"Invalid output_format: {self.output_format}. "
                f"Must be one of {valid_formats}"
            )


@dataclass
class TimingStats:
    """Timing statistics from benchmark runs."""
    times_ms: List[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    percentiles: Dict[int, float]  # {50: 1.2, 90: 1.5, ...}
    total_iterations: int
    warmup_iterations: int


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    peak_mb: float
    allocated_mb: float
    delta_mb: float
    bandwidth_gbps: Optional[float] = None


@dataclass
class AccuracyStats:
    """Detailed numerical accuracy statistics."""
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    tolerance_tier: str  # 'STRICT', 'STANDARD', 'RELAXED', 'LOOSE'
    passed: bool
    rtol_used: float
    atol_used: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "mean_rel_diff": self.mean_rel_diff,
            "tolerance_tier": self.tolerance_tier,
            "passed": self.passed,
            "rtol_used": self.rtol_used,
            "atol_used": self.atol_used,
        }


@dataclass
class ComparisonStats:
    """Comparison statistics between frameworks."""
    speedup: float  # pytorch_time / mlx_time (>1 = mlx faster)
    relative_performance: str  # "1.5x faster" or "0.8x (20% slower)"
    numerical_match: bool
    max_abs_diff: float
    accuracy: Optional[AccuracyStats] = None  # Detailed accuracy metrics


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    level: BenchmarkLevel
    input_config: Dict[str, Any]

    # MLX timing
    mlx_timing: TimingStats

    # Throughput
    throughput: float
    throughput_unit: str  # "samples/sec", "ops/sec", "GFLOPS"

    # Optional: PyTorch comparison
    pytorch_timing: Optional[TimingStats] = None
    comparison: Optional[ComparisonStats] = None

    # Optional: Memory profiling
    mlx_memory: Optional[MemoryStats] = None
    pytorch_memory: Optional[MemoryStats] = None

    # Metadata
    dtype: str = "float32"
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "level": self.level.value,
            "input_config": self.input_config,
            "dtype": self.dtype,
            "timestamp": self.timestamp,
            "mlx_compat": {
                "mean_ms": self.mlx_timing.mean_ms,
                "std_ms": self.mlx_timing.std_ms,
                "min_ms": self.mlx_timing.min_ms,
                "max_ms": self.mlx_timing.max_ms,
                "median_ms": self.mlx_timing.median_ms,
                "percentiles": self.mlx_timing.percentiles,
            },
            "throughput": self.throughput,
            "throughput_unit": self.throughput_unit,
        }

        if self.pytorch_timing is not None:
            result["pytorch"] = {
                "mean_ms": self.pytorch_timing.mean_ms,
                "std_ms": self.pytorch_timing.std_ms,
                "min_ms": self.pytorch_timing.min_ms,
                "max_ms": self.pytorch_timing.max_ms,
                "median_ms": self.pytorch_timing.median_ms,
                "percentiles": self.pytorch_timing.percentiles,
            }

        if self.comparison is not None:
            comparison_dict = {
                "speedup": self.comparison.speedup,
                "relative_performance": self.comparison.relative_performance,
                "numerical_match": self.comparison.numerical_match,
                "max_abs_diff": self.comparison.max_abs_diff,
            }
            if self.comparison.accuracy is not None:
                comparison_dict["accuracy"] = self.comparison.accuracy.to_dict()
            result["comparison"] = comparison_dict

        if self.mlx_memory is not None:
            result["mlx_memory"] = {
                "peak_mb": self.mlx_memory.peak_mb,
                "allocated_mb": self.mlx_memory.allocated_mb,
                "bandwidth_gbps": self.mlx_memory.bandwidth_gbps,
            }

        if self.error is not None:
            result["error"] = self.error

        return result


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""
    total_benchmarks: int
    by_level: Dict[str, int]
    faster_count: int
    slower_count: int
    average_speedup: float
    geometric_mean_speedup: float
    numerical_parity_passed: int
    numerical_parity_failed: int
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary for JSON serialization."""
        return {
            "total_benchmarks": self.total_benchmarks,
            "by_level": self.by_level,
            "comparison": {
                "faster_count": self.faster_count,
                "slower_count": self.slower_count,
                "average_speedup": self.average_speedup,
                "geometric_mean_speedup": self.geometric_mean_speedup,
            },
            "numerical_parity": {
                "passed": self.numerical_parity_passed,
                "failed": self.numerical_parity_failed,
            },
            "errors": self.errors,
        }
