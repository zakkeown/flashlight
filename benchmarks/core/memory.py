"""
Memory profiling for MLX unified memory and PyTorch.

MLX uses Apple's unified memory architecture where CPU and GPU
share the same memory pool. This module tracks memory usage
using process-level monitoring via psutil.
"""

import gc
import os
from dataclasses import dataclass
from typing import Callable, Optional, Any, Tuple

from benchmarks.core.config import MemoryStats


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    shared_mb: float  # Shared memory


class MLXMemoryTracker:
    """
    Memory tracker for MLX unified memory.

    MLX-Specific Considerations:
    - Unified memory architecture (shared CPU/GPU memory)
    - Lazy evaluation means memory is allocated on first use
    - No explicit GPU memory allocation tracking like CUDA
    - Use process-level tracking via psutil

    Example:
        tracker = MLXMemoryTracker()

        profile = tracker.profile_function(
            my_function, arg1, arg2,
            tensor_bytes=1024 * 1024 * 4,  # 4MB
        )

        print(f"Peak memory: {profile.peak_mb:.1f} MB")
        print(f"Bandwidth: {profile.bandwidth_gbps:.1f} GB/s")
    """

    def __init__(self, force_gc: bool = True):
        """
        Initialize tracker.

        Args:
            force_gc: Force garbage collection before profiling
        """
        self.force_gc = force_gc
        self._psutil_available = self._check_psutil()

    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False

    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory usage."""
        if not self._psutil_available:
            return MemorySnapshot(0.0, 0.0, 0.0)

        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        return MemorySnapshot(
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            shared_mb=getattr(mem_info, 'shared', 0) / (1024 * 1024),
        )

    def _sync_mlx(self) -> None:
        """Synchronize MLX operations."""
        try:
            import mlx.core as mx
            mx.eval(mx.array([0.0]))
        except ImportError:
            pass

    def profile_function(
        self,
        fn: Callable[..., Any],
        *args: Any,
        tensor_bytes: Optional[int] = None,
        time_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> MemoryStats:
        """
        Profile memory usage of a function.

        Args:
            fn: Function to profile
            args: Positional arguments for fn
            tensor_bytes: Total bytes for bandwidth calculation
            time_ms: Execution time for bandwidth calculation
            kwargs: Keyword arguments for fn

        Returns:
            MemoryStats with memory metrics
        """
        # Force GC and sync before measurement
        if self.force_gc:
            gc.collect()
        self._sync_mlx()

        # Get baseline
        before = self._get_memory_snapshot()

        # Execute function
        result = fn(*args, **kwargs)

        # Sync and measure after
        if hasattr(result, '_mlx_array'):
            import mlx.core as mx
            mx.eval(result._mlx_array)
        elif hasattr(result, 'numpy'):
            # Force evaluation
            _ = result.numpy()
        self._sync_mlx()

        after = self._get_memory_snapshot()

        # Calculate memory delta
        delta_mb = after.rss_mb - before.rss_mb
        peak_mb = after.rss_mb  # Approximate peak as final

        # Calculate bandwidth if timing provided
        bandwidth_gbps = None
        if tensor_bytes is not None and time_ms is not None and time_ms > 0:
            bandwidth_gbps = self.estimate_bandwidth(tensor_bytes, time_ms)

        return MemoryStats(
            peak_mb=peak_mb,
            allocated_mb=delta_mb,
            delta_mb=delta_mb,
            bandwidth_gbps=bandwidth_gbps,
        )

    def estimate_bandwidth(
        self,
        bytes_transferred: int,
        time_ms: float,
    ) -> float:
        """
        Estimate memory bandwidth in GB/s.

        Args:
            bytes_transferred: Total bytes read + written
            time_ms: Operation time in milliseconds

        Returns:
            Estimated bandwidth in GB/s
        """
        if time_ms <= 0:
            return 0.0

        time_seconds = time_ms / 1000.0
        bytes_per_second = bytes_transferred / time_seconds
        return bytes_per_second / (1024 ** 3)  # Convert to GB/s

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        snapshot = self._get_memory_snapshot()
        return snapshot.rss_mb


class PyTorchMemoryTracker:
    """
    Memory tracker for PyTorch MPS backend.

    MPS has limited memory introspection compared to CUDA,
    but we can track driver-allocated memory.
    """

    def __init__(self, device: str = "mps"):
        """
        Initialize tracker.

        Args:
            device: PyTorch device ("mps", "cuda", "cpu")
        """
        self.device = device
        self._pytorch_available = self._check_pytorch()

    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def _sync(self) -> None:
        """Synchronize device."""
        try:
            import torch
            if self.device == "mps":
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            elif self.device.startswith("cuda"):
                torch.cuda.synchronize()
        except (ImportError, AttributeError):
            pass

    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Get current PyTorch memory usage."""
        if not self._pytorch_available:
            return MemorySnapshot(0.0, 0.0, 0.0)

        try:
            import torch

            if self.device == "mps":
                # MPS has limited memory introspection
                if hasattr(torch.mps, 'driver_allocated_memory'):
                    allocated = torch.mps.driver_allocated_memory() / (1024 * 1024)
                else:
                    allocated = 0.0
                return MemorySnapshot(
                    rss_mb=allocated,
                    vms_mb=0.0,
                    shared_mb=0.0,
                )

            elif self.device.startswith("cuda"):
                # CUDA has rich memory tracking
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                return MemorySnapshot(
                    rss_mb=allocated,
                    vms_mb=reserved,
                    shared_mb=0.0,
                )

            else:
                # CPU - use process memory
                import psutil
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return MemorySnapshot(
                    rss_mb=mem_info.rss / (1024 * 1024),
                    vms_mb=mem_info.vms / (1024 * 1024),
                    shared_mb=0.0,
                )

        except Exception:
            return MemorySnapshot(0.0, 0.0, 0.0)

    def profile_function(
        self,
        fn: Callable[..., Any],
        *args: Any,
        tensor_bytes: Optional[int] = None,
        time_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> MemoryStats:
        """
        Profile memory usage of a PyTorch function.

        Args:
            fn: Function to profile
            args: Positional arguments for fn
            tensor_bytes: Total bytes for bandwidth calculation
            time_ms: Execution time for bandwidth calculation
            kwargs: Keyword arguments for fn

        Returns:
            MemoryStats with memory metrics
        """
        gc.collect()
        self._sync()

        before = self._get_memory_snapshot()

        result = fn(*args, **kwargs)

        self._sync()

        after = self._get_memory_snapshot()

        delta_mb = after.rss_mb - before.rss_mb
        peak_mb = after.rss_mb

        bandwidth_gbps = None
        if tensor_bytes is not None and time_ms is not None and time_ms > 0:
            time_seconds = time_ms / 1000.0
            bytes_per_second = tensor_bytes / time_seconds
            bandwidth_gbps = bytes_per_second / (1024 ** 3)

        return MemoryStats(
            peak_mb=peak_mb,
            allocated_mb=delta_mb,
            delta_mb=delta_mb,
            bandwidth_gbps=bandwidth_gbps,
        )


def calculate_tensor_bytes(
    shape: Tuple[int, ...],
    dtype: str = "float32",
) -> int:
    """
    Calculate total bytes for a tensor.

    Args:
        shape: Tensor shape
        dtype: Data type string

    Returns:
        Total bytes
    """
    dtype_sizes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "int64": 8,
        "int16": 2,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }

    element_size = dtype_sizes.get(dtype, 4)
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    return num_elements * element_size


def calculate_matmul_bytes(
    m: int, k: int, n: int,
    dtype: str = "float32",
) -> int:
    """
    Calculate bytes transferred for matmul.

    Args:
        m, k, n: Matrix dimensions (m x k) @ (k x n) = (m x n)
        dtype: Data type

    Returns:
        Bytes read + written
    """
    element_size = 4 if dtype == "float32" else 2

    # Bytes read: A (m*k) + B (k*n)
    # Bytes written: C (m*n)
    bytes_read = (m * k + k * n) * element_size
    bytes_written = m * n * element_size

    return bytes_read + bytes_written


def calculate_conv2d_bytes(
    batch: int,
    c_in: int,
    c_out: int,
    h: int,
    w: int,
    kernel_size: int,
    dtype: str = "float32",
) -> int:
    """
    Calculate bytes transferred for conv2d.

    Args:
        batch: Batch size
        c_in: Input channels
        c_out: Output channels
        h, w: Spatial dimensions
        kernel_size: Kernel size
        dtype: Data type

    Returns:
        Bytes read + written
    """
    element_size = 4 if dtype == "float32" else 2

    # Input: batch * c_in * h * w
    input_bytes = batch * c_in * h * w * element_size

    # Weights: c_out * c_in * kernel_size * kernel_size
    weight_bytes = c_out * c_in * kernel_size * kernel_size * element_size

    # Output: batch * c_out * h_out * w_out (approximately h * w for same padding)
    output_bytes = batch * c_out * h * w * element_size

    return input_bytes + weight_bytes + output_bytes
