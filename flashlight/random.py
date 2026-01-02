"""
Random Number Generation Module

PyTorch-compatible torch.random module for MLX.
Provides random number generation utilities and state management.

The implementation uses a Philox-based generator for state management
and reproducibility, while delegating bulk tensor generation to MLX
for performance (with deterministic seeding from Philox state).
"""

import warnings
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import mlx.core as mx

from .rng.philox import PhiloxEngine
from .rng.generator import Generator as PhiloxGenerator


class Generator:
    """
    Random number generator with PyTorch-compatible API.

    This class provides deterministic random number generation using
    the Philox algorithm (matching PyTorch's GPU implementation).
    For bulk tensor generation, it seeds MLX's RNG for performance
    while maintaining reproducibility.

    Args:
        device: Device for the generator (ignored in MLX - unified memory)

    Example:
        >>> g = flashlight.Generator()
        >>> g.manual_seed(42)
        >>> x = flashlight.randn(3, 3, generator=g)
    """

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._philox = PhiloxGenerator(device=device)
        self._initial_seed_value: Optional[int] = None

    def manual_seed(self, seed: int) -> "Generator":
        """
        Set the seed for the generator.

        Args:
            seed: The desired seed

        Returns:
            self for chaining
        """
        self._initial_seed_value = seed
        self._philox.manual_seed(seed)
        return self

    def seed(self) -> int:
        """
        Get a non-deterministic random number to seed the generator.

        Returns:
            A 64-bit number
        """
        new_seed = self._philox.seed()
        self._initial_seed_value = new_seed
        return new_seed

    def initial_seed(self) -> int:
        """
        Return the initial seed for generating random numbers.

        Returns:
            The seed value
        """
        return self._initial_seed_value if self._initial_seed_value is not None else 0

    def get_state(self) -> "Tensor":
        """
        Return the state of the generator as a ByteTensor.

        The state format is compatible with PyTorch MPS generator (44 bytes).

        Returns:
            A Tensor containing the serialized state (44 bytes, uint8)
        """
        return self._philox.get_state()

    def set_state(self, state: "Tensor") -> None:
        """
        Set the state of the generator from a ByteTensor.

        Args:
            state: A Tensor containing the serialized state (44 bytes)
        """
        self._philox.set_state(state)
        # Update initial seed from the restored state
        self._initial_seed_value = self._philox._initial_seed

    @property
    def device(self):
        """Return the device of the generator."""
        return self._philox.device

    # Internal methods for random generation

    def _next_uint32(self) -> int:
        """Get next uint32 from Philox engine."""
        return self._philox._next_uint32()

    def _next_uniform(self) -> float:
        """Get next uniform float in [0, 1)."""
        return self._philox._next_uniform()

    def _next_normal(self) -> float:
        """Get next standard normal value."""
        return self._philox._next_normal()

    def _seed_mlx(self) -> None:
        """
        Seed MLX's RNG using current Philox state for bulk generation.

        This allows us to use MLX's fast vectorized RNG while maintaining
        determinism based on our Philox state.
        """
        # Use next uint32 values to create a seed for MLX
        seed = self._next_uint32() ^ (self._next_uint32() << 32)
        mx.random.seed(seed & ((1 << 64) - 1))


# Default generator instance
default_generator = Generator()


def manual_seed(seed: int) -> Generator:
    """
    Set the seed for generating random numbers.

    Sets the seed for the global random number generator.
    Also seeds MLX's internal RNG for operations that use it directly.

    Args:
        seed: The desired seed

    Returns:
        The default Generator object

    Example:
        >>> flashlight.manual_seed(42)
        >>> x = flashlight.randn(3, 3)  # Reproducible
        >>> flashlight.manual_seed(42)
        >>> y = flashlight.randn(3, 3)  # Same as x
    """
    # Seed our Philox generator
    default_generator.manual_seed(seed)

    # Also seed MLX for operations that use it directly
    # (ensures reproducibility even for ops not using our generator)
    mx.random.seed(seed)

    return default_generator


def seed() -> int:
    """
    Set the seed for generating random numbers to a non-deterministic random number.

    Returns:
        The seed used
    """
    new_seed = default_generator.seed()
    mx.random.seed(new_seed)
    return new_seed


def initial_seed() -> int:
    """
    Return the initial seed for generating random numbers.

    Returns:
        The initial seed
    """
    return default_generator.initial_seed()


def get_rng_state() -> "Tensor":
    """
    Get the random number generator state as a ByteTensor.

    Returns:
        A Tensor containing the serialized RNG state (44 bytes, uint8)
    """
    return default_generator.get_state()


def set_rng_state(new_state: "Tensor") -> None:
    """
    Set the random number generator state from a ByteTensor.

    Args:
        new_state: A Tensor containing the serialized state (44 bytes)
    """
    default_generator.set_state(new_state)

    # Also reseed MLX with the initial seed for consistency
    if default_generator._initial_seed_value is not None:
        mx.random.seed(default_generator._initial_seed_value)


@contextmanager
def fork_rng(
    devices: Optional[List[Any]] = None,
    enabled: bool = True,
    device_type: str = "cuda",
    _caller: str = "fork_rng",
    _devices_kw: str = "devices",
):
    """
    Fork the RNG state, run some operations, then restore the original state.

    This is useful when you want to run operations that modify the RNG state
    but want to restore the original state afterward.

    Args:
        devices: List of devices to fork RNG for (ignored in MLX)
        enabled: Whether to actually fork the RNG
        device_type: Type of device (ignored in MLX)
        _caller: Internal parameter for error messages
        _devices_kw: Internal parameter for error messages

    Yields:
        None

    Example:
        >>> flashlight.manual_seed(42)
        >>> with flashlight.fork_rng():
        ...     x = flashlight.randn(3, 3)  # Uses forked state
        >>> y = flashlight.randn(3, 3)  # Original state restored
    """
    if not enabled:
        yield
        return

    # Save current state
    rng_state = get_rng_state()

    try:
        yield
    finally:
        # Restore state
        set_rng_state(rng_state)


__all__ = [
    "Generator",
    "default_generator",
    "manual_seed",
    "seed",
    "initial_seed",
    "get_rng_state",
    "set_rng_state",
    "fork_rng",
]
