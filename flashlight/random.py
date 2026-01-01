"""
Random Number Generation Module

PyTorch-compatible torch.random module for MLX.
Provides random number generation utilities and state management.
"""

from typing import Optional, Any, List, Union
from contextlib import contextmanager
import warnings

import mlx.core as mx


class Generator:
    """
    Random number generator.

    In MLX, this provides API compatibility. MLX uses global RNG state
    managed through mx.random.seed() and mx.random.key().

    Args:
        device: Device for the generator (ignored in MLX - unified memory)
    """

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._seed: Optional[int] = None
        self._key: Optional[mx.array] = None

    def manual_seed(self, seed: int) -> "Generator":
        """
        Set the seed for the generator.

        Args:
            seed: The desired seed

        Returns:
            self
        """
        self._seed = seed
        self._key = mx.random.key(seed)
        return self

    def seed(self) -> int:
        """
        Get a non-deterministic random number to seed the generator.

        Returns:
            A 64-bit number
        """
        import random
        new_seed = random.randint(0, 2**63 - 1)
        self._seed = new_seed
        self._key = mx.random.key(new_seed)
        return new_seed

    def initial_seed(self) -> int:
        """
        Return the initial seed for generating random numbers.

        Returns:
            The seed value
        """
        return self._seed if self._seed is not None else 0

    def get_state(self) -> dict:
        """
        Return the state of the generator.

        Returns:
            A dictionary containing the generator state
        """
        return {
            "seed": self._seed,
            "key": self._key,
        }

    def set_state(self, state: dict) -> None:
        """
        Set the state of the generator.

        Args:
            state: A dictionary containing the generator state
        """
        self._seed = state.get("seed")
        self._key = state.get("key")

    @property
    def device(self):
        """Return the device of the generator."""
        # Return a mock device object for compatibility
        return type("device", (), {"type": "mps"})()


# Default generator instance
default_generator = Generator()


def manual_seed(seed: int) -> Generator:
    """
    Set the seed for generating random numbers.

    Sets the seed for the global random number generator.

    Args:
        seed: The desired seed

    Returns:
        A Generator object
    """
    mx.random.seed(seed)
    default_generator.manual_seed(seed)
    return default_generator


def seed() -> int:
    """
    Set the seed for generating random numbers to a non-deterministic random number.

    Returns:
        The seed used
    """
    import random
    new_seed = random.randint(0, 2**63 - 1)
    mx.random.seed(new_seed)
    default_generator.manual_seed(new_seed)
    return new_seed


def initial_seed() -> int:
    """
    Return the initial seed for generating random numbers.

    Returns:
        The initial seed
    """
    return default_generator.initial_seed()


def get_rng_state() -> dict:
    """
    Get the random number generator state.

    Returns:
        A dictionary representing the RNG state
    """
    return default_generator.get_state()


def set_rng_state(new_state: dict) -> None:
    """
    Set the random number generator state.

    Args:
        new_state: The desired state dictionary
    """
    default_generator.set_state(new_state)
    if new_state.get("seed") is not None:
        mx.random.seed(new_state["seed"])


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
    'Generator',
    'default_generator',
    'manual_seed',
    'seed',
    'initial_seed',
    'get_rng_state',
    'set_rng_state',
    'fork_rng',
]
