"""
PyTorch-Compatible Random Number Generator

Provides a Generator class that wraps the Philox engine with PyTorch-compatible API.
"""

from typing import Any, Optional

from .philox import PhiloxEngine


class Generator:
    """
    Random number generator with PyTorch-compatible API.

    This class wraps the Philox engine to provide the same interface
    as torch.Generator. It maintains state that can be saved/restored
    and supports seeding for reproducibility.

    Args:
        device: Device for the generator (ignored in MLX - unified memory)

    Example:
        >>> g = Generator()
        >>> g.manual_seed(42)
        >>> # Use with random functions:
        >>> x = flashlight.randn(3, 3, generator=g)
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize generator with optional device specification."""
        self._device = device
        self._engine = PhiloxEngine()
        self._initial_seed: Optional[int] = None

    def manual_seed(self, seed: int) -> "Generator":
        """
        Set the seed for the generator.

        Args:
            seed: The desired seed (0 to 2^64-1)

        Returns:
            self for chaining
        """
        self._initial_seed = seed
        self._engine = PhiloxEngine(seed=seed)
        return self

    def seed(self) -> int:
        """
        Get a non-deterministic random number to seed the generator.

        Returns:
            A 64-bit random seed
        """
        import random
        import time

        # Use system entropy sources
        new_seed = random.randint(0, 2**63 - 1) ^ int(time.time() * 1000000)
        new_seed = new_seed & ((1 << 64) - 1)

        self._initial_seed = new_seed
        self._engine = PhiloxEngine(seed=new_seed)
        return new_seed

    def initial_seed(self) -> int:
        """
        Return the initial seed for generating random numbers.

        Returns:
            The initial seed value, or 0 if not set
        """
        return self._initial_seed if self._initial_seed is not None else 0

    def get_state(self) -> "Tensor":
        """
        Return the state of the generator as a ByteTensor.

        The state format is compatible with PyTorch MPS generator (44 bytes):
        - bytes [0:20]:  5 uint32s with value 1 (internal flags/magic)
        - bytes [20:28]: current seed as uint64 (little-endian)
        - bytes [28:36]: original seed as uint64 (little-endian)
        - bytes [36:44]: offset as uint64 (little-endian)

        Returns:
            A Tensor containing the serialized state (44 bytes, uint8)
        """
        # Import here to avoid circular imports
        from ..tensor import Tensor
        from ..dtype import uint8

        state = self._engine.get_state()

        # Build 44-byte state in PyTorch MPS format
        state_bytes = []

        # Helper to pack uint32 as little-endian bytes
        def pack_uint32(val):
            return [
                val & 0xFF,
                (val >> 8) & 0xFF,
                (val >> 16) & 0xFF,
                (val >> 24) & 0xFF,
            ]

        # Helper to pack uint64 as little-endian bytes
        def pack_uint64(val):
            return pack_uint32(val & 0xFFFFFFFF) + pack_uint32((val >> 32) & 0xFFFFFFFF)

        # bytes [0:20]: 5 uint32s with value 1 (magic numbers/flags)
        for _ in range(5):
            state_bytes.extend(pack_uint32(1))

        # bytes [20:28]: current seed as uint64
        seed = state["key"][0] | (state["key"][1] << 32)
        state_bytes.extend(pack_uint64(seed))

        # bytes [28:36]: original seed as uint64
        original_seed = self._initial_seed if self._initial_seed is not None else seed
        state_bytes.extend(pack_uint64(original_seed))

        # bytes [36:44]: offset as uint64
        # PyTorch MPS tracks offset as total random numbers consumed
        # Our Philox state: counter=N means block N-1 was just computed (for N>0)
        # state_idx=K means K values have been consumed from current block
        # Special case: counter=0, idx=0 means nothing consumed yet
        # After consuming values: counter=1, idx=1 means 1 value consumed
        #                        counter=1, idx=0 means 4 values consumed (block done)
        #                        counter=2, idx=1 means 5 values consumed
        block_offset = state["counter"][0] | (state["counter"][1] << 32)
        state_idx = state["state_idx"]

        if state_idx == 0:
            # All values from previous block consumed, or nothing consumed yet
            if block_offset == 0:
                offset = 0  # Nothing consumed
            else:
                offset = block_offset * 4  # block_offset blocks fully consumed
        else:
            # Partway through block at counter-1 (since counter increments when generating)
            offset = (block_offset - 1) * 4 + state_idx

        state_bytes.extend(pack_uint64(offset))

        return Tensor(state_bytes, dtype=uint8)

    def set_state(self, new_state: "Tensor") -> None:
        """
        Set the state of the generator from a ByteTensor.

        Accepts PyTorch MPS format (44 bytes):
        - bytes [0:20]:  5 uint32s (internal flags, ignored on restore)
        - bytes [20:28]: current seed as uint64
        - bytes [28:36]: original seed as uint64
        - bytes [36:44]: offset as uint64

        Args:
            new_state: A Tensor containing the serialized state (44 bytes)
        """
        # Convert tensor to list of bytes
        if hasattr(new_state, 'tolist'):
            state_bytes = new_state.tolist()
        else:
            state_bytes = list(new_state)

        def unpack_uint32(offset):
            return (
                int(state_bytes[offset]) |
                (int(state_bytes[offset + 1]) << 8) |
                (int(state_bytes[offset + 2]) << 16) |
                (int(state_bytes[offset + 3]) << 24)
            )

        def unpack_uint64(offset):
            return unpack_uint32(offset) | (unpack_uint32(offset + 4) << 32)

        # Parse MPS format
        # bytes [20:28]: seed
        seed = unpack_uint64(20)
        # bytes [28:36]: original seed
        original_seed = unpack_uint64(28)
        # bytes [36:44]: offset (total random numbers consumed)
        offset = unpack_uint64(36)

        # Convert MPS offset to Philox state
        # offset = total random numbers consumed
        # Need to figure out counter and state_idx
        #
        # If offset=0: counter=0, idx=0 (nothing consumed)
        # If offset=1: counter=1, idx=1 (first block generated, 1 value consumed)
        # If offset=4: counter=1, idx=0 (first block generated, all 4 consumed, ready for next)
        # If offset=5: counter=2, idx=1 (second block generated, 1 value consumed)

        if offset == 0:
            block_counter = 0
            state_idx = 0
            output = [0, 0, 0, 0]
        else:
            # Which block are we in? offset 1-4 is block 0, 5-8 is block 1, etc.
            block_number = (offset - 1) // 4
            # Position within block: 1->1, 2->2, 3->3, 4->0, 5->1, etc.
            state_idx = ((offset - 1) % 4) + 1
            if state_idx == 4:
                state_idx = 0
                block_counter = block_number + 1  # Ready for next block
                output = [0, 0, 0, 0]  # Will be regenerated on next call
            else:
                block_counter = block_number + 1  # Counter after generating this block
                # Regenerate the output for this block
                # _rand() uses the counter value passed to it, not self._counter
                key = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF]
                counter = [block_number, 0, 0, 0]  # Block N is generated with counter=N
                output = self._engine._rand(counter, key)

        # Set up the engine state
        state = {
            "key": [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF],
            "counter": [
                block_counter & 0xFFFFFFFF,
                (block_counter >> 32) & 0xFFFFFFFF,
                0,  # subsequence (not tracked in MPS format)
                0,
            ],
            "output": output,
            "state_idx": state_idx,
        }

        self._engine.set_state(state)
        self._initial_seed = original_seed

    @property
    def device(self) -> Any:
        """
        Return the device of the generator.

        For MLX compatibility, always returns a mock device with type "mps".
        """
        return type("device", (), {"type": "mps"})()

    # Internal methods for use by random factories

    def _next_uint32(self) -> int:
        """Get next uint32 from engine."""
        return self._engine()

    def _next_uniform(self) -> float:
        """Get next uniform float in [0, 1)."""
        return self._engine.uniform()

    def _next_normal(self) -> float:
        """Get next standard normal value."""
        return self._engine.normal()

    def _next_normal_pair(self):
        """Get next pair of standard normal values."""
        return self._engine.randn_pair()

    def fork(self, subsequence: int = None) -> "Generator":
        """
        Create a new generator forked from this one.

        Useful for parallel random number generation.

        Args:
            subsequence: Optional subsequence index. If None, uses next available.

        Returns:
            A new Generator with independent state
        """
        new_gen = Generator(device=self._device)
        new_gen._engine = self._engine.fork(subsequence)
        new_gen._initial_seed = self._initial_seed
        return new_gen


# Default global generator instance
default_generator = Generator()
