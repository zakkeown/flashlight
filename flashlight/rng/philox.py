"""
Philox 4x32-10 Random Number Generator

This implements PyTorch's exact Philox algorithm for byte-perfect RNG parity.
The Philox algorithm is a counter-based PRNG that produces high-quality random
numbers suitable for parallel computation.

Reference:
- http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- PyTorch: aten/src/ATen/core/PhiloxRNGEngine.h
"""

import math
from typing import List, Tuple


class PhiloxEngine:
    """
    Philox 4x32-10 PRNG matching PyTorch exactly.

    This counter-based RNG produces 2^192 total random values:
    - 2^64 possible seed values
    - 2^64 subsequences per seed
    - 2^64 values per subsequence

    Each call generates 4 uint32 values (128 bits), returned one at a time.

    Args:
        seed: Seed value (0 to 2^64-1). Default matches PyTorch.
        subsequence: Subsequence index for parallel generation.
        offset: How many 128-bit blocks to skip initially.

    Example:
        >>> engine = PhiloxEngine(seed=42)
        >>> value = engine()  # Get one uint32
        >>> uniform = engine.uniform()  # Get float in [0, 1)
        >>> normal = engine.normal()  # Get standard normal
    """

    # Philox constants (from PyTorch/Random123)
    PHILOX_10A = 0x9E3779B9  # Key increment constant 0
    PHILOX_10B = 0xBB67AE85  # Key increment constant 1
    PHILOX_SA = 0xD2511F53   # Multiplier constant 0
    PHILOX_SB = 0xCD9E8D57   # Multiplier constant 1

    # Mask for 32-bit operations
    UINT32_MAX = 0xFFFFFFFF

    def __init__(
        self,
        seed: int = 67280421310721,
        subsequence: int = 0,
        offset: int = 0,
    ):
        """Initialize Philox engine with seed, subsequence, and offset."""
        self._key: List[int] = [0, 0]
        self._counter: List[int] = [0, 0, 0, 0]
        self._output: List[int] = [0, 0, 0, 0]
        self._state_idx: int = 0

        self.reset_state(seed, subsequence)
        if offset > 0:
            self.incr_n(offset)

    def reset_state(self, seed: int = 67280421310721, subsequence: int = 0) -> None:
        """Reset the engine state with new seed and subsequence."""
        # Split 64-bit seed into two 32-bit parts
        self._key[0] = seed & self.UINT32_MAX
        self._key[1] = (seed >> 32) & self.UINT32_MAX

        # Counter: [offset_lo, offset_hi, subseq_lo, subseq_hi]
        self._counter = [0, 0, 0, 0]
        self._counter[2] = subsequence & self.UINT32_MAX
        self._counter[3] = (subsequence >> 32) & self.UINT32_MAX

        self._state_idx = 0

    def set_offset(self, offset: int) -> None:
        """Set the offset (counter[0:1]) directly."""
        self._counter[0] = offset & self.UINT32_MAX
        self._counter[1] = (offset >> 32) & self.UINT32_MAX

    def get_offset(self) -> int:
        """Get the current offset."""
        return self._counter[0] | (self._counter[1] << 32)

    def _mulhilo32(self, a: int, b: int) -> Tuple[int, int]:
        """
        Multiply two 32-bit integers, returning (low32, high32).

        This matches PyTorch's mulhilo32 function.
        """
        product = a * b
        lo = product & self.UINT32_MAX
        hi = (product >> 32) & self.UINT32_MAX
        return lo, hi

    def _single_round(self, ctr: List[int], key: List[int]) -> List[int]:
        """
        Perform one round of the Philox cipher.

        The round function is:
        - Multiply ctr[0] and ctr[2] by constants
        - XOR results with key and other counter elements
        - Shuffle the results
        """
        lo0, hi0 = self._mulhilo32(self.PHILOX_SA, ctr[0])
        lo1, hi1 = self._mulhilo32(self.PHILOX_SB, ctr[2])

        # Combine results with XOR and shuffle
        ret = [0, 0, 0, 0]
        ret[0] = (hi1 ^ ctr[1] ^ key[0]) & self.UINT32_MAX
        ret[1] = lo1 & self.UINT32_MAX
        ret[2] = (hi0 ^ ctr[3] ^ key[1]) & self.UINT32_MAX
        ret[3] = lo0 & self.UINT32_MAX

        return ret

    def _rand(self, counter: List[int], key: List[int], n_rounds: int = 10) -> List[int]:
        """
        Generate 4 random uint32 values using n_rounds of the Philox cipher.

        Default is 10 rounds for cryptographic strength.
        """
        ctr = counter.copy()
        k = key.copy()

        for _ in range(n_rounds - 1):
            ctr = self._single_round(ctr, k)
            # Bump the key
            k[0] = (k[0] + self.PHILOX_10A) & self.UINT32_MAX
            k[1] = (k[1] + self.PHILOX_10B) & self.UINT32_MAX

        # Final round without key bump
        return self._single_round(ctr, k)

    def incr(self) -> None:
        """Increment the counter by 1 (one 128-bit block)."""
        self._counter[0] = (self._counter[0] + 1) & self.UINT32_MAX
        if self._counter[0] != 0:
            return
        self._counter[1] = (self._counter[1] + 1) & self.UINT32_MAX
        if self._counter[1] != 0:
            return
        self._counter[2] = (self._counter[2] + 1) & self.UINT32_MAX
        if self._counter[2] != 0:
            return
        self._counter[3] = (self._counter[3] + 1) & self.UINT32_MAX

    def incr_n(self, n: int) -> None:
        """Skip n 128-bit blocks in the sequence."""
        nlo = n & self.UINT32_MAX
        nhi = (n >> 32) & self.UINT32_MAX

        old_counter_0 = self._counter[0]
        self._counter[0] = (self._counter[0] + nlo) & self.UINT32_MAX

        # Check for overflow in counter[0]
        if self._counter[0] < old_counter_0:
            nhi += 1

        old_counter_1 = self._counter[1]
        self._counter[1] = (self._counter[1] + nhi) & self.UINT32_MAX

        # Check for overflow in counter[1]
        if nhi > 0 and self._counter[1] < old_counter_1:
            self._counter[2] = (self._counter[2] + 1) & self.UINT32_MAX
            if self._counter[2] == 0:
                self._counter[3] = (self._counter[3] + 1) & self.UINT32_MAX

    def __call__(self, n_rounds: int = 10) -> int:
        """
        Generate one uint32 random value.

        Internally generates 4 values at a time and returns them sequentially.
        """
        if self._state_idx == 0:
            # Generate new batch of 4 values
            self._output = self._rand(self._counter, self._key, n_rounds)
            self.incr()

        result = self._output[self._state_idx]
        self._state_idx = (self._state_idx + 1) & 3
        return result

    def _uint32_to_uniform_float(self, value: int) -> float:
        """
        Convert uint32 to uniform float in [0, 1).

        Matches PyTorch's conversion exactly.
        """
        # Maximum value such that MAX_INT * scale < 1.0 (with float rounding)
        scale = 4.6566127342e-10
        return (value & 0x7FFFFFFF) * scale

    def uniform(self) -> float:
        """Generate a uniform random float in [0, 1)."""
        return self._uint32_to_uniform_float(self())

    def normal(self, n_rounds: int = 10) -> float:
        """
        Generate a standard normal random value using Box-Muller transform.

        Matches PyTorch's randn implementation.
        """
        if self._state_idx == 0:
            self._output = self._rand(self._counter, self._key, n_rounds)
            self.incr()

        # Box-Muller uses 2 uniform values
        # PyTorch uses output[0] and output[1]
        u1 = 1.0 - self._uint32_to_uniform_float(self._output[0])  # (0, 1]
        u2 = 1.0 - self._uint32_to_uniform_float(self._output[1])

        # Consume all 4 values from this batch to stay in sync with PyTorch
        self._state_idx = 0  # Reset to force new batch on next call

        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def randn_pair(self, n_rounds: int = 10) -> Tuple[float, float]:
        """
        Generate a pair of standard normal values using Box-Muller.

        More efficient than calling normal() twice since Box-Muller
        naturally produces pairs.
        """
        if self._state_idx == 0:
            self._output = self._rand(self._counter, self._key, n_rounds)
            self.incr()

        u1 = 1.0 - self._uint32_to_uniform_float(self._output[0])
        u2 = 1.0 - self._uint32_to_uniform_float(self._output[1])

        self._state_idx = 0

        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2

        return r * math.cos(theta), r * math.sin(theta)

    def get_state(self) -> dict:
        """Get the complete engine state for serialization."""
        return {
            "key": self._key.copy(),
            "counter": self._counter.copy(),
            "output": self._output.copy(),
            "state_idx": self._state_idx,
        }

    def set_state(self, state: dict) -> None:
        """Restore engine state from serialized form."""
        self._key = state["key"].copy()
        self._counter = state["counter"].copy()
        self._output = state["output"].copy()
        self._state_idx = state["state_idx"]

    def fork(self, subsequence: int = None) -> "PhiloxEngine":
        """
        Create a new engine with the same seed but different subsequence.

        Useful for parallel random number generation.
        """
        seed = self._key[0] | (self._key[1] << 32)
        if subsequence is None:
            # Use next subsequence
            current_subseq = self._counter[2] | (self._counter[3] << 32)
            subsequence = current_subseq + 1

        return PhiloxEngine(seed=seed, subsequence=subsequence)


# For API compatibility
Philox4_32 = PhiloxEngine
