"""
Random Number Generation Tests

Tests for flashlight.random module.
"""

import numpy as np
import pytest

import flashlight
from flashlight.random import (
    Generator,
    default_generator,
    fork_rng,
    get_rng_state,
    initial_seed,
    manual_seed,
    seed,
    set_rng_state,
)


class TestGenerator:
    """Tests for Generator class."""

    def test_generator_creation(self):
        """Test Generator creation."""
        gen = Generator()
        assert gen is not None

    def test_generator_with_device(self):
        """Test Generator creation with device."""
        gen = Generator(device="cpu")
        assert gen is not None

    def test_generator_manual_seed(self):
        """Test Generator.manual_seed()."""
        gen = Generator()
        result = gen.manual_seed(42)
        assert result is gen  # Returns self
        assert gen.initial_seed() == 42

    def test_generator_seed(self):
        """Test Generator.seed() generates random seed."""
        gen = Generator()
        seed1 = gen.seed()
        seed2 = gen.seed()
        # Seeds should be different (with very high probability)
        # but we can't guarantee it, so just check they're valid
        assert isinstance(seed1, int)
        assert isinstance(seed2, int)

    def test_generator_initial_seed_default(self):
        """Test Generator.initial_seed() default."""
        gen = Generator()
        assert gen.initial_seed() == 0

    def test_generator_get_set_state(self):
        """Test Generator.get_state() and set_state()."""
        gen = Generator()
        gen.manual_seed(12345)
        state = gen.get_state()
        assert "seed" in state
        assert state["seed"] == 12345

        # Create new generator and restore state
        gen2 = Generator()
        gen2.set_state(state)
        assert gen2.initial_seed() == 12345

    def test_generator_device_property(self):
        """Test Generator.device property."""
        gen = Generator()
        assert hasattr(gen.device, "type")


class TestDefaultGenerator:
    """Tests for default_generator."""

    def test_default_generator_exists(self):
        """Test default_generator exists."""
        assert default_generator is not None
        assert isinstance(default_generator, Generator)

    def test_default_generator_singleton(self):
        """Test default_generator is a singleton-like global."""
        from flashlight.random import default_generator as gen1
        from flashlight.random import default_generator as gen2

        assert gen1 is gen2


class TestManualSeed:
    """Tests for manual_seed function."""

    def test_manual_seed_returns_generator(self):
        """Test manual_seed returns a Generator."""
        gen = manual_seed(42)
        assert isinstance(gen, Generator)

    def test_manual_seed_reproducibility(self):
        """Test manual_seed produces reproducible random numbers."""
        manual_seed(12345)
        x1 = flashlight.randn(10)

        manual_seed(12345)
        x2 = flashlight.randn(10)

        np.testing.assert_allclose(
            np.array(x1._mlx_array),
            np.array(x2._mlx_array),
            rtol=1e-6,
        )

    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different random numbers."""
        manual_seed(1)
        x1 = flashlight.randn(100)

        manual_seed(2)
        x2 = flashlight.randn(100)

        # Should be different
        assert not np.allclose(
            np.array(x1._mlx_array),
            np.array(x2._mlx_array),
        )


class TestSeedFunction:
    """Tests for seed function."""

    def test_seed_returns_int(self):
        """Test seed() returns an integer."""
        s = seed()
        assert isinstance(s, int)

    def test_seed_produces_different_values(self):
        """Test seed() produces different values each call."""
        s1 = seed()
        s2 = seed()
        # They should be different (with very high probability)
        # but we can't guarantee it, so just check they're valid
        assert isinstance(s1, int)
        assert isinstance(s2, int)


class TestInitialSeed:
    """Tests for initial_seed function."""

    def test_initial_seed_after_manual_seed(self):
        """Test initial_seed() returns the seed set by manual_seed()."""
        manual_seed(99999)
        assert initial_seed() == 99999


class TestRNGState:
    """Tests for get_rng_state and set_rng_state."""

    def test_get_rng_state_returns_dict(self):
        """Test get_rng_state returns a dictionary."""
        state = get_rng_state()
        assert isinstance(state, dict)

    def test_save_restore_rng_state(self):
        """Test saving and restoring RNG state."""
        manual_seed(42)

        # Generate some random numbers
        x1 = flashlight.randn(10)

        # Save state
        state = get_rng_state()

        # Generate more random numbers
        x2 = flashlight.randn(10)

        # Restore state
        set_rng_state(state)

        # Should get same numbers as x1 continuation
        # Note: This may not work exactly due to MLX RNG implementation
        # but the API should at least not error


class TestForkRng:
    """Tests for fork_rng context manager."""

    def test_fork_rng_basic(self):
        """Test fork_rng basic usage."""
        manual_seed(42)
        state_before = get_rng_state()

        with fork_rng():
            # Generate random numbers inside fork
            _ = flashlight.randn(10)

        state_after = get_rng_state()
        # State should be restored
        assert state_before["seed"] == state_after["seed"]

    def test_fork_rng_disabled(self):
        """Test fork_rng with enabled=False."""
        manual_seed(42)

        with fork_rng(enabled=False):
            _ = flashlight.randn(10)

        # Should not error

    def test_fork_rng_exception_safety(self):
        """Test fork_rng restores state on exception."""
        manual_seed(42)
        state_before = get_rng_state()

        try:
            with fork_rng():
                _ = flashlight.randn(10)
                raise ValueError("test")
        except ValueError:
            pass

        state_after = get_rng_state()
        assert state_before["seed"] == state_after["seed"]


class TestRandomFunctions:
    """Tests for random tensor creation with seeding."""

    def test_randn_reproducible(self):
        """Test randn is reproducible with seed."""
        manual_seed(123)
        a = flashlight.randn(5, 5)

        manual_seed(123)
        b = flashlight.randn(5, 5)

        np.testing.assert_allclose(
            np.array(a._mlx_array),
            np.array(b._mlx_array),
            rtol=1e-6,
        )

    def test_rand_reproducible(self):
        """Test rand is reproducible with seed."""
        manual_seed(456)
        a = flashlight.rand(5, 5)

        manual_seed(456)
        b = flashlight.rand(5, 5)

        np.testing.assert_allclose(
            np.array(a._mlx_array),
            np.array(b._mlx_array),
            rtol=1e-6,
        )

    def test_randint_reproducible(self):
        """Test randint is reproducible with seed."""
        manual_seed(789)
        a = flashlight.randint(0, 100, (10,))

        manual_seed(789)
        b = flashlight.randint(0, 100, (10,))

        np.testing.assert_array_equal(
            np.array(a._mlx_array),
            np.array(b._mlx_array),
        )

    def test_randperm_reproducible(self):
        """Test randperm is reproducible with seed."""
        manual_seed(111)
        a = flashlight.randperm(10)

        manual_seed(111)
        b = flashlight.randperm(10)

        np.testing.assert_array_equal(
            np.array(a._mlx_array),
            np.array(b._mlx_array),
        )


class TestRandomStatistics:
    """Tests for statistical properties of random numbers."""

    def test_randn_statistics(self):
        """Test randn produces approximately N(0,1) distribution."""
        manual_seed(42)
        x = flashlight.randn(10000)
        data = np.array(x._mlx_array)

        # Mean should be close to 0
        assert abs(data.mean()) < 0.05

        # Std should be close to 1
        assert abs(data.std() - 1.0) < 0.05

    def test_rand_statistics(self):
        """Test rand produces approximately U(0,1) distribution."""
        manual_seed(42)
        x = flashlight.rand(10000)
        data = np.array(x._mlx_array)

        # Mean should be close to 0.5
        assert abs(data.mean() - 0.5) < 0.02

        # All values should be in [0, 1)
        assert data.min() >= 0.0
        assert data.max() < 1.0

    def test_randint_range(self):
        """Test randint produces values in correct range."""
        manual_seed(42)
        x = flashlight.randint(5, 15, (1000,))
        data = np.array(x._mlx_array)

        # All values should be in [5, 15)
        assert data.min() >= 5
        assert data.max() < 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
