"""
Tests for mlx_compat.utils.checkpoint module.

Tests gradient checkpointing functionality.
"""

import unittest
import warnings

import mlx.core as mx
import mlx_compat
from mlx_compat.utils.checkpoint import (
    checkpoint,
    checkpoint_sequential,
    check_backward_validity,
)


class TestCheckpointBasic(unittest.TestCase):
    """Basic checkpoint functionality tests."""

    def test_checkpoint_simple_function(self):
        """Test checkpoint with a simple function."""
        def simple_fn(x):
            return x * 2 + 1

        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = checkpoint(simple_fn, x)

        self.assertIsNotNone(result)
        expected = mlx_compat.tensor([3.0, 5.0, 7.0])
        mx.eval(result._mlx_array)
        mx.eval(expected._mlx_array)

        # Check values match
        for r, e in zip(result._mlx_array.tolist(), expected._mlx_array.tolist()):
            self.assertAlmostEqual(r, e, places=5)

    def test_checkpoint_preserves_output(self):
        """Test that checkpoint output matches non-checkpointed output."""
        def fn(x):
            y = x * 2
            z = y + 3
            return z * z

        x = mlx_compat.tensor([1.0, 2.0], requires_grad=True)

        # Non-checkpointed
        result_normal = fn(x)
        mx.eval(result_normal._mlx_array)

        # Checkpointed
        x2 = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        result_checkpointed = checkpoint(fn, x2)
        mx.eval(result_checkpointed._mlx_array)

        # Compare
        for r, c in zip(
            result_normal._mlx_array.tolist(),
            result_checkpointed._mlx_array.tolist()
        ):
            self.assertAlmostEqual(r, c, places=5)

    def test_checkpoint_with_no_grad_input(self):
        """Test checkpoint with input that doesn't require grad."""
        def fn(x):
            return x * 2

        x = mlx_compat.tensor([1.0, 2.0], requires_grad=False)

        # When no input requires grad, checkpoint just runs the function normally
        # It produces correct output
        result = checkpoint(fn, x)

        mx.eval(result._mlx_array)
        expected = [2.0, 4.0]
        for r, e in zip(result._mlx_array.tolist(), expected):
            self.assertAlmostEqual(r, e, places=5)

    def test_checkpoint_multiple_inputs(self):
        """Test checkpoint with multiple inputs."""
        def fn(a, b):
            return a * b + a

        a = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        b = mlx_compat.tensor([3.0, 4.0], requires_grad=True)

        result = checkpoint(fn, a, b)
        mx.eval(result._mlx_array)

        expected = [4.0, 10.0]  # 1*3+1=4, 2*4+2=10
        for r, e in zip(result._mlx_array.tolist(), expected):
            self.assertAlmostEqual(r, e, places=5)

    def test_checkpoint_with_non_tensor_args(self):
        """Test checkpoint with mixed tensor and non-tensor args."""
        def fn(x, multiplier):
            return x * multiplier

        x = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        result = checkpoint(fn, x, 3.0)
        mx.eval(result._mlx_array)

        expected = [3.0, 6.0]
        for r, e in zip(result._mlx_array.tolist(), expected):
            self.assertAlmostEqual(r, e, places=5)


class TestCheckpointSequential(unittest.TestCase):
    """Test checkpoint_sequential functionality."""

    def test_sequential_basic(self):
        """Test basic sequential checkpointing."""
        # Create simple layers (functions)
        layers = [
            lambda x: x * 2,
            lambda x: x + 1,
            lambda x: x * 3,
        ]

        x = mlx_compat.tensor([1.0], requires_grad=True)
        result = checkpoint_sequential(layers, segments=2, input=x)
        mx.eval(result._mlx_array)

        # (1 * 2 + 1) * 3 = 9
        expected = 9.0
        self.assertAlmostEqual(result._mlx_array.tolist()[0], expected, places=5)

    def test_sequential_single_segment(self):
        """Test with single segment."""
        layers = [
            lambda x: x * 2,
            lambda x: x + 1,
        ]

        x = mlx_compat.tensor([1.0], requires_grad=True)
        result = checkpoint_sequential(layers, segments=1, input=x)
        mx.eval(result._mlx_array)

        # (1 * 2) + 1 = 3
        expected = 3.0
        self.assertAlmostEqual(result._mlx_array.tolist()[0], expected, places=5)

    def test_sequential_more_segments_than_layers(self):
        """Test when segments > number of layers."""
        layers = [
            lambda x: x * 2,
            lambda x: x + 1,
        ]

        x = mlx_compat.tensor([1.0], requires_grad=True)
        # Should handle gracefully (uses num_layers segments)
        result = checkpoint_sequential(layers, segments=10, input=x)
        mx.eval(result._mlx_array)

        expected = 3.0
        self.assertAlmostEqual(result._mlx_array.tolist()[0], expected, places=5)

    def test_sequential_with_nn_modules(self):
        """Test sequential with nn.Module layers."""
        import mlx_compat.nn as nn

        layers = [
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        ]

        x = mlx_compat.randn(2, 10)
        x.requires_grad = True

        result = checkpoint_sequential(layers, segments=2, input=x)
        mx.eval(result._mlx_array)

        self.assertEqual(result.shape, (2, 5))

    def test_sequential_invalid_segments(self):
        """Test that invalid segments raises error."""
        layers = [lambda x: x]

        x = mlx_compat.tensor([1.0])

        with self.assertRaises(ValueError):
            checkpoint_sequential(layers, segments=0, input=x)

    def test_sequential_invalid_functions_type(self):
        """Test that non-iterable functions raises error."""
        x = mlx_compat.tensor([1.0])

        with self.assertRaises(TypeError):
            checkpoint_sequential(42, segments=1, input=x)


class TestCheckpointRNGPreservation(unittest.TestCase):
    """Test RNG state preservation in checkpointing."""

    def test_rng_preservation_enabled(self):
        """Test that RNG state is preserved when enabled."""
        def fn_with_random(x):
            noise = mlx_compat.randn(x.shape)
            return x + noise

        mlx_compat.random.manual_seed(42)
        x = mlx_compat.tensor([1.0, 2.0], requires_grad=True)

        # With RNG preservation, running twice should give same random values
        # (though this is hard to test directly, we verify no crash)
        result = checkpoint(fn_with_random, x, preserve_rng_state=True)
        mx.eval(result._mlx_array)

        self.assertEqual(result.shape, (2,))

    def test_rng_preservation_disabled(self):
        """Test checkpoint works with RNG preservation disabled."""
        def fn_with_random(x):
            noise = mlx_compat.randn(x.shape)
            return x + noise

        x = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        result = checkpoint(fn_with_random, x, preserve_rng_state=False)
        mx.eval(result._mlx_array)

        self.assertEqual(result.shape, (2,))


class TestCheckpointWarnings(unittest.TestCase):
    """Test checkpoint warnings."""

    def test_non_reentrant_warning(self):
        """Test warning for non-reentrant mode."""
        def fn(x):
            return x * 2

        x = mlx_compat.tensor([1.0], requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checkpoint(fn, x, use_reentrant=False)
            # Should warn about non-reentrant not fully supported
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("Non-reentrant" in msg or "reentrant" in msg.lower() for msg in warning_messages)
            )

    def test_context_fn_warning(self):
        """Test warning for context_fn parameter."""
        def fn(x):
            return x * 2

        x = mlx_compat.tensor([1.0], requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checkpoint(fn, x, context_fn=lambda: None)
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("context_fn" in msg for msg in warning_messages)
            )

    def test_determinism_check_warning(self):
        """Test warning for determinism_check parameter."""
        def fn(x):
            return x * 2

        x = mlx_compat.tensor([1.0], requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            checkpoint(fn, x, determinism_check="default")
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("determinism" in msg.lower() for msg in warning_messages)
            )


class TestCheckBackwardValidity(unittest.TestCase):
    """Test check_backward_validity function."""

    def test_warns_when_no_grad(self):
        """Test warning when no input requires grad."""
        x = mlx_compat.tensor([1.0], requires_grad=False)
        y = mlx_compat.tensor([2.0], requires_grad=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_backward_validity((x, y))
            self.assertGreater(len(w), 0)
            self.assertIn("requires_grad", str(w[0].message))

    def test_no_warning_when_has_grad(self):
        """Test no warning when input requires grad."""
        x = mlx_compat.tensor([1.0], requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_backward_validity((x,))
            grad_warnings = [
                warning for warning in w
                if "requires_grad" in str(warning.message)
            ]
            self.assertEqual(len(grad_warnings), 0)

    def test_mixed_inputs(self):
        """Test with mixed requires_grad inputs."""
        x = mlx_compat.tensor([1.0], requires_grad=True)
        y = mlx_compat.tensor([2.0], requires_grad=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_backward_validity((x, y))
            # Should not warn since at least one requires grad
            grad_warnings = [
                warning for warning in w
                if "requires_grad" in str(warning.message)
            ]
            self.assertEqual(len(grad_warnings), 0)


class TestCheckpointWithModules(unittest.TestCase):
    """Test checkpoint with nn.Module layers."""

    def test_checkpoint_linear(self):
        """Test checkpoint with Linear layer."""
        import mlx_compat.nn as nn

        linear = nn.Linear(10, 5)

        x = mlx_compat.randn(2, 10)
        x.requires_grad = True

        result = checkpoint(linear, x)
        mx.eval(result._mlx_array)

        self.assertEqual(result.shape, (2, 5))

    def test_checkpoint_nested_function(self):
        """Test checkpoint with nested function calls."""
        import mlx_compat.nn as nn

        linear1 = nn.Linear(10, 10)
        linear2 = nn.Linear(10, 5)

        def forward_fn(x):
            x = linear1(x)
            x = mlx_compat.relu(x)
            x = linear2(x)
            return x

        x = mlx_compat.randn(2, 10)
        x.requires_grad = True

        result = checkpoint(forward_fn, x)
        mx.eval(result._mlx_array)

        self.assertEqual(result.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
