"""
Test Phase 3: Custom Function

Tests the custom autograd Function class:
- Function subclassing
- SavedTensorsContext
- CustomFunctionBackward
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx.core as mx

    import flashlight
    from flashlight.autograd.function import Function

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestCustomFunction(TestCase):
    """Test custom autograd Function class."""

    def test_custom_relu(self):
        """Test custom ReLU implementation."""

        class MyReLU(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return mx.maximum(input._mlx_array, 0)

            @staticmethod
            def backward(ctx, grad_output):
                (input,) = ctx.saved_tensors
                mask = (input._mlx_array > 0).astype(grad_output._mlx_array.dtype)
                return flashlight.tensor(grad_output._mlx_array * mask)

        x = flashlight.tensor([-1.0, 0.5, -0.5, 2.0], requires_grad=True)
        y = MyReLU.apply(x)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_custom_scale(self):
        """Test custom scaling function."""

        class Scale(Function):
            @staticmethod
            def forward(ctx, input, scale):
                ctx.scale = scale
                return input._mlx_array * scale

            @staticmethod
            def backward(ctx, grad_output):
                return flashlight.tensor(grad_output._mlx_array * ctx.scale)

        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Scale.apply(x, 3.0)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.array([3.0, 3.0, 3.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_custom_square(self):
        """Test custom square function."""

        class Square(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input._mlx_array * input._mlx_array

            @staticmethod
            def backward(ctx, grad_output):
                (input,) = ctx.saved_tensors
                return flashlight.tensor(2 * input._mlx_array * grad_output._mlx_array)

        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Square.apply(x)
        loss = flashlight.sum(y)
        loss.backward()

        # d(x^2)/dx = 2x
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestSavedTensorsContext(TestCase):
    """Test SavedTensorsContext functionality."""

    def test_save_single_tensor(self):
        """Test saving a single tensor."""

        class SaveOne(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x._mlx_array + 1

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                self.assertIsNotNone(x)
                return grad_output

        x = flashlight.tensor([1.0, 2.0], requires_grad=True)
        y = SaveOne.apply(x)
        loss = flashlight.sum(y)
        loss.backward()

    def test_save_multiple_tensors(self):
        """Test saving multiple tensors."""

        class SaveMultiple(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x._mlx_array + y._mlx_array

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output, grad_output

        x = flashlight.tensor([1.0, 2.0], requires_grad=True)
        y = flashlight.tensor([3.0, 4.0], requires_grad=True)
        z = SaveMultiple.apply(x, y)
        loss = flashlight.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(2))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.ones(2))


@skipIfNoMLX
class TestCustomFunctionChain(TestCase):
    """Test chaining custom functions."""

    def test_chain_custom_functions(self):
        """Test chaining multiple custom functions."""

        class Double(Function):
            @staticmethod
            def forward(ctx, input):
                return input._mlx_array * 2

            @staticmethod
            def backward(ctx, grad_output):
                return flashlight.tensor(grad_output._mlx_array * 2)

        class AddOne(Function):
            @staticmethod
            def forward(ctx, input):
                return input._mlx_array + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Double.apply(x)
        z = AddOne.apply(y)
        loss = flashlight.sum(z)
        loss.backward()

        # Chain: z = 2x + 1, d/dx = 2
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
