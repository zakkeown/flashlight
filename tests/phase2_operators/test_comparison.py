"""
Tests for comparison operations (mlx_compat.ops.comparison).

Tests element-wise comparisons, allclose, isclose, maximum, minimum.
"""

import pytest
import numpy as np
import torch
import mlx_compat
from mlx_compat import Tensor


class TestElementwiseComparison:
    """Test basic element-wise comparison operations."""

    def test_eq_tensor_tensor(self):
        """Test eq with two tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 2.0])
        y = mlx_compat.tensor([1.0, 3.0, 3.0, 2.0])
        result = mlx_compat.eq(x, y)
        expected = torch.eq(torch.tensor([1.0, 2.0, 3.0, 2.0]), torch.tensor([1.0, 3.0, 3.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_eq_tensor_scalar(self):
        """Test eq with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 2.0])
        result = mlx_compat.eq(x, 2.0)
        expected = torch.eq(torch.tensor([1.0, 2.0, 3.0, 2.0]), 2.0)
        assert list(result.tolist()) == expected.tolist()

    def test_ne_tensor_tensor(self):
        """Test ne with two tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 2.0])
        y = mlx_compat.tensor([1.0, 3.0, 3.0, 2.0])
        result = mlx_compat.ne(x, y)
        expected = torch.ne(torch.tensor([1.0, 2.0, 3.0, 2.0]), torch.tensor([1.0, 3.0, 3.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_ne_tensor_scalar(self):
        """Test ne with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 2.0])
        result = mlx_compat.ne(x, 2.0)
        expected = torch.ne(torch.tensor([1.0, 2.0, 3.0, 2.0]), 2.0)
        assert list(result.tolist()) == expected.tolist()

    def test_lt_tensor_tensor(self):
        """Test lt with two tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0, 2.0])
        result = mlx_compat.lt(x, y)
        expected = torch.lt(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_lt_tensor_scalar(self):
        """Test lt with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        result = mlx_compat.lt(x, 2.5)
        expected = torch.lt(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2.5)
        assert list(result.tolist()) == expected.tolist()

    def test_le_tensor_tensor(self):
        """Test le with two tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0, 2.0])
        result = mlx_compat.le(x, y)
        expected = torch.le(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_le_tensor_scalar(self):
        """Test le with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        result = mlx_compat.le(x, 2.0)
        expected = torch.le(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2.0)
        assert list(result.tolist()) == expected.tolist()

    def test_gt_tensor_tensor(self):
        """Test gt with two tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0, 2.0])
        result = mlx_compat.gt(x, y)
        expected = torch.gt(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_gt_tensor_scalar(self):
        """Test gt with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        result = mlx_compat.gt(x, 2.5)
        expected = torch.gt(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2.5)
        assert list(result.tolist()) == expected.tolist()

    def test_ge_tensor_tensor(self):
        """Test ge with two tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0, 2.0])
        result = mlx_compat.ge(x, y)
        expected = torch.ge(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 2.0, 2.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_ge_tensor_scalar(self):
        """Test ge with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        result = mlx_compat.ge(x, 2.0)
        expected = torch.ge(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2.0)
        assert list(result.tolist()) == expected.tolist()


class TestAliases:
    """Test PyTorch-style aliases for comparison ops."""

    def test_greater(self):
        """Test greater alias for gt."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0])
        result1 = mlx_compat.greater(x, y)
        result2 = mlx_compat.gt(x, y)
        assert list(result1.tolist()) == list(result2.tolist())

    def test_greater_equal(self):
        """Test greater_equal alias for ge."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0])
        result1 = mlx_compat.greater_equal(x, y)
        result2 = mlx_compat.ge(x, y)
        assert list(result1.tolist()) == list(result2.tolist())

    def test_less(self):
        """Test less alias for lt."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0])
        result1 = mlx_compat.less(x, y)
        result2 = mlx_compat.lt(x, y)
        assert list(result1.tolist()) == list(result2.tolist())

    def test_less_equal(self):
        """Test less_equal alias for le."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0])
        result1 = mlx_compat.less_equal(x, y)
        result2 = mlx_compat.le(x, y)
        assert list(result1.tolist()) == list(result2.tolist())

    def test_not_equal(self):
        """Test not_equal alias for ne."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([2.0, 2.0, 2.0])
        result1 = mlx_compat.not_equal(x, y)
        result2 = mlx_compat.ne(x, y)
        assert list(result1.tolist()) == list(result2.tolist())


class TestEqual:
    """Test tensor equality checks."""

    def test_equal_same_tensors(self):
        """Test equal with identical tensors."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert mlx_compat.equal(x, y) == True

    def test_equal_different_values(self):
        """Test equal with different values."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 2.0], [3.0, 5.0]])
        assert mlx_compat.equal(x, y) == False

    def test_equal_different_shapes(self):
        """Test equal with different shapes."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        assert mlx_compat.equal(x, y) == False


class TestAllclose:
    """Test allclose comparison."""

    def test_allclose_exact_match(self):
        """Test allclose with exactly matching tensors."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([1.0, 2.0, 3.0])
        assert mlx_compat.allclose(x, y) == True

    def test_allclose_within_tolerance(self):
        """Test allclose with values within tolerance."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([1.0 + 1e-6, 2.0 - 1e-6, 3.0 + 1e-7])
        assert mlx_compat.allclose(x, y, rtol=1e-5, atol=1e-5) == True

    def test_allclose_outside_tolerance(self):
        """Test allclose with values outside tolerance."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([1.0, 2.0, 3.1])
        assert mlx_compat.allclose(x, y, rtol=1e-5, atol=1e-5) == False

    def test_allclose_relative_tolerance(self):
        """Test allclose with relative tolerance."""
        x = mlx_compat.tensor([100.0, 200.0])
        y = mlx_compat.tensor([100.01, 200.02])  # 0.01% difference
        assert mlx_compat.allclose(x, y, rtol=1e-3, atol=0) == True
        assert mlx_compat.allclose(x, y, rtol=1e-5, atol=0) == False


class TestIsclose:
    """Test isclose element-wise comparison."""

    def test_isclose_basic(self):
        """Test isclose basic functionality."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([1.0 + 1e-6, 2.0, 3.0 + 0.1])
        result = mlx_compat.isclose(x, y, rtol=1e-5, atol=1e-5)
        # First two should be close, third should not
        assert result[0].item() == True
        assert result[1].item() == True
        assert result[2].item() == False

    def test_isclose_parity(self):
        """Test isclose parity with PyTorch."""
        x_np = np.random.randn(10).astype(np.float32)
        y_np = x_np + np.random.randn(10).astype(np.float32) * 0.01

        x_mlx = mlx_compat.tensor(x_np)
        y_mlx = mlx_compat.tensor(y_np)
        x_torch = torch.tensor(x_np)
        y_torch = torch.tensor(y_np)

        result_mlx = mlx_compat.isclose(x_mlx, y_mlx, rtol=0.1, atol=0.01)
        result_torch = torch.isclose(x_torch, y_torch, rtol=0.1, atol=0.01)

        assert list(result_mlx.tolist()) == result_torch.tolist()


class TestMaximum:
    """Test maximum element-wise operation."""

    def test_maximum_tensor_tensor(self):
        """Test maximum with two tensors."""
        x = mlx_compat.tensor([1.0, 5.0, 3.0, 7.0])
        y = mlx_compat.tensor([2.0, 4.0, 6.0, 1.0])
        result = mlx_compat.maximum(x, y)
        expected = torch.maximum(torch.tensor([1.0, 5.0, 3.0, 7.0]), torch.tensor([2.0, 4.0, 6.0, 1.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_maximum_tensor_scalar(self):
        """Test maximum with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 5.0, 3.0, 7.0])
        result = mlx_compat.maximum(x, 4.0)
        expected = torch.maximum(torch.tensor([1.0, 5.0, 3.0, 7.0]), torch.tensor(4.0))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_maximum_requires_grad(self):
        """Test that maximum preserves requires_grad."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = mlx_compat.tensor([2.0, 1.0, 4.0])
        result = mlx_compat.maximum(x, y)
        assert result.requires_grad == True


class TestMinimum:
    """Test minimum element-wise operation."""

    def test_minimum_tensor_tensor(self):
        """Test minimum with two tensors."""
        x = mlx_compat.tensor([1.0, 5.0, 3.0, 7.0])
        y = mlx_compat.tensor([2.0, 4.0, 6.0, 1.0])
        result = mlx_compat.minimum(x, y)
        expected = torch.minimum(torch.tensor([1.0, 5.0, 3.0, 7.0]), torch.tensor([2.0, 4.0, 6.0, 1.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_minimum_tensor_scalar(self):
        """Test minimum with tensor and scalar."""
        x = mlx_compat.tensor([1.0, 5.0, 3.0, 7.0])
        result = mlx_compat.minimum(x, 4.0)
        expected = torch.minimum(torch.tensor([1.0, 5.0, 3.0, 7.0]), torch.tensor(4.0))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_minimum_requires_grad(self):
        """Test that minimum preserves requires_grad."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = mlx_compat.tensor([2.0, 1.0, 4.0])
        result = mlx_compat.minimum(x, y)
        assert result.requires_grad == True


@pytest.mark.parity
class TestComparisonParity:
    """Test PyTorch parity for all comparison operations."""

    def test_eq_parity_multidim(self):
        """Test eq parity with multi-dimensional tensors."""
        x_np = np.random.randint(0, 5, (3, 4, 5)).astype(np.float32)
        y_np = np.random.randint(0, 5, (3, 4, 5)).astype(np.float32)

        x_mlx = mlx_compat.tensor(x_np)
        y_mlx = mlx_compat.tensor(y_np)
        x_torch = torch.tensor(x_np)
        y_torch = torch.tensor(y_np)

        result_mlx = mlx_compat.eq(x_mlx, y_mlx)
        result_torch = torch.eq(x_torch, y_torch)

        assert np.array(result_mlx.tolist()).tolist() == result_torch.numpy().tolist()

    def test_ne_parity_multidim(self):
        """Test ne parity with multi-dimensional tensors."""
        x_np = np.random.randint(0, 5, (3, 4, 5)).astype(np.float32)
        y_np = np.random.randint(0, 5, (3, 4, 5)).astype(np.float32)

        x_mlx = mlx_compat.tensor(x_np)
        y_mlx = mlx_compat.tensor(y_np)
        x_torch = torch.tensor(x_np)
        y_torch = torch.tensor(y_np)

        result_mlx = mlx_compat.ne(x_mlx, y_mlx)
        result_torch = torch.ne(x_torch, y_torch)

        assert np.array(result_mlx.tolist()).tolist() == result_torch.numpy().tolist()

    def test_comparison_broadcast(self):
        """Test that comparisons work with broadcasting."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = mlx_compat.tensor([2.0, 3.0, 4.0])

        x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_torch = torch.tensor([2.0, 3.0, 4.0])

        # Test lt with broadcasting
        result_mlx = mlx_compat.lt(x, y)
        result_torch = torch.lt(x_torch, y_torch)
        assert np.array(result_mlx.tolist()).tolist() == result_torch.numpy().tolist()

        # Test ge with broadcasting
        result_mlx = mlx_compat.ge(x, y)
        result_torch = torch.ge(x_torch, y_torch)
        assert np.array(result_mlx.tolist()).tolist() == result_torch.numpy().tolist()

    def test_maximum_parity(self):
        """Test maximum parity with PyTorch."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        y_np = np.random.randn(3, 4, 5).astype(np.float32)

        x_mlx = mlx_compat.tensor(x_np)
        y_mlx = mlx_compat.tensor(y_np)
        x_torch = torch.tensor(x_np)
        y_torch = torch.tensor(y_np)

        result_mlx = mlx_compat.maximum(x_mlx, y_mlx)
        result_torch = torch.maximum(x_torch, y_torch)

        np.testing.assert_allclose(
            np.array(result_mlx.tolist()),
            result_torch.numpy(),
            rtol=1e-5
        )

    def test_minimum_parity(self):
        """Test minimum parity with PyTorch."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        y_np = np.random.randn(3, 4, 5).astype(np.float32)

        x_mlx = mlx_compat.tensor(x_np)
        y_mlx = mlx_compat.tensor(y_np)
        x_torch = torch.tensor(x_np)
        y_torch = torch.tensor(y_np)

        result_mlx = mlx_compat.minimum(x_mlx, y_mlx)
        result_torch = torch.minimum(x_torch, y_torch)

        np.testing.assert_allclose(
            np.array(result_mlx.tolist()),
            result_torch.numpy(),
            rtol=1e-5
        )
