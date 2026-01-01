"""
Tests for math functions (mlx_compat.ops.math_funcs).

Tests clamp, floor, ceil, round, trunc, logical ops, cumsum, cumprod, etc.
"""

import pytest
import numpy as np
import torch
import mlx_compat
from mlx_compat import Tensor


class TestClamp:
    """Test clamp and related functions."""

    def test_clamp_both_bounds(self):
        """Test clamp with both min and max."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        result = mlx_compat.clamp(x, min=-1.0, max=2.0)
        expected = torch.clamp(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]), min=-1.0, max=2.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_clamp_min_only(self):
        """Test clamp with only min bound."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.clamp(x, min=0.0)
        expected = torch.clamp(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), min=0.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_clamp_max_only(self):
        """Test clamp with only max bound."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.clamp(x, max=1.0)
        expected = torch.clamp(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), max=1.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_clamp_min_func(self):
        """Test clamp_min function."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.clamp_min(x, 0.0)
        expected = torch.clamp_min(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), 0.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_clamp_max_func(self):
        """Test clamp_max function."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.clamp_max(x, 1.0)
        expected = torch.clamp_max(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), 1.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestRounding:
    """Test rounding operations."""

    def test_floor(self):
        """Test floor function."""
        x = mlx_compat.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7])
        result = mlx_compat.floor(x)
        expected = torch.floor(torch.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_ceil(self):
        """Test ceil function."""
        x = mlx_compat.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7])
        result = mlx_compat.ceil(x)
        expected = torch.ceil(torch.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_round(self):
        """Test round function."""
        x = mlx_compat.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7])
        result = mlx_compat.round(x)
        expected = torch.round(torch.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_trunc(self):
        """Test trunc function."""
        x = mlx_compat.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7])
        result = mlx_compat.trunc(x)
        expected = torch.trunc(torch.tensor([-1.7, -1.2, -0.5, 0.0, 0.5, 1.2, 1.7]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_frac(self):
        """Test frac function."""
        x = mlx_compat.tensor([-1.7, -1.2, 0.0, 1.2, 1.7])
        result = mlx_compat.frac(x)
        expected = torch.frac(torch.tensor([-1.7, -1.2, 0.0, 1.2, 1.7]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestSign:
    """Test sign and related functions."""

    def test_sign(self):
        """Test sign function."""
        x = mlx_compat.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = mlx_compat.sign(x)
        expected = torch.sign(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_signbit(self):
        """Test signbit function."""
        # Test without negative zero since MLX may not preserve sign of zero
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 2.0])
        result = mlx_compat.signbit(x)
        expected = torch.signbit(torch.tensor([-2.0, -1.0, 0.0, 2.0]))
        assert list(result.tolist()) == expected.tolist()


class TestNaNInf:
    """Test NaN and Inf checking functions."""

    def test_isnan(self):
        """Test isnan function."""
        x = mlx_compat.tensor([1.0, float('nan'), 2.0, float('nan')])
        result = mlx_compat.isnan(x)
        expected = torch.isnan(torch.tensor([1.0, float('nan'), 2.0, float('nan')]))
        assert list(result.tolist()) == expected.tolist()

    def test_isinf(self):
        """Test isinf function."""
        x = mlx_compat.tensor([1.0, float('inf'), -float('inf'), 2.0])
        result = mlx_compat.isinf(x)
        expected = torch.isinf(torch.tensor([1.0, float('inf'), -float('inf'), 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_isfinite(self):
        """Test isfinite function."""
        x = mlx_compat.tensor([1.0, float('inf'), float('nan'), 2.0])
        result = mlx_compat.isfinite(x)
        expected = torch.isfinite(torch.tensor([1.0, float('inf'), float('nan'), 2.0]))
        assert list(result.tolist()) == expected.tolist()

    def test_nan_to_num(self):
        """Test nan_to_num function."""
        x = mlx_compat.tensor([1.0, float('nan'), float('inf'), -float('inf')])
        result = mlx_compat.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        expected = torch.nan_to_num(torch.tensor([1.0, float('nan'), float('inf'), -float('inf')]),
                                     nan=0.0, posinf=100.0, neginf=-100.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestLogicalOps:
    """Test logical operations."""

    def test_logical_and(self):
        """Test logical_and function."""
        x = mlx_compat.tensor([True, True, False, False])
        y = mlx_compat.tensor([True, False, True, False])
        result = mlx_compat.logical_and(x, y)
        expected = torch.logical_and(torch.tensor([True, True, False, False]),
                                      torch.tensor([True, False, True, False]))
        assert list(result.tolist()) == expected.tolist()

    def test_logical_or(self):
        """Test logical_or function."""
        x = mlx_compat.tensor([True, True, False, False])
        y = mlx_compat.tensor([True, False, True, False])
        result = mlx_compat.logical_or(x, y)
        expected = torch.logical_or(torch.tensor([True, True, False, False]),
                                     torch.tensor([True, False, True, False]))
        assert list(result.tolist()) == expected.tolist()

    def test_logical_not(self):
        """Test logical_not function."""
        x = mlx_compat.tensor([True, False, True, False])
        result = mlx_compat.logical_not(x)
        expected = torch.logical_not(torch.tensor([True, False, True, False]))
        assert list(result.tolist()) == expected.tolist()

    def test_logical_xor(self):
        """Test logical_xor function."""
        x = mlx_compat.tensor([True, True, False, False])
        y = mlx_compat.tensor([True, False, True, False])
        result = mlx_compat.logical_xor(x, y)
        expected = torch.logical_xor(torch.tensor([True, True, False, False]),
                                      torch.tensor([True, False, True, False]))
        assert list(result.tolist()) == expected.tolist()


class TestMathOps:
    """Test various math operations."""

    def test_reciprocal(self):
        """Test reciprocal function."""
        x = mlx_compat.tensor([1.0, 2.0, 4.0, 0.5])
        result = mlx_compat.reciprocal(x)
        expected = torch.reciprocal(torch.tensor([1.0, 2.0, 4.0, 0.5]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_rsqrt(self):
        """Test rsqrt function."""
        x = mlx_compat.tensor([1.0, 4.0, 9.0, 16.0])
        result = mlx_compat.rsqrt(x)
        expected = torch.rsqrt(torch.tensor([1.0, 4.0, 9.0, 16.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_square(self):
        """Test square function."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.square(x)
        expected = torch.square(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_lerp(self):
        """Test lerp function."""
        start = mlx_compat.tensor([0.0, 1.0, 2.0])
        end = mlx_compat.tensor([10.0, 10.0, 10.0])
        result = mlx_compat.lerp(start, end, 0.5)
        expected = torch.lerp(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([10.0, 10.0, 10.0]), 0.5)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_addcmul(self):
        """Test addcmul function."""
        t = mlx_compat.tensor([1.0, 2.0, 3.0])
        t1 = mlx_compat.tensor([1.0, 2.0, 3.0])
        t2 = mlx_compat.tensor([1.0, 2.0, 3.0])
        result = mlx_compat.addcmul(t, t1, t2, value=0.5)
        expected = torch.addcmul(torch.tensor([1.0, 2.0, 3.0]),
                                  torch.tensor([1.0, 2.0, 3.0]),
                                  torch.tensor([1.0, 2.0, 3.0]), value=0.5)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_addcdiv(self):
        """Test addcdiv function."""
        t = mlx_compat.tensor([1.0, 2.0, 3.0])
        t1 = mlx_compat.tensor([1.0, 2.0, 3.0])
        t2 = mlx_compat.tensor([2.0, 2.0, 2.0])
        result = mlx_compat.addcdiv(t, t1, t2, value=0.5)
        expected = torch.addcdiv(torch.tensor([1.0, 2.0, 3.0]),
                                  torch.tensor([1.0, 2.0, 3.0]),
                                  torch.tensor([2.0, 2.0, 2.0]), value=0.5)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestModularOps:
    """Test modular operations."""

    def test_fmod(self):
        """Test fmod function."""
        x = mlx_compat.tensor([5.0, -5.0, 5.0, -5.0])
        y = mlx_compat.tensor([3.0, 3.0, -3.0, -3.0])
        result = mlx_compat.fmod(x, y)
        expected = torch.fmod(torch.tensor([5.0, -5.0, 5.0, -5.0]),
                               torch.tensor([3.0, 3.0, -3.0, -3.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_remainder(self):
        """Test remainder function."""
        x = mlx_compat.tensor([5.0, -5.0, 5.0, -5.0])
        y = mlx_compat.tensor([3.0, 3.0, -3.0, -3.0])
        result = mlx_compat.remainder(x, y)
        expected = torch.remainder(torch.tensor([5.0, -5.0, 5.0, -5.0]),
                                    torch.tensor([3.0, 3.0, -3.0, -3.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_floor_divide(self):
        """Test floor_divide function."""
        x = mlx_compat.tensor([7.0, 8.0, 9.0])
        y = mlx_compat.tensor([3.0, 3.0, 3.0])
        result = mlx_compat.floor_divide(x, y)
        expected = torch.floor_divide(torch.tensor([7.0, 8.0, 9.0]), torch.tensor([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_true_divide(self):
        """Test true_divide function."""
        x = mlx_compat.tensor([7.0, 8.0, 9.0])
        y = mlx_compat.tensor([3.0, 3.0, 3.0])
        result = mlx_compat.true_divide(x, y)
        expected = torch.true_divide(torch.tensor([7.0, 8.0, 9.0]), torch.tensor([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestCumulativeOps:
    """Test cumulative operations."""

    def test_cumsum(self):
        """Test cumsum function."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = mlx_compat.cumsum(x, dim=1)
        expected = torch.cumsum(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dim=1)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_cumsum_dim0(self):
        """Test cumsum along dim 0."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = mlx_compat.cumsum(x, dim=0)
        expected = torch.cumsum(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dim=0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_cumprod(self):
        """Test cumprod function."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = mlx_compat.cumprod(x, dim=1)
        expected = torch.cumprod(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dim=1)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestAngleConversion:
    """Test angle conversion functions."""

    def test_deg2rad(self):
        """Test deg2rad function."""
        x = mlx_compat.tensor([0.0, 90.0, 180.0, 270.0, 360.0])
        result = mlx_compat.deg2rad(x)
        expected = torch.deg2rad(torch.tensor([0.0, 90.0, 180.0, 270.0, 360.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_rad2deg(self):
        """Test rad2deg function."""
        x = mlx_compat.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        result = mlx_compat.rad2deg(x)
        expected = torch.rad2deg(torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestDiff:
    """Test diff function."""

    def test_diff_1d(self):
        """Test diff on 1D tensor."""
        x = mlx_compat.tensor([1.0, 2.0, 4.0, 7.0, 11.0])
        result = mlx_compat.diff(x)
        expected = torch.diff(torch.tensor([1.0, 2.0, 4.0, 7.0, 11.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_diff_2d(self):
        """Test diff on 2D tensor."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 9.0]])
        result = mlx_compat.diff(x, dim=1)
        expected = torch.diff(torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 9.0]]), dim=1)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_diff_n2(self):
        """Test diff with n=2."""
        x = mlx_compat.tensor([1.0, 2.0, 4.0, 7.0, 11.0])
        result = mlx_compat.diff(x, n=2)
        expected = torch.diff(torch.tensor([1.0, 2.0, 4.0, 7.0, 11.0]), n=2)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestCountNonzero:
    """Test count_nonzero function."""

    def test_count_nonzero_all(self):
        """Test count_nonzero for entire tensor."""
        x = mlx_compat.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        result = mlx_compat.count_nonzero(x)
        expected = torch.count_nonzero(torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]))
        assert result.item() == expected.item()

    def test_count_nonzero_dim(self):
        """Test count_nonzero along dimension."""
        x = mlx_compat.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
        result = mlx_compat.count_nonzero(x, dim=1)
        expected = torch.count_nonzero(torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]]), dim=1)
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())


class TestLogit:
    """Test logit function."""

    def test_logit(self):
        """Test logit function."""
        x = mlx_compat.tensor([0.1, 0.5, 0.9])
        result = mlx_compat.logit(x)
        expected = torch.logit(torch.tensor([0.1, 0.5, 0.9]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_logit_with_eps(self):
        """Test logit with eps."""
        x = mlx_compat.tensor([0.0, 0.5, 1.0])
        result = mlx_compat.logit(x, eps=0.01)
        expected = torch.logit(torch.tensor([0.0, 0.5, 1.0]), eps=0.01)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestVander:
    """Test Vandermonde matrix function."""

    def test_vander(self):
        """Test vander function."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        result = mlx_compat.vander(x, N=4)
        expected = torch.vander(torch.tensor([1.0, 2.0, 3.0]), N=4)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_vander_increasing(self):
        """Test vander with increasing=True."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        result = mlx_compat.vander(x, N=4, increasing=True)
        expected = torch.vander(torch.tensor([1.0, 2.0, 3.0]), N=4, increasing=True)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestTriangularIndices:
    """Test triangular index functions."""

    def test_tril_indices(self):
        """Test tril_indices function."""
        result = mlx_compat.tril_indices(4, 4)
        expected = torch.tril_indices(4, 4)
        # Compare as tuples of arrays
        np.testing.assert_array_equal(np.array(result[0].tolist()), expected[0].numpy())
        np.testing.assert_array_equal(np.array(result[1].tolist()), expected[1].numpy())

    def test_triu_indices(self):
        """Test triu_indices function."""
        result = mlx_compat.triu_indices(4, 4)
        expected = torch.triu_indices(4, 4)
        np.testing.assert_array_equal(np.array(result[0].tolist()), expected[0].numpy())
        np.testing.assert_array_equal(np.array(result[1].tolist()), expected[1].numpy())


class TestBitwiseShift:
    """Test bitwise shift operations."""

    def test_bitwise_left_shift(self):
        """Test bitwise_left_shift function."""
        x = mlx_compat.tensor([1, 2, 4, 8], dtype=mlx_compat.int32)
        result = mlx_compat.bitwise_left_shift(x, 2)
        expected = torch.bitwise_left_shift(torch.tensor([1, 2, 4, 8], dtype=torch.int32), 2)
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())

    def test_bitwise_right_shift(self):
        """Test bitwise_right_shift function."""
        x = mlx_compat.tensor([4, 8, 16, 32], dtype=mlx_compat.int32)
        result = mlx_compat.bitwise_right_shift(x, 2)
        expected = torch.bitwise_right_shift(torch.tensor([4, 8, 16, 32], dtype=torch.int32), 2)
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())


class TestSubtract:
    """Test subtraction functions."""

    def test_rsub(self):
        """Test rsub function (other - input * alpha)."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        result = mlx_compat.rsub(x, 10.0, alpha=2.0)
        # rsub: other - alpha * input = 10 - 2 * [1, 2, 3] = [8, 6, 4]
        expected = [8.0, 6.0, 4.0]
        np.testing.assert_allclose(np.array(result.tolist()), expected, rtol=1e-5)

    def test_subtract(self):
        """Test subtract function."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.tensor([0.5, 1.0, 1.5])
        result = mlx_compat.subtract(x, y)
        expected = torch.subtract(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 1.0, 1.5]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestMiscFunctions:
    """Test miscellaneous functions."""

    def test_is_same_size(self):
        """Test is_same_size function."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[5.0, 6.0], [7.0, 8.0]])
        z = mlx_compat.tensor([1.0, 2.0, 3.0])
        assert mlx_compat.is_same_size(x, y) == True
        assert mlx_compat.is_same_size(x, z) == False

    def test_is_signed(self):
        """Test is_signed function."""
        x_float = mlx_compat.tensor([1.0, 2.0])
        x_int = mlx_compat.tensor([1, 2], dtype=mlx_compat.int32)
        # Float and signed int should return True
        assert mlx_compat.is_signed(x_float) == True
        assert mlx_compat.is_signed(x_int) == True

    def test_unravel_index(self):
        """Test unravel_index function."""
        indices = mlx_compat.tensor([0, 5, 11])
        result = mlx_compat.unravel_index(indices, (3, 4))
        expected = np.unravel_index([0, 5, 11], (3, 4))
        np.testing.assert_array_equal(np.array(result[0].tolist()), expected[0])
        np.testing.assert_array_equal(np.array(result[1].tolist()), expected[1])


@pytest.mark.parity
class TestMathFuncsParity:
    """Test PyTorch parity for math functions."""

    def test_clamp_parity_multidim(self):
        """Test clamp parity with multi-dimensional tensors."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        x_mlx = mlx_compat.tensor(x_np)
        x_torch = torch.tensor(x_np)

        result_mlx = mlx_compat.clamp(x_mlx, min=-0.5, max=0.5)
        result_torch = torch.clamp(x_torch, min=-0.5, max=0.5)

        np.testing.assert_allclose(
            np.array(result_mlx.tolist()),
            result_torch.numpy(),
            rtol=1e-5
        )

    def test_cumsum_parity_multidim(self):
        """Test cumsum parity with multi-dimensional tensors."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        x_mlx = mlx_compat.tensor(x_np)
        x_torch = torch.tensor(x_np)

        for dim in range(3):
            result_mlx = mlx_compat.cumsum(x_mlx, dim=dim)
            result_torch = torch.cumsum(x_torch, dim=dim)
            np.testing.assert_allclose(
                np.array(result_mlx.tolist()),
                result_torch.numpy(),
                rtol=1e-5
            )

    def test_diff_parity_multidim(self):
        """Test diff parity with multi-dimensional tensors."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        x_mlx = mlx_compat.tensor(x_np)
        x_torch = torch.tensor(x_np)

        for dim in range(3):
            result_mlx = mlx_compat.diff(x_mlx, dim=dim)
            result_torch = torch.diff(x_torch, dim=dim)
            np.testing.assert_allclose(
                np.array(result_mlx.tolist()),
                result_torch.numpy(),
                rtol=1e-5
            )
