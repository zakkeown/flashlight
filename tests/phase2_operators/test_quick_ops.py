"""
Tests for quick operations (flashlight.ops.quick_ops).

Tests bitwise ops, matrix ops, special ops, and utility functions.
"""

import pytest
import numpy as np
import torch
import flashlight
from flashlight import Tensor


class TestAtleast:
    """Test atleast_* functions."""

    def test_atleast_1d_scalar(self):
        """Test atleast_1d with scalar."""
        x = flashlight.tensor(5.0)
        result = flashlight.atleast_1d(x)
        assert result.ndim == 1
        assert result.shape == (1,)

    def test_atleast_1d_1d(self):
        """Test atleast_1d with 1d tensor."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        result = flashlight.atleast_1d(x)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_atleast_2d_1d(self):
        """Test atleast_2d with 1d tensor."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        result = flashlight.atleast_2d(x)
        assert result.ndim == 2

    def test_atleast_3d_2d(self):
        """Test atleast_3d with 2d tensor."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.atleast_3d(x)
        assert result.ndim == 3


class TestBitwiseOps:
    """Test bitwise operations."""

    def test_bitwise_and(self):
        """Test bitwise_and function."""
        x = flashlight.tensor([5, 3, 7], dtype=flashlight.int32)
        y = flashlight.tensor([3, 3, 3], dtype=flashlight.int32)
        result = flashlight.bitwise_and(x, y)
        expected = torch.bitwise_and(torch.tensor([5, 3, 7], dtype=torch.int32),
                                      torch.tensor([3, 3, 3], dtype=torch.int32))
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())

    def test_bitwise_or(self):
        """Test bitwise_or function."""
        x = flashlight.tensor([5, 3, 7], dtype=flashlight.int32)
        y = flashlight.tensor([3, 3, 3], dtype=flashlight.int32)
        result = flashlight.bitwise_or(x, y)
        expected = torch.bitwise_or(torch.tensor([5, 3, 7], dtype=torch.int32),
                                     torch.tensor([3, 3, 3], dtype=torch.int32))
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())

    def test_bitwise_xor(self):
        """Test bitwise_xor function."""
        x = flashlight.tensor([5, 3, 7], dtype=flashlight.int32)
        y = flashlight.tensor([3, 3, 3], dtype=flashlight.int32)
        result = flashlight.bitwise_xor(x, y)
        expected = torch.bitwise_xor(torch.tensor([5, 3, 7], dtype=torch.int32),
                                      torch.tensor([3, 3, 3], dtype=torch.int32))
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())

    def test_bitwise_not(self):
        """Test bitwise_not function."""
        x = flashlight.tensor([0, 1, 5], dtype=flashlight.int32)
        result = flashlight.bitwise_not(x)
        expected = torch.bitwise_not(torch.tensor([0, 1, 5], dtype=torch.int32))
        np.testing.assert_array_equal(np.array(result.tolist()), expected.numpy())


class TestBroadcast:
    """Test broadcast operations."""

    def test_broadcast_to(self):
        """Test broadcast_to function."""
        x = flashlight.tensor([[1.0], [2.0], [3.0]])
        result = flashlight.broadcast_to(x, (3, 4))
        expected = torch.broadcast_to(torch.tensor([[1.0], [2.0], [3.0]]), (3, 4))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestConcatenate:
    """Test concatenation."""

    def test_concatenate(self):
        """Test concatenate function."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[5.0, 6.0]])
        result = flashlight.concatenate([x, y], dim=0)
        expected = torch.concatenate([torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                                       torch.tensor([[5.0, 6.0]])], dim=0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestSpecialFunctions:
    """Test special math functions."""

    def test_erf(self):
        """Test erf function."""
        x = flashlight.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = flashlight.erf(x)
        expected = torch.erf(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_erfinv(self):
        """Test erfinv function."""
        x = flashlight.tensor([-0.5, 0.0, 0.5])
        result = flashlight.erfinv(x)
        expected = torch.erfinv(torch.tensor([-0.5, 0.0, 0.5]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-4)

    def test_sigmoid(self):
        """Test sigmoid function."""
        x = flashlight.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = flashlight.sigmoid(x)
        expected = torch.sigmoid(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_logaddexp(self):
        """Test logaddexp function."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        y = flashlight.tensor([4.0, 5.0, 6.0])
        result = flashlight.logaddexp(x, y)
        expected = torch.logaddexp(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestSoftmax:
    """Test softmax and related functions."""

    def test_softmax(self):
        """Test softmax function."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        result = flashlight.softmax(x, dim=1)
        expected = torch.softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), dim=1)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_logsumexp(self):
        """Test logsumexp function."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = flashlight.logsumexp(x, dim=1)
        expected = torch.logsumexp(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dim=1)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestMatrixOps:
    """Test matrix operations."""

    def test_addmm(self):
        """Test addmm function (beta*input + alpha*mat1@mat2)."""
        M = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        mat1 = flashlight.tensor([[1.0, 0.0], [0.0, 1.0]])
        mat2 = flashlight.tensor([[2.0, 1.0], [1.0, 2.0]])
        result = flashlight.addmm(M, mat1, mat2, beta=0.5, alpha=2.0)
        expected = torch.addmm(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                                torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
                                beta=0.5, alpha=2.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_mv(self):
        """Test mv function (matrix-vector product)."""
        mat = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        vec = flashlight.tensor([1.0, 2.0])
        result = flashlight.mv(mat, vec)
        expected = torch.mv(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([1.0, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_ger(self):
        """Test ger function (outer product)."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        y = flashlight.tensor([4.0, 5.0])
        result = flashlight.ger(x, y)
        expected = torch.ger(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_addr(self):
        """Test addr function (beta*input + alpha*vec1 x vec2)."""
        M = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vec1 = flashlight.tensor([1.0, 2.0, 3.0])
        vec2 = flashlight.tensor([1.0, 2.0])
        result = flashlight.addr(M, vec1, vec2, beta=0.5, alpha=2.0)
        expected = torch.addr(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                               torch.tensor([1.0, 2.0, 3.0]),
                               torch.tensor([1.0, 2.0]),
                               beta=0.5, alpha=2.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestDist:
    """Test distance computation."""

    def test_dist_l2(self):
        """Test dist with p=2 (Euclidean)."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        y = flashlight.tensor([4.0, 5.0, 6.0])
        result = flashlight.dist(x, y, p=2.0)
        expected = torch.dist(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), p=2.0)
        np.testing.assert_allclose(result.item(), expected.item(), rtol=1e-5)

    def test_dist_l1(self):
        """Test dist with p=1 (Manhattan)."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        y = flashlight.tensor([4.0, 5.0, 6.0])
        result = flashlight.dist(x, y, p=1.0)
        expected = torch.dist(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), p=1.0)
        np.testing.assert_allclose(result.item(), expected.item(), rtol=1e-5)


class TestNorm:
    """Test norm computations."""

    def test_frobenius_norm(self):
        """Test frobenius_norm function."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.frobenius_norm(x)
        expected = torch.linalg.norm(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), ord='fro')
        np.testing.assert_allclose(result.item(), expected.item(), rtol=1e-5)


class TestStats:
    """Test statistical functions."""

    def test_corrcoef(self):
        """Test corrcoef function."""
        x = flashlight.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = flashlight.corrcoef(x)
        expected = torch.corrcoef(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-4)

    def test_cov(self):
        """Test cov function."""
        x = flashlight.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = flashlight.cov(x)
        expected = torch.cov(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-4)


class TestNegativePositive:
    """Test negative and positive functions."""

    def test_negative(self):
        """Test negative function."""
        x = flashlight.tensor([-1.0, 0.0, 1.0, 2.0])
        result = flashlight.negative(x)
        expected = torch.negative(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)

    def test_positive(self):
        """Test positive function."""
        x = flashlight.tensor([-1.0, 0.0, 1.0, 2.0])
        result = flashlight.positive(x)
        expected = torch.positive(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestNumel:
    """Test numel function."""

    def test_numel(self):
        """Test numel function."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = flashlight.numel(x)
        expected = torch.numel(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert result == expected


class TestConj:
    """Test conj function."""

    def test_conj_real(self):
        """Test conj on real tensor."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        result = flashlight.conj(x)
        # For real tensors, conj should return same values
        np.testing.assert_allclose(np.array(result.tolist()), np.array([1.0, 2.0, 3.0]), rtol=1e-5)


class TestBatchMatrixOps:
    """Test batch matrix operations."""

    def test_baddbmm(self):
        """Test baddbmm function."""
        input_t = flashlight.zeros((3, 2, 4))
        batch1 = flashlight.randn((3, 2, 3))
        batch2 = flashlight.randn((3, 3, 4))
        result = flashlight.baddbmm(input_t, batch1, batch2)
        assert result.shape == (3, 2, 4)

    def test_addbmm(self):
        """Test addbmm function."""
        M = flashlight.zeros((2, 4))
        batch1 = flashlight.randn((3, 2, 3))
        batch2 = flashlight.randn((3, 3, 4))
        result = flashlight.addbmm(M, batch1, batch2)
        assert result.shape == (2, 4)


class TestChainMatmul:
    """Test chain_matmul function."""

    def test_chain_matmul(self):
        """Test chain_matmul function."""
        a = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = flashlight.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = flashlight.tensor([[9.0, 10.0], [11.0, 12.0]])
        result = flashlight.chain_matmul(a, b, c)
        expected = torch.chain_matmul(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                                       torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                                       torch.tensor([[9.0, 10.0], [11.0, 12.0]]))
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-4)


class TestFill:
    """Test fill function."""

    def test_fill(self):
        """Test fill_ function."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        flashlight.fill_(x, 5.0)
        expected = [[5.0, 5.0], [5.0, 5.0]]
        np.testing.assert_allclose(np.array(x.tolist()), expected, rtol=1e-5)


class TestConstantPad:
    """Test constant padding."""

    def test_constant_pad_nd(self):
        """Test constant_pad_nd function."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.constant_pad_nd(x, (1, 1, 1, 1), value=0.0)
        expected = torch.nn.functional.pad(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                                            (1, 1, 1, 1), mode='constant', value=0.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


class TestAddmv:
    """Test addmv function."""

    def test_addmv(self):
        """Test addmv (matrix-vector multiply with add)."""
        M = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vec = flashlight.tensor([1.0, 2.0, 3.0])
        input_t = flashlight.tensor([1.0, 2.0])
        result = flashlight.addmv(input_t, M, vec, beta=1.0, alpha=1.0)
        expected = torch.addmv(torch.tensor([1.0, 2.0]),
                                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                                torch.tensor([1.0, 2.0, 3.0]),
                                beta=1.0, alpha=1.0)
        np.testing.assert_allclose(np.array(result.tolist()), expected.numpy(), rtol=1e-5)


@pytest.mark.parity
class TestQuickOpsParity:
    """Test PyTorch parity for quick ops."""

    def test_erf_parity_multidim(self):
        """Test erf parity with multi-dimensional tensors."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)
        x_torch = torch.tensor(x_np)

        result_mlx = flashlight.erf(x_mlx)
        result_torch = torch.erf(x_torch)

        np.testing.assert_allclose(
            np.array(result_mlx.tolist()),
            result_torch.numpy(),
            rtol=1e-4  # MLX erf may have slightly different precision
        )

    def test_sigmoid_parity_multidim(self):
        """Test sigmoid parity with multi-dimensional tensors."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)
        x_torch = torch.tensor(x_np)

        result_mlx = flashlight.sigmoid(x_mlx)
        result_torch = torch.sigmoid(x_torch)

        np.testing.assert_allclose(
            np.array(result_mlx.tolist()),
            result_torch.numpy(),
            rtol=1e-5
        )

    def test_softmax_parity_multidim(self):
        """Test softmax parity with multi-dimensional tensors."""
        x_np = np.random.randn(3, 4, 5).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)
        x_torch = torch.tensor(x_np)

        for dim in range(3):
            result_mlx = flashlight.softmax(x_mlx, dim=dim)
            result_torch = torch.softmax(x_torch, dim=dim)
            np.testing.assert_allclose(
                np.array(result_mlx.tolist()),
                result_torch.numpy(),
                rtol=1e-5
            )

    def test_addmm_parity(self):
        """Test addmm parity with PyTorch."""
        M_np = np.random.randn(3, 4).astype(np.float32)
        mat1_np = np.random.randn(3, 5).astype(np.float32)
        mat2_np = np.random.randn(5, 4).astype(np.float32)

        M_mlx = flashlight.tensor(M_np)
        mat1_mlx = flashlight.tensor(mat1_np)
        mat2_mlx = flashlight.tensor(mat2_np)

        M_torch = torch.tensor(M_np)
        mat1_torch = torch.tensor(mat1_np)
        mat2_torch = torch.tensor(mat2_np)

        result_mlx = flashlight.addmm(M_mlx, mat1_mlx, mat2_mlx, beta=0.5, alpha=2.0)
        result_torch = torch.addmm(M_torch, mat1_torch, mat2_torch, beta=0.5, alpha=2.0)

        np.testing.assert_allclose(
            np.array(result_mlx.tolist()),
            result_torch.numpy(),
            rtol=1e-4
        )
