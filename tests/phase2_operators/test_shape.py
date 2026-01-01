"""
Test Phase 2: Shape Manipulation Operations

Tests shape manipulation operations (cat, stack, split, expand, repeat, etc.)
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestCat(TestCase):
    """Test concatenation."""

    def test_cat_1d(self):
        """Test concatenating 1D tensors."""
        a = flashlight.tensor([1.0, 2.0])
        b = flashlight.tensor([3.0, 4.0])
        c = flashlight.cat([a, b])
        np.testing.assert_array_equal(c.numpy(), np.array([1.0, 2.0, 3.0, 4.0]))

    def test_cat_2d_dim0(self):
        """Test concatenating along dim 0."""
        a = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = flashlight.tensor([[5.0, 6.0]])
        c = flashlight.cat([a, b], dim=0)
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_cat_2d_dim1(self):
        """Test concatenating along dim 1."""
        a = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = flashlight.tensor([[5.0], [6.0]])
        c = flashlight.cat([a, b], dim=1)
        expected = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_cat_requires_grad(self):
        """Test that cat propagates requires_grad."""
        a = flashlight.tensor([1.0, 2.0], requires_grad=True)
        b = flashlight.tensor([3.0, 4.0])
        c = flashlight.cat([a, b])
        self.assertTrue(c.requires_grad)


@skipIfNoMLX
class TestStack(TestCase):
    """Test stacking."""

    def test_stack_1d(self):
        """Test stacking 1D tensors."""
        a = flashlight.tensor([1.0, 2.0])
        b = flashlight.tensor([3.0, 4.0])
        c = flashlight.stack([a, b])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_stack_dim1(self):
        """Test stacking along dim 1."""
        a = flashlight.tensor([1.0, 2.0])
        b = flashlight.tensor([3.0, 4.0])
        c = flashlight.stack([a, b], dim=1)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_stack_requires_grad(self):
        """Test that stack propagates requires_grad."""
        a = flashlight.tensor([1.0, 2.0])
        b = flashlight.tensor([3.0, 4.0], requires_grad=True)
        c = flashlight.stack([a, b])
        self.assertTrue(c.requires_grad)


@skipIfNoMLX
class TestSplit(TestCase):
    """Test splitting."""

    def test_split_equal_chunks(self):
        """Test splitting into equal chunks."""
        x = flashlight.arange(10)
        chunks = flashlight.split(x, 3)
        self.assertEqual(len(chunks), 4)  # 3, 3, 3, 1
        np.testing.assert_array_equal(chunks[0].numpy(), np.array([0, 1, 2]))
        np.testing.assert_array_equal(chunks[1].numpy(), np.array([3, 4, 5]))
        np.testing.assert_array_equal(chunks[2].numpy(), np.array([6, 7, 8]))
        np.testing.assert_array_equal(chunks[3].numpy(), np.array([9]))

    def test_split_sections(self):
        """Test splitting with specified sections."""
        x = flashlight.arange(10)
        chunks = flashlight.split(x, [2, 5, 3])
        self.assertEqual(len(chunks), 3)
        np.testing.assert_array_equal(chunks[0].numpy(), np.array([0, 1]))
        np.testing.assert_array_equal(chunks[1].numpy(), np.array([2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(chunks[2].numpy(), np.array([7, 8, 9]))

    def test_split_2d(self):
        """Test splitting 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        chunks = flashlight.split(x, 2, dim=1)
        self.assertEqual(len(chunks), 2)
        self.assert_shape_equal(chunks[0].shape, (2, 2))


@skipIfNoMLX
class TestChunk(TestCase):
    """Test chunking."""

    def test_chunk(self):
        """Test chunking tensor."""
        x = flashlight.arange(10)
        chunks = flashlight.chunk(x, 3)
        self.assertEqual(len(chunks), 3)
        # Should split into chunks of size ceil(10/3) = 4
        np.testing.assert_array_equal(chunks[0].numpy(), np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(chunks[1].numpy(), np.array([4, 5, 6, 7]))
        np.testing.assert_array_equal(chunks[2].numpy(), np.array([8, 9]))

    def test_chunk_2d(self):
        """Test chunking 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        chunks = flashlight.chunk(x, 2, dim=0)
        self.assertEqual(len(chunks), 2)
        self.assert_shape_equal(chunks[0].shape, (2, 2))
        self.assert_shape_equal(chunks[1].shape, (1, 2))


@skipIfNoMLX
class TestExpand(TestCase):
    """Test expand operation."""

    def test_expand_1d_to_2d(self):
        """Test expanding 1D to 2D."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        y = flashlight.expand(x, 2, 3)
        self.assert_shape_equal(y.shape, (2, 3))
        expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_expand_with_negative_one(self):
        """Test expand with -1 (keep dimension)."""
        x = flashlight.tensor([[1.0], [2.0]])
        y = flashlight.expand(x, 2, 3)
        self.assert_shape_equal(y.shape, (2, 3))

    def test_expand_is_view(self):
        """Test that expand creates a view."""
        x = flashlight.tensor([1.0, 2.0])
        y = flashlight.expand(x, 3, 2)
        self.assertTrue(y.is_view)


@skipIfNoMLX
class TestRepeat(TestCase):
    """Test repeat operation."""

    def test_repeat_1d(self):
        """Test repeating 1D tensor."""
        x = flashlight.tensor([1.0, 2.0])
        y = flashlight.repeat(x, 3)
        np.testing.assert_array_equal(y.numpy(), np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]))

    def test_repeat_2d(self):
        """Test repeating 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0]])
        y = flashlight.repeat(x, 2, 3)
        self.assert_shape_equal(y.shape, (2, 6))

    def test_repeat_tuple_arg(self):
        """Test repeat with tuple argument."""
        x = flashlight.tensor([[1.0, 2.0]])
        y = flashlight.repeat(x, (2, 3))
        self.assert_shape_equal(y.shape, (2, 6))


@skipIfNoMLX
class TestRepeatInterleave(TestCase):
    """Test repeat_interleave operation."""

    def test_repeat_interleave_scalar(self):
        """Test repeat_interleave with scalar."""
        x = flashlight.tensor([1.0, 2.0, 3.0])
        y = flashlight.repeat_interleave(x, 2, dim=0)
        expected = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_repeat_interleave_no_dim(self):
        """Test repeat_interleave without dim (flatten)."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.repeat_interleave(x, 2)
        expected = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestGather(TestCase):
    """Test gather operation."""

    def test_gather_1d(self):
        """Test gather on 1D tensor."""
        x = flashlight.tensor([10.0, 20.0, 30.0, 40.0])
        idx = flashlight.tensor([0, 2, 1], dtype=flashlight.int32)
        y = flashlight.gather(x, 0, idx)
        np.testing.assert_array_equal(y.numpy(), np.array([10.0, 30.0, 20.0]))

    def test_gather_2d(self):
        """Test gather on 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = flashlight.tensor([[0, 1], [2, 0]], dtype=flashlight.int32)
        y = flashlight.gather(x, 0, idx)
        expected = np.array([[1.0, 4.0], [5.0, 2.0]])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestNarrow(TestCase):
    """Test narrow operation."""

    def test_narrow_1d(self):
        """Test narrow on 1D tensor."""
        x = flashlight.arange(10)
        y = flashlight.narrow(x, 0, 2, 5)
        np.testing.assert_array_equal(y.numpy(), np.array([2, 3, 4, 5, 6]))

    def test_narrow_2d(self):
        """Test narrow on 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = flashlight.narrow(x, 1, 1, 2)
        expected = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_narrow_is_view(self):
        """Test that narrow creates a view."""
        x = flashlight.arange(10)
        y = flashlight.narrow(x, 0, 2, 5)
        self.assertTrue(y.is_view)


@skipIfNoMLX
class TestSelect(TestCase):
    """Test select operation."""

    def test_select_2d(self):
        """Test select on 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = flashlight.select(x, 0, 1)
        np.testing.assert_array_equal(y.numpy(), np.array([4.0, 5.0, 6.0]))

    def test_select_reduces_dim(self):
        """Test that select reduces dimensionality."""
        x = flashlight.randn(3, 4, 5)
        y = flashlight.select(x, 1, 2)
        self.assert_shape_equal(y.shape, (3, 5))

    def test_select_is_view(self):
        """Test that select creates a view."""
        x = flashlight.randn(3, 4)
        y = flashlight.select(x, 0, 1)
        self.assertTrue(y.is_view)


@skipIfNoMLX
class TestUnbind(TestCase):
    """Test unbind operation."""

    def test_unbind_2d(self):
        """Test unbind on 2D tensor."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        results = flashlight.unbind(x, dim=0)
        self.assertEqual(len(results), 3)
        np.testing.assert_array_equal(results[0].numpy(), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(results[1].numpy(), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(results[2].numpy(), np.array([5.0, 6.0]))

    def test_unbind_default_dim(self):
        """Test unbind with default dim."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        results = flashlight.unbind(x)
        self.assertEqual(len(results), 2)


@skipIfNoMLX
class TestRoll(TestCase):
    """Test roll operation."""

    def test_roll_1d(self):
        """Test rolling 1D tensor."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = flashlight.roll(x, 2)
        # Elements shift right by 2, wrapping around
        expected = np.array([4.0, 5.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_roll_1d_negative(self):
        """Test rolling with negative shift."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = flashlight.roll(x, -2)
        # Elements shift left by 2
        expected = np.array([3.0, 4.0, 5.0, 1.0, 2.0])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_roll_2d_dim0(self):
        """Test rolling 2D tensor along dim 0."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = flashlight.roll(x, 1, dims=0)
        expected = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_roll_2d_dim1(self):
        """Test rolling 2D tensor along dim 1."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = flashlight.roll(x, 1, dims=1)
        expected = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_roll_multiple_dims(self):
        """Test rolling along multiple dimensions."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = flashlight.roll(x, shifts=(1, 2), dims=(0, 1))
        # Roll 1 along dim 0, then 2 along dim 1
        expected = np.array([[5.0, 6.0, 4.0], [2.0, 3.0, 1.0]])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestFlip(TestCase):
    """Test flip operation."""

    def test_flip_1d(self):
        """Test flipping 1D tensor."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = flashlight.flip(x, [0])
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_flip_2d_dim0(self):
        """Test flipping 2D tensor along dim 0."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = flashlight.flip(x, [0])
        expected = np.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_flip_2d_dim1(self):
        """Test flipping 2D tensor along dim 1."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = flashlight.flip(x, [1])
        expected = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_flip_multiple_dims(self):
        """Test flipping along multiple dimensions."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.flip(x, [0, 1])
        expected = np.array([[4.0, 3.0], [2.0, 1.0]])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestFliplr(TestCase):
    """Test fliplr operation."""

    def test_fliplr(self):
        """Test flipping left-right."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = flashlight.fliplr(x)
        expected = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestFlipud(TestCase):
    """Test flipud operation."""

    def test_flipud(self):
        """Test flipping up-down."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = flashlight.flipud(x)
        expected = np.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestRot90(TestCase):
    """Test rot90 operation."""

    def test_rot90_k1(self):
        """Test rotating 90 degrees once."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.rot90(x, k=1)
        expected = np.array([[2.0, 4.0], [1.0, 3.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_rot90_k2(self):
        """Test rotating 180 degrees."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.rot90(x, k=2)
        expected = np.array([[4.0, 3.0], [2.0, 1.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_rot90_k3(self):
        """Test rotating 270 degrees."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.rot90(x, k=3)
        expected = np.array([[3.0, 1.0], [4.0, 2.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_rot90_k0(self):
        """Test rotating 0 degrees (identity)."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.rot90(x, k=0)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_rot90_k4(self):
        """Test rotating 360 degrees (identity)."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.rot90(x, k=4)
        np.testing.assert_array_equal(y.numpy(), x.numpy())


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
