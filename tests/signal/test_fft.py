"""
Comprehensive tests for flashlight.fft module.

Tests all FFT functions with numerical parity against numpy.fft.
"""

import numpy as np
import pytest

import flashlight


class TestBasicFFT:
    """Tests for basic 1D FFT operations."""

    def test_fft_basic(self):
        """Test 1D FFT on basic signal."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.fft.fft(x)
        expected = np.fft.fft(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fft_with_n(self):
        """Test 1D FFT with specific length."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.fft.fft(x, n=8)
        expected = np.fft.fft(np.array([1.0, 2.0, 3.0, 4.0]), n=8)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_ifft_basic(self):
        """Test inverse 1D FFT."""
        x = flashlight.tensor([1.0 + 0j, 2.0 + 1j, 3.0 - 1j, 4.0 + 0j])
        result = flashlight.fft.ifft(x)
        expected = np.fft.ifft(np.array([1.0 + 0j, 2.0 + 1j, 3.0 - 1j, 4.0 + 0j]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fft_ifft_roundtrip(self):
        """Test FFT -> IFFT roundtrip."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = flashlight.fft.ifft(flashlight.fft.fft(x))
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        np.testing.assert_allclose(
            np.real(np.array(result._mlx_array)), expected, rtol=1e-5, atol=1e-6
        )


class TestRealFFT:
    """Tests for real-valued FFT operations."""

    def test_rfft_basic(self):
        """Test real FFT."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.fft.rfft(x)
        expected = np.fft.rfft(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_irfft_basic(self):
        """Test inverse real FFT."""
        # Start with complex FFT result
        x = np.array([10.0 + 0j, -2.0 + 2j, -2.0 + 0j])
        x_tensor = flashlight.tensor(x)
        result = flashlight.fft.irfft(x_tensor)
        expected = np.fft.irfft(x)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_rfft_irfft_roundtrip(self):
        """Test rfft -> irfft roundtrip."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.fft.irfft(flashlight.fft.rfft(x))
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)


class Test2DFFT:
    """Tests for 2D FFT operations."""

    def test_fft2_basic(self):
        """Test 2D FFT."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.fft.fft2(x)
        expected = np.fft.fft2(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_ifft2_basic(self):
        """Test inverse 2D FFT."""
        x = flashlight.tensor([[1.0 + 0j, 2.0 + 1j], [3.0 - 1j, 4.0 + 0j]])
        result = flashlight.fft.ifft2(x)
        expected = np.fft.ifft2(np.array([[1.0 + 0j, 2.0 + 1j], [3.0 - 1j, 4.0 + 0j]]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_rfft2_basic(self):
        """Test real 2D FFT."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = flashlight.fft.rfft2(x)
        expected = np.fft.rfft2(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_irfft2_basic(self):
        """Test inverse real 2D FFT."""
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        fft_result = np.fft.rfft2(x_np)
        x_tensor = flashlight.tensor(fft_result)
        # Need to specify the output shape s=(2, 3) to match original
        result = flashlight.fft.irfft2(x_tensor, s=(2, 3))
        np.testing.assert_allclose(np.array(result._mlx_array), x_np, rtol=1e-5, atol=1e-6)


class TestNDFFT:
    """Tests for N-dimensional FFT operations."""

    def test_fftn_basic(self):
        """Test N-D FFT."""
        x = flashlight.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = flashlight.fft.fftn(x)
        expected = np.fft.fftn(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_ifftn_basic(self):
        """Test inverse N-D FFT."""
        x_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        fft_result = np.fft.fftn(x_np)
        x_tensor = flashlight.tensor(fft_result)
        result = flashlight.fft.ifftn(x_tensor)
        np.testing.assert_allclose(np.real(np.array(result._mlx_array)), x_np, rtol=1e-5, atol=1e-6)

    def test_rfftn_basic(self):
        """Test real N-D FFT."""
        x = flashlight.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = flashlight.fft.rfftn(x)
        expected = np.fft.rfftn(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)


class TestFFTShift:
    """Tests for FFT shift operations."""

    def test_fftshift_1d(self):
        """Test 1D FFT shift."""
        x = flashlight.tensor([0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0])
        result = flashlight.fft.fftshift(x)
        expected = np.fft.fftshift(np.array([0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_ifftshift_1d(self):
        """Test inverse 1D FFT shift."""
        x = flashlight.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        result = flashlight.fft.ifftshift(x)
        expected = np.fft.ifftshift(np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fftshift_2d(self):
        """Test 2D FFT shift."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = flashlight.fft.fftshift(x)
        expected = np.fft.fftshift(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)


class TestFFTFrequencies:
    """Tests for FFT frequency functions."""

    def test_fftfreq_even(self):
        """Test fftfreq with even length."""
        result = flashlight.fft.fftfreq(8, d=0.1)
        expected = np.fft.fftfreq(8, d=0.1)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fftfreq_odd(self):
        """Test fftfreq with odd length."""
        result = flashlight.fft.fftfreq(9, d=0.1)
        expected = np.fft.fftfreq(9, d=0.1)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fftfreq_default_d(self):
        """Test fftfreq with default spacing."""
        result = flashlight.fft.fftfreq(10)
        expected = np.fft.fftfreq(10)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_rfftfreq_even(self):
        """Test rfftfreq with even length."""
        result = flashlight.fft.rfftfreq(8, d=0.1)
        expected = np.fft.rfftfreq(8, d=0.1)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_rfftfreq_odd(self):
        """Test rfftfreq with odd length."""
        result = flashlight.fft.rfftfreq(9, d=0.1)
        expected = np.fft.rfftfreq(9, d=0.1)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)


class TestHermitianFFT:
    """Tests for Hermitian FFT operations."""

    def test_hfft_basic(self):
        """Test Hermitian FFT."""
        # Hermitian-symmetric input (for real output)
        x = flashlight.tensor([1.0 + 0j, 2.0 + 1j, 3.0 + 0j, 2.0 - 1j])
        result = flashlight.fft.hfft(x)
        expected = np.fft.hfft(np.array([1.0 + 0j, 2.0 + 1j, 3.0 + 0j, 2.0 - 1j]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-4)

    def test_ihfft_basic(self):
        """Test inverse Hermitian FFT."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = flashlight.fft.ihfft(x)
        expected = np.fft.ihfft(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-4)


class TestFFTDimensions:
    """Tests for FFT with different dimensions."""

    def test_fft_along_dim(self):
        """Test FFT along specific dimension."""
        x = flashlight.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = flashlight.fft.fft(x, dim=1)
        expected = np.fft.fft(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), axis=1)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fft_along_dim0(self):
        """Test FFT along first dimension."""
        x = flashlight.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = flashlight.fft.fft(x, dim=0)
        expected = np.fft.fft(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), axis=0)
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)


class TestFFTSpecialCases:
    """Tests for FFT special cases."""

    def test_fft_length_1(self):
        """Test FFT on length-1 signal."""
        x = flashlight.tensor([5.0])
        result = flashlight.fft.fft(x)
        expected = np.fft.fft(np.array([5.0]))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6)

    def test_fft_power_of_2(self):
        """Test FFT on power-of-2 length."""
        x = flashlight.randn(256)
        result = flashlight.fft.fft(x)
        expected = np.fft.fft(np.array(x._mlx_array))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-5)

    def test_fft_non_power_of_2(self):
        """Test FFT on non-power-of-2 length."""
        x = flashlight.randn(100)
        result = flashlight.fft.fft(x)
        expected = np.fft.fft(np.array(x._mlx_array))
        np.testing.assert_allclose(np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-5)


class TestFFTProperties:
    """Tests for FFT mathematical properties."""

    def test_fft_linearity(self):
        """Test linearity: FFT(ax + by) = a*FFT(x) + b*FFT(y)."""
        x = flashlight.randn(16)
        y = flashlight.randn(16)
        a, b = 2.0, 3.0

        lhs = flashlight.fft.fft(a * x + b * y)
        rhs = a * flashlight.fft.fft(x) + b * flashlight.fft.fft(y)

        np.testing.assert_allclose(
            np.array(lhs._mlx_array), np.array(rhs._mlx_array), rtol=1e-4, atol=1e-5
        )

    def test_fft_parseval(self):
        """Test Parseval's theorem: sum|x|^2 = sum|X|^2 / n."""
        x = flashlight.randn(16)
        X = flashlight.fft.fft(x)

        time_energy = float(flashlight.sum(x * x).item())
        freq_energy = float(flashlight.sum(flashlight.real(X * flashlight.conj(X))).item()) / 16

        np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
