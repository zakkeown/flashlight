"""
FFT Parity Tests

Tests numerical parity between mlx_compat.fft and PyTorch torch.fft.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import mlx_compat


class TestBasicFFTParity:
    """Test basic 1D FFT parity with PyTorch."""

    @pytest.mark.parity
    def test_fft_parity(self):
        """Test fft matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(64).astype(np.float32)

        torch_out = torch.fft.fft(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.fft(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"fft mismatch: {max_diff}"

    @pytest.mark.parity
    def test_ifft_parity(self):
        """Test ifft matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(64).astype(np.float32) + 1j * np.random.randn(64).astype(np.float32)
        x_np = x_np.astype(np.complex64)

        torch_out = torch.fft.ifft(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.ifft(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"ifft mismatch: {max_diff}"

    @pytest.mark.parity
    def test_fft_with_n_parity(self):
        """Test fft with n parameter matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(32).astype(np.float32)

        torch_out = torch.fft.fft(torch.tensor(x_np), n=64)
        mlx_out = mlx_compat.fft.fft(mlx_compat.tensor(x_np), n=64)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"fft with n mismatch: {max_diff}"


class TestRealFFTParity:
    """Test real FFT parity with PyTorch."""

    @pytest.mark.parity
    def test_rfft_parity(self):
        """Test rfft matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(64).astype(np.float32)

        torch_out = torch.fft.rfft(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.rfft(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"rfft mismatch: {max_diff}"

    @pytest.mark.parity
    def test_irfft_parity(self):
        """Test irfft matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(33).astype(np.float32) + 1j * np.random.randn(33).astype(np.float32)
        x_np = x_np.astype(np.complex64)
        # First and last elements should be real for valid rfft output
        x_np[0] = x_np[0].real
        x_np[-1] = x_np[-1].real

        torch_out = torch.fft.irfft(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.irfft(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"irfft mismatch: {max_diff}"


class Test2DFFTParity:
    """Test 2D FFT parity with PyTorch."""

    @pytest.mark.parity
    def test_fft2_parity(self):
        """Test fft2 matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(16, 16).astype(np.float32)

        torch_out = torch.fft.fft2(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.fft2(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"fft2 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_ifft2_parity(self):
        """Test ifft2 matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(16, 16).astype(np.float32) + 1j * np.random.randn(16, 16).astype(np.float32)
        x_np = x_np.astype(np.complex64)

        torch_out = torch.fft.ifft2(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.ifft2(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"ifft2 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_rfft2_parity(self):
        """Test rfft2 matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(16, 16).astype(np.float32)

        torch_out = torch.fft.rfft2(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.rfft2(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"rfft2 mismatch: {max_diff}"


class TestNDFFTParity:
    """Test N-dimensional FFT parity with PyTorch."""

    @pytest.mark.parity
    def test_fftn_parity(self):
        """Test fftn matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(8, 8, 8).astype(np.float32)

        torch_out = torch.fft.fftn(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.fftn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"fftn mismatch: {max_diff}"

    @pytest.mark.parity
    def test_ifftn_parity(self):
        """Test ifftn matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(8, 8, 8).astype(np.float32) + 1j * np.random.randn(8, 8, 8).astype(np.float32)
        x_np = x_np.astype(np.complex64)

        torch_out = torch.fft.ifftn(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.ifftn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"ifftn mismatch: {max_diff}"

    @pytest.mark.parity
    def test_rfftn_parity(self):
        """Test rfftn matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(8, 8, 8).astype(np.float32)

        torch_out = torch.fft.rfftn(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.rfftn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"rfftn mismatch: {max_diff}"


class TestFFTShiftParity:
    """Test FFT shift parity with PyTorch."""

    @pytest.mark.parity
    def test_fftshift_1d_parity(self):
        """Test fftshift 1D matches PyTorch."""
        x_np = np.arange(10).astype(np.float32)

        torch_out = torch.fft.fftshift(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.fftshift(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6, f"fftshift 1D mismatch: {max_diff}"

    @pytest.mark.parity
    def test_fftshift_2d_parity(self):
        """Test fftshift 2D matches PyTorch."""
        x_np = np.arange(16).reshape(4, 4).astype(np.float32)

        torch_out = torch.fft.fftshift(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.fftshift(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6, f"fftshift 2D mismatch: {max_diff}"

    @pytest.mark.parity
    def test_ifftshift_parity(self):
        """Test ifftshift matches PyTorch."""
        x_np = np.arange(10).astype(np.float32)

        torch_out = torch.fft.ifftshift(torch.tensor(x_np))
        mlx_out = mlx_compat.fft.ifftshift(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6, f"ifftshift mismatch: {max_diff}"


class TestFFTFrequenciesParity:
    """Test FFT frequency functions parity with PyTorch."""

    @pytest.mark.parity
    def test_fftfreq_even_parity(self):
        """Test fftfreq with even length matches PyTorch."""
        torch_out = torch.fft.fftfreq(16, d=0.1)
        mlx_out = mlx_compat.fft.fftfreq(16, d=0.1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"fftfreq even mismatch: {max_diff}"

    @pytest.mark.parity
    def test_fftfreq_odd_parity(self):
        """Test fftfreq with odd length matches PyTorch."""
        torch_out = torch.fft.fftfreq(17, d=0.1)
        mlx_out = mlx_compat.fft.fftfreq(17, d=0.1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"fftfreq odd mismatch: {max_diff}"

    @pytest.mark.parity
    def test_rfftfreq_even_parity(self):
        """Test rfftfreq with even length matches PyTorch."""
        torch_out = torch.fft.rfftfreq(16, d=0.1)
        mlx_out = mlx_compat.fft.rfftfreq(16, d=0.1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"rfftfreq even mismatch: {max_diff}"

    @pytest.mark.parity
    def test_rfftfreq_odd_parity(self):
        """Test rfftfreq with odd length matches PyTorch."""
        torch_out = torch.fft.rfftfreq(17, d=0.1)
        mlx_out = mlx_compat.fft.rfftfreq(17, d=0.1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"rfftfreq odd mismatch: {max_diff}"


class TestFFTRoundtripParity:
    """Test FFT roundtrip properties match PyTorch."""

    @pytest.mark.parity
    def test_fft_ifft_roundtrip_parity(self):
        """Test FFT -> IFFT roundtrip matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(64).astype(np.float32)

        # PyTorch roundtrip
        torch_x = torch.tensor(x_np)
        torch_roundtrip = torch.fft.ifft(torch.fft.fft(torch_x))

        # MLX compat roundtrip
        mlx_x = mlx_compat.tensor(x_np)
        mlx_roundtrip = mlx_compat.fft.ifft(mlx_compat.fft.fft(mlx_x))

        # Compare roundtrip results
        max_diff = np.max(np.abs(torch_roundtrip.numpy() - np.array(mlx_roundtrip._mlx_array)))
        assert max_diff < 1e-5, f"FFT roundtrip mismatch: {max_diff}"

    @pytest.mark.parity
    def test_rfft_irfft_roundtrip_parity(self):
        """Test RFFT -> IRFFT roundtrip matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(64).astype(np.float32)

        # PyTorch roundtrip
        torch_x = torch.tensor(x_np)
        torch_roundtrip = torch.fft.irfft(torch.fft.rfft(torch_x))

        # MLX compat roundtrip
        mlx_x = mlx_compat.tensor(x_np)
        mlx_roundtrip = mlx_compat.fft.irfft(mlx_compat.fft.rfft(mlx_x))

        # Compare roundtrip results
        max_diff = np.max(np.abs(torch_roundtrip.numpy() - np.array(mlx_roundtrip._mlx_array)))
        assert max_diff < 1e-5, f"RFFT roundtrip mismatch: {max_diff}"


class TestFFTDimensionParity:
    """Test FFT along specific dimensions matches PyTorch."""

    @pytest.mark.parity
    def test_fft_dim_parity(self):
        """Test FFT along specific dimension matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(8, 16).astype(np.float32)

        # Along dim=0
        torch_out = torch.fft.fft(torch.tensor(x_np), dim=0)
        mlx_out = mlx_compat.fft.fft(mlx_compat.tensor(x_np), dim=0)
        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"fft dim=0 mismatch: {max_diff}"

        # Along dim=1
        torch_out = torch.fft.fft(torch.tensor(x_np), dim=1)
        mlx_out = mlx_compat.fft.fft(mlx_compat.tensor(x_np), dim=1)
        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"fft dim=1 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_rfft_dim_parity(self):
        """Test RFFT along specific dimension matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(8, 16).astype(np.float32)

        torch_out = torch.fft.rfft(torch.tensor(x_np), dim=1)
        mlx_out = mlx_compat.fft.rfft(mlx_compat.tensor(x_np), dim=1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"rfft dim mismatch: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
