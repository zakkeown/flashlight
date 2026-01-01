"""
Test Phase 7: Signal Processing - Window Functions

Comprehensive tests for flashlight.signal.windows module.
Includes numerical parity tests against PyTorch.
"""

import sys
sys.path.insert(0, '../..')

import unittest
import math

import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import torch
    import torch.signal.windows as tw
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import flashlight
    from flashlight import signal
    from flashlight.signal import windows
    import mlx.core as mx
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


def requires_torch(func):
    """Skip test if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        return unittest.skip("PyTorch not available")(func)
    return func


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@skipIfNoMLX
class TestBartlettWindow(TestCase):
    """Test bartlett window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.bartlett(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        # M=0 should return empty
        win0 = windows.bartlett(0)
        self.assertEqual(win0.shape, (0,))

        # M=1 should return [1.0]
        win1 = windows.bartlett(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric when sym=True."""
        win = windows.bartlett(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_periodic_mode(self):
        """Test periodic mode (sym=False)."""
        M = 64
        win_sym = windows.bartlett(M, sym=True)
        win_per = windows.bartlett(M, sym=False)
        self.assertEqual(win_sym.shape, (M,))
        self.assertEqual(win_per.shape, (M,))
        # Periodic window should be different from symmetric
        self.assertFalse(np.allclose(win_sym.numpy(), win_per.numpy()))

    def test_dtype(self):
        """Test dtype parameter."""
        win32 = windows.bartlett(64, dtype=flashlight.float32)
        self.assertEqual(win32.dtype, flashlight.float32)

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        win = windows.bartlett(64, requires_grad=True)
        self.assertTrue(win.requires_grad)
        win_no_grad = windows.bartlett(64, requires_grad=False)
        self.assertFalse(win_no_grad.requires_grad)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256, 1000]:
            for sym in [True, False]:
                mlx_win = windows.bartlett(M, sym=sym)
                torch_win = tw.bartlett(M, sym=sym)
                np.testing.assert_allclose(
                    mlx_win.numpy(),
                    torch_win.numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for M={M}, sym={sym}"
                )


@skipIfNoMLX
class TestBlackmanWindow(TestCase):
    """Test blackman window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.blackman(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.blackman(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.blackman(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric when sym=True."""
        win = windows.blackman(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_endpoints(self):
        """Test Blackman window endpoints are near zero."""
        win = windows.blackman(64, sym=True)
        arr = win.numpy()
        # Blackman window should be close to 0 at endpoints
        self.assertLess(abs(arr[0]), 1e-4)
        self.assertLess(abs(arr[-1]), 1e-4)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256, 1000]:
            for sym in [True, False]:
                mlx_win = windows.blackman(M, sym=sym)
                torch_win = tw.blackman(M, sym=sym)
                np.testing.assert_allclose(
                    mlx_win.numpy(),
                    torch_win.numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for M={M}, sym={sym}"
                )


@skipIfNoMLX
class TestCosineWindow(TestCase):
    """Test cosine (sine) window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.cosine(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.cosine(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.cosine(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric when sym=True."""
        win = windows.cosine(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256, 1000]:
            for sym in [True, False]:
                mlx_win = windows.cosine(M, sym=sym)
                torch_win = tw.cosine(M, sym=sym)
                np.testing.assert_allclose(
                    mlx_win.numpy(),
                    torch_win.numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for M={M}, sym={sym}"
                )


@skipIfNoMLX
class TestExponentialWindow(TestCase):
    """Test exponential (Poisson) window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.exponential(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.exponential(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.exponential(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_center_parameter(self):
        """Test center parameter."""
        M = 64
        # Default center should be in the middle
        win_default = windows.exponential(M)
        win_center = windows.exponential(M, center=31.5)
        # Should be similar (default is (M-1)/2 = 31.5)
        np.testing.assert_allclose(
            win_default.numpy(),
            win_center.numpy(),
            rtol=1e-5
        )

        # Custom center should shift the peak
        win_left = windows.exponential(M, center=10.0)
        arr = win_left.numpy()
        # Max should be near index 10
        self.assertEqual(np.argmax(arr), 10)

    def test_tau_parameter(self):
        """Test tau (decay) parameter."""
        M = 64
        # Larger tau = slower decay
        win_small_tau = windows.exponential(M, tau=1.0)
        win_large_tau = windows.exponential(M, tau=10.0)
        # With larger tau, endpoints should be larger (slower decay)
        self.assertGreater(
            float(win_large_tau.numpy()[0]),
            float(win_small_tau.numpy()[0])
        )

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256]:
            for sym in [True, False]:
                for tau in [0.5, 1.0, 5.0]:
                    mlx_win = windows.exponential(M, tau=tau, sym=sym)
                    torch_win = tw.exponential(M, tau=tau, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"Mismatch for M={M}, tau={tau}, sym={sym}"
                    )


@skipIfNoMLX
class TestGaussianWindow(TestCase):
    """Test gaussian window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.gaussian(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.gaussian(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.gaussian(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric."""
        win = windows.gaussian(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_std_parameter(self):
        """Test std (standard deviation) parameter."""
        M = 64
        # Larger std = wider window
        win_small_std = windows.gaussian(M, std=3.0)
        win_large_std = windows.gaussian(M, std=15.0)
        # With larger std, endpoints should be larger (slower falloff)
        self.assertGreater(
            float(win_large_std.numpy()[0]),
            float(win_small_std.numpy()[0])
        )

    def test_peak_at_center(self):
        """Test that peak is at center."""
        M = 65  # Odd for clear center
        win = windows.gaussian(M, std=10.0)
        arr = win.numpy()
        self.assertEqual(np.argmax(arr), 32)  # Center index

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256]:
            for sym in [True, False]:
                for std in [1.0, 5.0, 10.0]:
                    mlx_win = windows.gaussian(M, std=std, sym=sym)
                    torch_win = tw.gaussian(M, std=std, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"Mismatch for M={M}, std={std}, sym={sym}"
                    )


@skipIfNoMLX
class TestGeneralCosineWindow(TestCase):
    """Test general_cosine window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128]:
            win = windows.general_cosine(M, a=[0.5, 0.5])
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.general_cosine(0, a=[0.5, 0.5])
        self.assertEqual(win0.shape, (0,))

        win1 = windows.general_cosine(1, a=[0.5, 0.5])
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_hann_coefficients(self):
        """Test that Hann coefficients produce Hann window."""
        M = 64
        # Hann window: a = [0.5, 0.5]
        win_gc = windows.general_cosine(M, a=[0.5, 0.5])
        win_hann = windows.hann(M)
        np.testing.assert_allclose(
            win_gc.numpy(),
            win_hann.numpy(),
            rtol=1e-5
        )

    def test_blackman_coefficients(self):
        """Test that Blackman coefficients produce Blackman window."""
        M = 64
        # Blackman window: a = [0.42, 0.5, 0.08]
        win_gc = windows.general_cosine(M, a=[0.42, 0.5, 0.08])
        win_bm = windows.blackman(M)
        np.testing.assert_allclose(
            win_gc.numpy(),
            win_bm.numpy(),
            rtol=1e-5
        )

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        coefficients = [
            [0.5, 0.5],  # Hann
            [0.54, 0.46],  # Hamming
            [0.42, 0.5, 0.08],  # Blackman
            [0.3635819, 0.4891775, 0.1365995, 0.0106411],  # Nuttall
        ]
        for M in [10, 64, 128]:
            for sym in [True, False]:
                for a in coefficients:
                    mlx_win = windows.general_cosine(M, a=a, sym=sym)
                    torch_win = tw.general_cosine(M, a=a, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"Mismatch for M={M}, a={a}, sym={sym}"
                    )


@skipIfNoMLX
class TestGeneralHammingWindow(TestCase):
    """Test general_hamming window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.general_hamming(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.general_hamming(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.general_hamming(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_alpha_054_is_hamming(self):
        """Test alpha=0.54 produces Hamming window."""
        M = 64
        win_gh = windows.general_hamming(M, alpha=0.54)
        win_ham = windows.hamming(M)
        np.testing.assert_allclose(
            win_gh.numpy(),
            win_ham.numpy(),
            rtol=1e-5
        )

    def test_alpha_05_is_hann(self):
        """Test alpha=0.5 produces Hann window."""
        M = 64
        win_gh = windows.general_hamming(M, alpha=0.5)
        win_hann = windows.hann(M)
        np.testing.assert_allclose(
            win_gh.numpy(),
            win_hann.numpy(),
            rtol=1e-5
        )

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256]:
            for sym in [True, False]:
                for alpha in [0.5, 0.54, 0.6, 0.75]:
                    mlx_win = windows.general_hamming(M, alpha=alpha, sym=sym)
                    torch_win = tw.general_hamming(M, alpha=alpha, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"Mismatch for M={M}, alpha={alpha}, sym={sym}"
                    )


@skipIfNoMLX
class TestHammingWindow(TestCase):
    """Test hamming window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.hamming(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.hamming(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.hamming(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric."""
        win = windows.hamming(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_nonzero_endpoints(self):
        """Test Hamming window has non-zero endpoints."""
        win = windows.hamming(64, sym=True)
        arr = win.numpy()
        # Hamming window should have endpoints at 0.08 (approximately)
        self.assertGreater(abs(arr[0]), 0.05)
        self.assertGreater(abs(arr[-1]), 0.05)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256, 1000]:
            for sym in [True, False]:
                mlx_win = windows.hamming(M, sym=sym)
                torch_win = tw.hamming(M, sym=sym)
                np.testing.assert_allclose(
                    mlx_win.numpy(),
                    torch_win.numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for M={M}, sym={sym}"
                )


@skipIfNoMLX
class TestHannWindow(TestCase):
    """Test hann (hanning) window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.hann(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.hann(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.hann(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric."""
        win = windows.hann(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_zero_endpoints(self):
        """Test Hann window has zero endpoints."""
        win = windows.hann(64, sym=True)
        arr = win.numpy()
        # Hann window should have endpoints at 0
        self.assertLess(abs(arr[0]), 1e-6)
        self.assertLess(abs(arr[-1]), 1e-6)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256, 1000]:
            for sym in [True, False]:
                mlx_win = windows.hann(M, sym=sym)
                torch_win = tw.hann(M, sym=sym)
                np.testing.assert_allclose(
                    mlx_win.numpy(),
                    torch_win.numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for M={M}, sym={sym}"
                )


@skipIfNoMLX
class TestKaiserWindow(TestCase):
    """Test kaiser window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.kaiser(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.kaiser(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.kaiser(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric."""
        win = windows.kaiser(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_beta_parameter(self):
        """Test beta parameter affects window shape."""
        M = 64
        # Larger beta = narrower main lobe
        win_small_beta = windows.kaiser(M, beta=0.0)
        win_large_beta = windows.kaiser(M, beta=14.0)
        # With beta=0, should be rectangular (all ones)
        np.testing.assert_allclose(
            win_small_beta.numpy(),
            np.ones(M),
            rtol=1e-5
        )
        # With large beta, endpoints should be smaller
        self.assertLess(
            float(win_large_beta.numpy()[0]),
            float(win_small_beta.numpy()[0])
        )

    def test_various_betas(self):
        """Test various beta values."""
        M = 64
        for beta in [0.0, 4.0, 6.0, 8.0, 12.0, 14.0, 24.0]:
            win = windows.kaiser(M, beta=beta)
            self.assertEqual(win.shape, (M,))
            # All values should be positive
            self.assertTrue(np.all(win.numpy() >= 0))
            # Max should be 1 (at center)
            self.assertLessEqual(np.max(win.numpy()), 1.0 + 1e-5)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        # Kaiser uses Bessel I0 approximation, so use slightly relaxed tolerance
        for M in [10, 64, 128, 256]:
            for sym in [True, False]:
                for beta in [0.0, 6.0, 12.0, 14.0]:
                    mlx_win = windows.kaiser(M, beta=beta, sym=sym)
                    torch_win = tw.kaiser(M, beta=beta, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,  # Slightly relaxed due to I0 approximation
                        atol=1e-6,
                        err_msg=f"Mismatch for M={M}, beta={beta}, sym={sym}"
                    )


@skipIfNoMLX
class TestNuttallWindow(TestCase):
    """Test nuttall window function."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        for M in [10, 64, 128, 256]:
            win = windows.nuttall(M)
            self.assertEqual(win.shape, (M,))

    def test_edge_cases(self):
        """Test edge cases: M=0, M=1."""
        win0 = windows.nuttall(0)
        self.assertEqual(win0.shape, (0,))

        win1 = windows.nuttall(1)
        self.assertEqual(win1.shape, (1,))
        self.assertAlmostEqual(float(win1.numpy()[0]), 1.0, places=5)

    def test_symmetric_property(self):
        """Test window is symmetric."""
        win = windows.nuttall(64, sym=True)
        arr = win.numpy()
        np.testing.assert_allclose(arr, arr[::-1], rtol=1e-4, atol=1e-6)

    def test_near_zero_endpoints(self):
        """Test Nuttall window has small endpoints."""
        win = windows.nuttall(64, sym=True)
        arr = win.numpy()
        # Nuttall window should have endpoints much smaller than peak
        # Peak is ~1.0, endpoints are ~0.0003
        self.assertLess(abs(arr[0]), 1e-2)
        self.assertLess(abs(arr[-1]), 1e-2)

    @requires_torch
    def test_parity(self):
        """Test numerical parity with PyTorch."""
        for M in [10, 64, 128, 256, 1000]:
            for sym in [True, False]:
                mlx_win = windows.nuttall(M, sym=sym)
                torch_win = tw.nuttall(M, sym=sym)
                np.testing.assert_allclose(
                    mlx_win.numpy(),
                    torch_win.numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Mismatch for M={M}, sym={sym}"
                )


# =============================================================================
# Cross-Function Tests
# =============================================================================

@skipIfNoMLX
class TestWindowProperties(TestCase):
    """Test properties that should hold across all window functions."""

    def test_all_windows_same_shape(self):
        """Test all windows return correct shape."""
        M = 64
        window_funcs = [
            ('bartlett', {}),
            ('blackman', {}),
            ('cosine', {}),
            ('exponential', {}),
            ('gaussian', {}),
            ('general_cosine', {'a': [0.5, 0.5]}),
            ('general_hamming', {}),
            ('hamming', {}),
            ('hann', {}),
            ('kaiser', {}),
            ('nuttall', {}),
        ]
        for name, kwargs in window_funcs:
            func = getattr(windows, name)
            win = func(M, **kwargs)
            self.assertEqual(win.shape, (M,), f"{name} has wrong shape")

    def test_all_windows_empty_for_zero(self):
        """Test all windows return empty tensor for M=0."""
        window_funcs = [
            ('bartlett', {}),
            ('blackman', {}),
            ('cosine', {}),
            ('exponential', {}),
            ('gaussian', {}),
            ('general_cosine', {'a': [0.5, 0.5]}),
            ('general_hamming', {}),
            ('hamming', {}),
            ('hann', {}),
            ('kaiser', {}),
            ('nuttall', {}),
        ]
        for name, kwargs in window_funcs:
            func = getattr(windows, name)
            win = func(0, **kwargs)
            self.assertEqual(win.shape, (0,), f"{name} should return empty for M=0")

    def test_all_windows_one_for_single(self):
        """Test all windows return [1.0] for M=1."""
        window_funcs = [
            ('bartlett', {}),
            ('blackman', {}),
            ('cosine', {}),
            ('exponential', {}),
            ('gaussian', {}),
            ('general_cosine', {'a': [0.5, 0.5]}),
            ('general_hamming', {}),
            ('hamming', {}),
            ('hann', {}),
            ('kaiser', {}),
            ('nuttall', {}),
        ]
        for name, kwargs in window_funcs:
            func = getattr(windows, name)
            win = func(1, **kwargs)
            self.assertEqual(win.shape, (1,), f"{name} should have shape (1,) for M=1")
            self.assertAlmostEqual(
                float(win.numpy()[0]),
                1.0,
                places=5,
                msg=f"{name} should return [1.0] for M=1"
            )

    def test_all_windows_bounded(self):
        """Test all window values are in [0, 1]."""
        M = 64
        window_funcs = [
            ('bartlett', {}),
            ('blackman', {}),
            ('cosine', {}),
            ('exponential', {}),
            ('gaussian', {}),
            ('general_cosine', {'a': [0.5, 0.5]}),
            ('general_hamming', {}),
            ('hamming', {}),
            ('hann', {}),
            ('kaiser', {}),
            ('nuttall', {}),
        ]
        for name, kwargs in window_funcs:
            func = getattr(windows, name)
            win = func(M, **kwargs)
            arr = win.numpy()
            self.assertTrue(
                np.all(arr >= -1e-6),
                f"{name} has negative values"
            )
            self.assertTrue(
                np.all(arr <= 1.0 + 1e-6),
                f"{name} has values > 1"
            )


@skipIfNoMLX
class TestModuleExports(TestCase):
    """Test that signal.windows exports all functions correctly."""

    def test_windows_submodule_exists(self):
        """Test windows submodule is accessible."""
        self.assertTrue(hasattr(signal, 'windows'))

    def test_all_functions_exported(self):
        """Test all expected functions are exported."""
        expected_functions = [
            'bartlett',
            'blackman',
            'cosine',
            'exponential',
            'gaussian',
            'general_cosine',
            'general_hamming',
            'hamming',
            'hann',
            'kaiser',
            'nuttall',
        ]
        for name in expected_functions:
            self.assertTrue(
                hasattr(windows, name),
                f"{name} not found in windows module"
            )
            self.assertTrue(
                callable(getattr(windows, name)),
                f"{name} is not callable"
            )

    def test_all_list(self):
        """Test __all__ contains expected functions."""
        expected = [
            'bartlett',
            'blackman',
            'cosine',
            'exponential',
            'gaussian',
            'general_cosine',
            'general_hamming',
            'hamming',
            'hann',
            'kaiser',
            'nuttall',
        ]
        for name in expected:
            self.assertIn(name, windows.__all__)


# =============================================================================
# Large Scale Parity Tests
# =============================================================================

@skipIfNoMLX
class TestLargeScaleParity(TestCase):
    """Large scale parity tests for all windows."""

    @requires_torch
    def test_all_windows_parity_comprehensive(self):
        """Comprehensive parity test across all windows and sizes."""
        window_funcs = [
            ('bartlett', {}),
            ('blackman', {}),
            ('cosine', {}),
            ('exponential', {'tau': 1.0}),
            ('gaussian', {'std': 7.0}),
            ('hamming', {}),
            ('hann', {}),
            ('nuttall', {}),
        ]

        sizes = [10, 64, 128, 256, 512, 1024]
        sym_values = [True, False]

        for name, kwargs in window_funcs:
            mlx_func = getattr(windows, name)
            torch_func = getattr(tw, name)
            for M in sizes:
                for sym in sym_values:
                    mlx_win = mlx_func(M, sym=sym, **kwargs)
                    torch_win = torch_func(M, sym=sym, **kwargs)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"Parity failed: {name}, M={M}, sym={sym}"
                    )

    @requires_torch
    def test_kaiser_parity_comprehensive(self):
        """Comprehensive Kaiser window parity test."""
        sizes = [10, 64, 128, 256, 512]
        betas = [0.0, 4.0, 6.0, 8.0, 12.0, 14.0]
        sym_values = [True, False]

        for M in sizes:
            for beta in betas:
                for sym in sym_values:
                    mlx_win = windows.kaiser(M, beta=beta, sym=sym)
                    torch_win = tw.kaiser(M, beta=beta, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,  # Relaxed for I0 approximation
                        atol=1e-6,
                        err_msg=f"Kaiser parity failed: M={M}, beta={beta}, sym={sym}"
                    )

    @requires_torch
    def test_general_cosine_parity_comprehensive(self):
        """Comprehensive general_cosine parity test."""
        coefficient_sets = [
            [0.5, 0.5],  # Hann
            [0.54, 0.46],  # Hamming
            [0.42, 0.5, 0.08],  # Blackman
            [0.3635819, 0.4891775, 0.1365995, 0.0106411],  # Nuttall
            [1.0],  # Rectangular
            [0.25, 0.25, 0.25, 0.25],  # Custom
        ]
        sizes = [10, 64, 128, 256]

        for M in sizes:
            for a in coefficient_sets:
                for sym in [True, False]:
                    mlx_win = windows.general_cosine(M, a=a, sym=sym)
                    torch_win = tw.general_cosine(M, a=a, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"general_cosine parity failed: M={M}, a={a}, sym={sym}"
                    )

    @requires_torch
    def test_general_hamming_parity_comprehensive(self):
        """Comprehensive general_hamming parity test."""
        alphas = [0.0, 0.25, 0.5, 0.54, 0.75, 1.0]
        sizes = [10, 64, 128, 256]

        for M in sizes:
            for alpha in alphas:
                for sym in [True, False]:
                    mlx_win = windows.general_hamming(M, alpha=alpha, sym=sym)
                    torch_win = tw.general_hamming(M, alpha=alpha, sym=sym)
                    np.testing.assert_allclose(
                        mlx_win.numpy(),
                        torch_win.numpy(),
                        rtol=1e-4,
                        atol=1e-6,
                        err_msg=f"general_hamming parity failed: M={M}, alpha={alpha}, sym={sym}"
                    )


if __name__ == '__main__':
    unittest.main()
