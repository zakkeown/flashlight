"""
Layout Tests

Tests for flashlight.layout module - NCHW/NHWC layout management.
"""

import numpy as np
import pytest

import flashlight
from flashlight.layout import (
    Layout,
    convert_layout,
    ensure_layout,
    ensure_nchw,
    ensure_nhwc,
    get_output_layout,
    get_output_layout_1d,
    get_output_layout_3d,
    infer_layout,
    is_nhwc_mode,
    nchw_mode,
    nhwc_mode,
)


class TestLayoutEnum:
    """Tests for Layout enum."""

    def test_layout_values_exist(self):
        """Test that all layout values exist."""
        assert Layout.NCHW is not None
        assert Layout.NHWC is not None
        assert Layout.NCL is not None
        assert Layout.NLC is not None
        assert Layout.NCDHW is not None
        assert Layout.NDHWC is not None
        assert Layout.CONTIGUOUS is not None

    def test_layout_values_distinct(self):
        """Test that layout values are distinct."""
        layouts = [
            Layout.NCHW,
            Layout.NHWC,
            Layout.NCL,
            Layout.NLC,
            Layout.NCDHW,
            Layout.NDHWC,
            Layout.CONTIGUOUS,
        ]
        assert len(set(layouts)) == len(layouts)


class TestNHWCModeContext:
    """Tests for nhwc_mode context manager."""

    def test_default_is_nchw_mode(self):
        """Test that default mode is NCHW (not NHWC)."""
        assert is_nhwc_mode() is False

    def test_nhwc_mode_enables(self):
        """Test that nhwc_mode enables NHWC mode."""
        assert is_nhwc_mode() is False
        with nhwc_mode():
            assert is_nhwc_mode() is True
        assert is_nhwc_mode() is False

    def test_nhwc_mode_nested(self):
        """Test nested nhwc_mode contexts."""
        assert is_nhwc_mode() is False
        with nhwc_mode():
            assert is_nhwc_mode() is True
            with nhwc_mode():
                assert is_nhwc_mode() is True
            assert is_nhwc_mode() is True
        assert is_nhwc_mode() is False

    def test_nhwc_mode_as_decorator(self):
        """Test nhwc_mode as a decorator."""

        @nhwc_mode()
        def check_mode():
            return is_nhwc_mode()

        assert is_nhwc_mode() is False
        assert check_mode() is True
        assert is_nhwc_mode() is False

    def test_nhwc_mode_exception_handling(self):
        """Test that nhwc_mode restores state on exception."""
        assert is_nhwc_mode() is False
        try:
            with nhwc_mode():
                assert is_nhwc_mode() is True
                raise ValueError("test")
        except ValueError:
            pass
        assert is_nhwc_mode() is False


class TestNCHWModeContext:
    """Tests for nchw_mode context manager."""

    def test_nchw_mode_in_nhwc_context(self):
        """Test nchw_mode inside nhwc_mode context."""
        with nhwc_mode():
            assert is_nhwc_mode() is True
            with nchw_mode():
                assert is_nhwc_mode() is False
            assert is_nhwc_mode() is True

    def test_nchw_mode_standalone(self):
        """Test nchw_mode standalone (should be no-op)."""
        assert is_nhwc_mode() is False
        with nchw_mode():
            assert is_nhwc_mode() is False
        assert is_nhwc_mode() is False


class TestInferLayout:
    """Tests for infer_layout function."""

    def test_infer_layout_4d_default(self):
        """Test infer_layout for 4D tensor defaults to NCHW."""
        x = flashlight.randn(1, 3, 32, 32)
        layout = infer_layout(x)
        assert layout == Layout.NCHW

    def test_infer_layout_3d_default(self):
        """Test infer_layout for 3D tensor defaults to NCL."""
        x = flashlight.randn(1, 3, 32)
        layout = infer_layout(x)
        assert layout == Layout.NCL

    def test_infer_layout_5d_default(self):
        """Test infer_layout for 5D tensor defaults to NCDHW."""
        x = flashlight.randn(1, 3, 8, 8, 8)
        layout = infer_layout(x)
        assert layout == Layout.NCDHW

    def test_infer_layout_2d_contiguous(self):
        """Test infer_layout for 2D tensor returns CONTIGUOUS."""
        x = flashlight.randn(10, 20)
        layout = infer_layout(x)
        assert layout == Layout.CONTIGUOUS

    def test_infer_layout_1d_contiguous(self):
        """Test infer_layout for 1D tensor returns CONTIGUOUS."""
        x = flashlight.randn(100)
        layout = infer_layout(x)
        assert layout == Layout.CONTIGUOUS

    def test_infer_layout_explicit(self):
        """Test infer_layout respects explicit _layout attribute."""
        x = flashlight.randn(1, 32, 32, 3)
        x._layout = Layout.NHWC
        layout = infer_layout(x)
        assert layout == Layout.NHWC


class TestConvertLayout:
    """Tests for convert_layout function."""

    def test_convert_nchw_to_nhwc(self):
        """Test converting NCHW to NHWC."""
        # NCHW: [1, 3, 4, 5]
        x = flashlight.randn(1, 3, 4, 5)
        x._layout = Layout.NCHW
        result = convert_layout(x, Layout.NHWC)
        # Should become NHWC: [1, 4, 5, 3]
        assert result.shape == (1, 4, 5, 3)
        assert result._layout == Layout.NHWC

    def test_convert_nhwc_to_nchw(self):
        """Test converting NHWC to NCHW."""
        # NHWC: [1, 4, 5, 3]
        x = flashlight.randn(1, 4, 5, 3)
        x._layout = Layout.NHWC
        result = convert_layout(x, Layout.NCHW)
        # Should become NCHW: [1, 3, 4, 5]
        assert result.shape == (1, 3, 4, 5)
        assert result._layout == Layout.NCHW

    def test_convert_ncl_to_nlc(self):
        """Test converting NCL to NLC."""
        # NCL: [2, 8, 16]
        x = flashlight.randn(2, 8, 16)
        x._layout = Layout.NCL
        result = convert_layout(x, Layout.NLC)
        # Should become NLC: [2, 16, 8]
        assert result.shape == (2, 16, 8)
        assert result._layout == Layout.NLC

    def test_convert_nlc_to_ncl(self):
        """Test converting NLC to NCL."""
        # NLC: [2, 16, 8]
        x = flashlight.randn(2, 16, 8)
        x._layout = Layout.NLC
        result = convert_layout(x, Layout.NCL)
        # Should become NCL: [2, 8, 16]
        assert result.shape == (2, 8, 16)
        assert result._layout == Layout.NCL

    def test_convert_ncdhw_to_ndhwc(self):
        """Test converting NCDHW to NDHWC."""
        # NCDHW: [1, 3, 4, 5, 6]
        x = flashlight.randn(1, 3, 4, 5, 6)
        x._layout = Layout.NCDHW
        result = convert_layout(x, Layout.NDHWC)
        # Should become NDHWC: [1, 4, 5, 6, 3]
        assert result.shape == (1, 4, 5, 6, 3)
        assert result._layout == Layout.NDHWC

    def test_convert_same_layout_returns_same(self):
        """Test that converting to same layout returns same tensor."""
        x = flashlight.randn(1, 3, 4, 5)
        x._layout = Layout.NCHW
        result = convert_layout(x, Layout.NCHW)
        # Should return same tensor
        assert result is x

    def test_convert_contiguous_returns_same(self):
        """Test that contiguous tensors pass through unchanged."""
        x = flashlight.randn(10, 20)  # 2D - contiguous
        result = convert_layout(x, Layout.NCHW)
        assert result is x

    def test_convert_preserves_data(self):
        """Test that conversion preserves data values."""
        # Create NCHW tensor with known values
        x = flashlight.arange(24).reshape(1, 2, 3, 4).float()
        x._layout = Layout.NCHW
        result = convert_layout(x, Layout.NHWC)
        # Convert back
        back = convert_layout(result, Layout.NCHW)
        np.testing.assert_allclose(
            np.array(back._mlx_array),
            np.array(x._mlx_array),
            rtol=1e-6,
        )


class TestEnsureLayout:
    """Tests for ensure_layout and helpers."""

    def test_ensure_layout_converts_if_needed(self):
        """Test ensure_layout converts when necessary."""
        x = flashlight.randn(1, 3, 4, 5)
        x._layout = Layout.NCHW
        result = ensure_layout(x, Layout.NHWC)
        assert result.shape == (1, 4, 5, 3)
        assert result._layout == Layout.NHWC

    def test_ensure_layout_no_op_if_correct(self):
        """Test ensure_layout is no-op if already correct layout."""
        x = flashlight.randn(1, 3, 4, 5)
        x._layout = Layout.NCHW
        result = ensure_layout(x, Layout.NCHW)
        assert result is x

    def test_ensure_nhwc(self):
        """Test ensure_nhwc helper."""
        x = flashlight.randn(1, 3, 4, 5)
        x._layout = Layout.NCHW
        result = ensure_nhwc(x)
        assert result.shape == (1, 4, 5, 3)
        assert result._layout == Layout.NHWC

    def test_ensure_nchw(self):
        """Test ensure_nchw helper."""
        x = flashlight.randn(1, 4, 5, 3)
        x._layout = Layout.NHWC
        result = ensure_nchw(x)
        assert result.shape == (1, 3, 4, 5)
        assert result._layout == Layout.NCHW


class TestGetOutputLayout:
    """Tests for get_output_layout functions."""

    def test_get_output_layout_default(self):
        """Test get_output_layout returns NCHW by default."""
        assert get_output_layout() == Layout.NCHW

    def test_get_output_layout_in_nhwc_mode(self):
        """Test get_output_layout returns NHWC in nhwc_mode."""
        with nhwc_mode():
            assert get_output_layout() == Layout.NHWC

    def test_get_output_layout_1d_default(self):
        """Test get_output_layout_1d returns NCL by default."""
        assert get_output_layout_1d() == Layout.NCL

    def test_get_output_layout_1d_in_nhwc_mode(self):
        """Test get_output_layout_1d returns NLC in nhwc_mode."""
        with nhwc_mode():
            assert get_output_layout_1d() == Layout.NLC

    def test_get_output_layout_3d_default(self):
        """Test get_output_layout_3d returns NCDHW by default."""
        assert get_output_layout_3d() == Layout.NCDHW

    def test_get_output_layout_3d_in_nhwc_mode(self):
        """Test get_output_layout_3d returns NDHWC in nhwc_mode."""
        with nhwc_mode():
            assert get_output_layout_3d() == Layout.NDHWC


class TestLayoutWithConv:
    """Tests for layout handling with convolution operations."""

    def test_conv2d_default_mode(self):
        """Test Conv2d works in default NCHW mode."""
        conv = flashlight.nn.Conv2d(3, 16, 3, padding=1)
        x = flashlight.randn(1, 3, 32, 32)  # NCHW input
        output = conv(x)
        assert output.shape == (1, 16, 32, 32)  # NCHW output

    def test_conv2d_nhwc_mode(self):
        """Test Conv2d works in NHWC mode."""
        conv = flashlight.nn.Conv2d(3, 16, 3, padding=1)
        x = flashlight.randn(1, 3, 32, 32)  # NCHW input

        with nhwc_mode():
            output = conv(x)
            # In NHWC mode, the internal layout may be NHWC [N, H, W, C]
            # The output has the same total elements but may be in different layout
            total_elements = 1
            for dim in output.shape:
                total_elements *= dim
            assert total_elements == 1 * 16 * 32 * 32
            # Shape is either NCHW or NHWC depending on implementation
            assert output.shape in [(1, 16, 32, 32), (1, 32, 32, 16)]


class TestThreadSafety:
    """Tests for thread safety of layout mode."""

    def test_layout_mode_thread_local(self):
        """Test that layout mode is thread-local."""
        import threading

        results = {"main": None, "thread": None}

        def thread_func():
            # Should start in NCHW mode
            results["thread_start"] = is_nhwc_mode()
            with nhwc_mode():
                results["thread_inside"] = is_nhwc_mode()
            results["thread_end"] = is_nhwc_mode()

        # Main thread in NCHW mode
        assert is_nhwc_mode() is False

        # Start thread
        t = threading.Thread(target=thread_func)

        with nhwc_mode():
            # Main thread in NHWC mode
            assert is_nhwc_mode() is True
            t.start()
            t.join()

        # Check thread had independent state
        assert results["thread_start"] is False
        assert results["thread_inside"] is True
        assert results["thread_end"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
