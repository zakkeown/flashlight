"""
Automatic Mixed Precision (AMP) Tests

Tests for flashlight.amp module - autocast and GradScaler.
"""

import numpy as np
import pytest

import flashlight


class TestAutocastAvailability:
    """Tests for is_autocast_available function."""

    def test_autocast_available_cuda(self):
        """Test autocast is available for CUDA device type."""
        assert flashlight.amp.is_autocast_available("cuda") is True

    def test_autocast_available_cpu(self):
        """Test autocast is available for CPU device type."""
        assert flashlight.amp.is_autocast_available("cpu") is True

    def test_autocast_available_mps(self):
        """Test autocast is available for MPS device type."""
        assert flashlight.amp.is_autocast_available("mps") is True


class TestAutocast:
    """Tests for autocast context manager."""

    def test_autocast_context_manager(self):
        """Test autocast as context manager."""
        x = flashlight.randn(10, 10)
        with flashlight.amp.autocast():
            y = x + x
        assert y.shape == (10, 10)

    def test_autocast_enabled_false(self):
        """Test autocast with enabled=False."""
        x = flashlight.randn(10, 10)
        with flashlight.amp.autocast(enabled=False):
            y = x + x
        assert y.shape == (10, 10)

    def test_autocast_as_decorator(self):
        """Test autocast as decorator."""

        @flashlight.amp.autocast()
        def compute(x):
            return x * 2

        x = flashlight.randn(5, 5)
        result = compute(x)
        assert result.shape == (5, 5)

    def test_autocast_nested(self):
        """Test nested autocast contexts."""
        x = flashlight.randn(5, 5)
        with flashlight.amp.autocast():
            y = x + 1
            with flashlight.amp.autocast(enabled=False):
                z = y * 2
            w = z - 1
        assert w.shape == (5, 5)

    def test_autocast_device_types(self):
        """Test autocast with different device types."""
        x = flashlight.randn(5, 5)
        for device_type in ["cuda", "cpu", "mps"]:
            with flashlight.amp.autocast(device_type=device_type):
                y = x + x
            assert y.shape == (5, 5)


class TestGradScaler:
    """Tests for GradScaler class."""

    def test_gradscaler_creation(self):
        """Test GradScaler creation with defaults."""
        scaler = flashlight.amp.GradScaler()
        assert scaler.is_enabled() is True
        assert scaler.get_scale() == 65536.0
        assert scaler.get_growth_factor() == 2.0
        assert scaler.get_backoff_factor() == 0.5
        assert scaler.get_growth_interval() == 2000

    def test_gradscaler_custom_params(self):
        """Test GradScaler with custom parameters."""
        scaler = flashlight.amp.GradScaler(
            init_scale=1024.0,
            growth_factor=4.0,
            backoff_factor=0.25,
            growth_interval=1000,
        )
        assert scaler.get_scale() == 1024.0
        assert scaler.get_growth_factor() == 4.0
        assert scaler.get_backoff_factor() == 0.25
        assert scaler.get_growth_interval() == 1000

    def test_gradscaler_disabled(self):
        """Test GradScaler when disabled."""
        scaler = flashlight.amp.GradScaler(enabled=False)
        assert scaler.is_enabled() is False

    def test_gradscaler_scale_tensor(self):
        """Test GradScaler.scale() with single tensor."""
        scaler = flashlight.amp.GradScaler(init_scale=2.0)
        loss = flashlight.tensor([1.0, 2.0, 3.0])
        scaled_loss = scaler.scale(loss)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(np.array(scaled_loss._mlx_array), expected, rtol=1e-5)

    def test_gradscaler_scale_list(self):
        """Test GradScaler.scale() with list of tensors."""
        scaler = flashlight.amp.GradScaler(init_scale=3.0)
        losses = [flashlight.tensor([1.0]), flashlight.tensor([2.0])]
        scaled_losses = scaler.scale(losses)
        assert len(scaled_losses) == 2
        np.testing.assert_allclose(np.array(scaled_losses[0]._mlx_array), [3.0], rtol=1e-5)
        np.testing.assert_allclose(np.array(scaled_losses[1]._mlx_array), [6.0], rtol=1e-5)

    def test_gradscaler_scale_disabled(self):
        """Test GradScaler.scale() when disabled returns unchanged."""
        scaler = flashlight.amp.GradScaler(enabled=False)
        loss = flashlight.tensor([1.0, 2.0, 3.0])
        scaled_loss = scaler.scale(loss)
        # When disabled, should return original tensor
        np.testing.assert_allclose(np.array(scaled_loss._mlx_array), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_gradscaler_update_manual(self):
        """Test GradScaler.update() with manual scale."""
        scaler = flashlight.amp.GradScaler(init_scale=100.0)
        scaler.update(new_scale=200.0)
        assert scaler.get_scale() == 200.0

    def test_gradscaler_update_growth(self):
        """Test GradScaler.update() with automatic growth."""
        scaler = flashlight.amp.GradScaler(
            init_scale=100.0,
            growth_factor=2.0,
            growth_interval=2,
        )
        # Simulate successful steps
        scaler.update()  # growth_tracker = 1
        assert scaler.get_scale() == 100.0
        scaler.update()  # growth_tracker = 2, triggers growth
        assert scaler.get_scale() == 200.0

    def test_gradscaler_state_dict(self):
        """Test GradScaler state_dict and load_state_dict."""
        scaler = flashlight.amp.GradScaler(
            init_scale=512.0,
            growth_factor=3.0,
            backoff_factor=0.3,
            growth_interval=500,
        )
        state = scaler.state_dict()
        assert state["scale"] == 512.0
        assert state["growth_factor"] == 3.0
        assert state["backoff_factor"] == 0.3
        assert state["growth_interval"] == 500

        # Create new scaler and load state
        new_scaler = flashlight.amp.GradScaler()
        new_scaler.load_state_dict(state)
        assert new_scaler.get_scale() == 512.0
        assert new_scaler.get_growth_factor() == 3.0
        assert new_scaler.get_backoff_factor() == 0.3
        assert new_scaler.get_growth_interval() == 500


class TestCustomFwdBwd:
    """Tests for custom_fwd and custom_bwd decorators."""

    def test_custom_fwd_passthrough(self):
        """Test custom_fwd is a passthrough decorator."""

        @flashlight.amp.custom_fwd
        def forward(x):
            return x * 2

        x = flashlight.tensor([1.0, 2.0])
        result = forward(x)
        np.testing.assert_allclose(np.array(result._mlx_array), [2.0, 4.0], rtol=1e-5)

    def test_custom_fwd_with_args(self):
        """Test custom_fwd with arguments."""

        @flashlight.amp.custom_fwd(device_type="cuda", cast_inputs=None)
        def forward(x):
            return x + 1

        x = flashlight.tensor([1.0, 2.0])
        result = forward(x)
        np.testing.assert_allclose(np.array(result._mlx_array), [2.0, 3.0], rtol=1e-5)

    def test_custom_bwd_passthrough(self):
        """Test custom_bwd is a passthrough decorator."""

        @flashlight.amp.custom_bwd
        def backward(grad):
            return grad * 2

        grad = flashlight.tensor([1.0, 2.0])
        result = backward(grad)
        np.testing.assert_allclose(np.array(result._mlx_array), [2.0, 4.0], rtol=1e-5)


class TestSubmoduleCompatibility:
    """Tests for submodule compatibility classes."""

    def test_autocast_mode_submodule(self):
        """Test autocast_mode submodule."""
        assert flashlight.amp.autocast_mode.autocast is flashlight.amp.autocast
        assert flashlight.amp.autocast_mode.is_autocast_available("cuda") is True

    def test_grad_scaler_submodule(self):
        """Test grad_scaler submodule."""
        assert flashlight.amp.grad_scaler.GradScaler is flashlight.amp.GradScaler


class TestAMPIntegration:
    """Integration tests for AMP workflow."""

    def test_basic_amp_training_loop(self):
        """Test basic AMP training loop pattern."""
        scaler = flashlight.amp.GradScaler(init_scale=1024.0)

        # Simulate a training step
        x = flashlight.randn(4, 4)

        with flashlight.amp.autocast():
            # Forward pass
            y = x @ x.t()
            loss = y.sum()

        # Scale loss
        scaled_loss = scaler.scale(loss)
        assert scaled_loss.shape == ()

        # Update scaler
        scaler.update()

    def test_amp_with_model(self):
        """Test AMP with a simple model."""
        model = flashlight.nn.Linear(10, 5)
        x = flashlight.randn(2, 10)

        with flashlight.amp.autocast():
            output = model(x)

        assert output.shape == (2, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
