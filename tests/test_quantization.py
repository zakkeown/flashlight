"""
Tests for the Quantization Module.

Tests cover:
- QuantizedTensor creation and dequantization
- Observers (MinMax, MovingAverage, PerChannel)
- FakeQuantize for QAT
- prepare/convert utilities
"""

import pytest

import flashlight
from flashlight.quantization import (
    FakeQuantize,
    FakeQuantizeBase,
    LearnedFakeQuantize,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    NoopObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QuantStub,
    DeQuantStub,
    QuantizedTensor,
    convert,
    default_qat_qconfig,
    default_qconfig,
    dequantize,
    prepare,
    prepare_qat,
    quantize_per_tensor,
)


class TestQuantizedTensor:
    """Tests for QuantizedTensor class."""

    def test_quantize_basic(self):
        """Test basic quantization."""
        x = flashlight.randn(4, 64) * 5
        qt = quantize_per_tensor(x, bits=8)

        assert isinstance(qt, QuantizedTensor)
        assert qt.shape == (4, 64)
        assert qt.bits == 8

    def test_dequantize_roundtrip(self):
        """Test quantize -> dequantize roundtrip."""
        x = flashlight.randn(4, 64) * 5
        qt = quantize_per_tensor(x, bits=8)
        dq = qt.dequantize()

        assert dq.shape == x.shape
        # Max error should be bounded by quantization step size
        max_error = (x - dq).abs().max().item()
        assert max_error < 1.0  # Reasonable error bound

    def test_quantize_bits_2(self):
        """Test 2-bit quantization."""
        x = flashlight.randn(4, 64)
        qt = quantize_per_tensor(x, bits=2)

        assert qt.bits == 2
        dq = qt.dequantize()
        assert dq.shape == x.shape

    def test_quantize_bits_4(self):
        """Test 4-bit quantization."""
        x = flashlight.randn(4, 64)
        qt = quantize_per_tensor(x, bits=4)

        assert qt.bits == 4
        dq = qt.dequantize()
        assert dq.shape == x.shape

    def test_quantize_custom_group_size(self):
        """Test quantization with custom group size."""
        x = flashlight.randn(4, 128)
        qt = quantize_per_tensor(x, bits=8, group_size=128)

        assert qt.group_size == 128
        dq = qt.dequantize()
        assert dq.shape == x.shape


class TestObservers:
    """Tests for Observer classes."""

    def test_minmax_observer_basic(self):
        """Test MinMaxObserver tracks min/max."""
        obs = MinMaxObserver(bits=8)

        x = flashlight.tensor([[-5.0, 0.0, 5.0, 10.0]])
        obs(x)

        scale, zp = obs.calculate_qparams()
        assert scale.item() > 0
        assert isinstance(zp.item(), (int, float))

    def test_minmax_observer_accumulates(self):
        """Test MinMaxObserver accumulates over multiple calls."""
        obs = MinMaxObserver(bits=8)

        obs(flashlight.tensor([[0.0, 1.0]]))
        obs(flashlight.tensor([[-10.0, 5.0]]))
        obs(flashlight.tensor([[0.0, 20.0]]))

        # Should have tracked min=-10, max=20
        scale, zp = obs.calculate_qparams()
        assert scale.item() > 0

    def test_minmax_observer_reset(self):
        """Test MinMaxObserver can be reset."""
        obs = MinMaxObserver(bits=8)

        obs(flashlight.tensor([[0.0, 100.0]]))
        obs.reset_min_max_vals()

        with pytest.raises(RuntimeError):
            obs.calculate_qparams()  # No data observed

    def test_moving_average_observer(self):
        """Test MovingAverageMinMaxObserver."""
        obs = MovingAverageMinMaxObserver(bits=8, averaging_constant=0.5)

        obs(flashlight.tensor([[0.0, 10.0]]))
        obs(flashlight.tensor([[0.0, 20.0]]))

        scale, zp = obs.calculate_qparams()
        assert scale.item() > 0

    def test_per_channel_observer(self):
        """Test PerChannelMinMaxObserver."""
        obs = PerChannelMinMaxObserver(bits=8, ch_axis=0)

        x = flashlight.tensor([[0.0, 5.0], [-10.0, 10.0], [0.0, 1.0]])
        obs(x)

        scale, zp = obs.calculate_qparams()
        # Should have per-channel scales
        assert scale.shape[0] == 3

    def test_noop_observer(self):
        """Test NoopObserver does nothing."""
        obs = NoopObserver()

        x = flashlight.randn(4, 64)
        result = obs(x)

        # Should return input unchanged
        assert (result == x).all().item()


class TestFakeQuantize:
    """Tests for FakeQuantize module."""

    def test_fake_quantize_basic(self):
        """Test basic FakeQuantize forward pass."""
        fq = FakeQuantize(MinMaxObserver(bits=8))
        x = flashlight.randn(4, 64)

        y = fq(x)
        assert y.shape == x.shape

    def test_fake_quantize_stores_qparams(self):
        """Test FakeQuantize stores scale and zero_point after forward."""
        fq = FakeQuantize(MinMaxObserver(bits=8))
        x = flashlight.randn(4, 64)

        y = fq(x)

        assert fq.scale is not None
        assert fq.zero_point is not None

    def test_fake_quantize_disable(self):
        """Test disabling fake quantization."""
        fq = FakeQuantize(MinMaxObserver(bits=8))
        fq.disable_fake_quant()

        x = flashlight.randn(4, 64)
        y = fq(x)

        # With fake quant disabled, output should equal input
        assert ((y - x).abs() < 1e-6).all().item()

    def test_fake_quantize_disable_observer(self):
        """Test disabling observer."""
        fq = FakeQuantize(MinMaxObserver(bits=8))

        x = flashlight.randn(4, 64)
        fq(x)  # First forward with observer enabled

        fq.disable_observer()
        scale_before = fq.scale.tolist()  # Save scale values

        # Forward with different data
        y = flashlight.randn(4, 64) * 100
        fq(y)

        # Scale should not change since observer is disabled
        scale_after = fq.scale.tolist()
        assert scale_before == scale_after

    def test_fake_quantize_with_args(self):
        """Test FakeQuantize.with_args factory method."""
        fq = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            bits=8,
            averaging_constant=0.1,
        )

        x = flashlight.randn(4, 64)
        y = fq(x)

        assert y.shape == x.shape

    def test_fake_quantize_training_mode(self):
        """Test FakeQuantize respects training mode."""
        fq = FakeQuantize(MinMaxObserver(bits=8))
        x = flashlight.randn(4, 64)

        # In training mode
        fq.train()
        y_train = fq(x)

        # In eval mode
        fq.eval()
        y_eval = fq(x)

        # Both should produce valid output
        assert y_train.shape == x.shape
        assert y_eval.shape == x.shape


class TestLearnedFakeQuantize:
    """Tests for LearnedFakeQuantize module."""

    def test_learned_fake_quantize_basic(self):
        """Test basic LearnedFakeQuantize forward pass."""
        lfq = LearnedFakeQuantize()
        x = flashlight.randn(4, 64)

        y = lfq(x)
        assert y.shape == x.shape

    def test_learned_fake_quantize_has_parameters(self):
        """Test LearnedFakeQuantize has learnable scale and zero_point."""
        lfq = LearnedFakeQuantize()

        # scale and zero_point should be Parameters
        params = list(lfq.parameters())
        assert len(params) == 2

    def test_learned_fake_quantize_custom_init(self):
        """Test LearnedFakeQuantize with custom initial values."""
        lfq = LearnedFakeQuantize(init_scale=0.5, init_zero_point=10.0)

        assert lfq.scale.item() == pytest.approx(0.5)
        assert lfq.zero_point.item() == pytest.approx(10.0)


class TestStubs:
    """Tests for QuantStub and DeQuantStub."""

    def test_quant_stub_passthrough(self):
        """Test QuantStub is a passthrough before conversion."""
        qs = QuantStub()
        x = flashlight.randn(4, 64)

        y = qs(x)
        assert ((y - x).abs() < 1e-6).all().item()

    def test_dequant_stub_passthrough(self):
        """Test DeQuantStub is a passthrough."""
        dqs = DeQuantStub()
        x = flashlight.randn(4, 64)

        y = dqs(x)
        assert ((y - x).abs() < 1e-6).all().item()


class TestQConfig:
    """Tests for QConfig dataclass."""

    def test_default_qconfig(self):
        """Test default_qconfig creates valid observers."""
        qconfig = default_qconfig

        act_obs = qconfig.activation()
        weight_obs = qconfig.weight()

        assert isinstance(act_obs, MinMaxObserver)
        # Weight observer is PerChannelMinMaxObserver by default
        assert isinstance(weight_obs, PerChannelMinMaxObserver)

    def test_default_qat_qconfig(self):
        """Test default_qat_qconfig creates valid fake quantizers."""
        qconfig = default_qat_qconfig

        act_fq = qconfig.activation()
        weight_fq = qconfig.weight()

        assert isinstance(act_fq, FakeQuantize)
        assert isinstance(weight_fq, FakeQuantize)

    def test_custom_qconfig(self):
        """Test creating custom QConfig."""
        qconfig = QConfig(
            activation=lambda: MinMaxObserver(bits=4),
            weight=lambda: PerChannelMinMaxObserver(bits=4),
        )

        act_obs = qconfig.activation()
        weight_obs = qconfig.weight()

        assert isinstance(act_obs, MinMaxObserver)
        assert isinstance(weight_obs, PerChannelMinMaxObserver)


class TestPrepareConvert:
    """Tests for prepare, convert, and prepare_qat utilities."""

    def test_prepare_basic(self):
        """Test prepare inserts observers."""

        class SimpleModel(flashlight.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.linear = flashlight.nn.Linear(64, 32)
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                return self.dequant(x)

        model = SimpleModel()
        prepared = prepare(model, qconfig_mapping={"": default_qconfig})

        # Model should still work
        x = flashlight.randn(4, 64)
        y = prepared(x)
        assert y.shape == (4, 32)

    def test_prepare_qat_basic(self):
        """Test prepare_qat inserts fake quantizers."""

        class SimpleModel(flashlight.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.linear = flashlight.nn.Linear(64, 32)
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                return self.dequant(x)

        model = SimpleModel()
        prepared = prepare_qat(model, qconfig_mapping={"": default_qat_qconfig})

        # Model should still work in training mode
        prepared.train()
        x = flashlight.randn(4, 64)
        y = prepared(x)
        assert y.shape == (4, 32)


class TestIntegration:
    """Integration tests for quantization workflow."""

    def test_ptq_workflow(self):
        """Test post-training quantization workflow."""

        class SimpleModel(flashlight.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.linear = flashlight.nn.Linear(64, 64)
                self.relu = flashlight.nn.ReLU()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                x = self.relu(x)
                return self.dequant(x)

        # Create and prepare model
        model = SimpleModel()
        model = prepare(model, qconfig_mapping={"": default_qconfig})

        # Calibrate with sample data
        model.eval()
        for _ in range(10):
            x = flashlight.randn(4, 64)
            model(x)

        # Convert to quantized model
        quantized_model = convert(model)

        # Run inference
        x = flashlight.randn(4, 64)
        y = quantized_model(x)
        assert y.shape == (4, 64)

    def test_qat_workflow(self):
        """Test quantization-aware training workflow."""

        class SimpleModel(flashlight.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.linear = flashlight.nn.Linear(64, 64)
                self.relu = flashlight.nn.ReLU()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                x = self.relu(x)
                return self.dequant(x)

        # Create and prepare model for QAT
        model = SimpleModel()
        model = prepare_qat(model, qconfig_mapping={"": default_qat_qconfig})

        # Simulate training
        model.train()
        for _ in range(5):
            x = flashlight.randn(4, 64)
            y = model(x)
            # In real training, would compute loss and backprop

        # Convert to quantized model
        model.eval()
        quantized_model = convert(model)

        # Run inference
        x = flashlight.randn(4, 64)
        y = quantized_model(x)
        assert y.shape == (4, 64)
