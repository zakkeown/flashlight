"""
Tests for conv_transpose3d (3D transposed convolution).

Tests the functional API and nn.ConvTranspose3d layer against PyTorch for parity.
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.nn.functional as F_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import flashlight
import flashlight.nn as nn
from flashlight.ops.conv3d import conv_transpose3d


class TestConvTranspose3dBasic:
    """Basic functionality tests for conv_transpose3d."""

    def test_basic_forward_pass(self):
        """Test that forward pass runs without error."""
        x = flashlight.randn(2, 3, 4, 8, 8)  # NCDHW
        weight = flashlight.randn(3, 6, 3, 3, 3)  # C_in, C_out/groups, kD, kH, kW

        output = conv_transpose3d(x, weight)

        # Check output shape
        assert output.ndim == 5
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 6  # C_out

    def test_output_shape_stride_1(self):
        """Test output shape with stride=1."""
        N, C_in, D, H, W = 2, 3, 4, 8, 8
        C_out, kD, kH, kW = 6, 3, 3, 3

        x = flashlight.randn(N, C_in, D, H, W)
        weight = flashlight.randn(C_in, C_out, kD, kH, kW)

        output = conv_transpose3d(x, weight, stride=1, padding=0)

        # Output size formula for transposed conv:
        # output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        # With stride=1, padding=0, dilation=1, output_padding=0:
        # = (input_size - 1) * 1 - 0 + 1 * (kernel_size - 1) + 0 + 1
        # = input_size - 1 + kernel_size - 1 + 1
        # = input_size + kernel_size - 1
        expected_D = D + kD - 1  # 4 + 3 - 1 = 6
        expected_H = H + kH - 1  # 8 + 3 - 1 = 10
        expected_W = W + kW - 1  # 8 + 3 - 1 = 10

        assert output.shape == (N, C_out, expected_D, expected_H, expected_W)

    def test_output_shape_stride_2(self):
        """Test output shape with stride=2."""
        N, C_in, D, H, W = 2, 3, 4, 8, 8
        C_out, kD, kH, kW = 6, 3, 3, 3

        x = flashlight.randn(N, C_in, D, H, W)
        weight = flashlight.randn(C_in, C_out, kD, kH, kW)

        output = conv_transpose3d(x, weight, stride=2, padding=1)

        # With stride=2, padding=1, dilation=1, output_padding=0:
        # = (input_size - 1) * 2 - 2 * 1 + 1 * (kernel_size - 1) + 0 + 1
        # = 2 * input_size - 2 - 2 + kernel_size - 1 + 1
        # = 2 * input_size + kernel_size - 4
        expected_D = 2 * D + kD - 4  # 8 + 3 - 4 = 7
        expected_H = 2 * H + kH - 4  # 16 + 3 - 4 = 15
        expected_W = 2 * W + kW - 4  # 16 + 3 - 4 = 15

        assert output.shape == (N, C_out, expected_D, expected_H, expected_W)

    def test_with_bias(self):
        """Test forward pass with bias."""
        x = flashlight.randn(2, 3, 4, 8, 8)
        weight = flashlight.randn(3, 6, 3, 3, 3)
        bias = flashlight.randn(6)

        output = conv_transpose3d(x, weight, bias=bias)

        assert output.ndim == 5
        assert output.shape[1] == 6  # C_out

    def test_output_padding(self):
        """Test that output_padding increases output size."""
        x = flashlight.randn(2, 3, 4, 8, 8)
        weight = flashlight.randn(3, 6, 3, 3, 3)

        output_no_opad = conv_transpose3d(x, weight, stride=2, padding=1, output_padding=0)
        output_with_opad = conv_transpose3d(x, weight, stride=2, padding=1, output_padding=1)

        # Output padding adds to each spatial dimension
        assert output_with_opad.shape[2] == output_no_opad.shape[2] + 1
        assert output_with_opad.shape[3] == output_no_opad.shape[3] + 1
        assert output_with_opad.shape[4] == output_no_opad.shape[4] + 1


class TestConvTranspose3dLayer:
    """Tests for nn.ConvTranspose3d layer."""

    def test_layer_instantiation(self):
        """Test layer instantiation."""
        layer = nn.ConvTranspose3d(3, 6, kernel_size=3)

        assert layer.in_channels == 3
        assert layer.out_channels == 6
        assert layer.kernel_size == (3, 3, 3)
        assert layer.stride == (1, 1, 1)
        assert layer.padding == (0, 0, 0)
        assert layer.weight.shape == (3, 6, 3, 3, 3)
        assert layer.bias.shape == (6,)

    def test_layer_no_bias(self):
        """Test layer without bias."""
        layer = nn.ConvTranspose3d(3, 6, kernel_size=3, bias=False)

        assert layer.bias is None

    def test_layer_forward(self):
        """Test layer forward pass."""
        layer = nn.ConvTranspose3d(3, 6, kernel_size=3, stride=2, padding=1)
        x = flashlight.randn(2, 3, 4, 8, 8)

        output = layer(x)

        assert output.ndim == 5
        assert output.shape[0] == 2
        assert output.shape[1] == 6

    def test_layer_tuple_params(self):
        """Test layer with tuple parameters."""
        layer = nn.ConvTranspose3d(
            3, 6,
            kernel_size=(3, 4, 5),
            stride=(1, 2, 2),
            padding=(1, 1, 2)
        )

        assert layer.kernel_size == (3, 4, 5)
        assert layer.stride == (1, 2, 2)
        assert layer.padding == (1, 1, 2)

    def test_layer_output_padding(self):
        """Test layer with output_padding."""
        layer = nn.ConvTranspose3d(3, 6, kernel_size=3, stride=2, padding=1, output_padding=1)
        x = flashlight.randn(2, 3, 4, 8, 8)

        output = layer(x)

        # Verify output padding was applied
        layer_no_opad = nn.ConvTranspose3d(3, 6, kernel_size=3, stride=2, padding=1, output_padding=0)
        layer_no_opad.weight = layer.weight
        layer_no_opad.bias = layer.bias
        output_no_opad = layer_no_opad(x)

        assert output.shape[2] == output_no_opad.shape[2] + 1
        assert output.shape[3] == output_no_opad.shape[3] + 1
        assert output.shape[4] == output_no_opad.shape[4] + 1

    def test_extra_repr(self):
        """Test extra_repr for string representation."""
        layer = nn.ConvTranspose3d(3, 6, kernel_size=3, stride=2, padding=1)
        repr_str = layer.extra_repr()

        assert 'in_channels=3' in repr_str
        assert 'out_channels=6' in repr_str
        assert 'kernel_size=(3, 3, 3)' in repr_str


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestConvTranspose3dParity:
    """Parity tests comparing flashlight with PyTorch."""

    def test_basic_parity(self):
        """Test basic conv_transpose3d parity with PyTorch."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        # PyTorch weight: [C_in, C_out/groups, kD, kH, kW]
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        out_torch = F_torch.conv_transpose3d(x_torch, weight_torch)
        out_mlx = conv_transpose3d(x_mlx, weight_mlx)

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Basic conv_transpose3d parity failed"
        )

    def test_stride_parity(self):
        """Test conv_transpose3d with stride parity."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        for stride in [1, 2, (1, 2, 2)]:
            out_torch = F_torch.conv_transpose3d(x_torch, weight_torch, stride=stride)
            out_mlx = conv_transpose3d(x_mlx, weight_mlx, stride=stride)

            np.testing.assert_allclose(
                out_torch.numpy(),
                out_mlx.numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"Stride={stride} parity failed"
            )

    def test_padding_parity(self):
        """Test conv_transpose3d with padding parity."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        for padding in [0, 1, (1, 1, 2)]:
            out_torch = F_torch.conv_transpose3d(x_torch, weight_torch, padding=padding)
            out_mlx = conv_transpose3d(x_mlx, weight_mlx, padding=padding)

            np.testing.assert_allclose(
                out_torch.numpy(),
                out_mlx.numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"Padding={padding} parity failed"
            )

    def test_output_padding_parity(self):
        """Test conv_transpose3d with output_padding parity."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        # output_padding must be less than stride
        out_torch = F_torch.conv_transpose3d(
            x_torch, weight_torch, stride=2, padding=1, output_padding=1
        )
        out_mlx = conv_transpose3d(
            x_mlx, weight_mlx, stride=2, padding=1, output_padding=1
        )

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Output padding parity failed"
        )

    def test_bias_parity(self):
        """Test conv_transpose3d with bias parity."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)
        bias_np = np.random.randn(6).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)
        bias_torch = torch.tensor(bias_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)
        bias_mlx = flashlight.tensor(bias_np)

        out_torch = F_torch.conv_transpose3d(x_torch, weight_torch, bias_torch)
        out_mlx = conv_transpose3d(x_mlx, weight_mlx, bias_mlx)

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Bias parity failed"
        )

    def test_dilation_parity(self):
        """Test conv_transpose3d with dilation parity."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 6, 10, 10).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        for dilation in [1, 2]:
            out_torch = F_torch.conv_transpose3d(x_torch, weight_torch, dilation=dilation)
            out_mlx = conv_transpose3d(x_mlx, weight_mlx, dilation=dilation)

            np.testing.assert_allclose(
                out_torch.numpy(),
                out_mlx.numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"Dilation={dilation} parity failed"
            )

    def test_layer_parity(self):
        """Test nn.ConvTranspose3d layer parity with PyTorch."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)
        bias_np = np.random.randn(6).astype(np.float32)

        # PyTorch layer
        layer_torch = torch.nn.ConvTranspose3d(3, 6, kernel_size=3, stride=2, padding=1)
        with torch.no_grad():
            layer_torch.weight.copy_(torch.tensor(weight_np))
            layer_torch.bias.copy_(torch.tensor(bias_np))

        # MLX layer
        layer_mlx = nn.ConvTranspose3d(3, 6, kernel_size=3, stride=2, padding=1)
        layer_mlx.weight._mlx_array = flashlight.tensor(weight_np)._mlx_array
        layer_mlx.bias._mlx_array = flashlight.tensor(bias_np)._mlx_array

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = layer_torch(x_torch)
        out_mlx = layer_mlx(x_mlx)

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Layer parity failed"
        )

    def test_asymmetric_kernel_parity(self):
        """Test conv_transpose3d with asymmetric kernel."""
        np.random.seed(42)

        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        # Asymmetric kernel: (2, 3, 4)
        weight_np = np.random.randn(3, 6, 2, 3, 4).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        out_torch = F_torch.conv_transpose3d(x_torch, weight_torch, stride=(1, 2, 2), padding=(0, 1, 1))
        out_mlx = conv_transpose3d(x_mlx, weight_mlx, stride=(1, 2, 2), padding=(0, 1, 1))

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Asymmetric kernel parity failed"
        )

    def test_batch_size_1_parity(self):
        """Test conv_transpose3d with batch size 1."""
        np.random.seed(42)

        x_np = np.random.randn(1, 3, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(3, 6, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        out_torch = F_torch.conv_transpose3d(x_torch, weight_torch, stride=2, padding=1)
        out_mlx = conv_transpose3d(x_mlx, weight_mlx, stride=2, padding=1)

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Batch size 1 parity failed"
        )

    def test_single_channel_parity(self):
        """Test conv_transpose3d with single input/output channel."""
        np.random.seed(42)

        x_np = np.random.randn(2, 1, 4, 8, 8).astype(np.float32)
        weight_np = np.random.randn(1, 1, 3, 3, 3).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)

        out_torch = F_torch.conv_transpose3d(x_torch, weight_torch)
        out_mlx = conv_transpose3d(x_mlx, weight_mlx)

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Single channel parity failed"
        )

    def test_combined_params_parity(self):
        """Test conv_transpose3d with multiple parameters combined."""
        np.random.seed(42)

        x_np = np.random.randn(2, 4, 6, 10, 10).astype(np.float32)
        weight_np = np.random.randn(4, 8, 3, 4, 4).astype(np.float32)
        bias_np = np.random.randn(8).astype(np.float32)

        x_torch = torch.tensor(x_np)
        weight_torch = torch.tensor(weight_np)
        bias_torch = torch.tensor(bias_np)

        x_mlx = flashlight.tensor(x_np)
        weight_mlx = flashlight.tensor(weight_np)
        bias_mlx = flashlight.tensor(bias_np)

        out_torch = F_torch.conv_transpose3d(
            x_torch, weight_torch, bias_torch,
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 1, 1),
            dilation=1
        )
        out_mlx = conv_transpose3d(
            x_mlx, weight_mlx, bias_mlx,
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 1, 1),
            dilation=1
        )

        np.testing.assert_allclose(
            out_torch.numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Combined params parity failed"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
