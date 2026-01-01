"""
Tests for flashlight.utils.mobile_optimizer module.

Tests mobile optimization utilities.
"""

import unittest

import flashlight
import flashlight.nn as nn
from flashlight.utils.mobile_optimizer import (
    optimize_for_mobile,
    generate_mobile_module_lints,
    MobileOptimizerType,
    LintCode,
)


class TestMobileOptimizerType(unittest.TestCase):
    """Test MobileOptimizerType enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        self.assertEqual(MobileOptimizerType.CONV_BN_FUSION.value, "conv_bn_fusion")
        self.assertEqual(MobileOptimizerType.REMOVE_DROPOUT.value, "remove_dropout")
        self.assertEqual(MobileOptimizerType.FUSE_ADD_RELU.value, "fuse_add_relu")

    def test_all_types_exist(self):
        """Test all expected optimization types exist."""
        types = [
            MobileOptimizerType.CONV_BN_FUSION,
            MobileOptimizerType.INSERT_FOLD_PREPACK_OPS,
            MobileOptimizerType.REMOVE_DROPOUT,
            MobileOptimizerType.FUSE_ADD_RELU,
            MobileOptimizerType.FUSE_HARDSWISH,
            MobileOptimizerType.FUSE_CLAMP_MIN_MAX,
            MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
        ]
        self.assertEqual(len(types), 7)


class TestLintCode(unittest.TestCase):
    """Test LintCode enum."""

    def test_lint_codes(self):
        """Test lint codes have expected values."""
        self.assertEqual(LintCode.BUNDLED_INPUT.value, 1)
        self.assertEqual(LintCode.REQUIRES_GRAD.value, 2)
        self.assertEqual(LintCode.DROPOUT.value, 3)
        self.assertEqual(LintCode.BATCHNORM.value, 4)


class TestOptimizeForMobile(unittest.TestCase):
    """Test optimize_for_mobile function."""

    def test_basic_optimization(self):
        """Test basic model optimization."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        optimized = optimize_for_mobile(model)

        # Should return a model
        self.assertIsNotNone(optimized)

    def test_removes_dropout(self):
        """Test that dropout layers are removed."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
            nn.Linear(20, 10),
        )

        optimized = optimize_for_mobile(model)

        # Check that dropout is replaced with identity
        has_dropout = False
        for child in optimized.children():
            if isinstance(child, nn.Dropout):
                has_dropout = True
                break

        self.assertFalse(has_dropout)

    def test_preserves_structure(self):
        """Test that model structure is preserved."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        optimized = optimize_for_mobile(model)

        # Test forward pass still works
        x = flashlight.randn(2, 10)
        out = optimized(x)

        self.assertEqual(out.shape, (2, 5))

    def test_blocklist_dropout(self):
        """Test that blocklist prevents dropout removal."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
            nn.Linear(20, 10),
        )

        optimized = optimize_for_mobile(
            model,
            optimization_blocklist={MobileOptimizerType.REMOVE_DROPOUT},
        )

        # Dropout should still be present
        has_dropout = False
        for child in optimized.children():
            if isinstance(child, nn.Dropout):
                has_dropout = True
                break

        self.assertTrue(has_dropout)

    def test_sets_eval_mode(self):
        """Test that optimization sets model to eval mode."""
        model = nn.Linear(10, 5)
        model.train()

        optimized = optimize_for_mobile(model)

        self.assertFalse(optimized.training)


class TestConvBnFusion(unittest.TestCase):
    """Test Conv-BN fusion optimization."""

    def test_conv_bn_fusion_output_equivalence(self):
        """Test that fused model produces same output as unfused."""
        import numpy as np
        import mlx.core as mx

        # Create model with Conv + BN
        class ConvBnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        model = ConvBnModel()
        model.eval()

        # Create test input
        x = flashlight.randn(1, 3, 8, 8)

        # Get original output
        original_out = model(x)
        mx.eval(original_out._mlx_array)
        original_values = np.array(original_out._mlx_array.tolist())

        # Optimize with Conv-BN fusion
        optimized = optimize_for_mobile(model)

        # Verify BN is replaced with Identity
        self.assertIsInstance(optimized.bn, nn.Identity)

        # Get optimized output
        optimized_out = optimized(x)
        mx.eval(optimized_out._mlx_array)
        optimized_values = np.array(optimized_out._mlx_array.tolist())

        # Verify outputs are numerically equivalent
        np.testing.assert_allclose(
            original_values,
            optimized_values,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Conv-BN fusion produced different output"
        )

    def test_conv_bn_fusion_removes_bn(self):
        """Test that BN layers are replaced with Identity after fusion."""
        class ConvBnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.bn = nn.BatchNorm2d(16)

            def forward(self, x):
                return self.bn(self.conv(x))

        model = ConvBnModel()
        model.eval()

        optimized = optimize_for_mobile(model)

        # BN should be replaced with Identity
        self.assertIsInstance(optimized.bn, nn.Identity)

    def test_conv_bn_fusion_blocklist(self):
        """Test that fusion can be blocked."""
        class ConvBnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.bn = nn.BatchNorm2d(16)

            def forward(self, x):
                return self.bn(self.conv(x))

        model = ConvBnModel()
        model.eval()

        optimized = optimize_for_mobile(
            model,
            optimization_blocklist={MobileOptimizerType.CONV_BN_FUSION}
        )

        # BN should still be present
        self.assertIsInstance(optimized.bn, nn.BatchNorm2d)


class TestGenerateMobileModuleLints(unittest.TestCase):
    """Test generate_mobile_module_lints function."""

    def test_empty_lints_for_clean_model(self):
        """Test no lints for a clean inference model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        model.eval()

        # Disable requires_grad
        for param in model.parameters():
            param.requires_grad = False

        lints = generate_mobile_module_lints(model)

        # Should have no critical lints
        dropout_lints = [l for l in lints if l["code"] == LintCode.DROPOUT]
        self.assertEqual(len(dropout_lints), 0)

    def test_lint_dropout(self):
        """Test that dropout is flagged."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
            nn.Linear(20, 10),
        )

        lints = generate_mobile_module_lints(model)

        dropout_lints = [l for l in lints if l["code"] == LintCode.DROPOUT]
        self.assertGreater(len(dropout_lints), 0)

    def test_lint_requires_grad(self):
        """Test that requires_grad is flagged."""
        model = nn.Linear(10, 5)
        # Ensure requires_grad is True
        for param in model.parameters():
            param.requires_grad = True

        lints = generate_mobile_module_lints(model)

        grad_lints = [l for l in lints if l["code"] == LintCode.REQUIRES_GRAD]
        self.assertGreater(len(grad_lints), 0)

    def test_lint_batchnorm_training(self):
        """Test that training-mode batchnorm is flagged."""
        model = nn.BatchNorm2d(16)
        model.train()

        lints = generate_mobile_module_lints(model)

        bn_lints = [l for l in lints if l["code"] == LintCode.BATCHNORM]
        self.assertGreater(len(bn_lints), 0)

    def test_lint_batchnorm_eval_ok(self):
        """Test that eval-mode batchnorm is not flagged."""
        model = nn.BatchNorm2d(16)
        model.eval()

        lints = generate_mobile_module_lints(model)

        bn_lints = [l for l in lints if l["code"] == LintCode.BATCHNORM]
        self.assertEqual(len(bn_lints), 0)

    def test_lint_message_contains_name(self):
        """Test that lint messages contain module names."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
        )

        lints = generate_mobile_module_lints(model)

        # Find dropout lint
        dropout_lints = [l for l in lints if l["code"] == LintCode.DROPOUT]
        self.assertGreater(len(dropout_lints), 0)
        self.assertIn("name", dropout_lints[0])
        self.assertIn("message", dropout_lints[0])


class TestNestedModuleOptimization(unittest.TestCase):
    """Test optimization with nested modules."""

    def test_nested_dropout_removal(self):
        """Test dropout removal in nested modules."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                )
                self.output = nn.Linear(20, 5)

            def forward(self, x):
                x = self.block(x)
                return self.output(x)

        model = NestedModel()
        optimized = optimize_for_mobile(model)

        # Check no dropout in nested structure
        def has_dropout(module):
            if isinstance(module, nn.Dropout):
                return True
            for child in module.children():
                if has_dropout(child):
                    return True
            return False

        self.assertFalse(has_dropout(optimized))


if __name__ == "__main__":
    unittest.main()
