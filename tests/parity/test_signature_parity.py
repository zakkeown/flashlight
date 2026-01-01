"""
Signature parity tests comparing flashlight signatures against reference PyTorch source.

These tests use AST-based parsing of the reference PyTorch source to extract
signatures, bypassing the limitation that C++ builtins can't be introspected.
"""

from pathlib import Path

import pytest

from parity_check.introspection.signature import (
    compare_signatures,
    extract_signature,
    get_parameter_summary,
)
from parity_check.introspection.source_parser import (
    PYTORCH_REFERENCE_ROOT,
    extract_module_signatures,
    extract_signatures_from_file,
    get_source_signature,
)

# Skip tests if reference source not available
pytestmark = pytest.mark.skipif(
    not PYTORCH_REFERENCE_ROOT.exists(), reason="PyTorch reference source not available"
)


class TestSourceParserBasics:
    """Test basic source parser functionality."""

    def test_pytorch_reference_root_exists(self):
        """Verify reference source directory exists."""
        assert PYTORCH_REFERENCE_ROOT.exists()
        assert PYTORCH_REFERENCE_ROOT.is_dir()

    def test_extract_signatures_from_torch_init(self):
        """Test extracting signatures from torch/__init__.py."""
        init_path = PYTORCH_REFERENCE_ROOT / "torch" / "__init__.py"
        if init_path.exists():
            sigs = extract_signatures_from_file(init_path)
            # Should extract at least some signatures
            assert isinstance(sigs, dict)

    def test_extract_optimizer_signatures(self):
        """Test extracting signatures from optimizer files."""
        adam_path = PYTORCH_REFERENCE_ROOT / "torch" / "optim" / "adam.py"
        if adam_path.exists():
            sigs = extract_signatures_from_file(adam_path)
            assert "Adam" in sigs
            sig = sigs["Adam"]
            assert sig["extractable"] is True
            assert sig["source"] == "ast"
            # Adam should have params, lr, betas, eps, weight_decay, etc.
            param_names = [p["name"] for p in sig["parameters"]]
            assert "params" in param_names
            assert "lr" in param_names

    def test_get_source_signature_adam(self):
        """Test getting Adam optimizer signature from source."""
        sig = get_source_signature("torch.optim", "Adam")
        if sig is not None:
            assert sig["extractable"] is True
            param_names = [p["name"] for p in sig["parameters"]]
            assert "params" in param_names
            assert "lr" in param_names
            assert "betas" in param_names
            assert "eps" in param_names
            assert "weight_decay" in param_names


class TestOptimizerSignatureParity:
    """Test optimizer signature parity against reference source."""

    @pytest.fixture
    def mlx_optim(self):
        """Import flashlight.optim."""
        import flashlight.optim as optim

        return optim

    def test_sgd_signature_parity(self, mlx_optim):
        """Test SGD signature matches reference source."""
        ref_sig = get_source_signature("torch.optim", "SGD")
        if ref_sig is None:
            pytest.skip("SGD signature not extractable from reference source")

        mlx_sig = extract_signature(mlx_optim.SGD)
        assert mlx_sig is not None
        assert mlx_sig["extractable"]

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"SGD signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )

    def test_adam_signature_parity(self, mlx_optim):
        """Test Adam signature matches reference source."""
        ref_sig = get_source_signature("torch.optim", "Adam")
        if ref_sig is None:
            pytest.skip("Adam signature not extractable from reference source")

        mlx_sig = extract_signature(mlx_optim.Adam)
        assert mlx_sig is not None
        assert mlx_sig["extractable"]

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"Adam signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )

    def test_adamw_signature_parity(self, mlx_optim):
        """Test AdamW signature matches reference source."""
        ref_sig = get_source_signature("torch.optim", "AdamW")
        if ref_sig is None:
            pytest.skip("AdamW signature not extractable from reference source")

        mlx_sig = extract_signature(mlx_optim.AdamW)
        assert mlx_sig is not None
        assert mlx_sig["extractable"]

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"AdamW signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )

    def test_adagrad_signature_parity(self, mlx_optim):
        """Test Adagrad signature matches reference source."""
        ref_sig = get_source_signature("torch.optim", "Adagrad")
        if ref_sig is None:
            pytest.skip("Adagrad signature not extractable from reference source")

        mlx_sig = extract_signature(mlx_optim.Adagrad)
        assert mlx_sig is not None
        assert mlx_sig["extractable"]

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"Adagrad signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )

    def test_rmsprop_signature_parity(self, mlx_optim):
        """Test RMSprop signature matches reference source."""
        ref_sig = get_source_signature("torch.optim", "RMSprop")
        if ref_sig is None:
            pytest.skip("RMSprop signature not extractable from reference source")

        mlx_sig = extract_signature(mlx_optim.RMSprop)
        assert mlx_sig is not None
        assert mlx_sig["extractable"]

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"RMSprop signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )


class TestNNModuleSignatureParity:
    """Test nn.Module class signature parity against reference source."""

    @pytest.fixture
    def mlx_nn(self):
        """Import flashlight.nn."""
        import flashlight.nn as nn

        return nn

    def _test_nn_signature(self, mlx_nn, class_name, source_file):
        """Helper to test a single nn module signature."""
        source_path = PYTORCH_REFERENCE_ROOT / source_file
        if not source_path.exists():
            pytest.skip(f"Source file not found: {source_file}")

        sigs = extract_signatures_from_file(source_path)
        if class_name not in sigs:
            pytest.skip(f"{class_name} not found in {source_file}")

        ref_sig = sigs[class_name]

        if not hasattr(mlx_nn, class_name):
            pytest.skip(f"{class_name} not in flashlight.nn")

        mlx_class = getattr(mlx_nn, class_name)
        mlx_sig = extract_signature(mlx_class)

        if mlx_sig is None or not mlx_sig.get("extractable"):
            pytest.skip(f"{class_name} signature not extractable from flashlight")

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"{class_name} signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )

    def test_linear_signature_parity(self, mlx_nn):
        """Test Linear layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "Linear", "torch/nn/modules/linear.py")

    def test_conv2d_signature_parity(self, mlx_nn):
        """Test Conv2d layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "Conv2d", "torch/nn/modules/conv.py")

    def test_batchnorm2d_signature_parity(self, mlx_nn):
        """Test BatchNorm2d layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "BatchNorm2d", "torch/nn/modules/batchnorm.py")

    def test_layernorm_signature_parity(self, mlx_nn):
        """Test LayerNorm layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "LayerNorm", "torch/nn/modules/normalization.py")

    def test_dropout_signature_parity(self, mlx_nn):
        """Test Dropout layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "Dropout", "torch/nn/modules/dropout.py")

    def test_embedding_signature_parity(self, mlx_nn):
        """Test Embedding layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "Embedding", "torch/nn/modules/sparse.py")

    def test_lstm_signature_parity(self, mlx_nn):
        """Test LSTM layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "LSTM", "torch/nn/modules/rnn.py")

    def test_gru_signature_parity(self, mlx_nn):
        """Test GRU layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "GRU", "torch/nn/modules/rnn.py")

    def test_transformer_signature_parity(self, mlx_nn):
        """Test Transformer layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "Transformer", "torch/nn/modules/transformer.py")

    def test_multihead_attention_signature_parity(self, mlx_nn):
        """Test MultiheadAttention layer signature matches reference."""
        self._test_nn_signature(mlx_nn, "MultiheadAttention", "torch/nn/modules/activation.py")


class TestDataModuleSignatureParity:
    """Test data module signature parity against reference source."""

    @pytest.fixture
    def mlx_data(self):
        """Import flashlight data modules."""
        import flashlight.data as data

        return data

    def test_dataloader_signature_parity(self, mlx_data):
        """Test DataLoader signature matches reference."""
        source_path = PYTORCH_REFERENCE_ROOT / "torch" / "utils" / "data" / "dataloader.py"
        if not source_path.exists():
            pytest.skip("DataLoader source file not found")

        sigs = extract_signatures_from_file(source_path)
        if "DataLoader" not in sigs:
            pytest.skip("DataLoader not found in source")

        ref_sig = sigs["DataLoader"]
        mlx_sig = extract_signature(mlx_data.DataLoader)

        if mlx_sig is None or not mlx_sig.get("extractable"):
            pytest.skip("DataLoader signature not extractable from flashlight")

        comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
        assert comparison["matches"], (
            f"DataLoader signature mismatch:\n"
            f"  Reference: {get_parameter_summary(ref_sig)}\n"
            f"  flashlight: {get_parameter_summary(mlx_sig)}\n"
            f"  Differences: {comparison['differences']}"
        )


class TestModuleSignatureExtraction:
    """Test module-wide signature extraction."""

    def test_extract_torch_optim_signatures(self):
        """Test extracting all torch.optim signatures."""
        sigs = extract_module_signatures("torch.optim")
        # Should find multiple optimizers
        assert len(sigs) > 0
        # Check for expected optimizers
        expected = ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad"]
        found = [name for name in expected if name in sigs]
        assert len(found) > 0, f"Expected to find at least one of {expected}"

    def test_extract_torch_nn_modules_signatures(self):
        """Test extracting signatures from torch.nn.modules."""
        base_dir = PYTORCH_REFERENCE_ROOT / "torch" / "nn" / "modules"
        if not base_dir.exists():
            pytest.skip("torch.nn.modules not found in reference source")

        # Check a few specific files
        test_files = [
            ("linear.py", ["Linear", "Bilinear"]),
            ("conv.py", ["Conv1d", "Conv2d", "Conv3d"]),
            ("activation.py", ["ReLU", "GELU", "Sigmoid", "Tanh"]),
        ]

        for filename, expected_classes in test_files:
            filepath = base_dir / filename
            if filepath.exists():
                sigs = extract_signatures_from_file(filepath)
                for cls in expected_classes:
                    if cls in sigs:
                        sig = sigs[cls]
                        assert sig["extractable"] is True
                        assert sig["source"] == "ast"


class TestSignatureValidatorWithSource:
    """Test SignatureValidator with source-based fallback."""

    def test_validator_uses_source_fallback(self):
        """Test that validator uses source fallback for non-extractable signatures."""
        from parity_check.validators.signature_match import SignatureValidator

        # Create mock APIs where PyTorch signature is not extractable
        pytorch_apis = {
            "torch.optim": {
                "Adam": {
                    "type": "class",
                    "signature": {
                        "parameters": [],
                        "extractable": False,  # Simulate C++ builtin
                    },
                },
            }
        }

        # Get actual flashlight Adam signature
        import flashlight.optim as optim

        mlx_sig = extract_signature(optim.Adam)

        mlx_apis = {
            "torch.optim": {
                "Adam": {
                    "type": "class",
                    "signature": mlx_sig,
                },
            }
        }

        validator = SignatureValidator(pytorch_apis, mlx_apis, use_source_fallback=True)
        result = validator.validate()

        # With source fallback, should be able to compare
        # Either matches (if our impl is correct) or in mismatches (if not)
        # But should NOT be skipped
        skipped_adam = [s for s in result.skipped if s.get("api") == "Adam"]
        if len(skipped_adam) == 0:
            # Successfully used source fallback
            total = len(result.matches) + len(result.mismatches)
            assert total >= 1, "Adam should be compared using source fallback"


class TestFullSignatureAudit:
    """Run a full signature audit comparing all flashlight signatures to reference."""

    @pytest.mark.slow
    def test_full_optimizer_audit(self):
        """Audit all optimizer signatures."""
        import flashlight.optim as optim

        optimizers = ["SGD", "Adam", "AdamW", "Adagrad", "RMSprop", "Adadelta", "Adamax"]
        results = {}

        for opt_name in optimizers:
            if not hasattr(optim, opt_name):
                results[opt_name] = {"status": "missing", "reason": "Not in flashlight"}
                continue

            ref_sig = get_source_signature("torch.optim", opt_name)
            if ref_sig is None:
                results[opt_name] = {"status": "skipped", "reason": "Not in reference source"}
                continue

            mlx_sig = extract_signature(getattr(optim, opt_name))
            if mlx_sig is None or not mlx_sig.get("extractable"):
                results[opt_name] = {
                    "status": "skipped",
                    "reason": "Not extractable from flashlight",
                }
                continue

            comparison = compare_signatures(ref_sig, mlx_sig, strict_defaults=True)
            if comparison["matches"]:
                results[opt_name] = {"status": "pass"}
            else:
                results[opt_name] = {
                    "status": "fail",
                    "differences": comparison["differences"],
                    "ref": get_parameter_summary(ref_sig),
                    "mlx": get_parameter_summary(mlx_sig),
                }

        # Report results
        passed = sum(1 for r in results.values() if r["status"] == "pass")
        failed = sum(1 for r in results.values() if r["status"] == "fail")

        if failed > 0:
            fail_details = [
                f"{name}: {r['differences']}"
                for name, r in results.items()
                if r["status"] == "fail"
            ]
            pytest.fail(
                f"Optimizer signature audit: {passed} passed, {failed} failed\n"
                f"Failures:\n" + "\n".join(fail_details)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
