"""
Tests for validators.
"""

import pytest

from parity_check.validators.api_presence import (
    APIPresenceValidator,
    PresenceValidationResult,
)
from parity_check.validators.signature_match import (
    SignatureValidationResult,
    SignatureValidator,
)


class TestPresenceValidationResult:
    """Tests for PresenceValidationResult dataclass."""

    def test_empty_result(self):
        result = PresenceValidationResult()
        assert result.total_checked == 0
        assert result.coverage_percentage == 100.0
        assert result.is_complete is True

    def test_full_coverage(self):
        result = PresenceValidationResult(
            present=[{"module": "torch", "api": "tensor"}],
            missing=[],
            excluded=[],
        )
        assert result.total_checked == 1
        assert result.coverage_percentage == 100.0
        assert result.is_complete is True

    def test_partial_coverage(self):
        result = PresenceValidationResult(
            present=[{"module": "torch", "api": "tensor"}],
            missing=[{"module": "torch", "api": "zeros"}],
            excluded=[],
        )
        assert result.total_checked == 2
        assert result.coverage_percentage == 50.0
        assert result.is_complete is False

    def test_excluded_not_counted(self):
        result = PresenceValidationResult(
            present=[{"module": "torch", "api": "tensor"}],
            missing=[],
            excluded=[{"module": "torch", "api": "cuda"}],
        )
        # Excluded APIs don't count toward total_checked
        assert result.total_checked == 1
        assert result.coverage_percentage == 100.0


class TestAPIPresenceValidator:
    """Tests for APIPresenceValidator."""

    def test_all_present(self):
        pytorch_apis = {
            "torch": {
                "tensor": {"type": "function"},
                "zeros": {"type": "function"},
            }
        }
        mlx_apis = {
            "torch": {
                "tensor": {"type": "function"},
                "zeros": {"type": "function"},
            }
        }
        exclusions = {}

        validator = APIPresenceValidator(pytorch_apis, mlx_apis, exclusions)
        result = validator.validate()

        assert len(result.present) == 2
        assert len(result.missing) == 0
        assert result.is_complete is True

    def test_missing_api(self):
        pytorch_apis = {
            "torch": {
                "tensor": {"type": "function"},
                "zeros": {"type": "function"},
            }
        }
        mlx_apis = {
            "torch": {
                "tensor": {"type": "function"},
                # zeros is missing
            }
        }
        exclusions = {}

        validator = APIPresenceValidator(pytorch_apis, mlx_apis, exclusions)
        result = validator.validate()

        assert len(result.present) == 1
        assert len(result.missing) == 1
        assert result.missing[0]["api"] == "zeros"
        assert result.is_complete is False

    def test_excluded_api(self):
        pytorch_apis = {
            "torch": {
                "tensor": {"type": "function"},
                "cuda": {"type": "module"},
            }
        }
        mlx_apis = {
            "torch": {
                "tensor": {"type": "function"},
                # cuda is excluded (intentionally not implemented)
            }
        }
        exclusions = {
            "torch": {
                "cuda": "CUDA-specific module",
            }
        }

        validator = APIPresenceValidator(pytorch_apis, mlx_apis, exclusions)
        result = validator.validate()

        assert len(result.present) == 1
        assert len(result.missing) == 0
        assert len(result.excluded) == 1
        assert result.excluded[0]["api"] == "cuda"
        assert result.excluded[0]["reason"] == "CUDA-specific module"
        assert result.is_complete is True

    def test_multiple_modules(self):
        pytorch_apis = {
            "torch": {
                "tensor": {"type": "function"},
            },
            "torch.nn": {
                "Linear": {"type": "class"},
                "Conv2d": {"type": "class"},
            },
        }
        mlx_apis = {
            "torch": {
                "tensor": {"type": "function"},
            },
            "torch.nn": {
                "Linear": {"type": "class"},
                # Conv2d is missing
            },
        }
        exclusions = {}

        validator = APIPresenceValidator(pytorch_apis, mlx_apis, exclusions)
        result = validator.validate()

        assert len(result.present) == 2
        assert len(result.missing) == 1
        assert result.missing[0]["module"] == "torch.nn"
        assert result.missing[0]["api"] == "Conv2d"


class TestSignatureValidationResult:
    """Tests for SignatureValidationResult dataclass."""

    def test_empty_result(self):
        result = SignatureValidationResult()
        assert result.total_compared == 0
        assert result.match_percentage == 100.0
        assert result.is_complete is True

    def test_all_match(self):
        result = SignatureValidationResult(
            matches=[{"module": "torch", "api": "tensor"}],
            mismatches=[],
        )
        assert result.total_compared == 1
        assert result.match_percentage == 100.0
        assert result.is_complete is True

    def test_with_mismatches(self):
        result = SignatureValidationResult(
            matches=[{"module": "torch", "api": "tensor"}],
            mismatches=[{"module": "torch", "api": "zeros"}],
        )
        assert result.total_compared == 2
        assert result.match_percentage == 50.0
        assert result.is_complete is False


class TestSignatureValidator:
    """Tests for SignatureValidator."""

    def test_matching_signatures(self):
        pytorch_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [
                            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD", "default": None},
                        ],
                        "extractable": True,
                    },
                },
            }
        }
        mlx_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [
                            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD", "default": None},
                        ],
                        "extractable": True,
                    },
                },
            }
        }

        validator = SignatureValidator(pytorch_apis, mlx_apis)
        result = validator.validate()

        assert len(result.matches) == 1
        assert len(result.mismatches) == 0

    def test_mismatched_signatures(self):
        pytorch_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [
                            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD", "default": None},
                            {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
                        ],
                        "extractable": True,
                    },
                },
            }
        }
        mlx_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [
                            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD", "default": None},
                            # dtype is missing
                        ],
                        "extractable": True,
                    },
                },
            }
        }

        validator = SignatureValidator(pytorch_apis, mlx_apis)
        result = validator.validate()

        assert len(result.matches) == 0
        assert len(result.mismatches) == 1
        assert result.mismatches[0]["api"] == "zeros"

    def test_skips_missing_apis(self):
        pytorch_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [],
                        "extractable": True,
                    },
                },
            }
        }
        mlx_apis = {
            "torch": {
                # zeros is not present
            }
        }

        validator = SignatureValidator(pytorch_apis, mlx_apis)
        result = validator.validate()

        # Should skip APIs that aren't present
        assert len(result.matches) == 0
        assert len(result.mismatches) == 0

    def test_skips_non_extractable_signatures(self):
        pytorch_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [],
                        "extractable": False,
                    },
                },
            }
        }
        mlx_apis = {
            "torch": {
                "zeros": {
                    "type": "function",
                    "signature": {
                        "parameters": [],
                        "extractable": True,
                    },
                },
            }
        }

        # Disable source fallback to test the skip behavior for non-extractable signatures
        # (otherwise the source parser finds the real torch.zeros signature)
        validator = SignatureValidator(pytorch_apis, mlx_apis, use_source_fallback=False)
        result = validator.validate()

        assert len(result.skipped) == 1
        assert result.skipped[0]["reason"] == "Signature not extractable (builtin)"
