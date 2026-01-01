"""
Tests for API introspection utilities.
"""

import pytest

from parity_check.introspection.pytorch_api import (
    classify_api_type,
    get_public_names,
)
from parity_check.introspection.signature import (
    compare_signatures,
    extract_signature,
    get_parameter_summary,
)


class TestClassifyApiType:
    """Tests for classify_api_type function."""

    def test_classify_function(self):
        def sample_func():
            pass

        assert classify_api_type(sample_func) == "function"

    def test_classify_class(self):
        class SampleClass:
            pass

        assert classify_api_type(SampleClass) == "class"

    def test_classify_module(self):
        import os

        assert classify_api_type(os) == "module"

    def test_classify_constant(self):
        assert classify_api_type(42) == "constant"
        assert classify_api_type("string") == "constant"

    def test_classify_callable(self):
        # Callable objects should be classified as functions
        class CallableClass:
            def __call__(self):
                pass

        obj = CallableClass()
        assert classify_api_type(obj) == "function"


class TestExtractSignature:
    """Tests for extract_signature function."""

    def test_extract_simple_function(self):
        def simple(x, y):
            pass

        sig = extract_signature(simple)
        assert sig is not None
        assert sig["extractable"] is True
        assert len(sig["parameters"]) == 2
        assert sig["parameters"][0]["name"] == "x"
        assert sig["parameters"][1]["name"] == "y"

    def test_extract_function_with_defaults(self):
        def with_defaults(x, y=10, z="hello"):
            pass

        sig = extract_signature(with_defaults)
        assert sig is not None
        assert sig["parameters"][0]["default"] is None
        assert sig["parameters"][1]["default"] == "10"
        assert sig["parameters"][2]["default"] == "'hello'"

    def test_extract_function_with_args_kwargs(self):
        def with_var(*args, **kwargs):
            pass

        sig = extract_signature(with_var)
        assert sig is not None
        assert any(p["kind"] == "VAR_POSITIONAL" for p in sig["parameters"])
        assert any(p["kind"] == "VAR_KEYWORD" for p in sig["parameters"])

    def test_extract_class_init(self):
        class SampleClass:
            def __init__(self, a, b=5):
                pass

        sig = extract_signature(SampleClass)
        assert sig is not None
        assert sig["extractable"] is True
        # 'self' should be excluded
        assert len(sig["parameters"]) == 2
        assert sig["parameters"][0]["name"] == "a"
        assert sig["parameters"][1]["name"] == "b"

    def test_extract_builtin_not_extractable(self):
        # Built-in functions may not have extractable signatures
        sig = extract_signature(len)
        assert sig is not None
        # Should gracefully handle non-extractable signatures
        assert "extractable" in sig


class TestCompareSignatures:
    """Tests for compare_signatures function."""

    def test_identical_signatures_match(self):
        def func1(x, y, z=10):
            pass

        def func2(x, y, z=10):
            pass

        sig1 = extract_signature(func1)
        sig2 = extract_signature(func2)

        result = compare_signatures(sig1, sig2)
        assert result["matches"] is True
        assert len(result["differences"]) == 0

    def test_missing_parameter_detected(self):
        def func1(x, y, z):
            pass

        def func2(x, y):
            pass

        sig1 = extract_signature(func1)
        sig2 = extract_signature(func2)

        result = compare_signatures(sig1, sig2)
        assert result["matches"] is False
        assert any("Missing parameter" in d for d in result["differences"])

    def test_extra_required_parameter_detected(self):
        def func1(x, y):
            pass

        def func2(x, y, z):  # z is required (no default)
            pass

        sig1 = extract_signature(func1)
        sig2 = extract_signature(func2)

        result = compare_signatures(sig1, sig2)
        assert result["matches"] is False
        assert any("Extra required parameter" in d for d in result["differences"])

    def test_extra_optional_parameter_allowed(self):
        def func1(x, y):
            pass

        def func2(x, y, z=10):  # z has a default, so it's optional
            pass

        sig1 = extract_signature(func1)
        sig2 = extract_signature(func2)

        result = compare_signatures(sig1, sig2)
        # Extra optional parameters should be allowed
        assert result["matches"] is True

    def test_different_defaults_detected(self):
        def func1(x, y=10):
            pass

        def func2(x, y=20):
            pass

        sig1 = extract_signature(func1)
        sig2 = extract_signature(func2)

        result = compare_signatures(sig1, sig2, strict_defaults=True)
        assert result["matches"] is False
        assert any(
            "missing default" in d.lower() or "default" in d.lower() for d in result["differences"]
        )

    def test_none_signatures_handled(self):
        result = compare_signatures(None, None)
        assert result["matches"] is True

        sig = extract_signature(lambda x: x)
        result = compare_signatures(sig, None)
        assert result["matches"] is True


class TestGetParameterSummary:
    """Tests for get_parameter_summary function."""

    def test_simple_function(self):
        def simple(x, y):
            pass

        sig = extract_signature(simple)
        summary = get_parameter_summary(sig)
        assert summary == "(x, y)"

    def test_function_with_defaults(self):
        def with_defaults(x, y=10):
            pass

        sig = extract_signature(with_defaults)
        summary = get_parameter_summary(sig)
        assert "x" in summary
        assert "y=10" in summary

    def test_function_with_args_kwargs(self):
        def with_var(x, *args, **kwargs):
            pass

        sig = extract_signature(with_var)
        summary = get_parameter_summary(sig)
        assert "*args" in summary
        assert "**kwargs" in summary

    def test_none_signature(self):
        summary = get_parameter_summary(None)
        assert summary == "(...)"


class TestGetPublicNames:
    """Tests for get_public_names function."""

    def test_module_with_all(self):
        import os.path

        names = get_public_names(os.path)
        # os.path has __all__ defined
        assert isinstance(names, list)
        assert len(names) > 0

    def test_filters_private_names(self):
        class MockModule:
            public_name = 1
            _private_name = 2
            __dunder__ = 3

        names = get_public_names(MockModule)
        assert "public_name" in names
        assert "_private_name" not in names
        # Dunders are also filtered
        assert "__dunder__" not in names


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available",
)
class TestPyTorchIntrospection:
    """Tests for actual PyTorch API introspection."""

    def test_enumerate_torch_module(self):
        from parity_check.introspection.pytorch_api import enumerate_pytorch_api

        apis = enumerate_pytorch_api(["torch"])
        assert "torch" in apis
        assert len(apis["torch"]) > 0

        # Check some known APIs exist
        torch_apis = apis["torch"]
        assert "tensor" in torch_apis or "Tensor" in torch_apis

    def test_enumerate_torch_nn(self):
        from parity_check.introspection.pytorch_api import enumerate_pytorch_api

        apis = enumerate_pytorch_api(["torch.nn"])
        assert "torch.nn" in apis

        nn_apis = apis["torch.nn"]
        # Check some known nn modules
        assert "Linear" in nn_apis or "Module" in nn_apis
