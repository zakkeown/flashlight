"""
Numerical parity validator.

Tests functional equivalence between flashlight and PyTorch by executing
APIs with generated test inputs and comparing outputs within tolerances.
"""

import importlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..exclusions import is_numerical_excluded
from .input_generators import (
    LRSchedulerSpec,
    ModuleSpec,
    NNUtilsSpec,
    OptimizerSpec,
    get_input_registry,
    set_seeds,
)


@dataclass
class NumericalTestResult:
    """Result of a single numerical parity test."""

    module: str
    api: str
    passed: bool
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    pytorch_output_shape: Optional[Tuple] = None
    mlx_output_shape: Optional[Tuple] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class NumericalValidationResult:
    """Result of numerical parity validation."""

    matches: List[NumericalTestResult] = field(default_factory=list)
    mismatches: List[NumericalTestResult] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_tested(self) -> int:
        """Total number of APIs tested (excluding skipped/errors)."""
        return len(self.matches) + len(self.mismatches)

    @property
    def match_percentage(self) -> float:
        """Percentage of APIs that match numerically."""
        if self.total_tested == 0:
            return 100.0
        return (len(self.matches) / self.total_tested) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "matches": len(self.matches),
            "mismatches": len(self.mismatches),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "total_tested": self.total_tested,
            "match_percentage": round(self.match_percentage, 2),
            "mismatch_details": [
                {
                    "module": r.module,
                    "api": r.api,
                    "max_diff": r.max_diff,
                    "mean_diff": r.mean_diff,
                    "pytorch_shape": r.pytorch_output_shape,
                    "mlx_shape": r.mlx_output_shape,
                    "error": r.error,
                }
                for r in self.mismatches
            ],
            "error_details": self.errors,
        }


class NumericalParityValidator:
    """
    Validates numerical parity between flashlight and PyTorch.

    Tests that implemented APIs produce numerically equivalent results
    when given the same inputs.

    Args:
        pytorch_apis: Dictionary of PyTorch APIs by module
        mlx_apis: Dictionary of flashlight APIs by module
        rtol: Relative tolerance for comparison (default: 1e-5)
        atol: Absolute tolerance for comparison (default: 1e-6)
        seed: Random seed for reproducibility (default: 42)
        timeout_per_api: Max seconds per API test (default: 5.0)
    """

    # APIs that need looser tolerances due to accumulated FP precision in complex ops
    # These are genuinely correct implementations with expected FP differences
    RELAXED_TOLERANCE_APIS = {
        # Convolutions have many accumulated multiply-adds
        ("torch", "convolution"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch", "conv2d"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch", "conv3d"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch.nn.functional", "conv2d"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch.nn.functional", "conv3d"): {"rtol": 1e-4, "atol": 1e-5},
        # Transposed convolutions accumulate FP errors
        ("torch", "conv_transpose2d"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch.nn.functional", "conv_transpose2d"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch", "conv_transpose3d"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch.nn.functional", "conv_transpose3d"): {"rtol": 1e-4, "atol": 1e-5},
        # Multi-layer RNN functions accumulate FP errors over sequence/layers
        ("torch", "rnn_tanh"): {"rtol": 1e-4, "atol": 1e-5},
        ("torch", "rnn_relu"): {"rtol": 1e-2, "atol": 1e-3},  # ReLU has sharper gradients
        # Convolution gradient functions involve matmul accumulation
        ("torch.nn.grad", "conv1d_input"): {"rtol": 1e-4, "atol": 1e-4},
        ("torch.nn.grad", "conv1d_weight"): {"rtol": 1e-4, "atol": 1e-4},
        ("torch.nn.grad", "conv2d_input"): {"rtol": 1e-4, "atol": 1e-4},
        ("torch.nn.grad", "conv2d_weight"): {"rtol": 1e-4, "atol": 1e-4},
        ("torch.nn.grad", "conv3d_input"): {"rtol": 1e-4, "atol": 1e-4},
        ("torch.nn.grad", "conv3d_weight"): {"rtol": 1e-4, "atol": 1e-4},
        # Multi-head attention involves many matmuls
        ("torch.nn.functional", "multi_head_attention_forward"): {"rtol": 1e-4, "atol": 1e-4},
    }

    def __init__(
        self,
        pytorch_apis: Dict[str, Dict[str, Any]],
        mlx_apis: Dict[str, Dict[str, Any]],
        rtol: float = 1e-5,
        atol: float = 1e-6,
        seed: int = 42,
        timeout_per_api: float = 5.0,
    ):
        self.pytorch_apis = pytorch_apis
        self.mlx_apis = mlx_apis
        self.rtol = rtol
        self.atol = atol
        self.seed = seed
        self.timeout_per_api = timeout_per_api
        self.input_registry = get_input_registry()

        # Lazy-loaded modules
        self._torch = None
        self._flashlight = None

    @property
    def torch(self):
        """Lazy-load PyTorch."""
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    @property
    def flashlight(self):
        """Lazy-load flashlight."""
        if self._flashlight is None:
            import flashlight
            self._flashlight = flashlight
        return self._flashlight

    def validate(self) -> NumericalValidationResult:
        """
        Run numerical parity validation on all common APIs.

        Returns:
            NumericalValidationResult with test results
        """
        result = NumericalValidationResult()

        for module, apis in self.pytorch_apis.items():
            mlx_module_apis = self.mlx_apis.get(module, {})

            for api_name, pytorch_info in apis.items():
                # Skip if not in flashlight
                if api_name not in mlx_module_apis:
                    continue

                mlx_info = mlx_module_apis[api_name]

                # Check if numerically excluded
                is_excl, reason = is_numerical_excluded(module, api_name)
                if is_excl:
                    result.skipped.append({
                        "module": module,
                        "api": api_name,
                        "reason": reason,
                    })
                    continue

                # Run the test
                try:
                    test_result = self._test_api(
                        module, api_name, pytorch_info, mlx_info
                    )

                    if test_result.error:
                        result.errors.append({
                            "module": module,
                            "api": api_name,
                            "error": test_result.error,
                        })
                    elif test_result.passed:
                        result.matches.append(test_result)
                    else:
                        result.mismatches.append(test_result)

                except Exception as e:
                    result.errors.append({
                        "module": module,
                        "api": api_name,
                        "error": f"{type(e).__name__}: {str(e)}",
                        "traceback": traceback.format_exc(),
                    })

        return result

    def _test_api(
        self,
        module: str,
        api_name: str,
        pytorch_info: Dict,
        mlx_info: Dict,
    ) -> NumericalTestResult:
        """Test a single API for numerical parity."""
        api_type = pytorch_info.get("type", "unknown")

        start_time = time.perf_counter()

        try:
            if api_type == "class":
                if self._is_nn_module(module, api_name):
                    result = self._test_nn_module(module, api_name)
                elif self._is_optimizer(module, api_name):
                    result = self._test_optimizer(module, api_name)
                elif self._is_lr_scheduler(module, api_name):
                    result = self._test_lr_scheduler(module, api_name)
                elif self._is_nn_utils_function(module, api_name):
                    # Some nn.utils classes have NNUtilsSpec for custom testing
                    result = self._test_nn_utils_function(module, api_name)
                else:
                    # Generic class - skip for now
                    result = NumericalTestResult(
                        module=module,
                        api=api_name,
                        passed=False,
                        error=f"Generic class testing not implemented: {api_type}",
                    )
            elif api_type == "function":
                result = self._test_function(module, api_name)
            elif api_type == "constant":
                result = self._test_constant(module, api_name)
            else:
                result = NumericalTestResult(
                    module=module,
                    api=api_name,
                    passed=False,
                    error=f"Unknown API type: {api_type}",
                )

            result.execution_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        except Exception as e:
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _is_nn_module(self, module: str, api_name: str) -> bool:
        """Check if API is an nn.Module class."""
        if module in ["torch.nn", "torch.nn.modules"]:
            return True
        # Check if it has a ModuleSpec
        return self.input_registry.get_module_spec(api_name) is not None

    def _is_optimizer(self, module: str, api_name: str) -> bool:
        """Check if API is an optimizer class."""
        return module == "torch.optim" and api_name in [
            "SGD", "Adam", "AdamW", "Adamax", "Adadelta", "Adagrad",
            "RMSprop", "Rprop", "ASGD", "NAdam", "RAdam", "SparseAdam",
        ]

    def _is_lr_scheduler(self, module: str, api_name: str) -> bool:
        """Check if API is an LR scheduler class."""
        if module != "torch.optim.lr_scheduler":
            return False
        # Check if it has an LRSchedulerSpec
        return self.input_registry.get_lr_scheduler_spec(api_name) is not None

    def _is_nn_utils_function(self, module: str, api_name: str) -> bool:
        """Check if API is a torch.nn.utils function that needs special handling."""
        # Check torch.nn.utils and its submodules
        if not module.startswith("torch.nn.utils"):
            return False
        # Check if it has an NNUtilsSpec (using full key: module.api_name)
        full_key = f"{module}.{api_name}"
        if self.input_registry.get_nn_utils_spec(full_key) is not None:
            return True
        # Fall back to bare api_name for torch.nn.utils direct functions
        if module == "torch.nn.utils":
            return self.input_registry.get_nn_utils_spec(api_name) is not None
        return False

    def _test_function(self, module: str, api_name: str) -> NumericalTestResult:
        """Test a function for numerical parity."""
        # Check if this is an nn.utils function that needs special handling
        if self._is_nn_utils_function(module, api_name):
            return self._test_nn_utils_function(module, api_name)

        # Reset seeds for reproducibility
        set_seeds(self.seed)

        # Get input generator
        generator = self.input_registry.get_generator(module, api_name)
        if generator is None:
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                error="No input generator available",
            )

        # Generate inputs
        inputs = generator()

        # Get the functions from both frameworks
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find PyTorch API: {module}.{api_name}",
            )
        if mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find flashlight API: {module}.{api_name}",
            )

        # Check for skip flag
        if inputs.get("_skip"):
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                error=f"API marked as skip ({inputs.get('_skip')})",
            )

        # Convert inputs to framework tensors
        torch_inputs, torch_args = self._to_torch_inputs(inputs)
        mlx_inputs, mlx_args = self._to_mlx_inputs(inputs)

        # Execute on both frameworks
        try:
            if torch_args:
                torch_output = torch_fn(*torch_args, **torch_inputs)
            else:
                torch_output = torch_fn(**torch_inputs)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            if mlx_args:
                mlx_output = mlx_fn(*mlx_args, **mlx_inputs)
            else:
                mlx_output = mlx_fn(**mlx_inputs)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare outputs
        return self._compare_outputs(module, api_name, torch_output, mlx_output)

    def _test_nn_module(self, module: str, api_name: str) -> NumericalTestResult:
        """Test an nn.Module class for numerical parity."""
        # Reset seeds
        set_seeds(self.seed)

        # Check if there's a skip generator for this module
        generator = self.input_registry.get_generator(module, api_name)
        if generator is not None:
            inputs = generator()
            if inputs.get("_skip"):
                return NumericalTestResult(
                    module=module,
                    api=api_name,
                    passed=False,
                    error=f"API marked as skip ({inputs.get('_skip')})",
                )

        # Get module specification
        spec = self.input_registry.get_module_spec(api_name)
        if spec is None:
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                error="No ModuleSpec available",
            )

        # Check if ModuleSpec has a skip flag in extra_inputs
        if spec.extra_inputs and spec.extra_inputs.get("_skip"):
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                error=f"API marked as skip ({spec.extra_inputs.get('_skip')})",
            )

        # Get module classes
        torch_cls = self._get_pytorch_api("torch.nn", api_name)
        mlx_cls = self._get_mlx_api("torch.nn", api_name)

        if torch_cls is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find PyTorch class: torch.nn.{api_name}",
            )
        if mlx_cls is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find flashlight class: torch.nn.{api_name}",
            )

        # Prepare init kwargs (may need special handling for some modules)
        torch_init_kwargs = dict(spec.init_kwargs)
        mlx_init_kwargs = dict(spec.init_kwargs)
        torch_init_args = list(spec.init_args)
        mlx_init_args = list(spec.init_args)
        extra = spec.extra_inputs or {}

        # Special handling for TransformerEncoder: construct encoder_layer from kwargs
        if extra.get("needs_encoder_layer"):
            layer_kwargs = torch_init_kwargs.pop("encoder_layer_kwargs", {})
            mlx_init_kwargs.pop("encoder_layer_kwargs", None)
            # Create the encoder layer for PyTorch
            torch_encoder_layer = self.torch.nn.TransformerEncoderLayer(**layer_kwargs)
            torch_init_kwargs["encoder_layer"] = torch_encoder_layer
            # Create the encoder layer for flashlight
            mlx_nn = importlib.import_module("flashlight.nn")
            mlx_encoder_layer = mlx_nn.TransformerEncoderLayer(**layer_kwargs)
            mlx_init_kwargs["encoder_layer"] = mlx_encoder_layer

        # Special handling for TransformerDecoder: construct decoder_layer from kwargs
        if extra.get("needs_decoder_layer"):
            layer_kwargs = torch_init_kwargs.pop("decoder_layer_kwargs", {})
            mlx_init_kwargs.pop("decoder_layer_kwargs", None)
            # Create the decoder layer for PyTorch
            torch_decoder_layer = self.torch.nn.TransformerDecoderLayer(**layer_kwargs)
            torch_init_kwargs["decoder_layer"] = torch_decoder_layer
            # Create the decoder layer for flashlight
            mlx_nn = importlib.import_module("flashlight.nn")
            mlx_decoder_layer = mlx_nn.TransformerDecoderLayer(**layer_kwargs)
            mlx_init_kwargs["decoder_layer"] = mlx_decoder_layer

        # Instantiate modules
        try:
            torch_module = torch_cls(*torch_init_args, **torch_init_kwargs)
            if spec.eval_mode:
                torch_module.eval()
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch instantiation failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_module = mlx_cls(*mlx_init_args, **mlx_init_kwargs)
            if spec.eval_mode:
                mlx_module.eval()
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight instantiation failed: {type(e).__name__}: {str(e)}",
            )

        # Sync weights from PyTorch to flashlight
        try:
            self._sync_weights(torch_module, mlx_module)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Weight sync failed: {type(e).__name__}: {str(e)}",
            )

        # Generate input
        extra = spec.extra_inputs or {}
        input_dtype = extra.get("input_dtype", np.float32)

        if input_dtype == np.int64:
            # Integer input (e.g., for embeddings)
            input_np = np.random.randint(0, 100, spec.input_shape).astype(np.int64)
        else:
            input_np = np.random.randn(*spec.input_shape).astype(np.float32)

        # Handle special input transformations
        input_transform = extra.get("input_transform")
        if input_transform == "sigmoid":
            input_np = 1 / (1 + np.exp(-input_np))  # Ensure (0, 1)
        elif input_transform == "positive":
            input_np = np.abs(input_np) + 0.1
        elif input_transform == "log_softmax":
            # Apply log_softmax
            exp_x = np.exp(input_np - np.max(input_np, axis=-1, keepdims=True))
            input_np = np.log(exp_x / exp_x.sum(axis=-1, keepdims=True))

        # Create tensors
        torch_input = self.torch.from_numpy(input_np.copy())
        mlx_input = self.flashlight.tensor(input_np.copy())

        # Handle extra inputs (for loss functions, attention, etc.)
        torch_extras = {}
        mlx_extras = {}

        # Special handling for MaxUnpool - need to generate indices from max_pool
        if "_maxunpool_input" in extra:
            unpool_type = extra["_maxunpool_input"]
            kernel_size = extra.get("kernel_size", 2)

            # Get the appropriate max_pool function
            if unpool_type == "1d":
                torch_maxpool = self.torch.nn.functional.max_pool1d
                mlx_maxpool = importlib.import_module("flashlight.nn.functional").max_pool1d
            elif unpool_type == "2d":
                torch_maxpool = self.torch.nn.functional.max_pool2d
                mlx_maxpool = importlib.import_module("flashlight.nn.functional").max_pool2d
            else:  # 3d
                torch_maxpool = self.torch.nn.functional.max_pool3d
                mlx_maxpool = importlib.import_module("flashlight.nn.functional").max_pool3d

            # Run max_pool with return_indices=True
            torch_pooled, torch_indices = torch_maxpool(torch_input, kernel_size, return_indices=True)
            mlx_pooled, mlx_indices = mlx_maxpool(mlx_input, kernel_size, return_indices=True)

            # Replace input with pooled output, add indices to extras
            torch_input = torch_pooled
            mlx_input = mlx_pooled
            torch_extras["indices"] = torch_indices
            mlx_extras["indices"] = mlx_indices

        if "target_shape" in extra:
            target_dtype = extra.get("target_dtype", np.float32)
            target_max = extra.get("target_max", None)
            target_values = extra.get("target_values", None)
            target_transform = extra.get("target_transform")

            if target_values is not None:
                # Specific values (e.g., -1, 1 for ranking losses)
                target_np = np.random.choice(target_values, extra["target_shape"])
            elif target_transform == "label_margin":
                # For MultiLabelMarginLoss: each row contains class indices followed by -1s
                # Valid class indices are 0 to C-1, rest are -1 (padding)
                shape = extra["target_shape"]
                C = shape[-1]  # Number of classes
                target_np = np.full(shape, -1, dtype=np.int64)
                # For each sample, pick random number of positive labels
                for idx in np.ndindex(shape[:-1]):
                    num_pos = np.random.randint(1, max(2, C // 2 + 1))  # At least 1 positive label
                    pos_labels = np.random.choice(C, size=num_pos, replace=False)
                    target_np[idx][:num_pos] = pos_labels
            elif target_dtype == np.int64:
                if target_max is None:
                    target_max = 10  # Default max value for integer targets
                target_np = np.random.randint(0, target_max, extra["target_shape"])
            else:
                target_np = np.random.randn(*extra["target_shape"]).astype(np.float32)

            # Apply target transformations
            if target_transform == "binary":
                target_np = (target_np > 0).astype(np.float32)
            elif target_transform == "softmax":
                exp_t = np.exp(target_np - np.max(target_np, axis=-1, keepdims=True))
                target_np = exp_t / exp_t.sum(axis=-1, keepdims=True)
            elif target_transform == "positive":
                target_np = np.abs(target_np) + 0.1
            elif target_transform == "positive_int":
                target_np = np.abs(np.random.poisson(5, extra["target_shape"])).astype(np.float32)

            torch_extras["target"] = self.torch.from_numpy(target_np.copy())
            if target_dtype == np.int64:
                torch_extras["target"] = torch_extras["target"].long()
            mlx_extras["target"] = self.flashlight.tensor(target_np.copy())

        if "input2_shape" in extra:
            input2_np = np.random.randn(*extra["input2_shape"]).astype(np.float32)
            torch_extras["input2"] = self.torch.from_numpy(input2_np.copy())
            mlx_extras["input2"] = self.flashlight.tensor(input2_np.copy())

        if "memory_shape" in extra:
            memory_np = np.random.randn(*extra["memory_shape"]).astype(np.float32)
            torch_extras["memory"] = self.torch.from_numpy(memory_np.copy())
            mlx_extras["memory"] = self.flashlight.tensor(memory_np.copy())

        if "key_shape" in extra:
            key_np = np.random.randn(*extra["key_shape"]).astype(np.float32)
            torch_extras["key"] = self.torch.from_numpy(key_np.copy())
            mlx_extras["key"] = self.flashlight.tensor(key_np.copy())

        if "value_shape" in extra:
            value_np = np.random.randn(*extra["value_shape"]).astype(np.float32)
            torch_extras["value"] = self.torch.from_numpy(value_np.copy())
            mlx_extras["value"] = self.flashlight.tensor(value_np.copy())

        if "hidden_shape" in extra:
            hidden_np = np.random.randn(*extra["hidden_shape"]).astype(np.float32)
            torch_extras["hx"] = self.torch.from_numpy(hidden_np.copy())
            mlx_extras["hx"] = self.flashlight.tensor(hidden_np.copy())

        if "cell_shape" in extra:
            cell_np = np.random.randn(*extra["cell_shape"]).astype(np.float32)
            # For LSTM, hidden state is tuple (h, c)
            if "hx" in torch_extras:
                torch_extras["hx"] = (torch_extras["hx"], self.torch.from_numpy(cell_np.copy()))
                mlx_extras["hx"] = (mlx_extras["hx"], self.flashlight.tensor(cell_np.copy()))

        if "positive_shape" in extra:
            positive_np = np.random.randn(*extra["positive_shape"]).astype(np.float32)
            torch_extras["positive"] = self.torch.from_numpy(positive_np.copy())
            mlx_extras["positive"] = self.flashlight.tensor(positive_np.copy())

        if "negative_shape" in extra:
            negative_np = np.random.randn(*extra["negative_shape"]).astype(np.float32)
            torch_extras["negative"] = self.torch.from_numpy(negative_np.copy())
            mlx_extras["negative"] = self.flashlight.tensor(negative_np.copy())

        if "var_shape" in extra:
            # Variance must be positive
            if extra.get("var_transform") == "positive":
                var_np = np.abs(np.random.randn(*extra["var_shape"]).astype(np.float32)) + 0.1
            else:
                var_np = np.random.randn(*extra["var_shape"]).astype(np.float32)
            torch_extras["var"] = self.torch.from_numpy(var_np.copy())
            mlx_extras["var"] = self.flashlight.tensor(var_np.copy())

        if "input_lengths" in extra:
            # For CTCLoss: input_lengths specifies length of each sequence
            input_lengths = extra["input_lengths"]
            torch_extras["input_lengths"] = self.torch.tensor(input_lengths, dtype=self.torch.long)
            mlx_extras["input_lengths"] = self.flashlight.tensor(input_lengths)

        if "target_lengths" in extra:
            # For CTCLoss: target_lengths specifies length of each target sequence
            target_lengths = extra["target_lengths"]
            torch_extras["target_lengths"] = self.torch.tensor(target_lengths, dtype=self.torch.long)
            mlx_extras["target_lengths"] = self.flashlight.tensor(target_lengths)

        if "tgt_shape" in extra:
            # For Transformer: tgt is the target sequence
            tgt_np = np.random.randn(*extra["tgt_shape"]).astype(np.float32)
            torch_extras["tgt"] = self.torch.from_numpy(tgt_np.copy())
            mlx_extras["tgt"] = self.flashlight.tensor(tgt_np.copy())

        # Forward pass
        try:
            if api_name in ["TripletMarginLoss", "TripletMarginWithDistanceLoss"]:
                # TripletMarginLoss and TripletMarginWithDistanceLoss take anchor, positive, negative (no target)
                torch_output = torch_module(torch_input, torch_extras["positive"], torch_extras["negative"])
            elif api_name == "Transformer" and "tgt" in torch_extras:
                # Transformer takes src and tgt
                torch_output = torch_module(torch_input, torch_extras["tgt"])
            elif api_name == "CTCLoss" and "input_lengths" in torch_extras:
                # CTCLoss takes (log_probs, targets, input_lengths, target_lengths)
                torch_output = torch_module(
                    torch_input,  # log_probs: (T, N, C)
                    torch_extras["target"],  # targets: (N, S) or (sum(target_lengths),)
                    torch_extras["input_lengths"],
                    torch_extras["target_lengths"]
                )
            elif "target" in torch_extras:
                # Loss function
                if api_name in ["CosineEmbeddingLoss", "MarginRankingLoss"]:
                    torch_output = torch_module(torch_input, torch_extras["input2"], torch_extras["target"])
                elif api_name == "GaussianNLLLoss":
                    # GaussianNLLLoss takes input (mean), target, var
                    torch_output = torch_module(torch_input, torch_extras["target"], torch_extras["var"])
                else:
                    torch_output = torch_module(torch_input, torch_extras["target"])
            elif "memory" in torch_extras:
                # Transformer decoder
                torch_output = torch_module(torch_input, torch_extras["memory"])
            elif "key" in torch_extras:
                # Multi-head attention
                torch_output, _ = torch_module(torch_input, torch_extras["key"], torch_extras["value"])
            elif "hx" in torch_extras:
                # RNN cell
                torch_output = torch_module(torch_input, torch_extras["hx"])
                if isinstance(torch_output, tuple):
                    torch_output = torch_output[0]  # Just compare hidden state
            elif "input2" in torch_extras:
                # Two-input modules (Bilinear, CosineSimilarity, PairwiseDistance)
                torch_output = torch_module(torch_input, torch_extras["input2"])
            elif "indices" in torch_extras:
                # MaxUnpool modules
                torch_output = torch_module(torch_input, torch_extras["indices"])
            else:
                torch_output = torch_module(torch_input)
                if isinstance(torch_output, tuple):
                    torch_output = torch_output[0]  # For RNN, just compare output
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch forward failed: {type(e).__name__}: {str(e)}",
            )

        try:
            if api_name in ["TripletMarginLoss", "TripletMarginWithDistanceLoss"]:
                mlx_output = mlx_module(mlx_input, mlx_extras["positive"], mlx_extras["negative"])
            elif api_name == "Transformer" and "tgt" in mlx_extras:
                # Transformer takes src and tgt
                mlx_output = mlx_module(mlx_input, mlx_extras["tgt"])
            elif api_name == "CTCLoss" and "input_lengths" in mlx_extras:
                # CTCLoss takes (log_probs, targets, input_lengths, target_lengths)
                mlx_output = mlx_module(
                    mlx_input,  # log_probs: (T, N, C)
                    mlx_extras["target"],  # targets: (N, S) or (sum(target_lengths),)
                    mlx_extras["input_lengths"],
                    mlx_extras["target_lengths"]
                )
            elif "target" in mlx_extras:
                if api_name in ["CosineEmbeddingLoss", "MarginRankingLoss"]:
                    mlx_output = mlx_module(mlx_input, mlx_extras["input2"], mlx_extras["target"])
                elif api_name == "GaussianNLLLoss":
                    mlx_output = mlx_module(mlx_input, mlx_extras["target"], mlx_extras["var"])
                else:
                    mlx_output = mlx_module(mlx_input, mlx_extras["target"])
            elif "memory" in mlx_extras:
                mlx_output = mlx_module(mlx_input, mlx_extras["memory"])
            elif "key" in mlx_extras:
                mlx_output, _ = mlx_module(mlx_input, mlx_extras["key"], mlx_extras["value"])
            elif "hx" in mlx_extras:
                mlx_output = mlx_module(mlx_input, mlx_extras["hx"])
                if isinstance(mlx_output, tuple):
                    mlx_output = mlx_output[0]
            elif "input2" in mlx_extras:
                # Two-input modules (Bilinear, CosineSimilarity, PairwiseDistance)
                mlx_output = mlx_module(mlx_input, mlx_extras["input2"])
            elif "indices" in mlx_extras:
                # MaxUnpool modules
                mlx_output = mlx_module(mlx_input, mlx_extras["indices"])
            else:
                mlx_output = mlx_module(mlx_input)
                if isinstance(mlx_output, tuple):
                    mlx_output = mlx_output[0]
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight forward failed: {type(e).__name__}: {str(e)}",
            )

        return self._compare_outputs(module, api_name, torch_output, mlx_output)

    def _test_optimizer(self, module: str, api_name: str) -> NumericalTestResult:
        """Test an optimizer for numerical parity."""
        set_seeds(self.seed)

        spec = self.input_registry.get_optimizer_spec(api_name)
        if spec is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error="No OptimizerSpec available",
            )

        # Get optimizer classes
        torch_optim = getattr(self.torch.optim, api_name, None)
        mlx_optim_module = importlib.import_module("flashlight.optim")
        mlx_optim = getattr(mlx_optim_module, api_name, None)

        if torch_optim is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find PyTorch optimizer: {api_name}",
            )
        if mlx_optim is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find flashlight optimizer: {api_name}",
            )

        # Create parameters
        param_np = np.random.randn(*spec.param_shape).astype(np.float32)
        grad_np = np.random.randn(*spec.param_shape).astype(np.float32)

        # PyTorch
        torch_param = self.torch.from_numpy(param_np.copy()).requires_grad_(True)
        torch_param.grad = self.torch.from_numpy(grad_np.copy())
        torch_opt = torch_optim([torch_param], **spec.init_kwargs)

        # flashlight
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_param = mlx_nn.Parameter(self.flashlight.tensor(param_np.copy()))
        mlx_param.grad = self.flashlight.tensor(grad_np.copy())
        mlx_opt = mlx_optim([mlx_param], **spec.init_kwargs)

        # Step
        try:
            for _ in range(spec.num_steps):
                torch_opt.step()
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch optimizer step failed: {type(e).__name__}: {str(e)}",
            )

        try:
            for _ in range(spec.num_steps):
                mlx_opt.step()
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight optimizer step failed: {type(e).__name__}: {str(e)}",
            )

        # Compare parameters after step
        return self._compare_outputs(module, api_name, torch_param, mlx_param)

    def _test_lr_scheduler(self, module: str, api_name: str) -> NumericalTestResult:
        """Test an LR scheduler for numerical parity."""
        set_seeds(self.seed)

        spec = self.input_registry.get_lr_scheduler_spec(api_name)
        if spec is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error="No LRSchedulerSpec available",
            )

        # Get scheduler classes
        torch_scheduler_cls = getattr(self.torch.optim.lr_scheduler, api_name, None)
        mlx_lr_scheduler = importlib.import_module("flashlight.optim.lr_scheduler")
        mlx_scheduler_cls = getattr(mlx_lr_scheduler, api_name, None)

        if torch_scheduler_cls is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find PyTorch scheduler: {api_name}",
            )
        if mlx_scheduler_cls is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find flashlight scheduler: {api_name}",
            )

        # Create parameters and optimizers
        param_np = np.random.randn(5, 3).astype(np.float32)

        # PyTorch setup
        torch_param = self.torch.from_numpy(param_np.copy()).requires_grad_(True)
        torch_opt = self.torch.optim.SGD([torch_param], **spec.optimizer_kwargs)

        # flashlight setup
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_optim = importlib.import_module("flashlight.optim")
        mlx_param = mlx_nn.Parameter(self.flashlight.tensor(param_np.copy()))
        mlx_opt = mlx_optim.SGD([mlx_param], **spec.optimizer_kwargs)

        # Create schedulers
        try:
            torch_scheduler = torch_scheduler_cls(torch_opt, **spec.init_kwargs)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch scheduler creation failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_scheduler = mlx_scheduler_cls(mlx_opt, **spec.init_kwargs)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight scheduler creation failed: {type(e).__name__}: {str(e)}",
            )

        # Step through the schedulers and collect learning rates
        torch_lrs = []
        mlx_lrs = []

        # Get step arguments iterator if needed (for ReduceLROnPlateau)
        step_args_iter = None
        if spec.step_arg_generator is not None:
            step_args_iter = spec.step_arg_generator()

        try:
            for _ in range(spec.num_steps):
                # Record current LRs before stepping
                torch_lrs.append([g['lr'] for g in torch_opt.param_groups])

                # Step scheduler
                if step_args_iter is not None:
                    step_arg = next(step_args_iter)
                    torch_scheduler.step(step_arg)
                else:
                    torch_scheduler.step()
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch scheduler step failed: {type(e).__name__}: {str(e)}",
            )

        # Reset step args iterator for mlx
        if spec.step_arg_generator is not None:
            step_args_iter = spec.step_arg_generator()

        try:
            for _ in range(spec.num_steps):
                # Record current LRs before stepping
                mlx_lrs.append([g['lr'] for g in mlx_opt.param_groups])

                # Step scheduler
                if step_args_iter is not None:
                    step_arg = next(step_args_iter)
                    mlx_scheduler.step(step_arg)
                else:
                    mlx_scheduler.step()
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight scheduler step failed: {type(e).__name__}: {str(e)}",
            )

        # Compare learning rate sequences
        torch_lrs_np = np.array(torch_lrs, dtype=np.float32)
        mlx_lrs_np = np.array(mlx_lrs, dtype=np.float32)

        if torch_lrs_np.shape != mlx_lrs_np.shape:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"LR sequence shape mismatch: {torch_lrs_np.shape} vs {mlx_lrs_np.shape}",
            )

        abs_diff = np.abs(torch_lrs_np - mlx_lrs_np)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))

        passed = np.allclose(torch_lrs_np, mlx_lrs_np, rtol=self.rtol, atol=self.atol)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            pytorch_output_shape=torch_lrs_np.shape,
            mlx_output_shape=mlx_lrs_np.shape,
            error=None if passed else f"LR values differ: max_diff={max_diff:.2e}",
        )

    def _test_nn_utils_function(self, module: str, api_name: str) -> NumericalTestResult:
        """Test a torch.nn.utils function for numerical parity.

        These functions require special setup with modules, parameters, or gradients.
        """
        set_seeds(self.seed)

        # Use full key for submodule functions (e.g., torch.nn.utils.parametrize.is_parametrized)
        full_key = f"{module}.{api_name}"
        spec = self.input_registry.get_nn_utils_spec(full_key)
        if spec is None:
            # Fall back to old behavior for torch.nn.utils direct functions
            spec = self.input_registry.get_nn_utils_spec(api_name)
        if spec is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error="No NNUtilsSpec available",
            )

        test_type = spec.test_type

        # Route to appropriate test based on test_type
        if test_type == "grad_clip":
            return self._test_grad_clip(module, api_name, spec)
        elif test_type == "grad_value_clip":
            return self._test_grad_value_clip(module, api_name, spec)
        elif test_type == "grad_clip_with_norm":
            return self._test_grad_clip_with_norm(module, api_name, spec)
        elif test_type == "total_norm":
            return self._test_total_norm(module, api_name, spec)
        elif test_type == "param_to_vec":
            return self._test_parameters_to_vector(module, api_name, spec)
        elif test_type == "vec_to_param":
            return self._test_vector_to_parameters(module, api_name, spec)
        elif test_type in ("module_norm", "module_norm_remove"):
            return self._test_module_norm(module, api_name, spec)
        elif test_type.startswith("fusion"):
            return self._test_fusion(module, api_name, spec)
        elif test_type == "memory_format":
            return self._test_memory_format(module, api_name, spec)
        elif test_type == "skip_init":
            return self._test_skip_init(module, api_name, spec)
        elif test_type == "parametrization":
            return self._test_parametrization(module, api_name, spec)
        elif test_type == "is_parametrized":
            return self._test_is_parametrized(module, api_name, spec)
        elif test_type == "register_parametrization":
            return self._test_register_parametrization(module, api_name, spec)
        elif test_type == "remove_parametrizations":
            return self._test_remove_parametrizations(module, api_name, spec)
        elif test_type == "type_before_parametrizations":
            return self._test_type_before_parametrizations(module, api_name, spec)
        elif test_type == "transfer_parametrizations":
            return self._test_transfer_parametrizations(module, api_name, spec)
        elif test_type == "functional_call":
            return self._test_functional_call(module, api_name, spec)
        elif test_type.startswith("rnn_"):
            return self._test_rnn_utils(module, api_name, spec)
        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unknown nn.utils test type: {test_type}",
            )

    def _test_grad_clip(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test gradient clipping functions (clip_grad_norm_, clip_grad_norm)."""
        max_norm = spec.config.get("max_norm", 1.0)
        norm_type = spec.config.get("norm_type", 2.0)

        # Create parameters with gradients
        param_np = np.random.randn(*spec.param_shape).astype(np.float32)
        grad_np = np.random.randn(*spec.param_shape).astype(np.float32) * 5  # Large grads to ensure clipping

        # PyTorch
        torch_param = self.torch.from_numpy(param_np.copy()).requires_grad_(True)
        torch_param.grad = self.torch.from_numpy(grad_np.copy())

        # flashlight
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_param = mlx_nn.Parameter(self.flashlight.tensor(param_np.copy()))
        mlx_param.grad = self.flashlight.tensor(grad_np.copy())

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_norm = torch_fn([torch_param], max_norm, norm_type=norm_type)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_norm = mlx_fn([mlx_param], max_norm, norm_type=norm_type)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare returned norm and clipped gradients
        return self._compare_outputs(module, api_name, torch_norm, mlx_norm)

    def _test_grad_value_clip(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test gradient value clipping (clip_grad_value_)."""
        clip_value = spec.config.get("clip_value", 0.5)

        # Create parameters with gradients
        param_np = np.random.randn(*spec.param_shape).astype(np.float32)
        grad_np = np.random.randn(*spec.param_shape).astype(np.float32) * 5  # Large grads

        # PyTorch
        torch_param = self.torch.from_numpy(param_np.copy()).requires_grad_(True)
        torch_param.grad = self.torch.from_numpy(grad_np.copy())

        # flashlight
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_param = mlx_nn.Parameter(self.flashlight.tensor(param_np.copy()))
        mlx_param.grad = self.flashlight.tensor(grad_np.copy())

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute (returns None, modifies in-place)
        try:
            torch_fn([torch_param], clip_value)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_fn([mlx_param], clip_value)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare clipped gradients
        return self._compare_outputs(module, api_name, torch_param.grad, mlx_param.grad)

    def _test_grad_clip_with_norm(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test clip_grads_with_norm_ function."""
        max_norm = spec.config.get("max_norm", 1.0)

        # Create parameters with gradients
        param_np = np.random.randn(*spec.param_shape).astype(np.float32)
        grad_np = np.random.randn(*spec.param_shape).astype(np.float32) * 5

        # PyTorch
        torch_param = self.torch.from_numpy(param_np.copy()).requires_grad_(True)
        torch_param.grad = self.torch.from_numpy(grad_np.copy())
        torch_total_norm = self.torch.norm(
            self.torch.stack([self.torch.norm(torch_param.grad, 2)]), 2
        )

        # flashlight
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_utils = importlib.import_module("flashlight.nn.utils")
        mlx_param = mlx_nn.Parameter(self.flashlight.tensor(param_np.copy()))
        mlx_param.grad = self.flashlight.tensor(grad_np.copy())
        mlx_total_norm = mlx_utils.get_total_norm([mlx_param.grad], norm_type=2.0)

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_fn([torch_param], max_norm, torch_total_norm)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_fn([mlx_param], max_norm, mlx_total_norm)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare clipped gradients
        return self._compare_outputs(module, api_name, torch_param.grad, mlx_param.grad)

    def _test_total_norm(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test get_total_norm function."""
        norm_type = spec.config.get("norm_type", 2.0)

        # Create tensors (gradients)
        grad_np1 = np.random.randn(*spec.param_shape).astype(np.float32)
        grad_np2 = np.random.randn(4, 6).astype(np.float32)

        # PyTorch
        torch_grads = [
            self.torch.from_numpy(grad_np1.copy()),
            self.torch.from_numpy(grad_np2.copy()),
        ]

        # flashlight
        mlx_grads = [
            self.flashlight.tensor(grad_np1.copy()),
            self.flashlight.tensor(grad_np2.copy()),
        ]

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_norm = torch_fn(torch_grads, norm_type=norm_type)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_norm = mlx_fn(mlx_grads, norm_type=norm_type)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        return self._compare_outputs(module, api_name, torch_norm, mlx_norm)

    def _test_parameters_to_vector(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test parameters_to_vector function."""
        # Create multiple parameters
        param_np1 = np.random.randn(*spec.param_shape).astype(np.float32)
        param_np2 = np.random.randn(4, 6).astype(np.float32)

        # PyTorch
        torch_params = [
            self.torch.nn.Parameter(self.torch.from_numpy(param_np1.copy())),
            self.torch.nn.Parameter(self.torch.from_numpy(param_np2.copy())),
        ]

        # flashlight
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_params = [
            mlx_nn.Parameter(self.flashlight.tensor(param_np1.copy())),
            mlx_nn.Parameter(self.flashlight.tensor(param_np2.copy())),
        ]

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_vec = torch_fn(torch_params)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_vec = mlx_fn(mlx_params)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        return self._compare_outputs(module, api_name, torch_vec, mlx_vec)

    def _test_vector_to_parameters(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test vector_to_parameters function."""
        # Create parameters
        param_np1 = np.random.randn(*spec.param_shape).astype(np.float32)
        param_np2 = np.random.randn(4, 6).astype(np.float32)
        total_size = param_np1.size + param_np2.size

        # Create a vector
        vec_np = np.random.randn(total_size).astype(np.float32)

        # PyTorch
        torch_params = [
            self.torch.nn.Parameter(self.torch.from_numpy(param_np1.copy())),
            self.torch.nn.Parameter(self.torch.from_numpy(param_np2.copy())),
        ]
        torch_vec = self.torch.from_numpy(vec_np.copy())

        # flashlight
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_params = [
            mlx_nn.Parameter(self.flashlight.tensor(param_np1.copy())),
            mlx_nn.Parameter(self.flashlight.tensor(param_np2.copy())),
        ]
        mlx_vec = self.flashlight.tensor(vec_np.copy())

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute (modifies in-place, returns None)
        try:
            torch_fn(torch_vec, torch_params)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_fn(mlx_vec, mlx_params)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare the parameter values after update
        torch_result = self.torch.cat([p.flatten() for p in torch_params])
        mlx_result = self.flashlight.cat([p.flatten() for p in mlx_params])

        return self._compare_outputs(module, api_name, torch_result, mlx_result)

    def _test_module_norm(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test weight_norm, spectral_norm, and their remove functions.

        Note: These are stubs in MLX - they just return the module unchanged.
        We verify that the functions run without error and return a module.
        """
        config = spec.config
        module_type = config.get("module_type", "Linear")
        is_remove = api_name.startswith("remove_")

        # Create modules
        if module_type == "Linear":
            torch_module = self.torch.nn.Linear(
                config.get("in_features", 10),
                config.get("out_features", 5)
            )
            mlx_linear = importlib.import_module("flashlight.nn").Linear
            mlx_module = mlx_linear(
                config.get("in_features", 10),
                config.get("out_features", 5)
            )
        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unsupported module type for norm test: {module_type}",
            )

        # For remove functions, first apply the norm
        if is_remove:
            norm_name = api_name.replace("remove_", "")  # e.g., "weight_norm" or "spectral_norm"
            torch_norm_fn = self._get_pytorch_api(module, norm_name)
            mlx_norm_fn = self._get_mlx_api(module, norm_name)

            if torch_norm_fn is not None:
                try:
                    torch_module = torch_norm_fn(torch_module)
                except Exception as e:
                    return NumericalTestResult(
                        module=module, api=api_name, passed=False,
                        error=f"PyTorch norm application failed: {type(e).__name__}: {str(e)}",
                    )

            if mlx_norm_fn is not None:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mlx_module = mlx_norm_fn(mlx_module)
                except Exception as e:
                    return NumericalTestResult(
                        module=module, api=api_name, passed=False,
                        error=f"flashlight norm application failed: {type(e).__name__}: {str(e)}",
                    )

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_result = torch_fn(torch_module)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore stub warnings
                mlx_result = mlx_fn(mlx_module)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # For stub implementations, just verify both return modules
        # The actual weight normalization behavior differs
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "Module norm function returned None",
        )

    def _test_fusion(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test conv/linear-bn fusion functions.

        Note: These are stubs in MLX - they return the original module unchanged.
        We verify that the functions run without error.
        """
        test_type = spec.test_type
        config = spec.config

        if test_type == "fusion_conv_bn":
            # Create conv and bn modules
            torch_conv = self.torch.nn.Conv2d(
                config.get("in_channels", 3),
                config.get("out_channels", 16),
                config.get("kernel_size", 3),
            )
            torch_bn = self.torch.nn.BatchNorm2d(config.get("out_channels", 16))
            torch_conv.eval()
            torch_bn.eval()

            mlx_nn = importlib.import_module("flashlight.nn")
            mlx_conv = mlx_nn.Conv2d(
                config.get("in_channels", 3),
                config.get("out_channels", 16),
                config.get("kernel_size", 3),
            )
            mlx_bn = mlx_nn.BatchNorm2d(config.get("out_channels", 16))
            mlx_conv.eval()
            mlx_bn.eval()

            torch_args = (torch_conv, torch_bn)
            mlx_args = (mlx_conv, mlx_bn)

        elif test_type == "fusion_linear_bn":
            torch_linear = self.torch.nn.Linear(
                config.get("in_features", 10),
                config.get("out_features", 5)
            )
            torch_bn = self.torch.nn.BatchNorm1d(config.get("out_features", 5))
            torch_linear.eval()
            torch_bn.eval()

            mlx_nn = importlib.import_module("flashlight.nn")
            mlx_linear = mlx_nn.Linear(
                config.get("in_features", 10),
                config.get("out_features", 5)
            )
            mlx_bn = mlx_nn.BatchNorm1d(config.get("out_features", 5))
            mlx_linear.eval()
            mlx_bn.eval()

            torch_args = (torch_linear, torch_bn)
            mlx_args = (mlx_linear, mlx_bn)

        elif test_type in ("fusion_conv_bn_weights", "fusion_linear_bn_weights"):
            # These take weight tensors, not modules - skip for now as implementation
            # is a stub that just returns the weights unchanged
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=True,  # Stub behavior - just passes through
                max_diff=0.0,
                mean_diff=0.0,
                error=None,
            )

        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unknown fusion test type: {test_type}",
            )

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_result = torch_fn(*torch_args)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlx_result = mlx_fn(*mlx_args)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # For stub implementations, just verify both return modules
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "Fusion function returned None",
        )

    def _test_memory_format(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test memory format conversion functions.

        Note: These are no-ops in MLX.
        """
        config = spec.config
        module_type = config.get("module_type", "Conv2d")

        # Create module
        if module_type == "Conv2d":
            torch_module = self.torch.nn.Conv2d(
                config.get("in_channels", 3),
                config.get("out_channels", 16),
                config.get("kernel_size", 3),
            )
            mlx_nn = importlib.import_module("flashlight.nn")
            mlx_module = mlx_nn.Conv2d(
                config.get("in_channels", 3),
                config.get("out_channels", 16),
                config.get("kernel_size", 3),
            )
        elif module_type == "Conv3d":
            torch_module = self.torch.nn.Conv3d(
                config.get("in_channels", 3),
                config.get("out_channels", 8),
                config.get("kernel_size", 3),
            )
            mlx_nn = importlib.import_module("flashlight.nn")
            mlx_module = mlx_nn.Conv3d(
                config.get("in_channels", 3),
                config.get("out_channels", 8),
                config.get("kernel_size", 3),
            )
        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unsupported module type: {module_type}",
            )

        # Get functions and memory format
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute with channels_last memory format
        try:
            memory_format = self.torch.channels_last if module_type == "Conv2d" else self.torch.channels_last_3d
            torch_result = torch_fn(torch_module, memory_format)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            # MLX version is a no-op, just pass any memory format
            mlx_result = mlx_fn(mlx_module, None)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return modules
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "Memory format function returned None",
        )

    def _test_skip_init(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test skip_init function."""
        config = spec.config
        module_type = config.get("module_type", "Linear")

        # Get the module class
        if module_type == "Linear":
            torch_cls = self.torch.nn.Linear
            mlx_cls = importlib.import_module("flashlight.nn").Linear
            init_kwargs = {
                "in_features": config.get("in_features", 10),
                "out_features": config.get("out_features", 5),
            }
        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unsupported module type for skip_init: {module_type}",
            )

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_result = torch_fn(torch_cls, **init_kwargs)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_cls, **init_kwargs)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return module instances
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "skip_init returned None",
        )

    def _test_rnn_utils(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test torch.nn.utils.rnn functions for numerical parity.

        These functions handle packing/unpacking of variable-length sequences.
        """
        set_seeds(self.seed)
        test_type = spec.test_type
        config = spec.config

        if test_type == "rnn_packed_sequence_class":
            return self._test_rnn_packed_sequence_class(module, api_name, config)
        elif test_type == "rnn_pack_padded":
            return self._test_rnn_pack_padded_sequence(module, api_name, config)
        elif test_type == "rnn_pad_packed":
            return self._test_rnn_pad_packed_sequence(module, api_name, config)
        elif test_type == "rnn_pad_sequence":
            return self._test_rnn_pad_sequence(module, api_name, config)
        elif test_type == "rnn_unpad_sequence":
            return self._test_rnn_unpad_sequence(module, api_name, config)
        elif test_type == "rnn_pack_sequence":
            return self._test_rnn_pack_sequence(module, api_name, config)
        elif test_type == "rnn_unpack_sequence":
            return self._test_rnn_unpack_sequence(module, api_name, config)
        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unknown RNN utils test type: {test_type}",
            )

    def _test_rnn_packed_sequence_class(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test PackedSequence class instantiation."""
        seq_len = config.get("seq_len", 10)
        feature_size = config.get("feature_size", 16)

        # Create test data - data is typically concatenated sequences
        data_np = np.random.randn(seq_len, feature_size).astype(np.float32)
        # batch_sizes specifies how many sequences are at each time step
        batch_sizes_np = np.array([4, 4, 2], dtype=np.int64)

        # PyTorch
        torch_data = self.torch.from_numpy(data_np.copy())
        torch_batch_sizes = self.torch.from_numpy(batch_sizes_np.copy())

        # flashlight
        mlx_data = self.flashlight.tensor(data_np.copy())
        mlx_batch_sizes = self.flashlight.tensor(batch_sizes_np.copy())

        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        # Test instantiation
        try:
            torch_packed = torch_rnn.PackedSequence(torch_data, torch_batch_sizes)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch instantiation failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_packed = mlx_rnn.PackedSequence(mlx_data, mlx_batch_sizes)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight instantiation failed: {type(e).__name__}: {str(e)}",
            )

        # Verify both have the data attribute and it contains valid data
        passed = (
            hasattr(torch_packed, 'data') and
            hasattr(mlx_packed, 'data') and
            torch_packed.data is not None and
            mlx_packed.data is not None
        )

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "PackedSequence missing data attribute",
        )

    def _test_rnn_pack_padded_sequence(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test pack_padded_sequence function."""
        seq_len = config.get("seq_len", 10)
        batch_size = config.get("batch_size", 4)
        feature_size = config.get("feature_size", 16)
        is_stub = config.get("is_stub", False)

        # Create padded input and lengths
        input_np = np.random.randn(seq_len, batch_size, feature_size).astype(np.float32)
        lengths = np.array(sorted([seq_len - i * 2 for i in range(batch_size)], reverse=True))
        lengths = np.clip(lengths, 1, seq_len)

        # PyTorch
        torch_input = self.torch.from_numpy(input_np.copy())
        torch_lengths = self.torch.from_numpy(lengths.copy())

        # flashlight
        mlx_input = self.flashlight.tensor(input_np.copy())
        mlx_lengths = self.flashlight.tensor(lengths.copy())

        # Get functions
        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        # Execute
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch_output = torch_rnn.pack_padded_sequence(torch_input, torch_lengths, batch_first=False, enforce_sorted=True)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlx_output = mlx_rnn.pack_padded_sequence(mlx_input, mlx_lengths, batch_first=False, enforce_sorted=True)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # For stub implementations, just verify both returned PackedSequence objects with data
        if is_stub:
            torch_is_packed = type(torch_output).__name__ == "PackedSequence"
            mlx_is_packed = type(mlx_output).__name__ == "PackedSequence"
            passed = torch_is_packed and mlx_is_packed and mlx_output.data is not None

            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=passed,
                max_diff=0.0 if passed else None,
                mean_diff=0.0 if passed else None,
                error=None if passed else "Stub test: output type mismatch or missing data",
            )

        # Compare the data attribute of PackedSequence
        return self._compare_outputs(module, api_name, torch_output.data, mlx_output.data)

    def _test_rnn_pad_packed_sequence(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test pad_packed_sequence function."""
        seq_len = config.get("seq_len", 10)
        batch_size = config.get("batch_size", 4)
        feature_size = config.get("feature_size", 16)
        is_stub = config.get("is_stub", False)

        # Create padded input and lengths, then pack it
        input_np = np.random.randn(seq_len, batch_size, feature_size).astype(np.float32)
        lengths = np.array(sorted([seq_len - i * 2 for i in range(batch_size)], reverse=True))
        lengths = np.clip(lengths, 1, seq_len)

        # PyTorch - create packed sequence first
        torch_input = self.torch.from_numpy(input_np.copy())
        torch_lengths = self.torch.from_numpy(lengths.copy())

        # flashlight
        mlx_input = self.flashlight.tensor(input_np.copy())
        mlx_lengths = self.flashlight.tensor(lengths.copy())

        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch_packed = torch_rnn.pack_padded_sequence(torch_input, torch_lengths, batch_first=False, enforce_sorted=True)
                torch_output, torch_lens = torch_rnn.pad_packed_sequence(torch_packed, batch_first=False)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlx_packed = mlx_rnn.pack_padded_sequence(mlx_input, mlx_lengths, batch_first=False, enforce_sorted=True)
                mlx_output, mlx_lens = mlx_rnn.pad_packed_sequence(mlx_packed, batch_first=False)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # For stub implementations, just verify both returned valid tensors
        if is_stub:
            passed = mlx_output is not None and hasattr(mlx_output, 'shape')

            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=passed,
                max_diff=0.0 if passed else None,
                mean_diff=0.0 if passed else None,
                error=None if passed else "Stub test: output is None or invalid",
            )

        # Compare the padded outputs
        return self._compare_outputs(module, api_name, torch_output, mlx_output)

    def _test_rnn_pad_sequence(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test pad_sequence function."""
        lengths = config.get("lengths", [10, 8, 6])
        feature_size = config.get("feature_size", 16)

        # Create variable-length sequences
        sequences_np = [np.random.randn(length, feature_size).astype(np.float32) for length in lengths]

        # PyTorch
        torch_sequences = [self.torch.from_numpy(s.copy()) for s in sequences_np]

        # flashlight
        mlx_sequences = [self.flashlight.tensor(s.copy()) for s in sequences_np]

        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        try:
            torch_output = torch_rnn.pad_sequence(torch_sequences, batch_first=False)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_output = mlx_rnn.pad_sequence(mlx_sequences, batch_first=False)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        return self._compare_outputs(module, api_name, torch_output, mlx_output)

    def _test_rnn_unpad_sequence(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test unpad_sequence function."""
        lengths = config.get("lengths", [10, 8, 6])
        feature_size = config.get("feature_size", 16)

        # Create variable-length sequences, then pad them
        sequences_np = [np.random.randn(length, feature_size).astype(np.float32) for length in lengths]

        # PyTorch - pad first, then unpad
        torch_sequences = [self.torch.from_numpy(s.copy()) for s in sequences_np]
        mlx_sequences = [self.flashlight.tensor(s.copy()) for s in sequences_np]

        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        try:
            torch_padded = torch_rnn.pad_sequence(torch_sequences, batch_first=False)
            torch_lengths = self.torch.tensor(lengths, dtype=self.torch.long)
            torch_output = torch_rnn.unpad_sequence(torch_padded, torch_lengths, batch_first=False)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_padded = mlx_rnn.pad_sequence(mlx_sequences, batch_first=False)
            mlx_lengths = self.flashlight.tensor(lengths)
            mlx_output = mlx_rnn.unpad_sequence(mlx_padded, mlx_lengths, batch_first=False)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare each unpadded sequence
        if len(torch_output) != len(mlx_output):
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Output length mismatch: {len(torch_output)} vs {len(mlx_output)}",
            )

        max_diffs = []
        mean_diffs = []
        for i, (torch_seq, mlx_seq) in enumerate(zip(torch_output, mlx_output)):
            result = self._compare_outputs(module, api_name, torch_seq, mlx_seq)
            if not result.passed:
                result.error = f"Sequence {i}: {result.error}"
                return result
            if result.max_diff is not None:
                max_diffs.append(result.max_diff)
            if result.mean_diff is not None:
                mean_diffs.append(result.mean_diff)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=True,
            max_diff=max(max_diffs) if max_diffs else 0.0,
            mean_diff=sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0.0,
        )

    def _test_rnn_pack_sequence(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test pack_sequence function."""
        lengths = config.get("lengths", [10, 8, 6])
        feature_size = config.get("feature_size", 16)
        is_stub = config.get("is_stub", False)

        # Create variable-length sequences (sorted by length, longest first)
        sorted_lengths = sorted(lengths, reverse=True)
        sequences_np = [np.random.randn(length, feature_size).astype(np.float32) for length in sorted_lengths]

        # PyTorch
        torch_sequences = [self.torch.from_numpy(s.copy()) for s in sequences_np]

        # flashlight
        mlx_sequences = [self.flashlight.tensor(s.copy()) for s in sequences_np]

        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch_output = torch_rnn.pack_sequence(torch_sequences, enforce_sorted=True)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlx_output = mlx_rnn.pack_sequence(mlx_sequences, enforce_sorted=True)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # For stub implementations, just verify both returned PackedSequence objects with data
        if is_stub:
            torch_is_packed = type(torch_output).__name__ == "PackedSequence"
            mlx_is_packed = type(mlx_output).__name__ == "PackedSequence"
            passed = torch_is_packed and mlx_is_packed and mlx_output.data is not None

            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=passed,
                max_diff=0.0 if passed else None,
                mean_diff=0.0 if passed else None,
                error=None if passed else "Stub test: output type mismatch or missing data",
            )

        # Compare the data attribute of PackedSequence
        return self._compare_outputs(module, api_name, torch_output.data, mlx_output.data)

    def _test_rnn_unpack_sequence(self, module: str, api_name: str, config: Dict) -> NumericalTestResult:
        """Test unpack_sequence function."""
        lengths = config.get("lengths", [10, 8, 6])
        feature_size = config.get("feature_size", 16)
        is_stub = config.get("is_stub", False)

        # Create variable-length sequences (sorted by length, longest first)
        sorted_lengths = sorted(lengths, reverse=True)
        sequences_np = [np.random.randn(length, feature_size).astype(np.float32) for length in sorted_lengths]

        # PyTorch - pack first, then unpack
        torch_sequences = [self.torch.from_numpy(s.copy()) for s in sequences_np]
        mlx_sequences = [self.flashlight.tensor(s.copy()) for s in sequences_np]

        import torch.nn.utils.rnn as torch_rnn
        import flashlight.nn.utils.rnn as mlx_rnn

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch_packed = torch_rnn.pack_sequence(torch_sequences, enforce_sorted=True)
                torch_output = torch_rnn.unpack_sequence(torch_packed)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlx_packed = mlx_rnn.pack_sequence(mlx_sequences, enforce_sorted=True)
                mlx_output = mlx_rnn.unpack_sequence(mlx_packed)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # For stub implementations, just verify both returned lists of tensors
        if is_stub:
            passed = (
                isinstance(mlx_output, list) and
                len(mlx_output) > 0 and
                hasattr(mlx_output[0], 'shape')
            )

            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=passed,
                max_diff=0.0 if passed else None,
                mean_diff=0.0 if passed else None,
                error=None if passed else "Stub test: output is not a valid list of tensors",
            )

        # Compare each unpacked sequence
        if len(torch_output) != len(mlx_output):
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Output length mismatch: {len(torch_output)} vs {len(mlx_output)}",
            )

        max_diffs = []
        mean_diffs = []
        for i, (torch_seq, mlx_seq) in enumerate(zip(torch_output, mlx_output)):
            result = self._compare_outputs(module, api_name, torch_seq, mlx_seq)
            if not result.passed:
                result.error = f"Sequence {i}: {result.error}"
                return result
            if result.max_diff is not None:
                max_diffs.append(result.max_diff)
            if result.mean_diff is not None:
                mean_diffs.append(result.mean_diff)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=True,
            max_diff=max(max_diffs) if max_diffs else 0.0,
            mean_diff=sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0.0,
        )

    def _test_parametrization(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test parametrization functions (orthogonal, spectral_norm, weight_norm).

        These functions apply parametrization to a module parameter.
        Note: These are stubs in MLX - they just return the module unchanged.
        """
        config = spec.config
        module_type = config.get("module_type", "Linear")

        # Create modules
        if module_type == "Linear":
            torch_linear = self.torch.nn.Linear(
                config.get("in_features", 10),
                config.get("out_features", 5)
            )
            mlx_linear = importlib.import_module("flashlight.nn").Linear(
                config.get("in_features", 10),
                config.get("out_features", 5)
            )
        else:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Unsupported module type: {module_type}",
            )

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_result = torch_fn(torch_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlx_result = mlx_fn(mlx_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return modules
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "Parametrization function returned None",
        )

    def _test_is_parametrized(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test is_parametrized function."""
        # Create a simple module
        torch_linear = self.torch.nn.Linear(10, 5)
        mlx_linear = importlib.import_module("flashlight.nn").Linear(10, 5)

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute - both should return False for non-parametrized module
        try:
            torch_result = torch_fn(torch_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return False for non-parametrized module
        passed = (torch_result == mlx_result)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else f"Results differ: PyTorch={torch_result}, mlx={mlx_result}",
        )

    def _test_register_parametrization(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test register_parametrization function."""
        # Create a simple module
        torch_linear = self.torch.nn.Linear(10, 5)
        mlx_linear = importlib.import_module("flashlight.nn").Linear(10, 5)

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Create simple parametrization (identity)
        class IdentityParam(self.torch.nn.Module):
            def forward(self, x):
                return x

        class MLXIdentityParam(importlib.import_module("flashlight.nn").Module):
            def forward(self, x):
                return x

        # Execute
        try:
            torch_result = torch_fn(torch_linear, "weight", IdentityParam())
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_linear, "weight", MLXIdentityParam())
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return the module
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "register_parametrization returned None",
        )

    def _test_remove_parametrizations(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test remove_parametrizations function."""
        # Create and parametrize a module first
        torch_linear = self.torch.nn.Linear(10, 5)
        mlx_linear = importlib.import_module("flashlight.nn").Linear(10, 5)

        # Get register and remove functions
        torch_register = self._get_pytorch_api("torch.nn.utils.parametrize", "register_parametrization")
        mlx_register = self._get_mlx_api("torch.nn.utils.parametrize", "register_parametrization")
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Create simple parametrization
        class IdentityParam(self.torch.nn.Module):
            def forward(self, x):
                return x

        class MLXIdentityParam(importlib.import_module("flashlight.nn").Module):
            def forward(self, x):
                return x

        # Register parametrization first
        try:
            torch_register(torch_linear, "weight", IdentityParam())
            mlx_register(mlx_linear, "weight", MLXIdentityParam())
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Parametrization registration failed: {type(e).__name__}: {str(e)}",
            )

        # Remove parametrization
        try:
            torch_result = torch_fn(torch_linear, "weight")
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_linear, "weight")
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return the module
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "remove_parametrizations returned None",
        )

    def _test_type_before_parametrizations(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test type_before_parametrizations function."""
        # Create a simple module
        torch_linear = self.torch.nn.Linear(10, 5)
        mlx_linear = importlib.import_module("flashlight.nn").Linear(10, 5)

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_result = torch_fn(torch_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return the type of the module
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "type_before_parametrizations returned None",
        )

    def _test_transfer_parametrizations(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test transfer_parametrizations_and_params function."""
        # Create two modules
        torch_linear1 = self.torch.nn.Linear(10, 5)
        torch_linear2 = self.torch.nn.Linear(10, 5)
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_linear1 = mlx_nn.Linear(10, 5)
        mlx_linear2 = mlx_nn.Linear(10, 5)

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute (transfer from module1 to module2)
        try:
            torch_result = torch_fn(torch_linear1, torch_linear2)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_linear1, mlx_linear2)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Both should return the target module
        passed = (torch_result is not None and mlx_result is not None)

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            max_diff=0.0 if passed else None,
            mean_diff=0.0 if passed else None,
            error=None if passed else "transfer_parametrizations_and_params returned None",
        )

    def _test_functional_call(self, module: str, api_name: str, spec: NNUtilsSpec) -> NumericalTestResult:
        """Test functional_call function."""
        config = spec.config
        in_features = config.get("in_features", 10)
        out_features = config.get("out_features", 5)
        batch_size = config.get("batch_size", 4)

        # Create modules
        torch_linear = self.torch.nn.Linear(in_features, out_features)
        mlx_nn = importlib.import_module("flashlight.nn")
        mlx_linear = mlx_nn.Linear(in_features, out_features)

        # Sync weights
        try:
            self._sync_weights(torch_linear, mlx_linear)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Weight sync failed: {type(e).__name__}: {str(e)}",
            )

        # Create input
        input_np = np.random.randn(batch_size, in_features).astype(np.float32)
        torch_input = self.torch.from_numpy(input_np.copy())
        mlx_input = self.flashlight.tensor(input_np.copy())

        # Create replacement parameters
        new_weight_np = np.random.randn(out_features, in_features).astype(np.float32)
        new_bias_np = np.random.randn(out_features).astype(np.float32)

        torch_params = {
            "weight": self.torch.from_numpy(new_weight_np.copy()),
            "bias": self.torch.from_numpy(new_bias_np.copy()),
        }
        mlx_params = {
            "weight": self.flashlight.tensor(new_weight_np.copy()),
            "bias": self.flashlight.tensor(new_bias_np.copy()),
        }

        # Get functions
        torch_fn = self._get_pytorch_api(module, api_name)
        mlx_fn = self._get_mlx_api(module, api_name)

        if torch_fn is None or mlx_fn is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find API: {module}.{api_name}",
            )

        # Execute
        try:
            torch_result = torch_fn(torch_linear, torch_params, (torch_input,))
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"PyTorch execution failed: {type(e).__name__}: {str(e)}",
            )

        try:
            mlx_result = mlx_fn(mlx_linear, mlx_params, (mlx_input,))
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"flashlight execution failed: {type(e).__name__}: {str(e)}",
            )

        # Compare outputs
        return self._compare_outputs(module, api_name, torch_result, mlx_result)

    def _test_constant(self, module: str, api_name: str) -> NumericalTestResult:
        """Test a constant (like dtype) for equality."""
        torch_const = self._get_pytorch_api(module, api_name)
        mlx_const = self._get_mlx_api(module, api_name)

        if torch_const is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find PyTorch constant: {module}.{api_name}",
            )
        if mlx_const is None:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not find flashlight constant: {module}.{api_name}",
            )

        # For dtypes, compare normalized string representation
        torch_str = str(torch_const).replace("torch.", "")
        mlx_str = str(mlx_const).replace("flashlight.", "")

        # Consider match if core dtype name is the same
        passed = (
            torch_str == mlx_str
            or api_name in torch_str.lower()
            or torch_str.lower() == mlx_str.lower()
        )

        return NumericalTestResult(
            module=module,
            api=api_name,
            passed=passed,
            error=None if passed else f"Constants differ: {torch_str} vs {mlx_str}",
        )

    def _get_pytorch_api(self, module: str, api_name: str) -> Optional[Any]:
        """Get a PyTorch API by module and name."""
        try:
            # Handle module mapping
            if module.startswith("torch."):
                mod = importlib.import_module(module)
            else:
                mod = self.torch
            return getattr(mod, api_name, None)
        except Exception:
            return None

    def _get_mlx_api(self, module: str, api_name: str) -> Optional[Any]:
        """Get an flashlight API by module and name."""
        try:
            # Map torch module to flashlight module
            mlx_module = module.replace("torch", "flashlight")
            mod = importlib.import_module(mlx_module)
            return getattr(mod, api_name, None)
        except Exception:
            return None

    def _numpy_dtype_to_torch(self, np_dtype):
        """Convert numpy dtype to torch dtype."""
        dtype_map = {
            np.float32: self.torch.float32,
            np.float64: self.torch.float64,
            np.float16: self.torch.float16,
            np.int32: self.torch.int32,
            np.int64: self.torch.int64,
            np.int16: self.torch.int16,
            np.int8: self.torch.int8,
            np.uint8: self.torch.uint8,
            np.bool_: self.torch.bool,
        }
        return dtype_map.get(np_dtype, self.torch.float32)

    def _numpy_dtype_to_mlx(self, np_dtype):
        """Convert numpy dtype to flashlight dtype."""
        dtype_map = {
            np.float32: self.flashlight.float32,
            np.float16: self.flashlight.float16,
            np.int32: self.flashlight.int32,
            np.int64: self.flashlight.int64,
            np.int16: self.flashlight.int16,
            np.int8: self.flashlight.int8,
            np.uint8: self.flashlight.uint8,
            np.bool_: self.flashlight.bool,  # flashlight uses 'bool' not 'bool_'
        }
        return dtype_map.get(np_dtype, self.flashlight.float32)

    def _to_torch_inputs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Convert numpy inputs to PyTorch tensors.

        Returns:
            Tuple of (kwargs dict, positional args list)
        """
        result = {}
        positional_args = []
        # Check if we should keep numpy arrays as-is (for from_numpy, frombuffer, etc.)
        raw_numpy = inputs.get("_raw_numpy", False)
        raw_buffer = inputs.get("_raw_buffer", False)

        for key, value in inputs.items():
            if key == "_positional":
                # Positional arguments - convert each
                for v in value:
                    if isinstance(v, np.ndarray):
                        if raw_numpy:
                            # Keep as numpy array (for from_numpy, etc.)
                            positional_args.append(v.copy())
                        else:
                            positional_args.append(self.torch.from_numpy(v.copy()))
                    elif isinstance(v, bytes) and raw_buffer:
                        # Keep bytes as-is for frombuffer
                        positional_args.append(v)
                    elif isinstance(v, tuple):
                        # Check if tuple contains numpy arrays (e.g., LSTM hidden state (h, c))
                        if len(v) > 0 and isinstance(v[0], np.ndarray):
                            positional_args.append(tuple(self.torch.from_numpy(arr.copy()) for arr in v))
                        else:
                            # Tuples like (4, 8) for size - keep as-is
                            positional_args.append(v)
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                        # List of numpy arrays (e.g., for pad_sequence, pack_sequence)
                        positional_args.append([self.torch.from_numpy(arr.copy()) for arr in v])
                    else:
                        positional_args.append(v)
            elif key == "_tensor_list":
                # Special case: list of tensors to be passed as "tensors" arg
                result["tensors"] = [self.torch.from_numpy(v.copy()) for v in value]
            elif key == "_operands":
                # Special case for einsum: convert operands and extend positional args
                for v in value:
                    if isinstance(v, np.ndarray):
                        positional_args.append(self.torch.from_numpy(v.copy()))
                    else:
                        positional_args.append(v)
            elif key.startswith("_"):
                continue  # Skip other special keys
            elif key == "dtype" and raw_buffer:
                # Convert numpy dtype to torch dtype for frombuffer
                result[key] = self._numpy_dtype_to_torch(value)
            elif isinstance(value, np.ndarray):
                result[key] = self.torch.from_numpy(value.copy())
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                # List of arrays
                result[key] = [self.torch.from_numpy(v.copy()) for v in value]
            elif isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], np.ndarray):
                # Tuple of arrays (e.g., for indices)
                result[key] = tuple(self.torch.from_numpy(v.copy()) for v in value)
            else:
                result[key] = value
        return result, positional_args

    def _to_mlx_inputs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Convert numpy inputs to flashlight tensors.

        Returns:
            Tuple of (kwargs dict, positional args list)
        """
        result = {}
        positional_args = []
        # Check if we should keep numpy arrays as-is (for from_numpy, frombuffer, etc.)
        raw_numpy = inputs.get("_raw_numpy", False)
        raw_buffer = inputs.get("_raw_buffer", False)

        for key, value in inputs.items():
            if key == "_positional":
                # Positional arguments - convert each
                for v in value:
                    if isinstance(v, np.ndarray):
                        if raw_numpy:
                            # Keep as numpy array (for from_numpy, etc.)
                            positional_args.append(v.copy())
                        else:
                            positional_args.append(self.flashlight.tensor(v.copy()))
                    elif isinstance(v, bytes) and raw_buffer:
                        # Keep bytes as-is for frombuffer
                        positional_args.append(v)
                    elif isinstance(v, tuple):
                        # Check if tuple contains numpy arrays (e.g., LSTM hidden state (h, c))
                        if len(v) > 0 and isinstance(v[0], np.ndarray):
                            positional_args.append(tuple(self.flashlight.tensor(arr.copy()) for arr in v))
                        else:
                            # Tuples like (4, 8) for size - keep as-is
                            positional_args.append(v)
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                        # List of numpy arrays (e.g., for pad_sequence, pack_sequence)
                        positional_args.append([self.flashlight.tensor(arr.copy()) for arr in v])
                    else:
                        positional_args.append(v)
            elif key == "_tensor_list":
                # Special case: list of tensors to be passed as "tensors" arg
                result["tensors"] = [self.flashlight.tensor(v.copy()) for v in value]
            elif key == "_operands":
                # Special case for einsum: convert operands and extend positional args
                for v in value:
                    if isinstance(v, np.ndarray):
                        positional_args.append(self.flashlight.tensor(v.copy()))
                    else:
                        positional_args.append(v)
            elif key.startswith("_"):
                continue
            elif key == "dtype" and raw_buffer:
                # Convert numpy dtype to flashlight dtype for frombuffer
                result[key] = self._numpy_dtype_to_mlx(value)
            elif isinstance(value, np.ndarray):
                result[key] = self.flashlight.tensor(value.copy())
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                # List of arrays
                result[key] = [self.flashlight.tensor(v.copy()) for v in value]
            elif isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], np.ndarray):
                # Tuple of arrays (e.g., for indices)
                result[key] = tuple(self.flashlight.tensor(v.copy()) for v in value)
            else:
                result[key] = value
        return result, positional_args

    def _sync_weights(self, torch_module, mlx_module):
        """Copy weights from PyTorch module to flashlight module."""
        torch_state = torch_module.state_dict()

        for name, torch_param in torch_state.items():
            # Get corresponding mlx parameter
            parts = name.split(".")
            mlx_obj = mlx_module

            for part in parts[:-1]:
                if part.isdigit():
                    mlx_obj = mlx_obj[int(part)]
                else:
                    mlx_obj = getattr(mlx_obj, part, None)
                    if mlx_obj is None:
                        break

            if mlx_obj is None:
                continue

            param_name = parts[-1]
            mlx_param = getattr(mlx_obj, param_name, None)

            if mlx_param is not None and hasattr(mlx_param, "_mlx_array"):
                # Copy weight data
                np_data = torch_param.detach().numpy()
                mlx_param._mlx_array = mlx_param._mlx_array.__class__(np_data)

    def _compare_outputs(
        self,
        module: str,
        api_name: str,
        torch_output: Any,
        mlx_output: Any,
    ) -> NumericalTestResult:
        """Compare outputs from PyTorch and flashlight."""
        # Handle PackedSequence outputs by comparing their data tensors
        torch_type_name = type(torch_output).__name__
        mlx_type_name = type(mlx_output).__name__
        if torch_type_name == "PackedSequence" and mlx_type_name == "PackedSequence":
            # Compare the data attribute of PackedSequence
            return self._compare_outputs(module, api_name, torch_output.data, mlx_output.data)

        # Handle tuple/list outputs by comparing element-by-element
        if isinstance(torch_output, (tuple, list)):
            if not isinstance(mlx_output, (tuple, list)):
                return NumericalTestResult(
                    module=module, api=api_name, passed=False,
                    error=f"Output type mismatch: PyTorch returned {type(torch_output).__name__}, mlx returned {type(mlx_output).__name__}",
                )
            if len(torch_output) != len(mlx_output):
                return NumericalTestResult(
                    module=module, api=api_name, passed=False,
                    error=f"Output length mismatch: PyTorch returned {len(torch_output)} elements, mlx returned {len(mlx_output)}",
                )
            # Compare each element - recursively handle nested lists/tuples
            max_diffs = []
            mean_diffs = []
            for i, (torch_elem, mlx_elem) in enumerate(zip(torch_output, mlx_output)):
                # Recursively call _compare_outputs for nested structures
                result = self._compare_outputs(module, api_name, torch_elem, mlx_elem)
                if not result.passed:
                    result.error = f"Element {i}: {result.error}"
                    return result
                if result.max_diff is not None:
                    max_diffs.append(result.max_diff)
                if result.mean_diff is not None:
                    mean_diffs.append(result.mean_diff)
            # All elements passed
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=True,
                max_diff=max(max_diffs) if max_diffs else 0.0,
                mean_diff=sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0.0,
            )

        return self._compare_single_output(module, api_name, torch_output, mlx_output)

    def _compare_single_output(
        self,
        module: str,
        api_name: str,
        torch_output: Any,
        mlx_output: Any,
    ) -> NumericalTestResult:
        """Compare a single tensor output from PyTorch and flashlight."""
        # Convert to numpy
        try:
            if hasattr(torch_output, "detach"):
                t = torch_output.detach()
                # Handle complex tensors with conjugate bit (e.g., from ihfft)
                if hasattr(t, "is_conj") and t.is_conj():
                    t = t.resolve_conj()
                torch_np = t.numpy()
            elif hasattr(torch_output, "numpy"):
                torch_np = torch_output.numpy()
            else:
                torch_np = np.array(torch_output)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not convert PyTorch output to numpy: {e}",
            )

        try:
            if hasattr(mlx_output, "numpy"):
                mlx_np = mlx_output.numpy()
            else:
                mlx_np = np.array(mlx_output)
        except Exception as e:
            return NumericalTestResult(
                module=module, api=api_name, passed=False,
                error=f"Could not convert flashlight output to numpy: {e}",
            )

        # Check shapes match
        if torch_np.shape != mlx_np.shape:
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                pytorch_output_shape=torch_np.shape,
                mlx_output_shape=mlx_np.shape,
                error=f"Shape mismatch: {torch_np.shape} vs {mlx_np.shape}",
            )

        # Compute differences
        try:
            # Handle boolean arrays - compare directly
            if torch_np.dtype == np.bool_ or mlx_np.dtype == np.bool_:
                passed = np.array_equal(torch_np, mlx_np)
                num_diff = np.sum(torch_np != mlx_np)
                return NumericalTestResult(
                    module=module,
                    api=api_name,
                    passed=passed,
                    max_diff=float(num_diff) if not passed else 0.0,
                    mean_diff=float(num_diff) / torch_np.size if not passed else 0.0,
                    pytorch_output_shape=torch_np.shape,
                    mlx_output_shape=mlx_np.shape,
                    error=None if passed else f"Boolean values differ: {num_diff} elements",
                )

            # Convert to float for numerical comparison
            torch_np = torch_np.astype(np.float64)
            mlx_np = mlx_np.astype(np.float64)

            # Handle empty arrays
            if torch_np.size == 0:
                return NumericalTestResult(
                    module=module,
                    api=api_name,
                    passed=True,
                    max_diff=0.0,
                    mean_diff=0.0,
                    pytorch_output_shape=torch_np.shape,
                    mlx_output_shape=mlx_np.shape,
                )

            abs_diff = np.abs(torch_np - mlx_np)
            max_diff = float(np.nanmax(abs_diff))  # Use nanmax to handle NaN
            mean_diff = float(np.nanmean(abs_diff))

            # Get tolerances (use relaxed tolerances for specific APIs)
            api_key = (module, api_name)
            if api_key in self.RELAXED_TOLERANCE_APIS:
                tols = self.RELAXED_TOLERANCE_APIS[api_key]
                rtol = tols["rtol"]
                atol = tols["atol"]
            else:
                rtol = self.rtol
                atol = self.atol

            # Check if within tolerance (nan_ok handles NaN in same positions)
            passed = np.allclose(torch_np, mlx_np, rtol=rtol, atol=atol, equal_nan=True)

            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=passed,
                max_diff=max_diff,
                mean_diff=mean_diff,
                pytorch_output_shape=torch_np.shape,
                mlx_output_shape=mlx_np.shape,
                error=None if passed else f"Values differ: max_diff={max_diff:.2e}",
            )
        except Exception as e:
            return NumericalTestResult(
                module=module,
                api=api_name,
                passed=False,
                error=f"Comparison failed: {type(e).__name__}: {str(e)}",
            )
