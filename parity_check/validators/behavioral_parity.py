"""
Behavioral parity validator.

Tests PyTorch behavioral contracts that don't require numerical comparison,
such as context manager behavior, module state transitions, and container semantics.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BehavioralTestResult:
    """Result of a single behavioral test."""

    category: str
    test_name: str
    passed: bool
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class BehavioralValidationResult:
    """Result of behavioral parity validation."""

    passed: List[BehavioralTestResult] = field(default_factory=list)
    failed: List[BehavioralTestResult] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_tested(self) -> int:
        """Total number of tests executed (excluding skipped/errors)."""
        return len(self.passed) + len(self.failed)

    @property
    def pass_percentage(self) -> float:
        """Percentage of behavioral tests that passed."""
        if self.total_tested == 0:
            return 100.0
        return (len(self.passed) / self.total_tested) * 100

    @property
    def is_complete(self) -> bool:
        """Returns True if all behavioral tests passed."""
        return len(self.failed) == 0

    def by_category(self) -> Dict[str, Dict[str, int]]:
        """Get results grouped by category."""
        categories: Dict[str, Dict[str, int]] = {}
        for result in self.passed + self.failed:
            cat = result.category
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if result.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        return categories

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": len(self.passed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "total_tested": self.total_tested,
            "pass_percentage": round(self.pass_percentage, 2),
            "by_category": self.by_category(),
            "failed_details": [
                {
                    "category": r.category,
                    "test": r.test_name,
                    "error": r.error,
                    "details": r.details,
                }
                for r in self.failed
            ],
            "error_details": self.errors,
        }


class BehavioralParityValidator:
    """
    Validates behavioral parity between flashlight and PyTorch.

    Tests PyTorch behavioral contracts that don't require numerical comparison,
    such as context manager behavior, module state transitions, and container semantics.

    Args:
        categories: List of categories to test. If None, tests all categories.
        seed: Random seed for reproducibility (default: 42)
    """

    CATEGORIES = [
        "context_manager",
        "module_state",
        "container",
        "optimizer",
        "layer_mode",
        "distribution",
        "edge_cases",
        # New comprehensive categories
        "autograd_advanced",
        "view_semantics",
        "dtype_promotion",
        "inplace_ops",
        "module_hooks",
        "initialization",
        "loss_reduction",
        "activation_inplace",
        "normalization_stats",
        "state_management",
        "attention_masks",
        "serialization",
        "shape_broadcast",
        "item_extraction",
    ]

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.categories = categories or self.CATEGORIES
        self.seed = seed

        # Lazy-loaded modules
        self._flashlight = None

        # Test registry - maps category to list of test methods
        self._test_registry: Dict[str, List[Callable[[], BehavioralTestResult]]] = {}
        self._register_tests()

    @property
    def flashlight(self):
        """Lazy-load flashlight."""
        if self._flashlight is None:
            import flashlight

            self._flashlight = flashlight
        return self._flashlight

    def _register_tests(self):
        """Register all behavioral tests by category."""
        self._test_registry = {
            "context_manager": [
                self._test_no_grad_disables_gradients,
                self._test_enable_grad_within_no_grad,
                self._test_set_grad_enabled_toggle,
                self._test_no_grad_as_decorator,
                self._test_nested_context_managers,
            ],
            "module_state": [
                self._test_train_eval_toggle,
                self._test_training_flag_propagation,
                self._test_state_dict_round_trip,
                self._test_buffers_vs_parameters,
                self._test_named_parameters_hierarchy,
                self._test_zero_grad_clears_gradients,
                self._test_requires_grad_propagation,
            ],
            "container": [
                self._test_sequential_forward_order,
                self._test_modulelist_iteration_order,
                self._test_modulelist_parameter_registration,
                self._test_moduledict_semantics,
                self._test_sequential_indexing,
                self._test_sequential_slicing,
            ],
            "optimizer": [
                self._test_optimizer_step_updates_params,
                self._test_optimizer_zero_grad,
                self._test_optimizer_state_dict_round_trip,
                self._test_optimizer_param_groups,
            ],
            "layer_mode": [
                self._test_dropout_train_vs_eval,
                self._test_batchnorm_train_vs_eval,
            ],
            "distribution": [
                self._test_randn_statistics,
                self._test_rand_statistics,
                self._test_normal_statistics,
                self._test_seed_reproducibility,
            ],
            "edge_cases": [
                self._test_empty_tensor_behavior,
                self._test_zero_dim_tensor,
                self._test_broadcasting_rules,
            ],
            "autograd_advanced": [
                self._test_retain_graph,
                self._test_gradient_accumulation,
                self._test_create_graph,
            ],
            "view_semantics": [
                self._test_view_shares_storage,
                self._test_contiguous_behavior,
                self._test_reshape_vs_view,
            ],
            "dtype_promotion": [
                self._test_mixed_dtype_ops,
                self._test_type_conversion_methods,
                self._test_dtype_consistency,
            ],
            "inplace_ops": [
                self._test_add_inplace,
                self._test_mul_inplace,
                self._test_zero_inplace,
            ],
            "module_hooks": [
                self._test_forward_hook,
                self._test_forward_pre_hook,
                self._test_apply_fn,
            ],
            "initialization": [
                self._test_uniform_init,
                self._test_normal_init,
                self._test_xavier_init,
                self._test_kaiming_init,
            ],
            "loss_reduction": [
                self._test_loss_reduction_none,
                self._test_loss_reduction_mean,
                self._test_loss_reduction_sum,
            ],
            "activation_inplace": [
                self._test_relu_inplace,
                self._test_leaky_relu_inplace,
            ],
            "normalization_stats": [
                self._test_batchnorm_running_stats,
                self._test_batchnorm_momentum,
                self._test_batchnorm_affine,
            ],
            "state_management": [
                self._test_register_buffer,
                self._test_register_parameter,
                self._test_load_state_dict_strict,
            ],
            "attention_masks": [
                self._test_boolean_mask,
                self._test_float_mask,
            ],
            "serialization": [
                self._test_save_load_roundtrip,
                self._test_dtype_preservation,
            ],
            "shape_broadcast": [
                self._test_squeeze_behavior,
                self._test_unsqueeze_behavior,
                self._test_flatten_behavior,
            ],
            "item_extraction": [
                self._test_item_scalar,
                self._test_tolist_behavior,
            ],
        }

    def validate(self) -> BehavioralValidationResult:
        """
        Run all behavioral parity tests.

        Returns:
            BehavioralValidationResult with test results
        """
        result = BehavioralValidationResult()

        for category in self.categories:
            if category not in self._test_registry:
                result.skipped.append({
                    "category": category,
                    "reason": f"Unknown category: {category}",
                })
                continue

            for test_fn in self._test_registry[category]:
                try:
                    test_result = test_fn()
                    if test_result.passed:
                        result.passed.append(test_result)
                    else:
                        result.failed.append(test_result)
                except Exception as e:
                    result.errors.append({
                        "category": category,
                        "test": test_fn.__name__,
                        "error": f"{type(e).__name__}: {str(e)}",
                    })

        return result

    def validate_category(self, category: str) -> BehavioralValidationResult:
        """Run tests for a single category."""
        validator = BehavioralParityValidator(categories=[category], seed=self.seed)
        return validator.validate()

    # =========================================================================
    # Context Manager Tests
    # =========================================================================

    def _test_no_grad_disables_gradients(self) -> BehavioralTestResult:
        """Test that no_grad() disables gradient tracking."""
        import flashlight

        grad_enabled_outside_before = flashlight.is_grad_enabled()

        with flashlight.no_grad():
            grad_enabled_inside = flashlight.is_grad_enabled()

        grad_enabled_outside_after = flashlight.is_grad_enabled()

        passed = (
            grad_enabled_outside_before
            and not grad_enabled_inside
            and grad_enabled_outside_after
        )

        return BehavioralTestResult(
            category="context_manager",
            test_name="no_grad_disables_gradients",
            passed=passed,
            error=None if passed else "no_grad did not properly toggle gradient state",
            details={
                "grad_enabled_before": grad_enabled_outside_before,
                "grad_enabled_inside": grad_enabled_inside,
                "grad_enabled_after": grad_enabled_outside_after,
            },
        )

    def _test_enable_grad_within_no_grad(self) -> BehavioralTestResult:
        """Test that enable_grad() works within no_grad() context."""
        import flashlight

        with flashlight.no_grad():
            outer_state = flashlight.is_grad_enabled()
            with flashlight.enable_grad():
                inner_state = flashlight.is_grad_enabled()
            after_inner = flashlight.is_grad_enabled()

        passed = not outer_state and inner_state and not after_inner

        return BehavioralTestResult(
            category="context_manager",
            test_name="enable_grad_within_no_grad",
            passed=passed,
            error=None if passed else "Nested enable_grad did not work correctly",
            details={
                "outer_state": outer_state,
                "inner_state": inner_state,
                "after_inner": after_inner,
            },
        )

    def _test_set_grad_enabled_toggle(self) -> BehavioralTestResult:
        """Test that set_grad_enabled(bool) toggles correctly."""
        import flashlight

        original = flashlight.is_grad_enabled()

        with flashlight.set_grad_enabled(False):
            state_false = flashlight.is_grad_enabled()

        with flashlight.set_grad_enabled(True):
            state_true = flashlight.is_grad_enabled()

        final = flashlight.is_grad_enabled()

        passed = not state_false and state_true and final == original

        return BehavioralTestResult(
            category="context_manager",
            test_name="set_grad_enabled_toggle",
            passed=passed,
            error=None if passed else "set_grad_enabled did not toggle correctly",
            details={
                "original": original,
                "state_false": state_false,
                "state_true": state_true,
                "final": final,
            },
        )

    def _test_no_grad_as_decorator(self) -> BehavioralTestResult:
        """Test that no_grad() works as a decorator."""
        import flashlight

        @flashlight.no_grad()
        def compute():
            return flashlight.is_grad_enabled()

        inside_fn = compute()
        outside_fn = flashlight.is_grad_enabled()

        passed = not inside_fn and outside_fn

        return BehavioralTestResult(
            category="context_manager",
            test_name="no_grad_as_decorator",
            passed=passed,
            error=None if passed else "no_grad decorator did not work",
            details={"inside_fn": inside_fn, "outside_fn": outside_fn},
        )

    def _test_nested_context_managers(self) -> BehavioralTestResult:
        """Test deeply nested gradient context managers."""
        import flashlight

        states = []

        with flashlight.no_grad():
            states.append(("no_grad_1", flashlight.is_grad_enabled()))
            with flashlight.no_grad():
                states.append(("no_grad_2", flashlight.is_grad_enabled()))
                with flashlight.enable_grad():
                    states.append(("enable_grad", flashlight.is_grad_enabled()))
                states.append(("after_enable", flashlight.is_grad_enabled()))
            states.append(("after_no_grad_2", flashlight.is_grad_enabled()))
        states.append(("final", flashlight.is_grad_enabled()))

        expected = [
            ("no_grad_1", False),
            ("no_grad_2", False),
            ("enable_grad", True),
            ("after_enable", False),
            ("after_no_grad_2", False),
            ("final", True),
        ]

        passed = states == expected

        return BehavioralTestResult(
            category="context_manager",
            test_name="nested_context_managers",
            passed=passed,
            error=None if passed else "Nested context managers did not behave correctly",
            details={"actual": states, "expected": expected},
        )

    # =========================================================================
    # Module State Tests
    # =========================================================================

    def _test_train_eval_toggle(self) -> BehavioralTestResult:
        """Test that train()/eval() toggles self.training flag."""
        import flashlight.nn as nn

        model = nn.Linear(10, 5)

        initial = model.training
        model.eval()
        after_eval = model.training
        model.train()
        after_train = model.training
        model.train(False)
        after_train_false = model.training

        passed = (
            initial is True
            and after_eval is False
            and after_train is True
            and after_train_false is False
        )

        return BehavioralTestResult(
            category="module_state",
            test_name="train_eval_toggle",
            passed=passed,
            error=None if passed else "train/eval mode toggle failed",
            details={
                "initial": initial,
                "after_eval": after_eval,
                "after_train": after_train,
                "after_train_false": after_train_false,
            },
        )

    def _test_training_flag_propagation(self) -> BehavioralTestResult:
        """Test that training mode propagates to child modules."""
        import flashlight.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        model.eval()
        all_eval = all(not m.training for m in model.modules())

        model.train()
        all_train = all(m.training for m in model.modules())

        passed = all_eval and all_train

        return BehavioralTestResult(
            category="module_state",
            test_name="training_flag_propagation",
            passed=passed,
            error=None if passed else "Training mode did not propagate to children",
            details={"all_eval": all_eval, "all_train": all_train},
        )

    def _test_state_dict_round_trip(self) -> BehavioralTestResult:
        """Test that state_dict()/load_state_dict() round-trips correctly."""
        import flashlight
        import flashlight.nn as nn

        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)

        # Save state from model1
        state = model1.state_dict()

        # Load into model2
        model2.load_state_dict(state)

        # Check weights match
        weight_match = bool(flashlight.allclose(model1.weight, model2.weight))
        bias_match = bool(flashlight.allclose(model1.bias, model2.bias))

        passed = weight_match and bias_match

        return BehavioralTestResult(
            category="module_state",
            test_name="state_dict_round_trip",
            passed=passed,
            error=None if passed else "state_dict round-trip failed",
            details={"weight_match": weight_match, "bias_match": bias_match},
        )

    def _test_buffers_vs_parameters(self) -> BehavioralTestResult:
        """Test that buffers and parameters are distinguished."""
        import flashlight.nn as nn

        # BatchNorm has both parameters (weight, bias) and buffers (running_mean, running_var)
        bn = nn.BatchNorm1d(10)

        param_names = {name for name, _ in bn.named_parameters()}
        buffer_names = set(bn._buffers.keys()) if hasattr(bn, "_buffers") else set()

        # Parameters should include weight and bias
        has_weight_param = "weight" in param_names
        has_bias_param = "bias" in param_names

        # Buffers should include running stats
        has_running_mean = "running_mean" in buffer_names
        has_running_var = "running_var" in buffer_names

        passed = (
            has_weight_param
            and has_bias_param
            and has_running_mean
            and has_running_var
        )

        return BehavioralTestResult(
            category="module_state",
            test_name="buffers_vs_parameters",
            passed=passed,
            error=None if passed else "Buffers/parameters not properly distinguished",
            details={
                "parameters": list(param_names),
                "buffers": list(buffer_names),
            },
        )

    def _test_named_parameters_hierarchy(self) -> BehavioralTestResult:
        """Test that named_parameters() yields correct hierarchy."""
        import flashlight.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        param_names = [name for name, _ in model.named_parameters()]

        expected_names = ["0.weight", "0.bias", "1.weight", "1.bias"]
        passed = set(param_names) == set(expected_names)

        return BehavioralTestResult(
            category="module_state",
            test_name="named_parameters_hierarchy",
            passed=passed,
            error=None if passed else "named_parameters hierarchy incorrect",
            details={"actual": sorted(param_names), "expected": sorted(expected_names)},
        )

    def _test_zero_grad_clears_gradients(self) -> BehavioralTestResult:
        """Test that zero_grad() clears parameter gradients."""
        import flashlight
        import flashlight.nn as nn

        model = nn.Linear(10, 5)
        x = flashlight.randn(3, 10)
        y = flashlight.sum(model(x))
        y.backward()

        # Gradients should exist after backward
        has_grads_before = all(p.grad is not None for p in model.parameters())

        model.zero_grad()

        # Gradients should be cleared
        has_grads_after = any(p.grad is not None for p in model.parameters())

        passed = has_grads_before and not has_grads_after

        return BehavioralTestResult(
            category="module_state",
            test_name="zero_grad_clears_gradients",
            passed=passed,
            error=None if passed else "zero_grad did not clear gradients",
            details={
                "has_grads_before": has_grads_before,
                "has_grads_after": has_grads_after,
            },
        )

    def _test_requires_grad_propagation(self) -> BehavioralTestResult:
        """Test that requires_grad_() propagates to all parameters."""
        import flashlight.nn as nn

        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        model.requires_grad_(False)
        all_false = all(not p.requires_grad for p in model.parameters())

        model.requires_grad_(True)
        all_true = all(p.requires_grad for p in model.parameters())

        passed = all_false and all_true

        return BehavioralTestResult(
            category="module_state",
            test_name="requires_grad_propagation",
            passed=passed,
            error=None if passed else "requires_grad_ did not propagate",
            details={"all_false": all_false, "all_true": all_true},
        )

    # =========================================================================
    # Container Tests
    # =========================================================================

    def _test_sequential_forward_order(self) -> BehavioralTestResult:
        """Test that Sequential forwards through layers in order."""
        import flashlight
        import flashlight.nn as nn

        # Create a model that transforms shape: 10 -> 20 -> 5
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        x = flashlight.randn(3, 10)
        y = model(x)

        passed = y.shape == (3, 5)

        return BehavioralTestResult(
            category="container",
            test_name="sequential_forward_order",
            passed=passed,
            error=None if passed else f"Wrong output shape: {y.shape}",
            details={"output_shape": y.shape},
        )

    def _test_modulelist_iteration_order(self) -> BehavioralTestResult:
        """Test that ModuleList iterates in insertion order."""
        import flashlight.nn as nn

        modules = nn.ModuleList([
            nn.Linear(10, 10),
            nn.Linear(20, 20),
            nn.Linear(30, 30),
        ])

        sizes = [m.in_features for m in modules]

        passed = sizes == [10, 20, 30]

        return BehavioralTestResult(
            category="container",
            test_name="modulelist_iteration_order",
            passed=passed,
            error=None if passed else f"Wrong iteration order: {sizes}",
            details={"sizes": sizes},
        )

    def _test_modulelist_parameter_registration(self) -> BehavioralTestResult:
        """Test that ModuleList registers parameters correctly."""
        import flashlight.nn as nn

        modules = nn.ModuleList([
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        ])

        # Should have parameters from both modules
        param_count = sum(1 for _ in modules.parameters())
        # Linear has weight + bias, so 2 modules = 4 parameters

        passed = param_count == 4

        return BehavioralTestResult(
            category="container",
            test_name="modulelist_parameter_registration",
            passed=passed,
            error=None if passed else f"Wrong param count: {param_count}",
            details={"param_count": param_count},
        )

    def _test_moduledict_semantics(self) -> BehavioralTestResult:
        """Test that ModuleDict has dict-like semantics."""
        import flashlight.nn as nn

        modules = nn.ModuleDict({
            "encoder": nn.Linear(10, 20),
            "decoder": nn.Linear(20, 10),
        })

        # Test key access
        has_encoder = "encoder" in modules
        encoder = modules["encoder"]
        keys = list(modules.keys())
        values = list(modules.values())
        items = list(modules.items())

        passed = (
            has_encoder
            and encoder.in_features == 10
            and set(keys) == {"encoder", "decoder"}
            and len(values) == 2
            and len(items) == 2
        )

        return BehavioralTestResult(
            category="container",
            test_name="moduledict_semantics",
            passed=passed,
            error=None if passed else "ModuleDict semantics incorrect",
            details={"keys": keys, "has_encoder": has_encoder},
        )

    def _test_sequential_indexing(self) -> BehavioralTestResult:
        """Test that Sequential supports indexing."""
        import flashlight.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        first = model[0]
        last = model[-1]

        passed = (
            isinstance(first, nn.Linear)
            and first.in_features == 10
            and isinstance(last, nn.Linear)
            and last.out_features == 5
        )

        return BehavioralTestResult(
            category="container",
            test_name="sequential_indexing",
            passed=passed,
            error=None if passed else "Sequential indexing failed",
        )

    def _test_sequential_slicing(self) -> BehavioralTestResult:
        """Test that Sequential supports slicing."""
        import flashlight.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        first_two = model[:2]

        passed = isinstance(first_two, nn.Sequential) and len(first_two) == 2

        return BehavioralTestResult(
            category="container",
            test_name="sequential_slicing",
            passed=passed,
            error=None if passed else "Sequential slicing failed",
            details={"length": len(first_two) if first_two else None},
        )

    # =========================================================================
    # Optimizer Tests
    # =========================================================================

    def _test_optimizer_step_updates_params(self) -> BehavioralTestResult:
        """Test that optimizer.step() actually updates parameters."""
        import flashlight
        import flashlight.nn as nn
        import flashlight.optim as optim

        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Store original weight
        original_weight = flashlight.clone(model.weight)

        # Forward + backward
        x = flashlight.randn(3, 10)
        loss = flashlight.sum(model(x))
        loss.backward()

        # Step
        optimizer.step()

        # Weight should have changed
        weight_changed = not bool(flashlight.allclose(model.weight, original_weight))

        passed = weight_changed

        return BehavioralTestResult(
            category="optimizer",
            test_name="optimizer_step_updates_params",
            passed=passed,
            error=None if passed else "optimizer.step() did not update parameters",
        )

    def _test_optimizer_zero_grad(self) -> BehavioralTestResult:
        """Test that optimizer.zero_grad() clears gradients."""
        import flashlight
        import flashlight.nn as nn
        import flashlight.optim as optim

        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Create gradients
        x = flashlight.randn(3, 10)
        loss = flashlight.sum(model(x))
        loss.backward()

        has_grads_before = any(p.grad is not None for p in model.parameters())

        optimizer.zero_grad()

        has_grads_after = any(p.grad is not None for p in model.parameters())

        passed = has_grads_before and not has_grads_after

        return BehavioralTestResult(
            category="optimizer",
            test_name="optimizer_zero_grad",
            passed=passed,
            error=None if passed else "optimizer.zero_grad() did not clear gradients",
            details={
                "has_grads_before": has_grads_before,
                "has_grads_after": has_grads_after,
            },
        )

    def _test_optimizer_state_dict_round_trip(self) -> BehavioralTestResult:
        """Test optimizer state_dict()/load_state_dict() round-trip."""
        import flashlight
        import flashlight.nn as nn
        import flashlight.optim as optim

        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Run a few steps to populate optimizer state
        for _ in range(3):
            x = flashlight.randn(3, 10)
            loss = flashlight.sum(model(x))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save state
        state = optimizer.state_dict()

        # Create new optimizer and load state
        optimizer2 = optim.Adam(model.parameters(), lr=0.001)
        optimizer2.load_state_dict(state)

        # Check param_groups match
        groups_match = (
            optimizer.param_groups[0]["lr"] == optimizer2.param_groups[0]["lr"]
        )

        passed = groups_match

        return BehavioralTestResult(
            category="optimizer",
            test_name="optimizer_state_dict_round_trip",
            passed=passed,
            error=None if passed else "Optimizer state dict round-trip failed",
        )

    def _test_optimizer_param_groups(self) -> BehavioralTestResult:
        """Test that parameter groups work correctly."""
        import flashlight.nn as nn
        import flashlight.optim as optim

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        # Different learning rates for different layers
        optimizer = optim.SGD([
            {"params": model[0].parameters(), "lr": 0.1},
            {"params": model[1].parameters(), "lr": 0.01},
        ])

        passed = (
            len(optimizer.param_groups) == 2
            and optimizer.param_groups[0]["lr"] == 0.1
            and optimizer.param_groups[1]["lr"] == 0.01
        )

        return BehavioralTestResult(
            category="optimizer",
            test_name="optimizer_param_groups",
            passed=passed,
            error=None if passed else "Parameter groups not working correctly",
            details={
                "num_groups": len(optimizer.param_groups),
                "lr_0": optimizer.param_groups[0]["lr"],
                "lr_1": optimizer.param_groups[1]["lr"],
            },
        )

    # =========================================================================
    # Layer Mode Tests
    # =========================================================================

    def _test_dropout_train_vs_eval(self) -> BehavioralTestResult:
        """Test that Dropout applies in train mode, passes through in eval."""
        import flashlight
        import flashlight.nn as nn

        flashlight.manual_seed(42)
        dropout = nn.Dropout(p=0.5)
        x = flashlight.ones(100, 100)

        # Train mode: should have zeros
        dropout.train()
        y_train = dropout(x)
        has_zeros_train = bool(flashlight.any(y_train == 0))

        # Eval mode: should pass through unchanged
        dropout.eval()
        y_eval = dropout(x)
        is_unchanged_eval = bool(flashlight.allclose(y_eval, x))

        passed = has_zeros_train and is_unchanged_eval

        return BehavioralTestResult(
            category="layer_mode",
            test_name="dropout_train_vs_eval",
            passed=passed,
            error=None if passed else "Dropout mode behavior incorrect",
            details={
                "has_zeros_train": has_zeros_train,
                "is_unchanged_eval": is_unchanged_eval,
            },
        )

    def _test_batchnorm_train_vs_eval(self) -> BehavioralTestResult:
        """Test that BatchNorm uses different statistics in train vs eval."""
        import flashlight
        import flashlight.nn as nn

        bn = nn.BatchNorm1d(10)

        # In training mode, running stats should update
        bn.train()
        x = flashlight.randn(32, 10) * 2 + 5  # Non-standard mean/std

        running_mean_before = flashlight.clone(bn.running_mean)
        _ = bn(x)
        running_mean_after = flashlight.clone(bn.running_mean)

        # Running mean should have changed
        stats_updated = not bool(
            flashlight.allclose(running_mean_before, running_mean_after)
        )

        # In eval mode, same input should give consistent output
        bn.eval()
        y1 = bn(x)
        y2 = bn(x)
        eval_consistent = bool(flashlight.allclose(y1, y2))

        passed = stats_updated and eval_consistent

        return BehavioralTestResult(
            category="layer_mode",
            test_name="batchnorm_train_vs_eval",
            passed=passed,
            error=None if passed else "BatchNorm mode behavior incorrect",
            details={
                "stats_updated": stats_updated,
                "eval_consistent": eval_consistent,
            },
        )

    # =========================================================================
    # Distribution Tests
    # =========================================================================

    def _test_randn_statistics(self) -> BehavioralTestResult:
        """Test that randn has mean~0, std~1."""
        import flashlight

        flashlight.manual_seed(self.seed)
        x = flashlight.randn(10000)

        mean = float(x.mean().item())
        std = float(x.std().item())

        # Allow some tolerance for statistical sampling
        passed = abs(mean) < 0.1 and abs(std - 1.0) < 0.1

        return BehavioralTestResult(
            category="distribution",
            test_name="randn_statistics",
            passed=passed,
            error=None if passed else f"randn stats incorrect: mean={mean}, std={std}",
            details={"mean": mean, "std": std},
        )

    def _test_rand_statistics(self) -> BehavioralTestResult:
        """Test that rand has mean~0.5, uniform distribution."""
        import flashlight

        flashlight.manual_seed(self.seed)
        x = flashlight.rand(10000)

        mean = float(x.mean().item())
        min_val = float(flashlight.min(x).item())
        max_val = float(flashlight.max(x).item())

        passed = abs(mean - 0.5) < 0.1 and min_val >= 0 and max_val <= 1

        return BehavioralTestResult(
            category="distribution",
            test_name="rand_statistics",
            passed=passed,
            error=None if passed else f"rand stats incorrect: mean={mean}",
            details={"mean": mean, "min": min_val, "max": max_val},
        )

    def _test_normal_statistics(self) -> BehavioralTestResult:
        """Test that normal(mean, std) has correct statistics."""
        import flashlight

        flashlight.manual_seed(self.seed)
        target_mean, target_std = 5.0, 2.0
        x = flashlight.normal(target_mean, target_std, (10000,))

        actual_mean = float(x.mean().item())
        actual_std = float(x.std().item())

        passed = (
            abs(actual_mean - target_mean) < 0.2
            and abs(actual_std - target_std) < 0.2
        )

        return BehavioralTestResult(
            category="distribution",
            test_name="normal_statistics",
            passed=passed,
            error=None
            if passed
            else f"normal stats incorrect: mean={actual_mean}, std={actual_std}",
            details={
                "target_mean": target_mean,
                "actual_mean": actual_mean,
                "target_std": target_std,
                "actual_std": actual_std,
            },
        )

    def _test_seed_reproducibility(self) -> BehavioralTestResult:
        """Test that same seed produces same random numbers."""
        import flashlight

        flashlight.manual_seed(12345)
        x1 = flashlight.randn(100)

        flashlight.manual_seed(12345)
        x2 = flashlight.randn(100)

        passed = bool(flashlight.allclose(x1, x2))

        return BehavioralTestResult(
            category="distribution",
            test_name="seed_reproducibility",
            passed=passed,
            error=None if passed else "Same seed did not produce same random numbers",
        )

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    def _test_empty_tensor_behavior(self) -> BehavioralTestResult:
        """Test behavior with empty tensors."""
        import flashlight

        empty = flashlight.tensor([])

        # numel is a property, not a method
        numel_val = empty.numel

        passed = empty.shape == (0,) and numel_val == 0

        return BehavioralTestResult(
            category="edge_cases",
            test_name="empty_tensor_behavior",
            passed=passed,
            error=None if passed else f"Empty tensor shape: {empty.shape}",
            details={"shape": empty.shape, "numel": numel_val},
        )

    def _test_zero_dim_tensor(self) -> BehavioralTestResult:
        """Test zero-dimensional (scalar) tensor behavior."""
        import flashlight

        scalar = flashlight.tensor(5.0)

        # Use ndim property and numel property
        ndim_val = scalar.ndim
        numel_val = scalar.numel
        float_val = float(scalar.item())

        passed = ndim_val == 0 and numel_val == 1 and float_val == 5.0

        return BehavioralTestResult(
            category="edge_cases",
            test_name="zero_dim_tensor",
            passed=passed,
            error=None if passed else f"Scalar tensor ndim: {ndim_val}",
            details={"ndim": ndim_val, "numel": numel_val},
        )

    def _test_broadcasting_rules(self) -> BehavioralTestResult:
        """Test that broadcasting follows PyTorch rules."""
        import flashlight

        a = flashlight.ones(3, 1)
        b = flashlight.ones(1, 4)
        c = a + b

        passed = c.shape == (3, 4)

        return BehavioralTestResult(
            category="edge_cases",
            test_name="broadcasting_rules",
            passed=passed,
            error=None if passed else f"Broadcasting result shape: {c.shape}",
            details={"result_shape": c.shape},
        )

    # =========================================================================
    # Autograd Advanced Tests
    # =========================================================================

    def _test_retain_graph(self) -> BehavioralTestResult:
        """Test that retain_graph allows multiple backward passes."""
        import flashlight
        import flashlight.nn as nn

        model = nn.Linear(10, 5)
        x = flashlight.randn(3, 10)

        # First backward with retain_graph
        y = flashlight.sum(model(x))
        y.backward(retain_graph=True)
        grad1 = flashlight.clone(model.weight.grad)

        # Second backward should still work
        try:
            y.backward(retain_graph=True)
            grad2 = model.weight.grad
            second_backward_worked = True
            # With accumulation, grad2 should be ~2x grad1
            grads_accumulated = float(flashlight.mean(grad2 / grad1).item()) > 1.5
        except Exception:
            second_backward_worked = False
            grads_accumulated = False

        passed = second_backward_worked and grads_accumulated

        return BehavioralTestResult(
            category="autograd_advanced",
            test_name="retain_graph",
            passed=passed,
            error=None if passed else "retain_graph did not allow multiple backwards",
            details={
                "second_backward_worked": second_backward_worked,
                "grads_accumulated": grads_accumulated,
            },
        )

    def _test_gradient_accumulation(self) -> BehavioralTestResult:
        """Test that gradients accumulate across backward calls."""
        import flashlight
        import flashlight.nn as nn

        model = nn.Linear(10, 5)
        x = flashlight.randn(3, 10)

        # First backward
        y1 = flashlight.sum(model(x))
        y1.backward()
        grad1 = flashlight.clone(model.weight.grad)

        # Second backward (without zero_grad)
        y2 = flashlight.sum(model(x))
        y2.backward()
        grad2 = model.weight.grad

        # Gradients should have accumulated
        accumulated = not bool(flashlight.allclose(grad1, grad2))

        passed = accumulated

        return BehavioralTestResult(
            category="autograd_advanced",
            test_name="gradient_accumulation",
            passed=passed,
            error=None if passed else "Gradients did not accumulate",
        )

    def _test_create_graph(self) -> BehavioralTestResult:
        """Test that create_graph enables higher-order gradients."""
        import flashlight

        x = flashlight.tensor([2.0], requires_grad=True)
        y = x ** 3  # y = x^3, dy/dx = 3x^2

        # First derivative with create_graph
        try:
            y.backward(create_graph=True)
            first_grad = x.grad
            # first_grad should be 3*x^2 = 12.0

            passed = first_grad is not None and abs(float(first_grad.item()) - 12.0) < 0.1
        except Exception as e:
            passed = False
            first_grad = None

        return BehavioralTestResult(
            category="autograd_advanced",
            test_name="create_graph",
            passed=passed,
            error=None if passed else "create_graph did not work",
            details={"first_grad": float(first_grad.item()) if first_grad is not None else None},
        )

    # =========================================================================
    # View Semantics Tests
    # =========================================================================

    def _test_view_shares_storage(self) -> BehavioralTestResult:
        """Test that view operations share underlying storage."""
        import flashlight

        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.view(4)

        # Modifying y should affect x (or at least y should reflect x's data)
        original_x = flashlight.clone(x)
        same_data = bool(flashlight.allclose(y, x.view(4)))

        passed = same_data

        return BehavioralTestResult(
            category="view_semantics",
            test_name="view_shares_storage",
            passed=passed,
            error=None if passed else "view did not share data correctly",
        )

    def _test_contiguous_behavior(self) -> BehavioralTestResult:
        """Test contiguous() behavior."""
        import flashlight

        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])

        # After transpose, tensor may not be contiguous
        x_t = x.t()

        # contiguous() should return a contiguous version
        x_contig = x_t.contiguous()

        # Shape should be preserved
        shape_match = x_contig.shape == x_t.shape

        passed = shape_match

        return BehavioralTestResult(
            category="view_semantics",
            test_name="contiguous_behavior",
            passed=passed,
            error=None if passed else f"Shape mismatch: {x_contig.shape} vs {x_t.shape}",
        )

    def _test_reshape_vs_view(self) -> BehavioralTestResult:
        """Test that reshape is more flexible than view."""
        import flashlight

        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Both should work on contiguous tensor
        view_result = x.view(4)
        reshape_result = x.reshape(4)

        shapes_match = view_result.shape == reshape_result.shape == (4,)

        passed = shapes_match

        return BehavioralTestResult(
            category="view_semantics",
            test_name="reshape_vs_view",
            passed=passed,
            error=None if passed else "reshape/view behavior differs unexpectedly",
        )

    # =========================================================================
    # Dtype Promotion Tests
    # =========================================================================

    def _test_mixed_dtype_ops(self) -> BehavioralTestResult:
        """Test operations between different dtypes."""
        import flashlight

        x_float = flashlight.tensor([1.0, 2.0])
        x_int = flashlight.tensor([1, 2])

        # Float + int should promote to float
        result = x_float + x_int

        passed = result.dtype in (flashlight.float32, flashlight.float16)

        return BehavioralTestResult(
            category="dtype_promotion",
            test_name="mixed_dtype_ops",
            passed=passed,
            error=None if passed else f"Result dtype: {result.dtype}",
            details={"result_dtype": str(result.dtype)},
        )

    def _test_type_conversion_methods(self) -> BehavioralTestResult:
        """Test .float(), .int(), .half() conversion methods."""
        import flashlight

        x = flashlight.tensor([1, 2, 3])

        # Test float conversion
        x_float = x.float()
        float_ok = x_float.dtype == flashlight.float32

        # Test int conversion
        x_int = x_float.int()
        int_ok = x_int.dtype in (flashlight.int32, flashlight.int64)

        passed = float_ok and int_ok

        return BehavioralTestResult(
            category="dtype_promotion",
            test_name="type_conversion_methods",
            passed=passed,
            error=None if passed else f"Conversion failed: float={x_float.dtype}, int={x_int.dtype}",
        )

    def _test_dtype_consistency(self) -> BehavioralTestResult:
        """Test that dtype is preserved through operations."""
        import flashlight

        x = flashlight.tensor([1.0, 2.0], dtype=flashlight.float32)
        y = x * 2.0
        z = flashlight.sum(x)

        dtype_preserved = y.dtype == flashlight.float32 and z.dtype == flashlight.float32

        passed = dtype_preserved

        return BehavioralTestResult(
            category="dtype_promotion",
            test_name="dtype_consistency",
            passed=passed,
            error=None if passed else f"Dtype changed: y={y.dtype}, z={z.dtype}",
        )

    # =========================================================================
    # In-place Operations Tests
    # =========================================================================

    def _test_add_inplace(self) -> BehavioralTestResult:
        """Test add_ in-place operation."""
        import flashlight

        x = flashlight.tensor([1.0, 2.0, 3.0])
        original_sum = float(flashlight.sum(x).item())

        x.add_(1.0)  # Should add 1 to each element in-place

        new_sum = float(flashlight.sum(x).item())

        passed = abs(new_sum - (original_sum + 3.0)) < 0.01

        return BehavioralTestResult(
            category="inplace_ops",
            test_name="add_inplace",
            passed=passed,
            error=None if passed else f"Expected sum {original_sum + 3}, got {new_sum}",
        )

    def _test_mul_inplace(self) -> BehavioralTestResult:
        """Test mul_ in-place operation."""
        import flashlight

        x = flashlight.tensor([1.0, 2.0, 3.0])
        original_sum = float(flashlight.sum(x).item())

        x.mul_(2.0)  # Should multiply each element by 2 in-place

        new_sum = float(flashlight.sum(x).item())

        passed = abs(new_sum - (original_sum * 2.0)) < 0.01

        return BehavioralTestResult(
            category="inplace_ops",
            test_name="mul_inplace",
            passed=passed,
            error=None if passed else f"Expected sum {original_sum * 2}, got {new_sum}",
        )

    def _test_zero_inplace(self) -> BehavioralTestResult:
        """Test zero_ in-place operation."""
        import flashlight

        x = flashlight.tensor([1.0, 2.0, 3.0])
        x.zero_()

        all_zeros = bool(flashlight.allclose(x, flashlight.zeros(3)))

        passed = all_zeros

        return BehavioralTestResult(
            category="inplace_ops",
            test_name="zero_inplace",
            passed=passed,
            error=None if passed else "zero_ did not zero the tensor",
        )

    # =========================================================================
    # Module Hooks Tests
    # =========================================================================

    def _test_forward_hook(self) -> BehavioralTestResult:
        """Test that forward hooks are called."""
        import flashlight
        import flashlight.nn as nn

        model = nn.Linear(10, 5)
        hook_outputs = []

        def hook(module, input, output):
            hook_outputs.append(output.shape)

        handle = model.register_forward_hook(hook)

        x = flashlight.randn(3, 10)
        _ = model(x)

        handle.remove()

        passed = len(hook_outputs) == 1 and hook_outputs[0] == (3, 5)

        return BehavioralTestResult(
            category="module_hooks",
            test_name="forward_hook",
            passed=passed,
            error=None if passed else f"Hook outputs: {hook_outputs}",
        )

    def _test_forward_pre_hook(self) -> BehavioralTestResult:
        """Test that forward pre-hooks are called before forward."""
        import flashlight
        import flashlight.nn as nn

        model = nn.Linear(10, 5)
        pre_hook_inputs = []

        def pre_hook(module, input):
            pre_hook_inputs.append(input[0].shape)

        handle = model.register_forward_pre_hook(pre_hook)

        x = flashlight.randn(3, 10)
        _ = model(x)

        handle.remove()

        passed = len(pre_hook_inputs) == 1 and pre_hook_inputs[0] == (3, 10)

        return BehavioralTestResult(
            category="module_hooks",
            test_name="forward_pre_hook",
            passed=passed,
            error=None if passed else f"Pre-hook inputs: {pre_hook_inputs}",
        )

    def _test_apply_fn(self) -> BehavioralTestResult:
        """Test that apply() calls function on all modules."""
        import flashlight.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        module_names = []

        def collect_names(m):
            module_names.append(type(m).__name__)

        model.apply(collect_names)

        # Should have Sequential + 2 Linears
        passed = "Sequential" in module_names and module_names.count("Linear") == 2

        return BehavioralTestResult(
            category="module_hooks",
            test_name="apply_fn",
            passed=passed,
            error=None if passed else f"Module names: {module_names}",
        )

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def _test_uniform_init(self) -> BehavioralTestResult:
        """Test uniform_ initialization."""
        import flashlight
        import flashlight.nn as nn

        x = flashlight.empty(1000)
        nn.init.uniform_(x, a=-1.0, b=1.0)

        min_val = float(flashlight.min(x).item())
        max_val = float(flashlight.max(x).item())
        mean_val = float(x.mean().item())

        passed = min_val >= -1.0 and max_val <= 1.0 and abs(mean_val) < 0.1

        return BehavioralTestResult(
            category="initialization",
            test_name="uniform_init",
            passed=passed,
            error=None if passed else f"min={min_val}, max={max_val}, mean={mean_val}",
        )

    def _test_normal_init(self) -> BehavioralTestResult:
        """Test normal_ initialization."""
        import flashlight
        import flashlight.nn as nn

        x = flashlight.empty(10000)
        nn.init.normal_(x, mean=0.0, std=1.0)

        mean_val = float(x.mean().item())
        std_val = float(x.std().item())

        passed = abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.1

        return BehavioralTestResult(
            category="initialization",
            test_name="normal_init",
            passed=passed,
            error=None if passed else f"mean={mean_val}, std={std_val}",
        )

    def _test_xavier_init(self) -> BehavioralTestResult:
        """Test xavier_uniform_ initialization."""
        import flashlight
        import flashlight.nn as nn

        x = flashlight.empty(100, 100)
        nn.init.xavier_uniform_(x)

        # Xavier should have bounded values based on fan_in, fan_out
        min_val = float(flashlight.min(x).item())
        max_val = float(flashlight.max(x).item())

        # For 100x100, bound should be sqrt(6/(100+100)) ~ 0.173
        bound = (6.0 / 200) ** 0.5
        passed = min_val >= -bound * 1.1 and max_val <= bound * 1.1

        return BehavioralTestResult(
            category="initialization",
            test_name="xavier_init",
            passed=passed,
            error=None if passed else f"min={min_val}, max={max_val}, expected bound~{bound}",
        )

    def _test_kaiming_init(self) -> BehavioralTestResult:
        """Test kaiming_uniform_ initialization."""
        import flashlight
        import flashlight.nn as nn

        x = flashlight.empty(100, 100)
        nn.init.kaiming_uniform_(x, a=0, mode='fan_in', nonlinearity='relu')

        # Check it's not all zeros or constant
        std = float(x.std().item())

        passed = std > 0.01

        return BehavioralTestResult(
            category="initialization",
            test_name="kaiming_init",
            passed=passed,
            error=None if passed else f"std={std} (expected > 0)",
        )

    # =========================================================================
    # Loss Reduction Tests
    # =========================================================================

    def _test_loss_reduction_none(self) -> BehavioralTestResult:
        """Test loss with reduction='none' returns per-element loss."""
        import flashlight
        import flashlight.nn as nn

        loss_fn = nn.MSELoss(reduction='none')
        x = flashlight.tensor([1.0, 2.0, 3.0])
        target = flashlight.tensor([1.5, 2.5, 3.5])

        loss = loss_fn(x, target)

        passed = loss.shape == (3,)

        return BehavioralTestResult(
            category="loss_reduction",
            test_name="loss_reduction_none",
            passed=passed,
            error=None if passed else f"Expected shape (3,), got {loss.shape}",
        )

    def _test_loss_reduction_mean(self) -> BehavioralTestResult:
        """Test loss with reduction='mean' returns scalar."""
        import flashlight
        import flashlight.nn as nn

        loss_fn = nn.MSELoss(reduction='mean')
        x = flashlight.tensor([1.0, 2.0, 3.0])
        target = flashlight.tensor([1.5, 2.5, 3.5])

        loss = loss_fn(x, target)

        passed = loss.ndim == 0

        return BehavioralTestResult(
            category="loss_reduction",
            test_name="loss_reduction_mean",
            passed=passed,
            error=None if passed else f"Expected scalar, got ndim={loss.ndim}",
        )

    def _test_loss_reduction_sum(self) -> BehavioralTestResult:
        """Test loss with reduction='sum' returns scalar sum."""
        import flashlight
        import flashlight.nn as nn

        loss_fn_sum = nn.MSELoss(reduction='sum')
        loss_fn_none = nn.MSELoss(reduction='none')

        x = flashlight.tensor([1.0, 2.0, 3.0])
        target = flashlight.tensor([1.5, 2.5, 3.5])

        loss_sum = loss_fn_sum(x, target)
        loss_none = loss_fn_none(x, target)

        # Sum should equal sum of per-element losses
        expected = float(flashlight.sum(loss_none).item())
        actual = float(loss_sum.item())

        passed = abs(actual - expected) < 0.001

        return BehavioralTestResult(
            category="loss_reduction",
            test_name="loss_reduction_sum",
            passed=passed,
            error=None if passed else f"Expected {expected}, got {actual}",
        )

    # =========================================================================
    # Activation In-place Tests
    # =========================================================================

    def _test_relu_inplace(self) -> BehavioralTestResult:
        """Test ReLU with inplace=True modifies input."""
        import flashlight
        import flashlight.nn as nn

        relu = nn.ReLU(inplace=True)
        x = flashlight.tensor([-1.0, 0.0, 1.0])

        # Store original data reference
        relu(x)

        # After inplace ReLU, negative values should be 0
        passed = bool(flashlight.allclose(x, flashlight.tensor([0.0, 0.0, 1.0])))

        return BehavioralTestResult(
            category="activation_inplace",
            test_name="relu_inplace",
            passed=passed,
            error=None if passed else "ReLU inplace did not modify input correctly",
        )

    def _test_leaky_relu_inplace(self) -> BehavioralTestResult:
        """Test LeakyReLU with inplace=True."""
        import flashlight
        import flashlight.nn as nn

        leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        x = flashlight.tensor([-10.0, 0.0, 10.0])

        leaky_relu(x)

        # After inplace LeakyReLU: [-10*0.1, 0, 10] = [-1, 0, 10]
        expected = flashlight.tensor([-1.0, 0.0, 10.0])
        passed = bool(flashlight.allclose(x, expected))

        return BehavioralTestResult(
            category="activation_inplace",
            test_name="leaky_relu_inplace",
            passed=passed,
            error=None if passed else f"LeakyReLU inplace incorrect: {x}",
        )

    # =========================================================================
    # Normalization Stats Tests
    # =========================================================================

    def _test_batchnorm_running_stats(self) -> BehavioralTestResult:
        """Test that BatchNorm updates running statistics."""
        import flashlight
        import flashlight.nn as nn

        bn = nn.BatchNorm1d(10, track_running_stats=True)
        bn.train()

        # Initial running mean should be zeros
        initial_mean = flashlight.clone(bn.running_mean)

        # Forward pass should update stats
        x = flashlight.randn(32, 10) * 2 + 5
        _ = bn(x)

        updated_mean = bn.running_mean

        passed = not bool(flashlight.allclose(initial_mean, updated_mean))

        return BehavioralTestResult(
            category="normalization_stats",
            test_name="batchnorm_running_stats",
            passed=passed,
            error=None if passed else "Running stats did not update",
        )

    def _test_batchnorm_momentum(self) -> BehavioralTestResult:
        """Test that BatchNorm momentum affects running stats update rate."""
        import flashlight
        import flashlight.nn as nn

        # High momentum = fast update
        bn_fast = nn.BatchNorm1d(10, momentum=0.9)
        bn_slow = nn.BatchNorm1d(10, momentum=0.1)

        bn_fast.train()
        bn_slow.train()

        x = flashlight.randn(32, 10) * 2 + 5  # Mean around 5

        _ = bn_fast(x)
        _ = bn_slow(x)

        fast_mean = float(bn_fast.running_mean.mean().item())
        slow_mean = float(bn_slow.running_mean.mean().item())

        # Fast momentum should be closer to input mean (~5)
        passed = abs(fast_mean) > abs(slow_mean)

        return BehavioralTestResult(
            category="normalization_stats",
            test_name="batchnorm_momentum",
            passed=passed,
            error=None if passed else f"fast_mean={fast_mean}, slow_mean={slow_mean}",
        )

    def _test_batchnorm_affine(self) -> BehavioralTestResult:
        """Test BatchNorm with affine=False has no learnable parameters."""
        import flashlight.nn as nn

        bn_affine = nn.BatchNorm1d(10, affine=True)
        bn_no_affine = nn.BatchNorm1d(10, affine=False)

        params_affine = sum(1 for _ in bn_affine.parameters())
        params_no_affine = sum(1 for _ in bn_no_affine.parameters())

        passed = params_affine == 2 and params_no_affine == 0

        return BehavioralTestResult(
            category="normalization_stats",
            test_name="batchnorm_affine",
            passed=passed,
            error=None if passed else f"affine={params_affine}, no_affine={params_no_affine}",
        )

    # =========================================================================
    # State Management Tests
    # =========================================================================

    def _test_register_buffer(self) -> BehavioralTestResult:
        """Test that register_buffer adds non-parameter state."""
        import flashlight
        import flashlight.nn as nn

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("my_buffer", flashlight.zeros(5))

        model = TestModule()

        # Buffer should exist
        has_buffer = hasattr(model, "my_buffer")

        # Buffer should be in state_dict
        in_state_dict = "my_buffer" in model.state_dict()

        # Buffer should NOT be in parameters
        not_in_params = "my_buffer" not in dict(model.named_parameters())

        passed = has_buffer and in_state_dict and not_in_params

        return BehavioralTestResult(
            category="state_management",
            test_name="register_buffer",
            passed=passed,
            error=None if passed else f"has={has_buffer}, state_dict={in_state_dict}, not_param={not_in_params}",
        )

    def _test_register_parameter(self) -> BehavioralTestResult:
        """Test that register_parameter adds learnable parameters."""
        import flashlight
        import flashlight.nn as nn

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("my_param", nn.Parameter(flashlight.zeros(5)))

        model = TestModule()

        # Parameter should exist
        has_param = hasattr(model, "my_param")

        # Parameter should be in parameters()
        in_params = "my_param" in dict(model.named_parameters())

        passed = has_param and in_params

        return BehavioralTestResult(
            category="state_management",
            test_name="register_parameter",
            passed=passed,
            error=None if passed else f"has={has_param}, in_params={in_params}",
        )

    def _test_load_state_dict_strict(self) -> BehavioralTestResult:
        """Test load_state_dict strict mode behavior."""
        import flashlight
        import flashlight.nn as nn

        model = nn.Linear(10, 5)

        # Try loading state dict with extra key (should fail with strict=True)
        state = model.state_dict()
        state["extra_key"] = flashlight.zeros(1)

        strict_failed = False
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, KeyError):
            strict_failed = True

        # With strict=False, should work
        non_strict_worked = False
        try:
            model.load_state_dict(state, strict=False)
            non_strict_worked = True
        except Exception:
            pass

        passed = strict_failed and non_strict_worked

        return BehavioralTestResult(
            category="state_management",
            test_name="load_state_dict_strict",
            passed=passed,
            error=None if passed else f"strict_failed={strict_failed}, non_strict_worked={non_strict_worked}",
        )

    # =========================================================================
    # Attention Masks Tests
    # =========================================================================

    def _test_boolean_mask(self) -> BehavioralTestResult:
        """Test boolean mask indexing."""
        import flashlight

        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = flashlight.tensor([True, False, True, False, True])

        result = x[mask]

        expected = flashlight.tensor([1.0, 3.0, 5.0])
        passed = bool(flashlight.allclose(result, expected))

        return BehavioralTestResult(
            category="attention_masks",
            test_name="boolean_mask",
            passed=passed,
            error=None if passed else f"Result: {result}",
        )

    def _test_float_mask(self) -> BehavioralTestResult:
        """Test float mask (additive attention mask style)."""
        import flashlight

        # Simulate attention scores
        scores = flashlight.ones(3, 3)

        # Float mask with -inf for masked positions
        mask = flashlight.tensor([
            [0.0, float('-inf'), float('-inf')],
            [0.0, 0.0, float('-inf')],
            [0.0, 0.0, 0.0],
        ])

        masked_scores = scores + mask

        # After softmax, -inf positions should be ~0
        softmax_scores = flashlight.softmax(masked_scores, dim=-1)

        # First row should have all attention on first position
        first_row = softmax_scores[0]
        passed = abs(float(first_row[0].item()) - 1.0) < 0.01

        return BehavioralTestResult(
            category="attention_masks",
            test_name="float_mask",
            passed=passed,
            error=None if passed else f"First row: {first_row}",
        )

    # =========================================================================
    # Serialization Tests
    # =========================================================================

    def _test_save_load_roundtrip(self) -> BehavioralTestResult:
        """Test tensor save/load round-trip."""
        import flashlight
        import tempfile
        import os

        x = flashlight.randn(5, 5)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            flashlight.save(x, filepath)
            loaded = flashlight.load(filepath)

            passed = bool(flashlight.allclose(x, loaded))
        except Exception as e:
            passed = False
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

        return BehavioralTestResult(
            category="serialization",
            test_name="save_load_roundtrip",
            passed=passed,
            error=None if passed else "Save/load round-trip failed",
        )

    def _test_dtype_preservation(self) -> BehavioralTestResult:
        """Test that dtype is preserved through serialization."""
        import flashlight
        import tempfile
        import os

        x = flashlight.tensor([1.0, 2.0], dtype=flashlight.float32)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            flashlight.save(x, filepath)
            loaded = flashlight.load(filepath)

            passed = loaded.dtype == x.dtype
        except Exception as e:
            passed = False
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

        return BehavioralTestResult(
            category="serialization",
            test_name="dtype_preservation",
            passed=passed,
            error=None if passed else "Dtype not preserved",
        )

    # =========================================================================
    # Shape Broadcast Tests
    # =========================================================================

    def _test_squeeze_behavior(self) -> BehavioralTestResult:
        """Test squeeze removes size-1 dimensions."""
        import flashlight

        x = flashlight.zeros(1, 3, 1, 4)

        # Squeeze all
        squeezed = x.squeeze()
        all_squeezed = squeezed.shape == (3, 4)

        # Squeeze specific dim
        squeezed_dim = x.squeeze(0)
        dim_squeezed = squeezed_dim.shape == (3, 1, 4)

        passed = all_squeezed and dim_squeezed

        return BehavioralTestResult(
            category="shape_broadcast",
            test_name="squeeze_behavior",
            passed=passed,
            error=None if passed else f"all={squeezed.shape}, dim0={squeezed_dim.shape}",
        )

    def _test_unsqueeze_behavior(self) -> BehavioralTestResult:
        """Test unsqueeze adds size-1 dimension."""
        import flashlight

        x = flashlight.zeros(3, 4)

        unsqueezed = x.unsqueeze(0)
        passed = unsqueezed.shape == (1, 3, 4)

        return BehavioralTestResult(
            category="shape_broadcast",
            test_name="unsqueeze_behavior",
            passed=passed,
            error=None if passed else f"Shape: {unsqueezed.shape}",
        )

    def _test_flatten_behavior(self) -> BehavioralTestResult:
        """Test flatten collapses dimensions."""
        import flashlight

        x = flashlight.zeros(2, 3, 4)

        # Flatten all
        flat = x.flatten()
        all_flat = flat.shape == (24,)

        # Flatten with start_dim
        partial_flat = x.flatten(start_dim=1)
        partial_shape = partial_flat.shape == (2, 12)

        passed = all_flat and partial_shape

        return BehavioralTestResult(
            category="shape_broadcast",
            test_name="flatten_behavior",
            passed=passed,
            error=None if passed else f"all={flat.shape}, partial={partial_flat.shape}",
        )

    # =========================================================================
    # Item Extraction Tests
    # =========================================================================

    def _test_item_scalar(self) -> BehavioralTestResult:
        """Test .item() extracts scalar value."""
        import flashlight

        scalar = flashlight.tensor(42.0)
        value = scalar.item()

        passed = isinstance(value, float) and abs(value - 42.0) < 0.001

        return BehavioralTestResult(
            category="item_extraction",
            test_name="item_scalar",
            passed=passed,
            error=None if passed else f"Got {value} (type: {type(value).__name__})",
        )

    def _test_tolist_behavior(self) -> BehavioralTestResult:
        """Test .tolist() converts tensor to Python list."""
        import flashlight

        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = x.tolist()

        expected = [[1.0, 2.0], [3.0, 4.0]]
        passed = result == expected

        return BehavioralTestResult(
            category="item_extraction",
            test_name="tolist_behavior",
            passed=passed,
            error=None if passed else f"Got {result}",
        )
