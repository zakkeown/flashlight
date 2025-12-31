"""Tests for behavioral parity validator."""

import pytest

from parity_check.validators.behavioral_parity import (
    BehavioralParityValidator,
    BehavioralTestResult,
    BehavioralValidationResult,
)


class TestBehavioralTestResult:
    """Tests for BehavioralTestResult dataclass."""

    def test_passed_result(self):
        result = BehavioralTestResult(
            category="context_manager",
            test_name="no_grad_disables_gradients",
            passed=True,
        )
        assert result.category == "context_manager"
        assert result.test_name == "no_grad_disables_gradients"
        assert result.passed is True
        assert result.error is None
        assert result.details is None

    def test_failed_result_with_error(self):
        result = BehavioralTestResult(
            category="module_state",
            test_name="train_eval_toggle",
            passed=False,
            error="train/eval mode toggle failed",
            details={"initial": True, "after_eval": True},
        )
        assert result.passed is False
        assert result.error == "train/eval mode toggle failed"
        assert result.details["initial"] is True


class TestBehavioralValidationResult:
    """Tests for BehavioralValidationResult dataclass."""

    def test_empty_result(self):
        result = BehavioralValidationResult()
        assert result.total_tested == 0
        assert result.pass_percentage == 100.0
        assert result.is_complete is True

    def test_all_passed(self):
        result = BehavioralValidationResult(
            passed=[
                BehavioralTestResult("cat1", "test1", True),
                BehavioralTestResult("cat1", "test2", True),
            ],
            failed=[],
        )
        assert result.total_tested == 2
        assert result.pass_percentage == 100.0
        assert result.is_complete is True

    def test_with_failures(self):
        result = BehavioralValidationResult(
            passed=[BehavioralTestResult("cat1", "test1", True)],
            failed=[BehavioralTestResult("cat1", "test2", False, error="failed")],
        )
        assert result.total_tested == 2
        assert result.pass_percentage == 50.0
        assert result.is_complete is False

    def test_by_category(self):
        result = BehavioralValidationResult(
            passed=[
                BehavioralTestResult("context_manager", "test1", True),
                BehavioralTestResult("module_state", "test1", True),
            ],
            failed=[
                BehavioralTestResult("context_manager", "test2", False),
            ],
        )
        categories = result.by_category()
        assert categories["context_manager"]["passed"] == 1
        assert categories["context_manager"]["failed"] == 1
        assert categories["module_state"]["passed"] == 1
        assert "module_state" not in categories or categories["module_state"].get("failed", 0) == 0

    def test_to_dict_serialization(self):
        result = BehavioralValidationResult(
            passed=[BehavioralTestResult("cat1", "test1", True)],
            failed=[BehavioralTestResult("cat1", "test2", False, error="error msg")],
            skipped=[{"category": "unknown", "reason": "unknown category"}],
            errors=[{"category": "cat2", "test": "test3", "error": "Exception: boom"}],
        )
        data = result.to_dict()

        assert data["passed"] == 1
        assert data["failed"] == 1
        assert data["skipped"] == 1
        assert data["errors"] == 1
        assert data["total_tested"] == 2
        assert data["pass_percentage"] == 50.0
        assert "by_category" in data
        assert len(data["failed_details"]) == 1
        assert data["failed_details"][0]["error"] == "error msg"


class TestBehavioralParityValidator:
    """Tests for BehavioralParityValidator."""

    def test_categories_list(self):
        """Test that CATEGORIES contains expected categories."""
        expected = [
            "context_manager",
            "module_state",
            "container",
            "optimizer",
            "layer_mode",
            "distribution",
            "edge_cases",
            # Comprehensive categories
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
        assert BehavioralParityValidator.CATEGORIES == expected

    def test_validate_single_category(self):
        """Test validating a single category."""
        validator = BehavioralParityValidator(categories=["edge_cases"])
        result = validator.validate()

        assert isinstance(result, BehavioralValidationResult)
        # Should only have edge_cases tests
        categories = result.by_category()
        assert "edge_cases" in categories
        # Should not have other categories
        for cat in ["context_manager", "module_state"]:
            assert cat not in categories

    def test_validate_multiple_categories(self):
        """Test validating multiple categories."""
        validator = BehavioralParityValidator(
            categories=["edge_cases", "distribution"]
        )
        result = validator.validate()

        categories = result.by_category()
        # Could have either or both if tests pass/fail
        assert result.total_tested > 0

    def test_unknown_category_skipped(self):
        """Test that unknown categories are skipped."""
        validator = BehavioralParityValidator(categories=["unknown_category"])
        result = validator.validate()

        assert len(result.skipped) == 1
        assert result.skipped[0]["category"] == "unknown_category"
        assert result.total_tested == 0

    def test_validate_category_method(self):
        """Test validate_category convenience method."""
        validator = BehavioralParityValidator()
        result = validator.validate_category("edge_cases")

        assert isinstance(result, BehavioralValidationResult)
        categories = result.by_category()
        assert "edge_cases" in categories

    def test_default_categories(self):
        """Test that default includes all categories."""
        validator = BehavioralParityValidator()
        assert validator.categories == BehavioralParityValidator.CATEGORIES

    def test_custom_seed(self):
        """Test custom seed is used."""
        validator = BehavioralParityValidator(seed=12345)
        assert validator.seed == 12345


class TestContextManagerBehavior:
    """Integration tests for context manager behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["context_manager"])

    def test_all_context_manager_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "no_grad_disables_gradients",
            "enable_grad_within_no_grad",
            "set_grad_enabled_toggle",
            "no_grad_as_decorator",
            "nested_context_managers",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"


class TestModuleStateBehavior:
    """Integration tests for module state behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["module_state"])

    def test_all_module_state_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "train_eval_toggle",
            "training_flag_propagation",
            "state_dict_round_trip",
            "buffers_vs_parameters",
            "named_parameters_hierarchy",
            "zero_grad_clears_gradients",
            "requires_grad_propagation",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"


class TestContainerBehavior:
    """Integration tests for container behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["container"])

    def test_all_container_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "sequential_forward_order",
            "modulelist_iteration_order",
            "modulelist_parameter_registration",
            "moduledict_semantics",
            "sequential_indexing",
            "sequential_slicing",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"


class TestOptimizerBehavior:
    """Integration tests for optimizer behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["optimizer"])

    def test_all_optimizer_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "optimizer_step_updates_params",
            "optimizer_zero_grad",
            "optimizer_state_dict_round_trip",
            "optimizer_param_groups",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"


class TestLayerModeBehavior:
    """Integration tests for layer mode behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["layer_mode"])

    def test_all_layer_mode_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "dropout_train_vs_eval",
            "batchnorm_train_vs_eval",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"


class TestDistributionBehavior:
    """Integration tests for distribution behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["distribution"])

    def test_all_distribution_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "randn_statistics",
            "rand_statistics",
            "normal_statistics",
            "seed_reproducibility",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"


class TestEdgeCaseBehavior:
    """Integration tests for edge case behavioral tests."""

    @pytest.fixture
    def validator(self):
        return BehavioralParityValidator(categories=["edge_cases"])

    def test_all_edge_case_tests_run(self, validator):
        result = validator.validate()

        test_names = [r.test_name for r in result.passed + result.failed]
        expected_tests = [
            "empty_tensor_behavior",
            "zero_dim_tensor",
            "broadcasting_rules",
        ]
        for test in expected_tests:
            assert test in test_names, f"Missing test: {test}"
