"""
Test Phase 3: Context Managers

Tests gradient context managers:
- no_grad
- enable_grad
- set_grad_enabled
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    from flashlight.autograd.context import is_grad_enabled
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestNoGrad(TestCase):
    """Test no_grad context manager."""

    def test_no_grad_basic(self):
        """Test that no_grad disables gradient tracking."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)

        with flashlight.no_grad():
            y = x * 2
            self.assertFalse(y.requires_grad)

    def test_no_grad_nested(self):
        """Test nested no_grad contexts."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        with flashlight.no_grad():
            y = x * 2
            self.assertFalse(y.requires_grad)

            with flashlight.no_grad():
                z = y + 1
                self.assertFalse(z.requires_grad)

    def test_no_grad_decorator(self):
        """Test no_grad as decorator."""
        @flashlight.no_grad()
        def func(x):
            return x * 2

        x = flashlight.tensor([1.0, 2.0], requires_grad=True)
        y = func(x)
        self.assertFalse(y.requires_grad)

    def test_no_grad_restores_state(self):
        """Test that no_grad restores previous state."""
        self.assertTrue(is_grad_enabled())

        with flashlight.no_grad():
            self.assertFalse(is_grad_enabled())

        self.assertTrue(is_grad_enabled())


@skipIfNoMLX
class TestEnableGrad(TestCase):
    """Test enable_grad context manager."""

    def test_enable_grad_in_no_grad(self):
        """Test enable_grad inside no_grad."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        with flashlight.no_grad():
            self.assertFalse(is_grad_enabled())

            with flashlight.enable_grad():
                self.assertTrue(is_grad_enabled())
                y = x * 2
                self.assertTrue(y.requires_grad)

            self.assertFalse(is_grad_enabled())

    def test_enable_grad_decorator(self):
        """Test enable_grad as decorator."""
        @flashlight.enable_grad()
        def func(x):
            return x * 2

        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        with flashlight.no_grad():
            y = func(x)
            # Inside the decorated function, grad is enabled
            self.assertTrue(y.requires_grad)


@skipIfNoMLX
class TestSetGradEnabled(TestCase):
    """Test set_grad_enabled context manager."""

    def test_set_grad_enabled_true(self):
        """Test set_grad_enabled(True)."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        with flashlight.set_grad_enabled(True):
            y = x * 2
            self.assertTrue(y.requires_grad)

    def test_set_grad_enabled_false(self):
        """Test set_grad_enabled(False)."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        with flashlight.set_grad_enabled(False):
            y = x * 2
            self.assertFalse(y.requires_grad)

    def test_set_grad_enabled_restores(self):
        """Test that set_grad_enabled restores state."""
        self.assertTrue(is_grad_enabled())

        with flashlight.set_grad_enabled(False):
            self.assertFalse(is_grad_enabled())

            with flashlight.set_grad_enabled(True):
                self.assertTrue(is_grad_enabled())

            self.assertFalse(is_grad_enabled())

        self.assertTrue(is_grad_enabled())


@skipIfNoMLX
class TestIsGradEnabled(TestCase):
    """Test is_grad_enabled function."""

    def test_default_enabled(self):
        """Test that gradients are enabled by default."""
        self.assertTrue(is_grad_enabled())

    def test_changes_with_context(self):
        """Test is_grad_enabled changes with context managers."""
        self.assertTrue(is_grad_enabled())

        with flashlight.no_grad():
            self.assertFalse(is_grad_enabled())

        self.assertTrue(is_grad_enabled())


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
