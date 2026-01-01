"""
Tests for mlx_compat.utils.hooks module.

Tests RemovableHandle, BackwardHook, and hook utilities.
"""

import unittest
import warnings
from collections import OrderedDict

from mlx_compat.utils.hooks import (
    RemovableHandle,
    BackwardHook,
    unserializable_hook,
    warn_if_has_hooks,
)


class TestRemovableHandle(unittest.TestCase):
    """Test RemovableHandle class."""

    def test_remove_hook(self):
        """Test that remove() properly removes hook from dict."""
        hooks = OrderedDict()

        # Simulate adding a hook
        def my_hook(x):
            return x * 2

        handle = RemovableHandle(hooks)
        hooks[handle.id] = my_hook

        self.assertIn(handle.id, hooks)
        handle.remove()
        self.assertNotIn(handle.id, hooks)

    def test_remove_idempotent(self):
        """Test that calling remove() multiple times is safe."""
        hooks = OrderedDict()
        handle = RemovableHandle(hooks)
        hooks[handle.id] = lambda x: x

        handle.remove()
        handle.remove()  # Should not raise
        self.assertNotIn(handle.id, hooks)

    def test_context_manager(self):
        """Test with statement support."""
        hooks = OrderedDict()

        with RemovableHandle(hooks) as handle:
            hooks[handle.id] = lambda x: x
            self.assertIn(handle.id, hooks)

        # After exiting context, hook should be removed
        self.assertNotIn(handle.id, hooks)

    def test_unique_ids(self):
        """Test that each handle gets a unique ID."""
        hooks = OrderedDict()

        handle1 = RemovableHandle(hooks)
        handle2 = RemovableHandle(hooks)
        handle3 = RemovableHandle(hooks)

        self.assertNotEqual(handle1.id, handle2.id)
        self.assertNotEqual(handle2.id, handle3.id)
        self.assertNotEqual(handle1.id, handle3.id)

    def test_extra_dict(self):
        """Test extra_dict parameter for removing from multiple dicts."""
        main_hooks = OrderedDict()
        extra_hooks = OrderedDict()

        handle = RemovableHandle(main_hooks, extra_dict=extra_hooks)
        main_hooks[handle.id] = lambda x: x
        extra_hooks[handle.id] = "extra_data"

        handle.remove()

        self.assertNotIn(handle.id, main_hooks)
        self.assertNotIn(handle.id, extra_hooks)

    def test_extra_dict_list(self):
        """Test extra_dict with a list of dictionaries."""
        main_hooks = OrderedDict()
        extra1 = OrderedDict()
        extra2 = OrderedDict()

        handle = RemovableHandle(main_hooks, extra_dict=[extra1, extra2])
        main_hooks[handle.id] = lambda x: x
        extra1[handle.id] = "data1"
        extra2[handle.id] = "data2"

        handle.remove()

        self.assertNotIn(handle.id, main_hooks)
        self.assertNotIn(handle.id, extra1)
        self.assertNotIn(handle.id, extra2)

    def test_getstate_setstate(self):
        """Test pickle serialization support."""
        hooks = OrderedDict()
        handle = RemovableHandle(hooks)
        hooks[handle.id] = lambda x: x

        state = handle.__getstate__()
        self.assertIsInstance(state, tuple)
        # State has 2 or 3 elements depending on extra_dict_ref
        self.assertGreaterEqual(len(state), 2)

    def test_weakref_safety(self):
        """Test that weakrefs don't cause issues when dict is deleted."""
        hooks = OrderedDict()
        handle = RemovableHandle(hooks)
        hooks[handle.id] = lambda x: x

        # Delete the hooks dict
        del hooks

        # remove() should handle the dead weakref gracefully
        handle.remove()  # Should not raise


class TestUnserializableHook(unittest.TestCase):
    """Test unserializable_hook decorator."""

    def test_decorator_marks_function(self):
        """Test that decorator adds the marker attribute."""
        @unserializable_hook
        def my_hook(grad):
            return grad * 2

        self.assertTrue(hasattr(my_hook, "__mlx_unserializable__"))
        self.assertTrue(my_hook.__mlx_unserializable__)

    def test_function_still_callable(self):
        """Test that decorated function still works."""
        @unserializable_hook
        def my_hook(x):
            return x * 2

        self.assertEqual(my_hook(5), 10)

    def test_preserves_function_identity(self):
        """Test that decorator returns the same function."""
        def my_hook(x):
            return x

        decorated = unserializable_hook(my_hook)
        self.assertIs(decorated, my_hook)


class TestWarnIfHasHooks(unittest.TestCase):
    """Test warn_if_has_hooks function."""

    def test_no_warning_without_hooks(self):
        """Test no warning when tensor has no hooks."""
        class MockTensor:
            pass

        tensor = MockTensor()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_has_hooks(tensor)
            self.assertEqual(len(w), 0)

    def test_no_warning_empty_hooks(self):
        """Test no warning when hooks dict is empty."""
        class MockTensor:
            _backward_hooks = {}

        tensor = MockTensor()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_has_hooks(tensor)
            self.assertEqual(len(w), 0)

    def test_warning_with_serializable_hook(self):
        """Test warning for hooks without unserializable marker."""
        def my_hook(grad):
            return grad

        class MockTensor:
            _backward_hooks = {0: my_hook}

        tensor = MockTensor()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_has_hooks(tensor)
            self.assertEqual(len(w), 1)
            self.assertIn("backward hook", str(w[0].message))

    def test_no_warning_with_unserializable_hook(self):
        """Test no warning for hooks with unserializable marker."""
        @unserializable_hook
        def my_hook(grad):
            return grad

        class MockTensor:
            _backward_hooks = {0: my_hook}

        tensor = MockTensor()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_has_hooks(tensor)
            self.assertEqual(len(w), 0)


class TestBackwardHook(unittest.TestCase):
    """Test BackwardHook class."""

    def test_init(self):
        """Test BackwardHook initialization."""
        module = object()
        user_hooks = {}
        user_pre_hooks = {}

        hook = BackwardHook(module, user_hooks, user_pre_hooks)

        self.assertIs(hook.module, module)
        self.assertIs(hook.user_hooks, user_hooks)
        self.assertIs(hook.user_pre_hooks, user_pre_hooks)

    def test_setup_input_hook(self):
        """Test input hook setup."""
        import mlx_compat

        module = object()
        hook = BackwardHook(module, {}, {})

        # Create some tensors
        t1 = mlx_compat.tensor([1.0, 2.0])
        t2 = mlx_compat.tensor([3.0, 4.0])
        args = (t1, "non_tensor", t2)

        result = hook.setup_input_hook(args)

        self.assertIs(result, args)
        self.assertEqual(hook.n_inputs, 3)
        self.assertEqual(hook.input_tensors_index, [0, 2])

    def test_setup_output_hook_single(self):
        """Test output hook setup with single tensor."""
        import mlx_compat

        module = object()
        hook = BackwardHook(module, {}, {})

        output = mlx_compat.tensor([1.0, 2.0])
        result = hook.setup_output_hook(output)

        self.assertIs(result, output)
        self.assertEqual(hook.n_outputs, 1)
        self.assertEqual(hook.output_tensors_index, [0])

    def test_setup_output_hook_tuple(self):
        """Test output hook setup with tuple of outputs."""
        import mlx_compat

        module = object()
        hook = BackwardHook(module, {}, {})

        t1 = mlx_compat.tensor([1.0])
        t2 = mlx_compat.tensor([2.0])
        outputs = (t1, t2)

        result = hook.setup_output_hook(outputs)

        self.assertEqual(result, outputs)
        self.assertEqual(hook.n_outputs, 2)
        self.assertEqual(hook.output_tensors_index, [0, 1])

    def test_pack_with_none(self):
        """Test _pack_with_none helper."""
        module = object()
        hook = BackwardHook(module, {}, {})

        result = hook._pack_with_none([0, 2], ("a", "b"), 4)

        self.assertEqual(result, ("a", None, "b", None))

    def test_unpack_none(self):
        """Test _unpack_none helper."""
        module = object()
        hook = BackwardHook(module, {}, {})

        result = hook._unpack_none([0, 2], ("a", "b", "c", "d"))

        self.assertEqual(result, ("a", "c"))

    def test_call_executes_hooks(self):
        """Test that __call__ executes hooks."""
        import mlx_compat

        module = object()

        # Track hook calls
        call_log = []

        def hook1(mod, grad_in, grad_out):
            call_log.append("hook1")
            return grad_in

        def hook2(mod, grad_in, grad_out):
            call_log.append("hook2")
            return grad_in

        user_hooks = {0: hook1, 1: hook2}
        backward_hook = BackwardHook(module, user_hooks, {})

        grad_in = (mlx_compat.tensor([1.0]),)
        grad_out = (mlx_compat.tensor([2.0]),)

        result = backward_hook(grad_in, grad_out)

        self.assertEqual(call_log, ["hook1", "hook2"])

    def test_call_with_pre_hooks(self):
        """Test that pre-hooks are called before hooks."""
        module = object()

        call_log = []

        def pre_hook(mod, grad_in, grad_out):
            call_log.append("pre")
            return grad_in

        def post_hook(mod, grad_in, grad_out):
            call_log.append("post")
            return grad_in

        backward_hook = BackwardHook(module, {0: post_hook}, {0: pre_hook})

        result = backward_hook((), ())

        self.assertEqual(call_log, ["pre", "post"])


if __name__ == "__main__":
    unittest.main()
