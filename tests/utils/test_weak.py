"""
Tests for flashlight.utils.weak module.

Tests weak reference utilities for tensors.
"""

import gc
import unittest
import weakref

from flashlight.utils.weak import (
    TensorWeakRef,
    WeakIdKeyDictionary,
    WeakIdRef,
    WeakRef,
    WeakTensorKeyDictionary,
    ref,
)


# Helper class that supports weak references (lists don't)
class WeakReferenceable:
    """Simple class that supports weak references."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, WeakReferenceable):
            return self.value == other.value
        return False


class TestWeakRef(unittest.TestCase):
    """Test WeakRef class."""

    def test_basic_weakref(self):
        """Test basic weak reference functionality."""
        obj = WeakReferenceable([1, 2, 3])
        wr = WeakRef(obj)

        # Should be able to dereference
        self.assertEqual(wr(), obj)
        self.assertIs(wr(), obj)

    def test_weakref_dead(self):
        """Test weak reference after object is deleted."""
        obj = WeakReferenceable([1, 2, 3])
        wr = WeakRef(obj)

        del obj
        gc.collect()

        # Should return None
        self.assertIsNone(wr())

    def test_callback(self):
        """Test weak reference callback."""
        callback_called = []

        def callback(ref):
            callback_called.append(True)

        obj = WeakReferenceable([1, 2, 3])
        wr = WeakRef(obj, callback)

        del obj
        gc.collect()

        self.assertEqual(len(callback_called), 1)

    def test_ref_property(self):
        """Test ref property returns underlying weakref."""
        obj = WeakReferenceable([1, 2, 3])
        wr = WeakRef(obj)

        self.assertIsInstance(wr.ref, weakref.ref)


class TestTensorWeakRef(unittest.TestCase):
    """Test TensorWeakRef class."""

    def test_tensor_weakref(self):
        """Test weak reference to a tensor."""
        import flashlight

        tensor = flashlight.tensor([1.0, 2.0, 3.0])
        wr = TensorWeakRef(tensor)

        # tensor_id should be set
        self.assertEqual(wr.tensor_id, id(tensor))

    def test_tensor_id_preserved(self):
        """Test tensor_id is preserved even after deletion."""
        import flashlight

        tensor = flashlight.tensor([1.0, 2.0])
        original_id = id(tensor)
        wr = TensorWeakRef(tensor)

        del tensor
        gc.collect()

        # ID should still be available
        self.assertEqual(wr.tensor_id, original_id)


class TestWeakIdRef(unittest.TestCase):
    """Test WeakIdRef class."""

    def test_hash_by_id(self):
        """Test hashing based on object id."""
        obj1 = WeakReferenceable([1, 2, 3])
        obj2 = WeakReferenceable([1, 2, 3])  # Same content, different object

        ref1 = WeakIdRef(obj1)
        ref2 = WeakIdRef(obj2)

        # Different objects should have different hashes
        self.assertNotEqual(hash(ref1), hash(ref2))

    def test_equality_by_id(self):
        """Test equality based on object id."""
        obj = WeakReferenceable([1, 2, 3])
        ref1 = WeakIdRef(obj)
        ref2 = WeakIdRef(obj)

        # Same object should be equal
        self.assertEqual(ref1, ref2)

        other = WeakReferenceable([1, 2, 3])
        ref3 = WeakIdRef(other)

        # Different objects should not be equal
        self.assertNotEqual(ref1, ref3)

    def test_key_id_property(self):
        """Test key_id property."""
        obj = WeakReferenceable([1, 2, 3])
        wr = WeakIdRef(obj)

        self.assertEqual(wr.key_id, id(obj))


class TestWeakIdKeyDictionary(unittest.TestCase):
    """Test WeakIdKeyDictionary class."""

    def test_basic_operations(self):
        """Test basic dict operations."""
        d = WeakIdKeyDictionary()

        key = WeakReferenceable([1, 2, 3])
        d[key] = "value"

        self.assertEqual(d[key], "value")
        self.assertIn(key, d)
        self.assertEqual(len(d), 1)

    def test_delete_item(self):
        """Test deleting items."""
        d = WeakIdKeyDictionary()

        key = WeakReferenceable([1, 2, 3])
        d[key] = "value"

        del d[key]
        self.assertNotIn(key, d)
        self.assertEqual(len(d), 0)

    def test_key_not_found(self):
        """Test KeyError for missing keys."""
        d = WeakIdKeyDictionary()

        with self.assertRaises(KeyError):
            _ = d[WeakReferenceable([1, 2, 3])]

    def test_get_with_default(self):
        """Test get method with default."""
        d = WeakIdKeyDictionary()

        key = WeakReferenceable([1, 2, 3])
        d[key] = "value"

        self.assertEqual(d.get(key), "value")
        self.assertEqual(d.get(WeakReferenceable([4, 5, 6]), "default"), "default")

    def test_automatic_cleanup(self):
        """Test automatic cleanup when key is garbage collected."""
        d = WeakIdKeyDictionary()

        key = WeakReferenceable([1, 2, 3])
        key_id = id(key)
        d[key] = "value"

        self.assertEqual(len(d), 1)

        del key
        gc.collect()

        # Length should be 0 after cleanup
        self.assertEqual(len(d), 0)

    def test_iteration(self):
        """Test iterating over keys."""
        d = WeakIdKeyDictionary()

        key1 = WeakReferenceable([1])
        key2 = WeakReferenceable([2])
        key3 = WeakReferenceable([3])

        d[key1] = "a"
        d[key2] = "b"
        d[key3] = "c"

        keys = list(d)
        self.assertEqual(len(keys), 3)

    def test_contains_dead_reference(self):
        """Test contains returns False for dead references."""
        d = WeakIdKeyDictionary()

        key = WeakReferenceable([1, 2, 3])
        d[key] = "value"

        # Create another reference to same content but different id
        other = WeakReferenceable([1, 2, 3])

        # other should not be in dict
        self.assertNotIn(other, d)

    def test_init_with_data(self):
        """Test initialization with existing data."""
        key1 = WeakReferenceable([1])
        key2 = WeakReferenceable([2])

        d = WeakIdKeyDictionary()
        d[key1] = "a"
        d[key2] = "b"

        self.assertEqual(d[key1], "a")
        self.assertEqual(d[key2], "b")

    def test_multiple_values_same_key(self):
        """Test overwriting values."""
        d = WeakIdKeyDictionary()

        key = WeakReferenceable([1, 2, 3])
        d[key] = "first"
        d[key] = "second"

        self.assertEqual(d[key], "second")
        self.assertEqual(len(d), 1)


class TestWeakTensorKeyDictionary(unittest.TestCase):
    """Test WeakTensorKeyDictionary class."""

    def test_tensor_as_key(self):
        """Test using tensors as keys."""
        import flashlight

        d = WeakTensorKeyDictionary()

        tensor = flashlight.tensor([1.0, 2.0])
        d[tensor] = "value"

        self.assertEqual(d[tensor], "value")

    def test_repr(self):
        """Test string representation."""
        d = WeakTensorKeyDictionary()
        repr_str = repr(d)

        self.assertIn("WeakTensorKeyDictionary", repr_str)
        self.assertIn("0 items", repr_str)

    def test_multiple_tensors(self):
        """Test multiple tensor keys."""
        import flashlight

        d = WeakTensorKeyDictionary()

        t1 = flashlight.tensor([1.0])
        t2 = flashlight.tensor([2.0])

        d[t1] = "one"
        d[t2] = "two"

        self.assertEqual(d[t1], "one")
        self.assertEqual(d[t2], "two")
        self.assertEqual(len(d), 2)


class TestRefAlias(unittest.TestCase):
    """Test ref alias."""

    def test_ref_is_weakref_ref(self):
        """Test that ref is weakref.ref."""
        self.assertIs(ref, weakref.ref)

    def test_ref_works(self):
        """Test that ref can be used with proper objects."""
        obj = WeakReferenceable([1, 2, 3])
        r = ref(obj)

        self.assertEqual(r(), obj)


if __name__ == "__main__":
    unittest.main()
