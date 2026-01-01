"""
Weak reference utilities for tensors.

Provides weak reference containers for memory-efficient tensor caching.
"""

import weakref
from collections.abc import MutableMapping, Mapping
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar, Generic

# Type variables
K = TypeVar("K")
V = TypeVar("V")


class WeakRef:
    """
    Weak reference wrapper with optional callback.

    This is a simple wrapper around weakref.ref for consistency.
    """

    def __init__(self, obj: Any, callback: Optional[Callable] = None):
        """
        Create a weak reference.

        Args:
            obj: Object to create weak reference to.
            callback: Optional callback when object is garbage collected.
        """
        self._ref = weakref.ref(obj, callback)

    def __call__(self) -> Optional[Any]:
        """Dereference the weak reference."""
        return self._ref()

    @property
    def ref(self) -> weakref.ref:
        """Get the underlying weakref.ref object."""
        return self._ref


class TensorWeakRef:
    """
    Weak reference specifically for tensors.

    Handles the special case where tensors may not be directly weak-referenceable
    by storing a reference to the underlying data.
    """

    def __init__(self, tensor: Any):
        """
        Create a weak reference to a tensor.

        Args:
            tensor: Tensor to create weak reference to.
        """
        self._tensor_ref: Optional[weakref.ref] = None
        self._tensor_id: int = id(tensor)

        try:
            self._tensor_ref = weakref.ref(tensor)
        except TypeError:
            # Tensor type doesn't support weak references
            # Store None and rely on id for comparison
            pass

    def __call__(self) -> Optional[Any]:
        """Dereference the weak reference."""
        if self._tensor_ref is not None:
            return self._tensor_ref()
        return None

    @property
    def tensor_id(self) -> int:
        """Get the original tensor's id."""
        return self._tensor_id


class WeakIdRef(WeakRef):
    """
    Weak reference that uses object id for hashing/equality.

    This allows using objects as dictionary keys based on identity.
    """

    def __init__(self, obj: Any, callback: Optional[Callable] = None):
        """
        Create a weak id reference.

        Args:
            obj: Object to create weak reference to.
            callback: Optional callback when object is garbage collected.
        """
        super().__init__(obj, callback)
        self._id = id(obj)

    def __hash__(self) -> int:
        """Hash based on object id."""
        return hash(self._id)

    def __eq__(self, other: Any) -> bool:
        """Compare based on object id."""
        if isinstance(other, WeakIdRef):
            return self._id == other._id
        return False

    @property
    def key_id(self) -> int:
        """Get the object id used as key."""
        return self._id


class WeakIdKeyDictionary(MutableMapping):
    """
    Dictionary with weak references to keys, using id for identity.

    Keys are stored as weak references and automatically removed when
    the key object is garbage collected.
    """

    def __init__(self, data: Optional[Dict] = None):
        """
        Create a weak id key dictionary.

        Args:
            data: Optional initial data to populate dictionary.
        """
        self._data: Dict[int, tuple] = {}  # id -> (weakref, value)

        if data:
            for key, value in data.items():
                self[key] = value

    def _remove_dead(self, key_id: int) -> None:
        """Callback to remove dead references."""
        self._data.pop(key_id, None)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an item."""
        key_id = id(key)

        def callback(ref: weakref.ref) -> None:
            self._remove_dead(key_id)

        try:
            ref = weakref.ref(key, callback)
            self._data[key_id] = (ref, value)
        except TypeError:
            # Key doesn't support weak references, store strong reference
            self._data[key_id] = (key, value)

    def __getitem__(self, key: Any) -> Any:
        """Get an item."""
        key_id = id(key)
        if key_id not in self._data:
            raise KeyError(key)

        ref_or_obj, value = self._data[key_id]

        # Check if it's a weakref
        if isinstance(ref_or_obj, weakref.ref):
            obj = ref_or_obj()
            if obj is None:
                # Reference died
                del self._data[key_id]
                raise KeyError(key)
        return value

    def __delitem__(self, key: Any) -> None:
        """Delete an item."""
        key_id = id(key)
        if key_id not in self._data:
            raise KeyError(key)
        del self._data[key_id]

    def __iter__(self) -> Iterator:
        """Iterate over keys."""
        # Collect live keys first to avoid modification during iteration
        live_keys = []
        dead_ids = []

        for key_id, (ref_or_obj, _) in list(self._data.items()):
            if isinstance(ref_or_obj, weakref.ref):
                obj = ref_or_obj()
                if obj is not None:
                    live_keys.append(obj)
                else:
                    dead_ids.append(key_id)
            else:
                live_keys.append(ref_or_obj)

        # Clean up dead references
        for key_id in dead_ids:
            self._data.pop(key_id, None)

        return iter(live_keys)

    def __len__(self) -> int:
        """Get number of items."""
        # Count only live references
        count = 0
        dead_ids = []

        for key_id, (ref_or_obj, _) in list(self._data.items()):
            if isinstance(ref_or_obj, weakref.ref):
                if ref_or_obj() is not None:
                    count += 1
                else:
                    dead_ids.append(key_id)
            else:
                count += 1

        # Clean up dead references
        for key_id in dead_ids:
            self._data.pop(key_id, None)

        return count

    def __contains__(self, key: Any) -> bool:
        """Check if key exists."""
        key_id = id(key)
        if key_id not in self._data:
            return False

        ref_or_obj, _ = self._data[key_id]
        if isinstance(ref_or_obj, weakref.ref):
            if ref_or_obj() is None:
                del self._data[key_id]
                return False
        return True

    def get(self, key: Any, default: Any = None) -> Any:
        """Get an item with default."""
        try:
            return self[key]
        except KeyError:
            return default


class WeakTensorKeyDictionary(WeakIdKeyDictionary):
    """
    Dictionary specifically for tensor keys.

    This is a specialized version of WeakIdKeyDictionary that handles
    tensors which may or may not support weak references.
    """

    def __repr__(self) -> str:
        """String representation."""
        return f"WeakTensorKeyDictionary({len(self)} items)"


# Re-export weakref.ref for convenience
ref = weakref.ref


# Alias for compatibility
Tensor = None  # Will be set when mlx_compat is imported
try:
    from mlx_compat.tensor import Tensor
except ImportError:
    pass
