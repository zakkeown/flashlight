"""
Backward Pass Engine

Implements the core autograd engine that:
1. Builds computation graphs during forward pass
2. Performs topological sort of gradient functions
3. Executes backward pass to compute gradients
4. Accumulates gradients for tensors used multiple times
"""

import threading
from collections import defaultdict
from typing import Dict, List, Optional, Set

import mlx.core as mx

# Thread-local storage for backward pass state
_backward_state = threading.local()


def _get_create_graph() -> bool:
    """Get whether create_graph is enabled for current backward pass."""
    return getattr(_backward_state, "create_graph", False)


def _set_create_graph(value: bool):
    """Set whether create_graph is enabled."""
    _backward_state.create_graph = value


class backward_context:
    """
    Context manager for backward pass state.

    Used internally to track create_graph setting during backward.
    """

    def __init__(self, create_graph: bool = False):
        self.create_graph = create_graph

    def __enter__(self):
        self.prev = _get_create_graph()
        _set_create_graph(self.create_graph)
        return self

    def __exit__(self, *args):
        _set_create_graph(self.prev)


def topological_sort(root_grad_fn):
    """
    Perform topological sort of gradient functions in the computation graph.

    Args:
        root_grad_fn: The gradient function of the output tensor

    Returns:
        List of gradient functions in topological order (root first)
    """
    visited = set()
    topo_order = []

    def dfs(grad_fn):
        if grad_fn is None or grad_fn in visited:
            return

        # Check if this grad_fn has been released (inputs set to None)
        # This can happen when a previous backward() released the graph
        # but a parameter tensor still holds a reference to the old grad_fn
        if grad_fn.inputs is None:
            return

        visited.add(grad_fn)

        # Visit dependencies first (inputs' gradient functions)
        for input_tensor in grad_fn.inputs:
            if hasattr(input_tensor, "_grad_fn") and input_tensor._grad_fn is not None:
                dfs(input_tensor._grad_fn)

        # Add current node after dependencies
        topo_order.append(grad_fn)

    dfs(root_grad_fn)
    return topo_order


def _check_for_nan_inf(tensor, name: str, grad_fn_name: str):
    """
    Check if tensor contains NaN or Inf values.

    Args:
        tensor: Tensor to check
        name: Name for the value being checked (e.g., "grad 0")
        grad_fn_name: Name of the gradient function that produced this

    Raises:
        RuntimeError: If NaN or Inf values are detected
    """
    arr = tensor._mlx_array
    mx.eval(arr)  # Force evaluation to check values

    has_nan = bool(mx.any(mx.isnan(arr)).item())
    has_inf = bool(mx.any(mx.isinf(arr)).item())

    if has_nan or has_inf:
        anomaly_type = "NaN" if has_nan else "Inf"
        raise RuntimeError(
            f"Function '{grad_fn_name}' returned {anomaly_type} in {name}. "
            f"Enable anomaly detection with flashlight.autograd.detect_anomaly() "
            f"for more debugging information."
        )


def backward(tensor, gradient=None, retain_graph=False, create_graph=False):
    """
    Execute backward pass from a tensor.

    This is the main entry point for backpropagation. It:
    1. Verifies the tensor requires gradients
    2. Initializes the output gradient
    3. Performs topological sort
    4. Executes gradient functions in reverse order
    5. Accumulates gradients for leaf tensors

    Args:
        tensor: Tensor to backpropagate from (usually a scalar loss)
        gradient: Initial gradient (defaults to ones). Must match tensor shape.
        retain_graph: If True, keep computation graph for multiple backward passes
        create_graph: If True, create gradient computation graph (for higher-order derivatives)

    Raises:
        RuntimeError: If tensor doesn't require gradients or is not a scalar without gradient argument
    """
    from ..tensor import Tensor
    from .anomaly_mode import is_anomaly_check_nan_enabled, is_anomaly_enabled

    # Validation
    if not tensor.requires_grad:
        raise RuntimeError("Cannot backward through a tensor that doesn't require gradients")

    # Check if graph was already released
    if tensor._grad_fn is not None and getattr(tensor._grad_fn, '_released', False):
        raise RuntimeError(
            "Trying to backward through the graph a second time. "
            "Use retain_graph=True if you need to backward through the graph multiple times."
        )

    # For non-scalar tensors, gradient must be provided
    if tensor.numel != 1 and gradient is None:
        raise RuntimeError(
            f"grad can be implicitly created only for scalar outputs. "
            f"Got tensor of shape {tensor.shape}"
        )

    # Initialize gradient for output tensor
    if gradient is None:
        # Scalar case - gradient is 1
        gradient = Tensor._from_mlx_array(mx.ones(tensor.shape, dtype=tensor.dtype._mlx_dtype))
    elif not isinstance(gradient, Tensor):
        # Convert to Tensor if needed
        gradient = Tensor(gradient, dtype=tensor.dtype)

    # Check gradient shape matches
    if gradient.shape != tensor.shape:
        raise RuntimeError(
            f"Gradient shape {gradient.shape} doesn't match tensor shape {tensor.shape}"
        )

    # Determine if we should check for NaN/Inf
    check_nan = is_anomaly_enabled() and is_anomaly_check_nan_enabled()

    # If this is a leaf tensor, just set the gradient
    if tensor._grad_fn is None:
        if tensor.grad is None:
            tensor.grad = gradient
        else:
            # Accumulate gradient
            tensor.grad = Tensor._from_mlx_array(tensor.grad._mlx_array + gradient._mlx_array)
        return

    # Topological sort to get execution order
    topo_order = topological_sort(tensor._grad_fn)

    # Dictionary to accumulate gradients for each tensor
    # Key: tensor id, Value: accumulated gradient (raw MLX array for performance)
    # We store raw mx.array to avoid creating Tensor wrappers during accumulation
    grads: Dict[int, mx.array] = {}
    grads[id(tensor)] = gradient._mlx_array

    # Execute backward functions in reverse topological order
    # Use backward_context to make create_graph available to grad functions
    with backward_context(create_graph=create_graph):
        for grad_fn in reversed(topo_order):
            # Get gradient flowing into this function's output
            output_tensor = grad_fn.output_tensor
            if id(output_tensor) not in grads:
                # This shouldn't happen in a well-formed graph
                continue

            # Wrap gradient for grad_fn.apply (which expects Tensor)
            grad_output = Tensor._from_mlx_array(grads[id(output_tensor)])

            # Call the gradient function to compute input gradients
            try:
                input_grads = grad_fn.apply(grad_output)
            except Exception as e:
                raise RuntimeError(
                    f"Error in backward pass for {grad_fn.__class__.__name__}: {e}"
                ) from e

            # Ensure we have a tuple of gradients
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)

            # Check for NaN/Inf if anomaly detection is enabled
            if check_nan:
                for i, grad in enumerate(input_grads):
                    if grad is not None:
                        _check_for_nan_inf(grad, f"grad {i}", grad_fn.__class__.__name__)

            # Accumulate gradients for each input tensor
            for input_tensor, grad in zip(grad_fn.inputs, input_grads):
                if grad is None:
                    # This input doesn't need gradients
                    continue

                if not input_tensor.requires_grad:
                    # Skip tensors that don't require gradients
                    continue

                # Call registered hooks on this tensor
                # Hooks can modify the gradient by returning a new value
                if hasattr(input_tensor, "_hooks") and input_tensor._hooks:
                    for hook_id, hook in input_tensor._hooks:
                        hook_result = hook(grad)
                        if hook_result is not None:
                            grad = hook_result

                # Accumulate gradient using raw MLX arrays
                tensor_id = id(input_tensor)
                grad_arr = grad._mlx_array
                if tensor_id in grads:
                    # Add to existing gradient (raw array addition)
                    grads[tensor_id] = grads[tensor_id] + grad_arr
                else:
                    # First gradient for this tensor
                    grads[tensor_id] = grad_arr

                # Handle retain_grad for non-leaf tensors
                if (
                    hasattr(input_tensor, "_retain_grad")
                    and input_tensor._retain_grad
                    and input_tensor._grad_fn is not None  # Non-leaf
                ):
                    # Store gradient for this non-leaf tensor
                    if input_tensor.grad is None:
                        input_tensor.grad = Tensor._from_mlx_array(grads[tensor_id])
                    else:
                        input_tensor.grad = Tensor._from_mlx_array(
                            input_tensor.grad._mlx_array + grads[tensor_id]
                        )

    # Assign gradients to leaf tensors
    # Keep track of which tensors we've already assigned to avoid double-counting
    assigned = set()
    for grad_fn in topo_order:
        for input_tensor in grad_fn.inputs:
            # Only assign to leaf tensors (those without grad_fn)
            tensor_id = id(input_tensor)
            if (
                input_tensor._grad_fn is None
                and input_tensor.requires_grad
                and tensor_id not in assigned
                and tensor_id in grads
            ):

                assigned.add(tensor_id)
                # Wrap only at final assignment to tensor.grad
                if input_tensor.grad is None:
                    input_tensor.grad = Tensor._from_mlx_array(grads[tensor_id])
                else:
                    # Accumulate with existing gradient
                    input_tensor.grad = Tensor._from_mlx_array(
                        input_tensor.grad._mlx_array + grads[tensor_id]
                    )

    # Clean up computation graph if not retaining
    if not retain_graph:
        _release_graph(topo_order, tensor)


def _release_graph(topo_order, root_tensor):
    """
    Release computation graph nodes to free memory.

    This clears references in gradient functions to allow garbage collection.

    Args:
        topo_order: List of gradient functions in topological order
        root_tensor: The root tensor of the backward pass
    """
    for grad_fn in topo_order:
        # Clear inputs to break reference cycles
        grad_fn.inputs = None
        # Clear output tensor reference
        grad_fn.output_tensor = None
        # Mark as released to prevent reuse
        grad_fn._released = True

    # Clear the grad_fn from root tensor
    root_tensor._grad_fn = None


def zero_grad(tensor):
    """
    Zero out the gradient of a tensor.

    Args:
        tensor: Tensor whose gradient to zero
    """
    tensor.grad = None


def zero_grad_all(tensors):
    """
    Zero out gradients for a list of tensors.

    Args:
        tensors: Iterable of tensors
    """
    for tensor in tensors:
        zero_grad(tensor)
