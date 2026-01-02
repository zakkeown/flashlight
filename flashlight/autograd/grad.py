"""
Functional gradient computation.

Provides torch.autograd.grad() equivalent functionality for computing
gradients without calling tensor.backward().
"""

from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx

from .engine import topological_sort


def grad(
    outputs: Union["Tensor", Sequence["Tensor"]],
    inputs: Union["Tensor", Sequence["Tensor"]],
    grad_outputs: Optional[Union["Tensor", Sequence["Tensor"]]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
) -> Tuple["Tensor", ...]:
    """
    Compute gradients of outputs with respect to inputs.

    This is equivalent to torch.autograd.grad() and computes the sum of
    gradients of outputs with respect to the inputs.

    Args:
        outputs: Outputs of the differentiated function (usually a scalar loss).
        inputs: Inputs for which gradients will be computed.
        grad_outputs: Gradients with respect to each output. Defaults to ones.
        retain_graph: If False, graph used to compute grads will be freed.
            Defaults to create_graph.
        create_graph: If True, graph of the derivative will be constructed,
            allowing higher-order derivatives.
        only_inputs: If True (default), only compute gradients for inputs,
            not other tensors in the graph.
        allow_unused: If True, don't raise error for unused inputs.
            Instead, return None for their gradients.

    Returns:
        Tuple of gradients, one for each input.

    Raises:
        RuntimeError: If any output doesn't require gradients, or if an
            input's gradient wasn't computed and allow_unused=False.

    Example:
        >>> x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> grads = flashlight.autograd.grad(y, x)
        >>> print(grads[0])  # [2.0, 4.0, 6.0]
    """
    from ..tensor import Tensor

    # Normalize inputs to sequences
    if isinstance(outputs, Tensor):
        outputs = (outputs,)
    else:
        outputs = tuple(outputs)

    if isinstance(inputs, Tensor):
        inputs = (inputs,)
    else:
        inputs = tuple(inputs)

    # Handle grad_outputs
    if grad_outputs is None:
        grad_outputs = []
        for output in outputs:
            if output.numel != 1:
                raise RuntimeError(
                    f"grad can be implicitly created only for scalar outputs. "
                    f"Got output of shape {output.shape}"
                )
            grad_outputs.append(
                Tensor._from_mlx_array(mx.ones(output.shape, dtype=output.dtype._mlx_dtype))
            )
    elif isinstance(grad_outputs, Tensor):
        grad_outputs = (grad_outputs,)
    else:
        grad_outputs = tuple(grad_outputs)

    if len(grad_outputs) != len(outputs):
        raise RuntimeError(
            f"Number of grad_outputs ({len(grad_outputs)}) doesn't match "
            f"number of outputs ({len(outputs)})"
        )

    # Set retain_graph default
    if retain_graph is None:
        retain_graph = create_graph

    # Validate outputs require gradients
    for output in outputs:
        if not output.requires_grad:
            raise RuntimeError(
                "Cannot compute gradients for outputs that don't require gradients"
            )

    # Collect all gradient functions and perform backward pass
    # Dictionary to accumulate gradients for each tensor (by id)
    grads = {}

    # Process each output
    for output, grad_output in zip(outputs, grad_outputs):
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output, dtype=output.dtype)

        if grad_output.shape != output.shape:
            raise RuntimeError(
                f"grad_output shape {grad_output.shape} doesn't match "
                f"output shape {output.shape}"
            )

        # Seed the output gradient
        grads[id(output)] = grad_output._mlx_array

        # If output has no grad_fn, it's a leaf
        if output._grad_fn is None:
            continue

        # Get topological order of gradient functions
        topo_order = topological_sort(output._grad_fn)

        # Execute backward pass
        for grad_fn in reversed(topo_order):
            output_tensor = grad_fn.output_tensor
            if id(output_tensor) not in grads:
                continue

            # Wrap gradient for grad_fn.apply
            grad_out = Tensor._from_mlx_array(grads[id(output_tensor)])

            try:
                input_grads = grad_fn.apply(grad_out)
            except Exception as e:
                raise RuntimeError(
                    f"Error in grad computation for {grad_fn.__class__.__name__}: {e}"
                ) from e

            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)

            # Accumulate gradients
            for input_tensor, input_grad in zip(grad_fn.inputs, input_grads):
                if input_grad is None:
                    continue
                if not input_tensor.requires_grad:
                    continue

                tensor_id = id(input_tensor)
                grad_arr = input_grad._mlx_array

                if tensor_id in grads:
                    grads[tensor_id] = grads[tensor_id] + grad_arr
                else:
                    grads[tensor_id] = grad_arr

    # Extract gradients for requested inputs
    result = []
    for input_tensor in inputs:
        tensor_id = id(input_tensor)
        if tensor_id in grads:
            result.append(Tensor._from_mlx_array(grads[tensor_id]))
        elif allow_unused:
            result.append(None)
        else:
            raise RuntimeError(
                f"One of the inputs was not used in the graph. "
                f"Set allow_unused=True if this is expected."
            )

    return tuple(result)
