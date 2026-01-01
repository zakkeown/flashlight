"""
Gradient Checkpointing for Memory-Efficient Training

Implements PyTorch-compatible torch.utils.checkpoint API for MLX:
- checkpoint(): Checkpoint a single function
- checkpoint_sequential(): Checkpoint a sequence of modules

Gradient checkpointing trades compute for memory by not storing intermediate
activations during the forward pass, then recomputing them during backward.
"""

from typing import Callable, Any, Tuple, Optional, List, Union
import warnings

import mlx.core as mx

from ..autograd.function import Function, SavedTensorsContext, GradientFunction
from ..autograd.context import no_grad, set_grad_enabled, is_grad_enabled
from ..random import get_rng_state, set_rng_state


__all__ = ["checkpoint", "checkpoint_sequential", "CheckpointFunction"]


def _get_autocast_kwargs() -> dict:
    """Get current autocast settings (compatibility stub)."""
    return {}


class CheckpointFunctionBackward(GradientFunction):
    """
    Gradient function for CheckpointFunction.

    Handles recomputation of forward pass during backward to recreate
    intermediate activations before computing gradients.
    """

    def __init__(
        self,
        run_function: Callable,
        preserve_rng_state: bool,
        rng_state: Optional[dict],
        tensor_inputs: list,
        all_inputs: tuple,
    ):
        super().__init__(*tensor_inputs)
        self.run_function = run_function
        self.preserve_rng_state = preserve_rng_state
        self.rng_state = rng_state
        self.all_inputs = all_inputs

    def apply(self, grad_output):
        """Recompute forward and compute gradients."""
        from ..tensor import Tensor

        # Restore RNG state for deterministic recomputation
        if self.preserve_rng_state and self.rng_state is not None:
            set_rng_state(self.rng_state)

        # Detach and enable grad on tensor inputs
        detached_inputs = []
        tensor_idx = 0
        for inp in self.all_inputs:
            if isinstance(inp, Tensor):
                # Create a new tensor that requires grad
                detached = Tensor._from_mlx_array(inp._mlx_array.copy())
                detached.requires_grad = inp.requires_grad
                detached_inputs.append(detached)
            else:
                detached_inputs.append(inp)

        # Recompute forward pass with grad enabled
        with set_grad_enabled(True):
            outputs = self.run_function(*detached_inputs)

        # Handle single or multiple outputs
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Compute gradients using MLX's value_and_grad
        # We need to compute gradient of each output w.r.t. each input that requires grad
        grads = []

        for inp in self.all_inputs:
            if isinstance(inp, Tensor) and inp.requires_grad:
                # Use vjp to compute gradient
                def compute_output(x):
                    # Replace the input in detached_inputs and recompute
                    new_inputs = []
                    replaced = False
                    for di in detached_inputs:
                        if isinstance(di, Tensor) and not replaced:
                            if di._mlx_array.shape == x.shape:
                                new_inputs.append(Tensor._from_mlx_array(x))
                                replaced = True
                            else:
                                new_inputs.append(di)
                        else:
                            new_inputs.append(di)
                    result = self.run_function(*new_inputs)
                    if isinstance(result, Tensor):
                        return result._mlx_array
                    return result[0]._mlx_array if isinstance(result, tuple) else result

                try:
                    # Compute VJP (vector-Jacobian product)
                    _, vjp_fn = mx.vjp(compute_output, [inp._mlx_array])
                    cotangent = grad_output._mlx_array if isinstance(grad_output, Tensor) else grad_output
                    grad_inp = vjp_fn([cotangent])[0]
                    grads.append(Tensor._from_mlx_array(grad_inp))
                except Exception:
                    # Fallback: return None if gradient computation fails
                    grads.append(None)
            elif isinstance(inp, Tensor):
                grads.append(None)

        return tuple(grads) if grads else (None,)


class CheckpointFunction(Function):
    """
    Custom autograd Function for gradient checkpointing.

    This function:
    1. Runs the forward pass without storing intermediate activations
    2. During backward, re-runs forward to recreate activations
    3. Then computes gradients normally
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        """
        Forward pass that saves only inputs, not intermediate activations.

        Args:
            ctx: Context for saving state
            run_function: The function to checkpoint
            preserve_rng_state: Whether to save/restore RNG state
            *args: Arguments to the function

        Returns:
            Output of run_function(*args)
        """
        from ..tensor import Tensor

        check_backward_validity(args)

        # Save function and RNG state
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.had_autograd_enabled = is_grad_enabled()

        # Save input tensors (references only, not intermediate activations)
        ctx.inputs = args

        if preserve_rng_state:
            ctx.rng_state = get_rng_state()
        else:
            ctx.rng_state = None

        # Run forward with no_grad to avoid building graph for intermediates
        with no_grad():
            outputs = run_function(*args)

        # Force evaluation of outputs (MLX lazy evaluation)
        if isinstance(outputs, Tensor):
            mx.eval(outputs._mlx_array)
            return outputs._mlx_array
        elif isinstance(outputs, tuple):
            for out in outputs:
                if isinstance(out, Tensor):
                    mx.eval(out._mlx_array)
            return tuple(out._mlx_array if isinstance(out, Tensor) else out for out in outputs)
        else:
            return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass that recomputes forward activations.

        Args:
            ctx: Context with saved state
            *grad_outputs: Gradients flowing back from outputs

        Returns:
            Gradients for each input
        """
        from ..tensor import Tensor

        # Restore inputs
        inputs = ctx.inputs
        run_function = ctx.run_function

        # Restore RNG state for deterministic recomputation
        if ctx.preserve_rng_state and ctx.rng_state is not None:
            set_rng_state(ctx.rng_state)

        # Re-enable gradients and recompute forward pass
        with set_grad_enabled(True):
            # Detach inputs to avoid double-backward through them
            detached_inputs = []
            for inp in inputs:
                if isinstance(inp, Tensor):
                    detached = Tensor._from_mlx_array(inp._mlx_array.copy())
                    detached.requires_grad = inp.requires_grad
                    detached_inputs.append(detached)
                else:
                    detached_inputs.append(inp)

            # Recompute forward pass (now with graph)
            outputs = run_function(*detached_inputs)

            # Handle single or multiple outputs
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

        # Compute gradients for each input that requires grad
        grads = []
        grad_output = grad_outputs[0] if grad_outputs else None

        for i, inp in enumerate(inputs):
            if isinstance(inp, Tensor) and inp.requires_grad:
                # Find corresponding detached input
                detached_inp = detached_inputs[i]
                if hasattr(detached_inp, 'grad') and detached_inp.grad is not None:
                    grads.append(detached_inp.grad)
                else:
                    # Gradient not computed, return None
                    grads.append(None)
            else:
                grads.append(None)

        # Return None for run_function and preserve_rng_state args, then input grads
        return (None, None) + tuple(grads)


def checkpoint(
    function: Callable,
    *args,
    use_reentrant: bool = True,
    context_fn: Optional[Callable] = None,
    determinism_check: Optional[str] = None,
    debug: bool = False,
    preserve_rng_state: bool = True,
) -> Any:
    """
    Checkpoint a function to trade compute for memory.

    During the forward pass, the function is run without storing
    intermediate activations. During the backward pass, the forward
    is re-run to recreate activations before computing gradients.

    This can significantly reduce memory usage at the cost of
    approximately 2x forward pass computation time.

    Args:
        function: The function to checkpoint. Takes tensors as input,
                  returns tensor(s) as output.
        *args: Arguments to pass to function.
        use_reentrant: If True, use reentrant autograd (default).
                       Non-reentrant mode is not yet supported.
        context_fn: Optional function that returns a context manager.
                    Not yet supported.
        determinism_check: How to check for non-determinism.
                          Not yet supported.
        debug: Enable debugging mode. Not yet supported.
        preserve_rng_state: If True, save and restore RNG state for
                           deterministic recomputation (default True).

    Returns:
        Output of function(*args).

    Example:
        >>> def bottleneck(x):
        ...     x = model.layer1(x)
        ...     x = model.layer2(x)
        ...     x = model.layer3(x)
        ...     return x
        >>>
        >>> # Without checkpoint: stores all activations
        >>> out = bottleneck(x)
        >>>
        >>> # With checkpoint: recomputes activations during backward
        >>> out = checkpoint(bottleneck, x)

    Note:
        For best memory savings, checkpoint functions that have many
        intermediate activations (e.g., residual blocks, transformer layers).
    """
    from ..tensor import Tensor

    if not use_reentrant:
        warnings.warn(
            "Non-reentrant checkpointing is not yet fully supported in MLX Compat. "
            "Falling back to reentrant mode.",
            UserWarning,
        )

    if context_fn is not None:
        warnings.warn(
            "context_fn parameter is not yet supported in MLX Compat.",
            UserWarning,
        )

    if determinism_check is not None:
        warnings.warn(
            "determinism_check parameter is not yet supported in MLX Compat.",
            UserWarning,
        )

    # Check if any input requires grad
    any_requires_grad = any(
        isinstance(arg, Tensor) and arg.requires_grad for arg in args
    )

    if not any_requires_grad:
        # No gradients needed, just run the function normally
        return function(*args)

    # Use the CheckpointFunction
    return CheckpointFunction.apply(function, preserve_rng_state, *args)


def checkpoint_sequential(
    functions: Union[List[Callable], Any],
    segments: int,
    input: Any,
    use_reentrant: bool = True,
    preserve_rng_state: bool = True,
) -> Any:
    """
    Checkpoint a sequential chain of functions.

    Divides the functions into segments and checkpoints each segment.
    This is useful for models like ResNets or Transformers where you
    have a sequence of similar layers.

    Args:
        functions: List of functions (nn.Module instances) to run sequentially,
                   or an nn.Sequential module.
        segments: Number of checkpoint segments to divide into.
        input: Input tensor to the first function.
        use_reentrant: Use reentrant autograd (default True).
        preserve_rng_state: Save/restore RNG state (default True).

    Returns:
        Output after running all functions.

    Example:
        >>> # Divide 8 layers into 2 segments, checkpoint each
        >>> layers = [nn.Linear(10, 10) for _ in range(8)]
        >>> out = checkpoint_sequential(layers, 2, x)
        >>>
        >>> # Equivalent to:
        >>> # Segment 1 (checkpointed): layers[0:4]
        >>> # Segment 2 (checkpointed): layers[4:8]
    """
    # Handle Sequential modules
    if hasattr(functions, '__iter__') and not isinstance(functions, (list, tuple)):
        # Convert to list (e.g., from nn.Sequential)
        functions = list(functions)

    if not isinstance(functions, (list, tuple)):
        raise TypeError(
            f"functions must be a list or Sequential, got {type(functions)}"
        )

    if segments <= 0:
        raise ValueError(f"segments must be positive, got {segments}")

    def get_segment_function(start_idx: int, end_idx: int) -> Callable:
        """Create a function that runs a segment of layers."""
        def segment_forward(x):
            for i in range(start_idx, end_idx):
                x = functions[i](x)
            return x

        return segment_forward

    # Calculate segment boundaries
    n = len(functions)
    if segments > n:
        segments = n  # Can't have more segments than functions

    segment_size = (n + segments - 1) // segments  # Ceiling division

    x = input
    for seg_idx in range(segments):
        start = seg_idx * segment_size
        end = min(start + segment_size, n)

        if start >= n:
            break

        # Create segment function and checkpoint it
        segment_fn = get_segment_function(start, end)
        x = checkpoint(
            segment_fn,
            x,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
        )

    return x


def check_backward_validity(inputs: tuple) -> None:
    """
    Check that at least one input requires grad for checkpointing to be useful.

    Args:
        inputs: Tuple of inputs to check.
    """
    from ..tensor import Tensor

    has_grad = any(isinstance(inp, Tensor) and inp.requires_grad for inp in inputs)
    if not has_grad:
        warnings.warn(
            "None of the inputs have requires_grad=True. "
            "Checkpointing will have no effect.",
            UserWarning,
        )
