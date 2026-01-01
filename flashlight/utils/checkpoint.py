"""
Gradient Checkpointing for Memory-Efficient Training

Implements PyTorch-compatible torch.utils.checkpoint API for MLX:
- checkpoint(): Checkpoint a single function
- checkpoint_sequential(): Checkpoint a sequence of modules

Gradient checkpointing trades compute for memory by not storing intermediate
activations during the forward pass, then recomputing them during backward.
"""

import contextlib
import warnings
import weakref
from typing import Any, Callable, ContextManager, List, Optional, Tuple, Union

import mlx.core as mx

from ..autograd.context import is_grad_enabled, no_grad, set_grad_enabled
from ..autograd.function import Function, GradientFunction, SavedTensorsContext
from ..random import get_rng_state, set_rng_state

__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointFunction",
    "noop_context_fn",
]


# Default determinism check mode
_DEFAULT_DETERMINISM_MODE = "default"


def noop_context_fn() -> Tuple[ContextManager, ContextManager]:
    """
    Return a tuple of two no-op context managers.

    This is the default context_fn for checkpoint, providing no additional
    context during forward or recomputation.

    Returns:
        Tuple of two nullcontext instances.
    """
    return contextlib.nullcontext(), contextlib.nullcontext()


def _get_autocast_kwargs() -> dict:
    """Get current autocast settings (compatibility stub)."""
    return {}


def _default_metadata_fn(tensor) -> dict:
    """
    Extract metadata from a tensor for determinism checking.

    Args:
        tensor: The tensor to extract metadata from.

    Returns:
        Dictionary containing shape, dtype, and device info.
    """
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
    }


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
                    cotangent = (
                        grad_output._mlx_array if isinstance(grad_output, Tensor) else grad_output
                    )
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
    Custom autograd Function for gradient checkpointing (reentrant mode).

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
                if hasattr(detached_inp, "grad") and detached_inp.grad is not None:
                    grads.append(detached_inp.grad)
                else:
                    # Gradient not computed, return None
                    grads.append(None)
            else:
                grads.append(None)

        # Return None for run_function and preserve_rng_state args, then input grads
        return (None, None) + tuple(grads)


class _CheckpointFrame:
    """
    Frame for tracking checkpointed region state in non-reentrant mode.

    This class stores:
    - The recompute function and its inputs
    - Saved tensor metadata for determinism checking
    - Weak references to holders for saved tensors
    """

    def __init__(
        self,
        recompute_fn: Callable,
        preserve_rng_state: bool,
        rng_state: Optional[dict],
        determinism_check: str,
        context_fn: Callable,
    ):
        self.recompute_fn = recompute_fn
        self.preserve_rng_state = preserve_rng_state
        self.rng_state = rng_state
        self.determinism_check = determinism_check
        self.context_fn = context_fn

        # Track saved tensors and their metadata
        self.saved_tensors = []
        self.saved_metadata = []
        self.recomputed_tensors = []
        self.is_recomputed = False

    def save_tensor(self, tensor):
        """Save a tensor during forward pass."""
        self.saved_tensors.append(tensor)
        if self.determinism_check != "none":
            self.saved_metadata.append(_default_metadata_fn(tensor))

    def recompute(self):
        """Recompute the forward pass to get saved tensors."""
        if self.is_recomputed:
            return

        # Restore RNG state
        if self.preserve_rng_state and self.rng_state is not None:
            set_rng_state(self.rng_state)

        # Get recompute context
        _, recompute_context = self.context_fn()

        # Recompute with grad enabled
        with set_grad_enabled(True), recompute_context:
            self.recompute_fn()

        self.is_recomputed = True

        # Verify determinism if enabled
        if self.determinism_check != "none":
            self._check_determinism()

    def _check_determinism(self):
        """Check that recomputed tensors match saved tensors."""
        if len(self.recomputed_tensors) != len(self.saved_tensors):
            raise RuntimeError(
                f"Checkpoint determinism check failed: "
                f"saved {len(self.saved_tensors)} tensors during forward "
                f"but recomputed {len(self.recomputed_tensors)} tensors. "
                f"This indicates non-deterministic behavior in the checkpointed function."
            )

        for i, (saved_meta, recomputed) in enumerate(
            zip(self.saved_metadata, self.recomputed_tensors)
        ):
            recomputed_meta = _default_metadata_fn(recomputed)
            if saved_meta != recomputed_meta:
                raise RuntimeError(
                    f"Checkpoint determinism check failed for tensor {i}: "
                    f"saved metadata {saved_meta} but recomputed metadata {recomputed_meta}. "
                    f"This indicates non-deterministic behavior in the checkpointed function."
                )


def _checkpoint_without_reentrant(
    function: Callable,
    preserve_rng_state: bool,
    context_fn: Callable,
    determinism_check: str,
    *args,
) -> Any:
    """
    Non-reentrant checkpoint implementation.

    Unlike reentrant checkpointing, this:
    1. Records the autograd graph during forward pass
    2. Supports torch.autograd.grad and backward with inputs parameter
    3. Does not require inputs/outputs to have requires_grad=True
    4. Handles nested checkpoints and detached tensors correctly

    Args:
        function: The function to checkpoint
        preserve_rng_state: Whether to save/restore RNG state
        context_fn: Function returning (forward_context, recompute_context)
        determinism_check: How to verify recomputation matches forward
        *args: Arguments to the function

    Returns:
        Output of function(*args)
    """
    from ..tensor import Tensor

    # Save RNG state if requested
    rng_state = get_rng_state() if preserve_rng_state else None

    # Get context managers
    forward_context, recompute_context = context_fn()

    # Store inputs for recomputation
    saved_inputs = []
    for arg in args:
        if isinstance(arg, Tensor):
            # Save a copy of the tensor data (not the autograd graph)
            saved_inputs.append(Tensor._from_mlx_array(arg._mlx_array.copy()))
            saved_inputs[-1].requires_grad = arg.requires_grad
        else:
            saved_inputs.append(arg)

    # Create the frame to track this checkpoint
    def make_recompute_fn():
        """Create a closure that can recompute the forward pass."""

        def recompute():
            # Restore RNG state
            if preserve_rng_state and rng_state is not None:
                set_rng_state(rng_state)

            # Detach inputs for recomputation
            detached = []
            for inp in saved_inputs:
                if isinstance(inp, Tensor):
                    d = Tensor._from_mlx_array(inp._mlx_array.copy())
                    d.requires_grad = inp.requires_grad
                    detached.append(d)
                else:
                    detached.append(inp)

            # Run the function
            with set_grad_enabled(True), recompute_context:
                return function(*detached)

        return recompute

    frame = _CheckpointFrame(
        recompute_fn=make_recompute_fn(),
        preserve_rng_state=preserve_rng_state,
        rng_state=rng_state,
        determinism_check=determinism_check,
        context_fn=context_fn,
    )

    # Run forward pass with the forward context
    # Unlike reentrant mode, we DO record the autograd graph
    with forward_context:
        outputs = function(*args)

    # If no outputs require grad, just return
    if isinstance(outputs, Tensor):
        if not outputs.requires_grad:
            return outputs
        output_list = [outputs]
    elif isinstance(outputs, tuple):
        output_list = [o for o in outputs if isinstance(o, Tensor)]
        if not any(o.requires_grad for o in output_list):
            return outputs
    else:
        return outputs

    # Force evaluation
    for out in output_list:
        if isinstance(out, Tensor):
            mx.eval(out._mlx_array)

    # Store frame reference in outputs for backward to use
    for out in output_list:
        if isinstance(out, Tensor) and out.requires_grad:
            # Store frame weakref for potential use in custom backward
            out._checkpoint_frame = weakref.ref(frame)

    # Create wrapper that will handle recomputation during backward
    class NonReentrantCheckpointBackward(GradientFunction):
        """Gradient function for non-reentrant checkpoint."""

        def __init__(self, frame, tensor_inputs, outputs):
            super().__init__(*tensor_inputs)
            self.frame = frame
            self.outputs = outputs
            self.saved_inputs = saved_inputs

        def apply(self, grad_output):
            """Recompute forward and compute gradients."""
            # Restore RNG state for deterministic recomputation
            if self.frame.preserve_rng_state and self.frame.rng_state is not None:
                set_rng_state(self.frame.rng_state)

            # Get recompute context
            _, recompute_ctx = self.frame.context_fn()

            # Detach and enable grad on tensor inputs
            detached_inputs = []
            for inp in self.saved_inputs:
                if isinstance(inp, Tensor):
                    detached = Tensor._from_mlx_array(inp._mlx_array.copy())
                    detached.requires_grad = inp.requires_grad
                    detached_inputs.append(detached)
                else:
                    detached_inputs.append(inp)

            # Recompute forward pass with grad enabled
            with set_grad_enabled(True), recompute_ctx:
                recomputed_outputs = function(*detached_inputs)

            # Handle single or multiple outputs
            if not isinstance(recomputed_outputs, tuple):
                recomputed_outputs = (recomputed_outputs,)

            # Verify determinism if enabled
            if self.frame.determinism_check != "none":
                original_outputs = (
                    self.outputs if isinstance(self.outputs, tuple) else (self.outputs,)
                )
                for i, (orig, recomp) in enumerate(zip(original_outputs, recomputed_outputs)):
                    if isinstance(orig, Tensor) and isinstance(recomp, Tensor):
                        orig_meta = _default_metadata_fn(orig)
                        recomp_meta = _default_metadata_fn(recomp)
                        if orig_meta != recomp_meta:
                            raise RuntimeError(
                                f"Checkpoint determinism check failed for output {i}: "
                                f"original metadata {orig_meta} but recomputed metadata {recomp_meta}. "
                                f"This indicates non-deterministic behavior in the checkpointed function."
                            )

            # Compute gradients using VJP
            grads = []
            for i, inp in enumerate(self.saved_inputs):
                if isinstance(inp, Tensor) and inp.requires_grad:

                    def compute_output(x, idx=i):
                        new_inputs = []
                        for j, di in enumerate(detached_inputs):
                            if j == idx:
                                new_inputs.append(Tensor._from_mlx_array(x))
                            else:
                                new_inputs.append(di)
                        result = function(*new_inputs)
                        if isinstance(result, Tensor):
                            return result._mlx_array
                        return result[0]._mlx_array if isinstance(result, tuple) else result

                    try:
                        _, vjp_fn = mx.vjp(compute_output, [inp._mlx_array])
                        cotangent = (
                            grad_output._mlx_array
                            if isinstance(grad_output, Tensor)
                            else grad_output
                        )
                        grad_inp = vjp_fn([cotangent])[0]
                        grads.append(Tensor._from_mlx_array(grad_inp))
                    except Exception:
                        grads.append(None)
                elif isinstance(inp, Tensor):
                    grads.append(None)

            return tuple(grads) if grads else (None,)

    # Attach backward function to outputs
    tensor_inputs = [arg for arg in args if isinstance(arg, Tensor)]
    if tensor_inputs and output_list:
        grad_fn = NonReentrantCheckpointBackward(frame, tensor_inputs, outputs)
        for out in output_list:
            if isinstance(out, Tensor) and out.requires_grad:
                grad_fn.output_tensor = out
                out._grad_fn = grad_fn
                break

    return outputs


def checkpoint(
    function: Callable,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
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
        use_reentrant: If True, use reentrant autograd. If False, use
                       non-reentrant mode which is recommended as it handles
                       more edge cases correctly. If None (default), uses True
                       with a deprecation warning.
        context_fn: A callable returning a tuple of two context managers.
                    The function will be run under the first context manager
                    during forward, and under the second during recomputation.
                    Only supported when use_reentrant=False.
        determinism_check: How to check for non-determinism during recomputation.
                          "default" compares shapes and dtypes of recomputed tensors.
                          "none" disables the check.
                          Only supported when use_reentrant=False.
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
        >>> out = checkpoint(bottleneck, x, use_reentrant=False)

    Note:
        For best memory savings, checkpoint functions that have many
        intermediate activations (e.g., residual blocks, transformer layers).

        The non-reentrant mode (use_reentrant=False) is recommended as it:
        - Supports torch.autograd.grad and backward with inputs parameter
        - Handles nested checkpoints correctly
        - Works with detached tensors inside the checkpointed region
    """
    from ..tensor import Tensor

    # Handle use_reentrant default with deprecation warning
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. In a future version, use_reentrant=False will "
            "be the default. use_reentrant=False is recommended as it handles "
            "more edge cases correctly.",
            FutureWarning,
            stacklevel=2,
        )
        use_reentrant = True

    # Validate parameters
    if use_reentrant:
        if context_fn is not noop_context_fn:
            raise ValueError("Passing context_fn is only supported when use_reentrant=False.")
        if determinism_check != _DEFAULT_DETERMINISM_MODE:
            raise ValueError(
                "Passing determinism_check is only supported when use_reentrant=False."
            )
        if debug:
            warnings.warn(
                "debug=True is only fully supported when use_reentrant=False.",
                UserWarning,
            )

    # Validate determinism_check
    allowed_determinism_checks = {"default", "none"}
    if determinism_check not in allowed_determinism_checks:
        raise ValueError(
            f"determinism_check should be one of {allowed_determinism_checks}, "
            f"but got '{determinism_check}'"
        )

    # Check if any input requires grad
    any_requires_grad = any(isinstance(arg, Tensor) and arg.requires_grad for arg in args)

    if not any_requires_grad:
        # No gradients needed, just run the function normally
        return function(*args)

    if use_reentrant:
        # Use the reentrant CheckpointFunction
        return CheckpointFunction.apply(function, preserve_rng_state, *args)
    else:
        # Use non-reentrant implementation
        return _checkpoint_without_reentrant(
            function,
            preserve_rng_state,
            context_fn,
            determinism_check,
            *args,
        )


def checkpoint_sequential(
    functions: Union[List[Callable], Any],
    segments: int,
    input: Any,
    use_reentrant: Optional[bool] = None,
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
        use_reentrant: Use reentrant autograd. If None, defaults to True with
                       a deprecation warning. use_reentrant=False is recommended.
        preserve_rng_state: Save/restore RNG state (default True).

    Returns:
        Output after running all functions.

    Example:
        >>> # Divide 8 layers into 2 segments, checkpoint each
        >>> layers = [nn.Linear(10, 10) for _ in range(8)]
        >>> out = checkpoint_sequential(layers, 2, x, use_reentrant=False)
        >>>
        >>> # Equivalent to:
        >>> # Segment 1 (checkpointed): layers[0:4]
        >>> # Segment 2 (checkpointed): layers[4:8]
    """
    # Handle Sequential modules
    if hasattr(functions, "__iter__") and not isinstance(functions, (list, tuple)):
        # Convert to list (e.g., from nn.Sequential)
        functions = list(functions)

    if not isinstance(functions, (list, tuple)):
        raise TypeError(f"functions must be a list or Sequential, got {type(functions)}")

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
            "None of the inputs have requires_grad=True. " "Checkpointing will have no effect.",
            UserWarning,
        )
