"""
Gradient Functions and Custom Function Base Class

This module implements:
1. GradientFunction: Base class for all gradient functions (backward passes)
2. Function: User-facing base class for custom autograd operations
3. Specific gradient functions for all 55 operators
"""

from typing import Any, Optional, Tuple

import mlx.core as mx


def _unbroadcast(grad, target_shape):
    """
    Unbroadcast gradient to match target shape by summing along broadcast dimensions.

    Args:
        grad: Gradient tensor (may have been broadcast during forward pass)
        target_shape: Original shape before broadcasting

    Returns:
        Gradient tensor with target_shape
    """
    from ..tensor import Tensor

    # Sum along dimensions that were added by broadcasting
    ndim_added = len(grad.shape) - len(target_shape)
    if ndim_added > 0:
        # Sum along the leading dimensions that were added
        for _ in range(ndim_added):
            grad = Tensor._from_mlx_array(mx.sum(grad._mlx_array, axis=0))

    # Sum along dimensions that were size 1 in target_shape
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim > 1:
            grad = Tensor._from_mlx_array(mx.sum(grad._mlx_array, axis=i, keepdims=True))

    return grad


class GradientFunction:
    """
    Base class for gradient functions.

    Each operator creates a GradientFunction instance during forward pass
    that stores necessary information for computing gradients during backward pass.
    """

    def __init__(self, *inputs):
        """
        Args:
            *inputs: Input tensors that produced the output
        """
        self.inputs = inputs
        self.next_functions = []  # For building computation graph
        self.output_tensor = None  # Set by operator after creation

    def apply(self, grad_output):
        """
        Compute gradients with respect to inputs.

        Args:
            grad_output: Gradient flowing back from output

        Returns:
            Tuple of gradients for each input (None if input doesn't require grad)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply()")


class SavedTensorsContext:
    """Context for saving tensors in custom Functions."""

    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self.saved_tensors = tensors


class Function:
    """
    Base class for custom autograd functions.

    Users subclass this and implement forward() and backward() methods.

    Example:
        class MyReLU(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return mx.maximum(input._mlx_array, 0)

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                return grad_output * (input._mlx_array > 0)
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Forward pass computation.

        Args:
            ctx: Context object for saving tensors
            *args: Input tensors and other arguments
            **kwargs: Keyword arguments

        Returns:
            Output tensor(s)
        """
        raise NotImplementedError("Must implement forward()")

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass computation.

        Args:
            ctx: Context object with saved tensors
            *grad_outputs: Gradients flowing back from outputs

        Returns:
            Gradients for each input (None for non-tensor inputs)
        """
        raise NotImplementedError("Must implement backward()")

    @classmethod
    def apply(cls, *args, **kwargs):
        """
        Apply the custom function (called by users).

        This method:
        1. Creates a context
        2. Calls forward()
        3. Wraps result in Tensor
        4. Attaches gradient function
        """
        from ..tensor import Tensor

        ctx = SavedTensorsContext()

        # Call forward pass
        output_data = cls.forward(ctx, *args, **kwargs)

        # Wrap in Tensor
        if isinstance(output_data, mx.array):
            output = Tensor._from_mlx_array(output_data)
        else:
            output = output_data

        # Attach gradient function if any input requires grad
        requires_grad = any(isinstance(arg, Tensor) and arg.requires_grad for arg in args)

        if requires_grad:
            output.requires_grad = True
            # Create custom gradient function
            grad_fn = CustomFunctionBackward(cls, ctx, args)
            grad_fn.output_tensor = output
            output._grad_fn = grad_fn

        return output


class CustomFunctionBackward(GradientFunction):
    """Gradient function for custom Function subclasses."""

    def __init__(self, function_cls, ctx, inputs):
        from ..tensor import Tensor

        # Only pass Tensor inputs to the base class
        tensor_inputs = [arg for arg in inputs if isinstance(arg, Tensor)]
        super().__init__(*tensor_inputs)
        self.function_cls = function_cls
        self.ctx = ctx
        self.all_inputs = inputs  # Keep all inputs for backward

    def apply(self, grad_output):
        """Call the user-defined backward function."""
        grads = self.function_cls.backward(self.ctx, grad_output)

        # Ensure we return a tuple
        if not isinstance(grads, tuple):
            grads = (grads,)

        # Pad with None for inputs that don't need gradients (just for tensor inputs)
        if len(grads) < len(self.inputs):
            grads = grads + (None,) * (len(self.inputs) - len(grads))

        return grads


# ============================================================================
# Gradient Functions for Arithmetic Operators
# ============================================================================


class AddBackward(GradientFunction):
    """Gradient for addition: d(a+b)/da = 1, d(a+b)/db = 1"""

    def __init__(self, input_a, input_b, alpha=1):
        super().__init__(input_a, input_b)
        self.alpha = alpha
        self.input_a_shape = input_a.shape
        self.input_b_shape = input_b.shape

    def apply(self, grad_output):
        import mlx.core as mx

        from ..tensor import Tensor

        grad_a = None
        grad_b = None

        if self.inputs[0].requires_grad:
            grad_a = grad_output
            # Handle broadcasting: sum out dimensions that were broadcast
            if grad_a.shape != self.input_a_shape:
                grad_a = _unbroadcast(grad_a, self.input_a_shape)

        if self.inputs[1].requires_grad:
            grad_b = grad_output * self.alpha
            # Handle broadcasting
            if grad_b.shape != self.input_b_shape:
                grad_b = _unbroadcast(grad_b, self.input_b_shape)

        return grad_a, grad_b


class SubBackward(GradientFunction):
    """Gradient for subtraction: d(a-b)/da = 1, d(a-b)/db = -1"""

    def __init__(self, input_a, input_b, alpha=1):
        super().__init__(input_a, input_b)
        self.alpha = alpha
        self.input_a_shape = input_a.shape
        self.input_b_shape = input_b.shape

    def apply(self, grad_output):
        from ..tensor import Tensor

        grad_a = None
        grad_b = None

        if self.inputs[0].requires_grad:
            grad_a = grad_output
            # Handle broadcasting: sum out dimensions that were broadcast
            if grad_a.shape != self.input_a_shape:
                grad_a = _unbroadcast(grad_a, self.input_a_shape)

        if self.inputs[1].requires_grad:
            grad_b = Tensor._from_mlx_array(-grad_output._mlx_array * self.alpha)
            # Handle broadcasting: sum out dimensions that were broadcast
            if grad_b.shape != self.input_b_shape:
                grad_b = _unbroadcast(grad_b, self.input_b_shape)

        return grad_a, grad_b


class MulBackward(GradientFunction):
    """Gradient for multiplication: d(a*b)/da = b, d(a*b)/db = a"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        grad_a = None
        grad_b = None

        if self.inputs[0].requires_grad:
            grad_a = Tensor._from_mlx_array(grad_output._mlx_array * self.inputs[1]._mlx_array)

        if self.inputs[1].requires_grad:
            grad_b = Tensor._from_mlx_array(grad_output._mlx_array * self.inputs[0]._mlx_array)

        return grad_a, grad_b


class DivBackward(GradientFunction):
    """Gradient for division: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        grad_a = None
        grad_b = None

        if self.inputs[0].requires_grad:
            grad_a = Tensor._from_mlx_array(grad_output._mlx_array / self.inputs[1]._mlx_array)

        if self.inputs[1].requires_grad:
            grad_b = Tensor._from_mlx_array(
                -grad_output._mlx_array
                * self.inputs[0]._mlx_array
                / (self.inputs[1]._mlx_array ** 2)
            )

        return grad_a, grad_b


class MatmulBackward(GradientFunction):
    """Gradient for matrix multiplication: d(AB)/dA = grad @ B.T, d(AB)/dB = A.T @ grad"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        grad_a = None
        grad_b = None

        if self.inputs[0].requires_grad:
            # grad_a = grad_output @ input_b.T
            grad_a = Tensor._from_mlx_array(
                mx.matmul(grad_output._mlx_array, mx.swapaxes(self.inputs[1]._mlx_array, -1, -2))
            )

        if self.inputs[1].requires_grad:
            # grad_b = input_a.T @ grad_output
            grad_b = Tensor._from_mlx_array(
                mx.matmul(mx.swapaxes(self.inputs[0]._mlx_array, -1, -2), grad_output._mlx_array)
            )

        return grad_a, grad_b


class PowBackward(GradientFunction):
    """Gradient for power: d(a^b)/da = b * a^(b-1), d(a^b)/db = a^b * log(a)"""

    def __init__(self, input_a, input_b, result):
        super().__init__(input_a, input_b)
        self.result = result  # Save result for efficiency

    def apply(self, grad_output):
        from ..tensor import Tensor

        grad_a = None
        grad_b = None

        if self.inputs[0].requires_grad:
            # d/da = b * a^(b-1) = b * result / a
            grad_a = Tensor._from_mlx_array(
                grad_output._mlx_array
                * self.inputs[1]._mlx_array
                * self.result._mlx_array
                / self.inputs[0]._mlx_array
            )

        if self.inputs[1].requires_grad:
            # d/db = a^b * log(a) = result * log(a)
            grad_b = Tensor._from_mlx_array(
                grad_output._mlx_array * self.result._mlx_array * mx.log(self.inputs[0]._mlx_array)
            )

        return grad_a, grad_b


class SqrtBackward(GradientFunction):
    """Gradient for sqrt: d(sqrt(x))/dx = 1/(2*sqrt(x))"""

    def __init__(self, input_tensor, result):
        super().__init__(input_tensor)
        self.result = result

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = 1/(2*sqrt(x)) = 1/(2*result)
        grad = Tensor._from_mlx_array(grad_output._mlx_array / (2 * self.result._mlx_array))
        return (grad,)


class ExpBackward(GradientFunction):
    """Gradient for exp: d(exp(x))/dx = exp(x)"""

    def __init__(self, input_tensor, result):
        super().__init__(input_tensor)
        self.result = result

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = exp(x) = result
        grad = Tensor._from_mlx_array(grad_output._mlx_array * self.result._mlx_array)
        return (grad,)


class LogBackward(GradientFunction):
    """Gradient for log: d(log(x))/dx = 1/x"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = 1/x
        grad = Tensor._from_mlx_array(grad_output._mlx_array / self.inputs[0]._mlx_array)
        return (grad,)


class AbsBackward(GradientFunction):
    """Gradient for abs: d(|x|)/dx = sign(x)"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = sign(x)
        grad = Tensor._from_mlx_array(grad_output._mlx_array * mx.sign(self.inputs[0]._mlx_array))
        return (grad,)


class NegBackward(GradientFunction):
    """Gradient for negation: d(-x)/dx = -1"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = -1
        grad = Tensor._from_mlx_array(-grad_output._mlx_array)
        return (grad,)


# ============================================================================
# Gradient Functions for Activation Functions
# ============================================================================


class ReLUBackward(GradientFunction):
    """Gradient for ReLU: d(relu(x))/dx = 1 if x > 0 else 0"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = 1 if x > 0 else 0
        mask = (self.inputs[0]._mlx_array > 0).astype(grad_output._mlx_array.dtype)
        grad = Tensor._from_mlx_array(grad_output._mlx_array * mask)
        return (grad,)


class SigmoidBackward(GradientFunction):
    """Gradient for sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))"""

    def __init__(self, input_tensor, result):
        super().__init__(input_tensor)
        self.result = result

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = sigmoid(x) * (1 - sigmoid(x))
        grad = Tensor._from_mlx_array(
            grad_output._mlx_array * self.result._mlx_array * (1 - self.result._mlx_array)
        )
        return (grad,)


class TanhBackward(GradientFunction):
    """Gradient for tanh: d(tanh(x))/dx = 1 - tanh^2(x)"""

    def __init__(self, input_tensor, result):
        super().__init__(input_tensor)
        self.result = result

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = 1 - tanh^2(x) = 1 - result^2
        grad = Tensor._from_mlx_array(grad_output._mlx_array * (1 - self.result._mlx_array**2))
        return (grad,)


class SoftmaxBackward(GradientFunction):
    """Gradient for softmax: complex Jacobian-vector product"""

    def __init__(self, input_tensor, result, dim):
        super().__init__(input_tensor)
        self.result = result
        self.dim = dim

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Softmax gradient: grad_input = softmax * (grad_output - sum(softmax * grad_output))
        s = self.result._mlx_array
        grad_out = grad_output._mlx_array

        sum_term = mx.sum(s * grad_out, axis=self.dim, keepdims=True)
        grad = Tensor._from_mlx_array(s * (grad_out - sum_term))
        return (grad,)


class LogSoftmaxBackward(GradientFunction):
    """Gradient for log_softmax"""

    def __init__(self, input_tensor, result, dim):
        super().__init__(input_tensor)
        self.result = result
        self.dim = dim

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # log_softmax gradient: grad_input = grad_output - exp(log_softmax) * sum(grad_output)
        log_s = self.result._mlx_array
        grad_out = grad_output._mlx_array

        sum_term = mx.sum(grad_out, axis=self.dim, keepdims=True)
        grad = Tensor._from_mlx_array(grad_out - mx.exp(log_s) * sum_term)
        return (grad,)


class SiLUBackward(GradientFunction):
    """Gradient for SiLU/Swish: d(x * sigmoid(x))/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""

    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Compute sigmoid
        sigmoid = 1 / (1 + mx.exp(-self.inputs[0]._mlx_array))

        # d/dx = sigmoid + x * sigmoid * (1 - sigmoid)
        grad = Tensor._from_mlx_array(
            grad_output._mlx_array * (sigmoid + self.inputs[0]._mlx_array * sigmoid * (1 - sigmoid))
        )
        return (grad,)


class LeakyReLUBackward(GradientFunction):
    """Gradient for Leaky ReLU: d/dx = 1 if x > 0 else negative_slope"""

    def __init__(self, input_tensor, negative_slope):
        super().__init__(input_tensor)
        self.negative_slope = negative_slope

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = 1 if x > 0 else negative_slope
        mask = (self.inputs[0]._mlx_array > 0).astype(grad_output._mlx_array.dtype)
        grad_mask = mask + (1 - mask) * self.negative_slope
        grad = Tensor._from_mlx_array(grad_output._mlx_array * grad_mask)
        return (grad,)


class ELUBackward(GradientFunction):
    """Gradient for ELU: d/dx = 1 if x > 0 else alpha * exp(x)"""

    def __init__(self, input_tensor, result, alpha):
        super().__init__(input_tensor)
        self.result = result
        self.alpha = alpha

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = 1 if x > 0 else alpha * exp(x) = result + alpha
        mask = (self.inputs[0]._mlx_array > 0).astype(grad_output._mlx_array.dtype)
        grad_mask = mask + (1 - mask) * (self.result._mlx_array + self.alpha)
        grad = Tensor._from_mlx_array(grad_output._mlx_array * grad_mask)
        return (grad,)


class GELUBackward(GradientFunction):
    """Gradient for GELU using analytical formula.

    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    d/dx GELU(x) = Phi(x) + x * phi(x)
    where phi(x) is the PDF of standard normal = exp(-x^2/2) / sqrt(2*pi)
    """

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        x = self.inputs[0]._mlx_array

        # Use the tanh approximation for GELU gradient
        # This matches the implementation in mlx.nn.gelu
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        a = 0.044715

        # tanh_arg = sqrt(2/pi) * (x + a * x^3)
        tanh_arg = sqrt_2_over_pi * (x + a * x * x * x)
        tanh_val = mx.tanh(tanh_arg)

        # d/dx tanh(arg) = sech^2(arg) * d(arg)/dx
        # d(arg)/dx = sqrt(2/pi) * (1 + 3*a*x^2)
        sech2 = 1 - tanh_val * tanh_val
        darg_dx = sqrt_2_over_pi * (1 + 3 * a * x * x)

        # GELU(x) = 0.5 * x * (1 + tanh(arg))
        # d/dx GELU(x) = 0.5 * (1 + tanh(arg)) + 0.5 * x * sech^2(arg) * darg_dx
        grad_value = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * darg_dx

        grad = Tensor._from_mlx_array(grad_output._mlx_array * grad_value)
        return (grad,)


# ============================================================================
# Gradient Functions for Reduction Operators
# ============================================================================


class SumBackward(GradientFunction):
    """Gradient for sum: broadcasts gradient back to input shape"""

    def __init__(self, input_tensor, dim, keepdim):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
        self.input_shape = input_tensor.shape

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Broadcast gradient back to input shape
        grad = grad_output._mlx_array

        if not self.keepdim and self.dim is not None:
            # Need to unsqueeze the dimensions that were reduced
            if isinstance(self.dim, (tuple, list)):
                for d in sorted(self.dim):
                    grad = mx.expand_dims(grad, axis=d)
            else:
                grad = mx.expand_dims(grad, axis=self.dim)

        # Broadcast to input shape
        grad = mx.broadcast_to(grad, self.input_shape)
        return (Tensor._from_mlx_array(grad),)


class MeanBackward(GradientFunction):
    """Gradient for mean: broadcasts gradient / count back to input shape"""

    def __init__(self, input_tensor, dim, keepdim):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
        self.input_shape = input_tensor.shape

        # Calculate count of elements being averaged
        if dim is None:
            self.count = 1
            for s in input_tensor.shape:
                self.count *= s
        elif isinstance(dim, (tuple, list)):
            self.count = 1
            for d in dim:
                self.count *= input_tensor.shape[d]
        else:
            self.count = input_tensor.shape[dim]

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Divide by count and broadcast back
        grad = grad_output._mlx_array / self.count

        if not self.keepdim and self.dim is not None:
            if isinstance(self.dim, (tuple, list)):
                for d in sorted(self.dim):
                    grad = mx.expand_dims(grad, axis=d)
            else:
                grad = mx.expand_dims(grad, axis=self.dim)

        grad = mx.broadcast_to(grad, self.input_shape)
        return (Tensor._from_mlx_array(grad),)


class MaxBackward(GradientFunction):
    """Gradient for max: gradient distributed equally among tied maximum elements (PyTorch behavior)"""

    def __init__(self, input_tensor, dim, keepdim, indices=None):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
        self.indices = indices
        self.input_shape = input_tensor.shape

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        input_arr = self.inputs[0]._mlx_array

        if self.dim is None:
            # Global max: distribute gradient among all max elements
            max_val = mx.max(input_arr)
            is_max = mx.equal(input_arr, max_val).astype(input_arr.dtype)

            # Count number of max elements to distribute gradient
            num_max = mx.sum(is_max)

            # Distribute gradient equally among all max elements
            grad_scalar = grad_output._mlx_array.flatten()[0]
            grad_in = is_max * (grad_scalar / num_max)

            return (Tensor._from_mlx_array(grad_in),)
        elif isinstance(self.dim, (tuple, list)):
            # Multi-dimension max (e.g., dim=(1, 2))
            max_vals = mx.max(input_arr, axis=self.dim, keepdims=True)
            is_max = mx.equal(input_arr, max_vals).astype(input_arr.dtype)

            # Count ties along reduced dimensions
            num_max = mx.sum(is_max, axis=self.dim, keepdims=True)

            # Expand grad_output to match input shape if keepdim=False
            grad_out = grad_output._mlx_array
            if not self.keepdim:
                for d in sorted(self.dim):
                    grad_out = mx.expand_dims(grad_out, axis=d)

            # Distribute gradient equally among ties
            grad_in = is_max * mx.broadcast_to(grad_out / num_max, self.input_shape)

            return (Tensor._from_mlx_array(grad_in),)
        else:
            # Single dimension max
            max_vals = mx.max(input_arr, axis=self.dim, keepdims=True)
            is_max = mx.equal(input_arr, max_vals).astype(input_arr.dtype)

            # Count ties along dimension
            num_max = mx.sum(is_max, axis=self.dim, keepdims=True)

            # Expand grad_output to match input shape if keepdim=False
            grad_out = grad_output._mlx_array
            if not self.keepdim:
                grad_out = mx.expand_dims(grad_out, axis=self.dim)

            # Distribute gradient equally among ties
            grad_in = is_max * mx.broadcast_to(grad_out / num_max, self.input_shape)

            return (Tensor._from_mlx_array(grad_in),)


# Placeholder backward functions for other operators
# These will need full implementations


class CatBackward(GradientFunction):
    """Gradient for concatenation: split gradient back to inputs"""

    def __init__(self, tensors, dim):
        super().__init__(*tensors)
        self.dim = dim
        self.sizes = [t.shape[dim] for t in tensors]

    def apply(self, grad_output):
        from ..tensor import Tensor

        # Split gradient along concat dimension
        grads = []
        start = 0
        for size in self.sizes:
            # Use slicing to extract the gradient for this input
            indices = [slice(None)] * grad_output.ndim
            indices[self.dim] = slice(start, start + size)
            grad_slice = grad_output._mlx_array[tuple(indices)]
            grads.append(
                Tensor._from_mlx_array(grad_slice)
                if self.inputs[len(grads)].requires_grad
                else None
            )
            start += size
        return tuple(grads)


class ViewBackward(GradientFunction):
    """Gradient for view/reshape: reshape gradient back to input shape"""

    def __init__(self, input_tensor):
        super().__init__(input_tensor)
        self.input_shape = input_tensor.shape

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)
        grad = Tensor._from_mlx_array(grad_output._mlx_array.reshape(self.input_shape))
        return (grad,)


class TransposeBackward(GradientFunction):
    """Gradient for transpose: transpose gradient back"""

    def __init__(self, input_tensor, dim0, dim1):
        super().__init__(input_tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        import mlx.core as mx

        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Transpose gradient back using the same dimensions
        ndim = len(grad_output.shape)
        axes = list(range(ndim))
        axes[self.dim0], axes[self.dim1] = axes[self.dim1], self.dim0

        grad = Tensor._from_mlx_array(mx.transpose(grad_output._mlx_array, axes))
        return (grad,)


# ============================================================================
# Gradient Functions for Trigonometric Functions
# ============================================================================


class SinBackward(GradientFunction):
    """Gradient for sin: d(sin(x))/dx = cos(x)"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = cos(x)
        grad = Tensor._from_mlx_array(grad_output._mlx_array * mx.cos(self.inputs[0]._mlx_array))
        return (grad,)


class CosBackward(GradientFunction):
    """Gradient for cos: d(cos(x))/dx = -sin(x)"""

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # d/dx = -sin(x)
        grad = Tensor._from_mlx_array(-grad_output._mlx_array * mx.sin(self.inputs[0]._mlx_array))
        return (grad,)


# ============================================================================
# Gradient Functions for Convolution Operations
# ============================================================================


class Conv2dBackward(GradientFunction):
    """
    Gradient for conv2d.

    For y = conv2d(x, w, bias):
    - d(loss)/d(x) = conv_transpose2d(d(loss)/d(y), w)
    - d(loss)/d(w) = conv2d(x.T, d(loss)/d(y).T).T (correlation)
    - d(loss)/d(bias) = sum(d(loss)/d(y), dims=[0,2,3])
    """

    def __init__(self, input_tensor, weight, bias, stride, padding, dilation, groups, nhwc_native):
        inputs = [input_tensor, weight] if bias is None else [input_tensor, weight, bias]
        super().__init__(*inputs)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.nhwc_native = nhwc_native
        self.has_bias = bias is not None
        # Store shapes for gradient computation
        self.input_shape = input_tensor.shape
        self.weight_shape = weight.shape

    def apply(self, grad_output):
        from ..layout import Layout
        from ..tensor import Tensor

        input_tensor = self.inputs[0]
        weight = self.inputs[1]

        grad_input = None
        grad_weight = None
        grad_bias = None

        # Get grad_output in NHWC format for MLX operations
        if self.nhwc_native:
            grad_nhwc = grad_output._mlx_array
        else:
            # grad_output is NCHW, convert to NHWC
            grad_nhwc = mx.transpose(grad_output._mlx_array, [0, 2, 3, 1])

        # Get input in NHWC format
        if (
            self.nhwc_native
            and hasattr(input_tensor, "_layout")
            and input_tensor._layout == Layout.NHWC
        ):
            input_nhwc = input_tensor._mlx_array
        else:
            input_nhwc = mx.transpose(input_tensor._mlx_array, [0, 2, 3, 1])

        # Weight is stored as [out, in, kH, kW], MLX wants [out, kH, kW, in]
        weight_mlx = mx.transpose(weight._mlx_array, [0, 2, 3, 1])

        if input_tensor.requires_grad:
            # Use MLX's vector-Jacobian product for correct gradient computation
            # Define the forward function in NHWC format
            def conv_fn(x):
                return mx.conv2d(
                    x,
                    weight_mlx,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )

            # Compute gradient using MLX's vjp (vector-Jacobian product)
            # vjp returns (primals, vjps) where vjps = cotangents @ jacobian
            _, grad_input_list = mx.vjp(conv_fn, [input_nhwc], [grad_nhwc])
            grad_input_nhwc = grad_input_list[0]

            # Convert back to NCHW if needed
            if self.nhwc_native:
                grad_input = Tensor._from_mlx_array(grad_input_nhwc, layout=Layout.NHWC)
            else:
                grad_input_nchw = mx.transpose(grad_input_nhwc, [0, 3, 1, 2])
                grad_input = Tensor._from_mlx_array(grad_input_nchw)

        if weight.requires_grad:
            # Compute weight gradient using MLX's vjp
            def conv_wrt_weight(w):
                return mx.conv2d(
                    input_nhwc,
                    w,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )

            _, grad_weight_list = mx.vjp(conv_wrt_weight, [weight_mlx], [grad_nhwc])
            grad_weight_mlx = grad_weight_list[0]

            # Convert from MLX format [out, kH, kW, in] to PyTorch format [out, in, kH, kW]
            grad_weight_arr = mx.transpose(grad_weight_mlx, [0, 3, 1, 2])
            grad_weight = Tensor._from_mlx_array(grad_weight_arr)

        if self.has_bias and self.inputs[2].requires_grad:
            # Gradient w.r.t. bias: sum over batch and spatial dimensions
            # grad_output shape: [N, H, W, C] in NHWC
            grad_bias_arr = mx.sum(grad_nhwc, axis=(0, 1, 2))
            grad_bias = Tensor._from_mlx_array(grad_bias_arr)

        if self.has_bias:
            return (grad_input, grad_weight, grad_bias)
        else:
            return (grad_input, grad_weight)


# ============================================================================
# Gradient Functions for Pooling Operations
# ============================================================================


class MaxPool2dBackward(GradientFunction):
    """
    Gradient for max_pool2d.

    Uses MLX's native gradient computation via vjp for maximum performance.
    MLX's pooling layers are differentiable, so we leverage that instead of
    implementing a manual backward pass.
    """

    def __init__(
        self, input_tensor, kernel_size, stride, padding, nhwc_native, input_nhwc, output_nhwc
    ):
        super().__init__(input_tensor)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nhwc_native = nhwc_native
        # Store the input in NHWC format for backward pass
        self.input_nhwc = input_nhwc
        self.input_shape = input_tensor.shape

    def apply(self, grad_output):
        import mlx.nn as mxnn

        from ..layout import Layout
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Get grad_output in NHWC format
        if self.nhwc_native:
            grad_nhwc = grad_output._mlx_array
        else:
            grad_nhwc = mx.transpose(grad_output._mlx_array, [0, 2, 3, 1])

        # Create the MLX pooling layer with same parameters
        pool = mxnn.MaxPool2d(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

        # Use MLX's native vjp (vector-Jacobian product) to compute gradients
        # This is much faster than our manual Python loop implementation
        _, vjp_fn = mx.vjp(pool, [self.input_nhwc], [grad_nhwc])
        grad_input_nhwc = vjp_fn[0]

        # Convert back to NCHW if needed
        if self.nhwc_native:
            grad_input = Tensor._from_mlx_array(grad_input_nhwc, layout=Layout.NHWC)
        else:
            grad_input_nchw = mx.transpose(grad_input_nhwc, [0, 3, 1, 2])
            grad_input = Tensor._from_mlx_array(grad_input_nchw)

        return (grad_input,)


class AvgPool2dBackward(GradientFunction):
    """
    Gradient for avg_pool2d.

    Uses MLX's native gradient computation via vjp for maximum performance.
    MLX's pooling layers are differentiable, so we leverage that instead of
    implementing a manual backward pass.
    """

    def __init__(
        self,
        input_tensor,
        kernel_size,
        stride,
        padding,
        nhwc_native,
        divisor_override=None,
        count_include_pad=True,
        input_nhwc=None,
    ):
        super().__init__(input_tensor)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nhwc_native = nhwc_native
        self.divisor_override = divisor_override
        self.count_include_pad = count_include_pad
        self.input_shape = input_tensor.shape
        # Store the input in NHWC format for backward pass
        self.input_nhwc = input_nhwc

    def apply(self, grad_output):
        import mlx.nn as mxnn

        from ..layout import Layout
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # Get grad_output in NHWC format
        if self.nhwc_native:
            grad_nhwc = grad_output._mlx_array
        else:
            grad_nhwc = mx.transpose(grad_output._mlx_array, [0, 2, 3, 1])

        kH, kW = self.kernel_size

        # Handle divisor_override by scaling gradient
        if self.divisor_override is not None:
            # MLX AvgPool2d divides by kernel area, so adjust for custom divisor
            kernel_area = kH * kW
            grad_nhwc = grad_nhwc * (kernel_area / self.divisor_override)

        # Create the MLX pooling layer with same parameters
        pool = mxnn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

        # Use MLX's native vjp (vector-Jacobian product) to compute gradients
        # This is much faster than our manual Python loop implementation
        _, vjp_fn = mx.vjp(pool, [self.input_nhwc], [grad_nhwc])
        grad_input_nhwc = vjp_fn[0]

        # Convert back to NCHW if needed
        if self.nhwc_native:
            grad_input = Tensor._from_mlx_array(grad_input_nhwc, layout=Layout.NHWC)
        else:
            grad_input_nchw = mx.transpose(grad_input_nhwc, [0, 3, 1, 2])
            grad_input = Tensor._from_mlx_array(grad_input_nchw)

        return (grad_input,)


class EmbeddingBackward(GradientFunction):
    """
    Gradient for embedding lookup.

    The gradient for the weight matrix is computed by scattering the output
    gradients back to the appropriate rows of the weight matrix.

    When sparse=True (simulated), we use a more memory-efficient approach that
    only computes gradients for the unique accessed indices. This avoids
    creating a full (num_lookups x num_embeddings) one-hot matrix, which can
    be very large for big vocabularies.

    Note: MLX doesn't have native sparse tensor support, so the final gradient
    is still a dense tensor, but the computation is more memory-efficient.
    """

    def __init__(
        self, weight, indices, num_embeddings, embedding_dim, padding_idx=None, sparse=False
    ):
        super().__init__(weight)
        self.indices = indices  # Store indices for backward (as MLX array)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.sparse = sparse

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        # grad_output has shape (*indices.shape, embedding_dim)
        # We need to scatter these gradients back to weight matrix rows

        grad_out_arr = grad_output._mlx_array
        indices_arr = self.indices

        # Flatten grad_output and indices for scatter operation
        # grad_out_arr: (*batch, embedding_dim) -> (num_lookups, embedding_dim)
        num_lookups = indices_arr.size
        grad_flat = grad_out_arr.reshape((num_lookups, self.embedding_dim))
        indices_flat = indices_arr.flatten().astype(mx.int32)

        if self.sparse:
            # Sparse mode: only compute gradients for unique accessed indices
            # This is more memory efficient for large vocabularies with sparse access
            unique_indices, inverse_indices = mx.unique(indices_flat, return_inverse=True)
            num_unique = unique_indices.size

            # Initialize gradient weight matrix
            grad_weight = mx.zeros(
                (self.num_embeddings, self.embedding_dim), dtype=grad_out_arr.dtype
            )

            # Accumulate gradients for each unique index
            # Using scatter_add style operation
            for i in range(num_unique):
                target_idx = int(unique_indices[i])
                # Find all positions that map to this index
                mask = (inverse_indices == i).astype(grad_flat.dtype)
                mask = mx.expand_dims(mask, axis=-1)
                accumulated_grad = mx.sum(grad_flat * mask, axis=0)
                grad_weight = grad_weight.at[target_idx].add(accumulated_grad)
        else:
            # Dense mode: use vectorized one-hot matrix approach
            # This is faster but uses more memory: O(num_lookups * num_embeddings)
            # Create one-hot via identity matrix indexing: one_hot[i] = eye[indices[i]]
            eye = mx.eye(self.num_embeddings, dtype=grad_out_arr.dtype)
            one_hot = eye[indices_flat]  # (num_lookups, num_embeddings)

            # grad_weight = one_hot.T @ grad_flat
            # This accumulates gradients for duplicate indices automatically via matrix multiply
            grad_weight = mx.matmul(mx.transpose(one_hot), grad_flat)

        # Zero out gradient for padding_idx if specified
        if self.padding_idx is not None:
            grad_weight = grad_weight.at[self.padding_idx, :].add(-grad_weight[self.padding_idx, :])

        return (Tensor._from_mlx_array(grad_weight),)


class EmbeddingBagBackward(GradientFunction):
    """
    Gradient for embedding_bag operation.

    The gradient for the weight matrix is computed by scattering the aggregated
    gradients back to the appropriate rows of the weight matrix, accounting for
    the bag structure (sum, mean, or max aggregation).

    When sparse=True (simulated), we use a more memory-efficient approach that
    only computes and stores gradients for the accessed indices, rather than
    creating a full gradient matrix for the entire vocabulary.
    """

    def __init__(
        self,
        weight,
        indices,
        offsets,
        num_embeddings,
        embedding_dim,
        mode="mean",
        padding_idx=None,
        sparse=False,
        per_sample_weights=None,
        include_last_offset=False,
        is_2d_input=False,
        bag_size=None,
    ):
        super().__init__(weight)
        self.indices = indices  # MLX array of indices
        self.offsets = offsets  # MLX array of offsets (or None for 2D input)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        self.sparse = sparse
        self.per_sample_weights = per_sample_weights  # MLX array or None
        self.include_last_offset = include_last_offset
        self.is_2d_input = is_2d_input
        self.bag_size = bag_size  # For 2D input, the size of each bag

    def apply(self, grad_output):
        from ..tensor import Tensor

        if not self.inputs[0].requires_grad:
            return (None,)

        grad_out_arr = grad_output._mlx_array  # Shape: (num_bags, embedding_dim)
        indices_arr = self.indices

        if self.is_2d_input:
            # 2D input: indices shape is (num_bags, bag_size)
            num_bags, bag_size = indices_arr.shape

            # Expand grad_output to match bag structure
            # grad_out_arr: (num_bags, embedding_dim) -> (num_bags, bag_size, embedding_dim)
            grad_expanded = mx.broadcast_to(
                mx.expand_dims(grad_out_arr, axis=1), (num_bags, bag_size, self.embedding_dim)
            )

            # Apply per_sample_weights scaling if present
            if self.per_sample_weights is not None:
                psw = mx.expand_dims(self.per_sample_weights, axis=-1)  # (num_bags, bag_size, 1)
                grad_expanded = grad_expanded * psw

            # For mean mode, divide by bag size (or count of non-padding)
            if self.mode == "mean":
                if self.padding_idx is not None:
                    # Count non-padding elements per bag
                    mask = (indices_arr != self.padding_idx).astype(mx.float32)
                    count = mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1.0)
                    count = mx.expand_dims(count, axis=-1)  # (num_bags, 1, 1)
                    grad_expanded = grad_expanded / count
                else:
                    grad_expanded = grad_expanded / bag_size

            # Handle padding_idx - zero out gradient for padding indices
            if self.padding_idx is not None:
                mask = (indices_arr != self.padding_idx).astype(mx.float32)
                mask = mx.expand_dims(mask, axis=-1)  # (num_bags, bag_size, 1)
                grad_expanded = grad_expanded * mask

            # Flatten for scatter operation
            indices_flat = indices_arr.flatten().astype(mx.int32)
            grad_flat = grad_expanded.reshape((-1, self.embedding_dim))

        else:
            # 1D input with offsets
            offsets_arr = self.offsets.astype(mx.int32)
            indices_arr = indices_arr.astype(mx.int32)

            # Determine bag boundaries
            if self.include_last_offset:
                bag_boundaries = offsets_arr
                num_bags = len(offsets_arr) - 1
            else:
                bag_boundaries = mx.concatenate([offsets_arr, mx.array([indices_arr.size])])
                num_bags = len(offsets_arr)

            # Build gradient for each index
            grad_list = []
            for i in range(num_bags):
                start = int(bag_boundaries[i])
                end = int(bag_boundaries[i + 1])
                bag_len = end - start

                if bag_len == 0:
                    continue

                # Gradient for this bag's output
                bag_grad = grad_out_arr[i]  # (embedding_dim,)

                # Expand to match bag size
                bag_grad_expanded = mx.broadcast_to(
                    mx.expand_dims(bag_grad, axis=0), (bag_len, self.embedding_dim)
                )

                # Apply per_sample_weights scaling if present
                if self.per_sample_weights is not None:
                    psw = self.per_sample_weights[start:end]
                    psw = mx.expand_dims(psw, axis=-1)
                    bag_grad_expanded = bag_grad_expanded * psw

                # For mean mode, divide by bag size
                if self.mode == "mean":
                    if self.padding_idx is not None:
                        bag_indices = indices_arr[start:end]
                        mask = (bag_indices != self.padding_idx).astype(mx.float32)
                        count = mx.maximum(mx.sum(mask), 1.0)
                        bag_grad_expanded = bag_grad_expanded / count
                        # Also apply mask
                        mask = mx.expand_dims(mask, axis=-1)
                        bag_grad_expanded = bag_grad_expanded * mask
                    else:
                        bag_grad_expanded = bag_grad_expanded / bag_len
                elif self.padding_idx is not None:
                    # Zero out padding indices for non-mean modes too
                    bag_indices = indices_arr[start:end]
                    mask = (bag_indices != self.padding_idx).astype(mx.float32)
                    mask = mx.expand_dims(mask, axis=-1)
                    bag_grad_expanded = bag_grad_expanded * mask

                grad_list.append(bag_grad_expanded)

            if len(grad_list) == 0:
                # No non-empty bags, return zero gradient
                grad_weight = mx.zeros(
                    (self.num_embeddings, self.embedding_dim), dtype=grad_out_arr.dtype
                )
                return (Tensor._from_mlx_array(grad_weight),)

            grad_flat = mx.concatenate(grad_list, axis=0)
            indices_flat = indices_arr

        # Scatter gradients to weight matrix using one-hot approach
        if self.sparse:
            # Sparse mode: only compute gradients for unique accessed indices
            # This is more memory efficient for large vocabularies
            unique_indices = mx.unique(indices_flat)[0]
            num_unique = unique_indices.size

            # Create mapping from unique indices to their positions
            # and accumulate gradients for each unique index
            grad_weight = mx.zeros(
                (self.num_embeddings, self.embedding_dim), dtype=grad_out_arr.dtype
            )

            # For each unique index, sum up all gradients that go to it
            for idx in range(num_unique):
                target_idx = int(unique_indices[idx])
                mask = (indices_flat == target_idx).astype(grad_flat.dtype)
                mask = mx.expand_dims(mask, axis=-1)
                accumulated_grad = mx.sum(grad_flat * mask, axis=0)
                grad_weight = grad_weight.at[target_idx].add(accumulated_grad)
        else:
            # Dense mode: use vectorized one-hot approach (faster but more memory)
            eye = mx.eye(self.num_embeddings, dtype=grad_out_arr.dtype)
            one_hot = eye[indices_flat]  # (num_lookups, num_embeddings)
            grad_weight = mx.matmul(mx.transpose(one_hot), grad_flat)

        # Zero out gradient for padding_idx
        if self.padding_idx is not None:
            grad_weight = grad_weight.at[self.padding_idx, :].add(-grad_weight[self.padding_idx, :])

        return (Tensor._from_mlx_array(grad_weight),)
