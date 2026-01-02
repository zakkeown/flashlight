"""
Quantization Stubs

QuantStub and DeQuantStub mark the boundaries of quantized regions in a model.
"""

from ..nn.module import Module
from ..tensor import Tensor


class QuantStub(Module):
    """
    Stub module to mark the entry point into a quantized region.

    In forward pass, this is a no-op. During quantization preparation,
    this gets replaced with actual quantization logic.

    Example:
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.quant = QuantStub()
        ...         self.linear = nn.Linear(10, 20)
        ...         self.dequant = DeQuantStub()
        ...
        ...     def forward(self, x):
        ...         x = self.quant(x)
        ...         x = self.linear(x)
        ...         x = self.dequant(x)
        ...         return x
    """

    def __init__(self, qconfig=None):
        super().__init__()
        self.qconfig = qconfig

    def forward(self, x: Tensor) -> Tensor:
        """Pass through unchanged."""
        return x


class DeQuantStub(Module):
    """
    Stub module to mark the exit point from a quantized region.

    In forward pass, this is a no-op. During quantization preparation,
    this marks where dequantization should occur.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Pass through unchanged."""
        return x


class QuantWrapper(Module):
    """
    Wrapper that adds QuantStub and DeQuantStub around a module.

    Useful for wrapping modules that should be quantized without
    modifying their code.

    Example:
        >>> model = nn.Linear(10, 20)
        >>> wrapped = QuantWrapper(model)
        >>> # Now wrapped.quant, wrapped.module, wrapped.dequant exist
    """

    def __init__(self, module: Module, qconfig=None):
        super().__init__()
        self.quant = QuantStub(qconfig=qconfig)
        self.module = module
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        """Quantize, process, dequantize."""
        x = self.quant(x)
        x = self.module(x)
        x = self.dequant(x)
        return x
