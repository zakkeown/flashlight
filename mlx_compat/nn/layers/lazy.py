"""
Lazy Modules

Lazy versions of layers that infer dimensions from the first input.
These modules allow creating layers without specifying input dimensions upfront.
"""

from typing import Any, Optional, Tuple, Union
from ..module import Module
from ..parameter import UninitializedParameter
from .linear import Linear
from .conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from .normalization import BatchNorm1d, BatchNorm2d, BatchNorm3d, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d


class LazyModuleMixin:
    """
    Mixin for modules with lazy initialization.

    Lazy modules defer parameter initialization until the first forward pass,
    allowing automatic inference of input dimensions.
    """

    _is_lazy: bool = True

    def has_uninitialized_params(self) -> bool:
        """Check if module has any uninitialized parameters."""
        for param in self.parameters():
            if isinstance(param, UninitializedParameter):
                return True
        return False


class LazyLinear(LazyModuleMixin, Module):
    """
    A Linear module where in_features is inferred from the first input.

    Args:
        out_features: Size of each output sample
        bias: If True, adds a learnable bias (default: True)
        device: Ignored (MLX uses unified memory)
        dtype: Data type for the layer
    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_features = out_features
        self.use_bias = bias
        self._initialized = False
        # Don't set _linear = None here - it would shadow the _modules entry

    @property
    def in_features(self):
        """Return in_features if initialized, else None."""
        if self._initialized and '_linear' in self._modules:
            return self._modules['_linear'].in_features
        return None

    def _initialize(self, input):
        """Initialize the Linear layer with inferred in_features."""
        in_features = input.shape[-1]
        self._linear = Linear(in_features, self.out_features, bias=self.use_bias)
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_linear'](input)


class LazyConv1d(LazyModuleMixin, Module):
    """A Conv1d module where in_channels is inferred from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self._initialized = False
        # Don't set _conv = None here - it would shadow the _modules entry

    @property
    def in_channels(self):
        """Return in_channels if initialized, else None."""
        if self._initialized and '_conv' in self._modules:
            return self._modules['_conv'].in_channels
        return None

    def _initialize(self, input):
        in_channels = input.shape[1]  # NCHW format
        self._conv = Conv1d(
            in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, bias=self.use_bias, padding_mode=self.padding_mode
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_conv'](input)


class LazyConv2d(LazyModuleMixin, Module):
    """A Conv2d module where in_channels is inferred from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self._initialized = False
        # Don't set _conv = None here - it would shadow the _modules entry

    @property
    def in_channels(self):
        """Return in_channels if initialized, else None."""
        if self._initialized and '_conv' in self._modules:
            return self._modules['_conv'].in_channels
        return None

    def _initialize(self, input):
        in_channels = input.shape[1]  # NCHW format
        self._conv = Conv2d(
            in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, bias=self.use_bias, padding_mode=self.padding_mode
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_conv'](input)


class LazyConv3d(LazyModuleMixin, Module):
    """A Conv3d module where in_channels is inferred from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self._initialized = False
        # Don't set _conv = None here - it would shadow the _modules entry

    @property
    def in_channels(self):
        """Return in_channels if initialized, else None."""
        if self._initialized and '_conv' in self._modules:
            return self._modules['_conv'].in_channels
        return None

    def _initialize(self, input):
        in_channels = input.shape[1]  # NCDHW format
        self._conv = Conv3d(
            in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, bias=self.use_bias, padding_mode=self.padding_mode
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_conv'](input)


class LazyConvTranspose1d(LazyModuleMixin, Module):
    """A ConvTranspose1d module where in_channels is inferred from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self._initialized = False
        # Don't set _conv = None here - it would shadow the _modules entry

    @property
    def in_channels(self):
        """Return in_channels if initialized, else None."""
        if self._initialized and '_conv' in self._modules:
            return self._modules['_conv'].in_channels
        return None

    def _initialize(self, input):
        in_channels = input.shape[1]
        self._conv = ConvTranspose1d(
            in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, bias=self.use_bias, dilation=self.dilation,
            padding_mode=self.padding_mode
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_conv'](input)


class LazyConvTranspose2d(LazyModuleMixin, Module):
    """A ConvTranspose2d module where in_channels is inferred from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self._initialized = False
        # Don't set _conv = None here - it would shadow the _modules entry

    @property
    def in_channels(self):
        """Return in_channels if initialized, else None."""
        if self._initialized and '_conv' in self._modules:
            return self._modules['_conv'].in_channels
        return None

    def _initialize(self, input):
        in_channels = input.shape[1]
        self._conv = ConvTranspose2d(
            in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, bias=self.use_bias, dilation=self.dilation,
            padding_mode=self.padding_mode
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_conv'](input)


class LazyConvTranspose3d(LazyModuleMixin, Module):
    """A ConvTranspose3d module where in_channels is inferred from the first input."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self._initialized = False
        # Don't set _conv = None here - it would shadow the _modules entry

    @property
    def in_channels(self):
        """Return in_channels if initialized, else None."""
        if self._initialized and '_conv' in self._modules:
            return self._modules['_conv'].in_channels
        return None

    def _initialize(self, input):
        in_channels = input.shape[1]
        self._conv = ConvTranspose3d(
            in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, bias=self.use_bias, dilation=self.dilation,
            padding_mode=self.padding_mode
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_conv'](input)


class LazyBatchNorm1d(LazyModuleMixin, Module):
    """A BatchNorm1d module where num_features is inferred from the first input."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._initialized = False
        # Don't set _bn = None here - it would shadow the _modules entry

    @property
    def num_features(self):
        """Return num_features if initialized, else None."""
        if self._initialized and '_bn' in self._modules:
            return self._modules['_bn'].num_features
        return None

    def _initialize(self, input):
        num_features = input.shape[1]
        self._bn = BatchNorm1d(
            num_features, eps=self.eps, momentum=self.momentum,
            affine=self.affine, track_running_stats=self.track_running_stats
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_bn'](input)


class LazyBatchNorm2d(LazyModuleMixin, Module):
    """A BatchNorm2d module where num_features is inferred from the first input."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._initialized = False
        # Don't set _bn = None here - it would shadow the _modules entry

    @property
    def num_features(self):
        """Return num_features if initialized, else None."""
        if self._initialized and '_bn' in self._modules:
            return self._modules['_bn'].num_features
        return None

    def _initialize(self, input):
        num_features = input.shape[1]
        self._bn = BatchNorm2d(
            num_features, eps=self.eps, momentum=self.momentum,
            affine=self.affine, track_running_stats=self.track_running_stats
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_bn'](input)


class LazyBatchNorm3d(LazyModuleMixin, Module):
    """A BatchNorm3d module where num_features is inferred from the first input."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._initialized = False
        # Don't set _bn = None here - it would shadow the _modules entry

    @property
    def num_features(self):
        """Return num_features if initialized, else None."""
        if self._initialized and '_bn' in self._modules:
            return self._modules['_bn'].num_features
        return None

    def _initialize(self, input):
        num_features = input.shape[1]
        self._bn = BatchNorm3d(
            num_features, eps=self.eps, momentum=self.momentum,
            affine=self.affine, track_running_stats=self.track_running_stats
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_bn'](input)


class LazyInstanceNorm1d(LazyModuleMixin, Module):
    """An InstanceNorm1d module where num_features is inferred from the first input."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._initialized = False
        # Don't set _norm = None here - it would shadow the _modules entry

    @property
    def num_features(self):
        """Return num_features if initialized, else None."""
        if self._initialized and '_norm' in self._modules:
            return self._modules['_norm'].num_features
        return None

    def _initialize(self, input):
        num_features = input.shape[1]
        self._norm = InstanceNorm1d(
            num_features, eps=self.eps, momentum=self.momentum,
            affine=self.affine, track_running_stats=self.track_running_stats
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_norm'](input)


class LazyInstanceNorm2d(LazyModuleMixin, Module):
    """An InstanceNorm2d module where num_features is inferred from the first input."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._initialized = False
        # Don't set _norm = None here - it would shadow the _modules entry

    @property
    def num_features(self):
        """Return num_features if initialized, else None."""
        if self._initialized and '_norm' in self._modules:
            return self._modules['_norm'].num_features
        return None

    def _initialize(self, input):
        num_features = input.shape[1]
        self._norm = InstanceNorm2d(
            num_features, eps=self.eps, momentum=self.momentum,
            affine=self.affine, track_running_stats=self.track_running_stats
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_norm'](input)


class LazyInstanceNorm3d(LazyModuleMixin, Module):
    """An InstanceNorm3d module where num_features is inferred from the first input."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._initialized = False
        # Don't set _norm = None here - it would shadow the _modules entry

    @property
    def num_features(self):
        """Return num_features if initialized, else None."""
        if self._initialized and '_norm' in self._modules:
            return self._modules['_norm'].num_features
        return None

    def _initialize(self, input):
        num_features = input.shape[1]
        self._norm = InstanceNorm3d(
            num_features, eps=self.eps, momentum=self.momentum,
            affine=self.affine, track_running_stats=self.track_running_stats
        )
        self._initialized = True

    def forward(self, input):
        if not self._initialized:
            self._initialize(input)
        return self._modules['_norm'](input)


__all__ = [
    'LazyModuleMixin',
    'LazyLinear',
    'LazyConv1d', 'LazyConv2d', 'LazyConv3d',
    'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d',
    'LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d',
    'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d',
]
