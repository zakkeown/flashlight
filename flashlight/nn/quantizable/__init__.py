"""
Quantizable Modules

PyTorch-compatible torch.nn.quantizable module stub.
MLX does not currently support quantization, so this module
provides stub implementations for API compatibility.
"""

import warnings

__all__ = [
    "LSTM",
    "GRU",
    "LSTMCell",
    "GRUCell",
    "MultiheadAttention",
    "modules",
]


def _quantizable_not_supported():
    """Raise warning about quantizable modules not being supported."""
    warnings.warn(
        "Quantizable modules are not supported in MLX. "
        "Quantizable modules will behave like their non-quantizable counterparts.",
        UserWarning,
    )


from ..layers.attention import MultiheadAttention as _MultiheadAttention

# Import regular layers as quantizable versions
from ..layers.rnn import GRU as _GRU
from ..layers.rnn import LSTM as _LSTM
from ..layers.rnn import GRUCell as _GRUCell
from ..layers.rnn import LSTMCell as _LSTMCell


class LSTM(_LSTM):
    """Quantizable LSTM stub - behaves like regular LSTM in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


class GRU(_GRU):
    """Quantizable GRU stub - behaves like regular GRU in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


class LSTMCell(_LSTMCell):
    """Quantizable LSTMCell stub - behaves like regular LSTMCell in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


class GRUCell(_GRUCell):
    """Quantizable GRUCell stub - behaves like regular GRUCell in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


class MultiheadAttention(_MultiheadAttention):
    """Quantizable MultiheadAttention stub - behaves like regular MultiheadAttention in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


# Import modules submodule for compatibility
from . import modules
