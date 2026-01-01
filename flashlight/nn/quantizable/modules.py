"""
Quantizable Modules - re-exports from parent package.

PyTorch-compatible torch.nn.quantizable.modules submodule.
Directly imports the same classes that are in the parent __init__.py
"""

import warnings

from ..layers.attention import MultiheadAttention as _MultiheadAttention
from ..layers.rnn import LSTM as _LSTM
from ..layers.rnn import LSTMCell as _LSTMCell


def _quantizable_not_supported():
    """Raise warning about quantizable modules not being supported."""
    warnings.warn(
        "Quantizable modules are not supported in MLX. "
        "Quantizable modules will behave like their non-quantizable counterparts.",
        UserWarning,
    )


class LSTM(_LSTM):
    """Quantizable LSTM stub - behaves like regular LSTM in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


class LSTMCell(_LSTMCell):
    """Quantizable LSTMCell stub - behaves like regular LSTMCell in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


class MultiheadAttention(_MultiheadAttention):
    """Quantizable MultiheadAttention stub - behaves like regular MultiheadAttention in MLX."""

    def __init__(self, *args, **kwargs):
        _quantizable_not_supported()
        super().__init__(*args, **kwargs)


__all__ = [
    "LSTM",
    "LSTMCell",
    "MultiheadAttention",
]
