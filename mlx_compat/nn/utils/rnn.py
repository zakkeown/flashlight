"""
RNN Utilities

PyTorch-compatible torch.nn.utils.rnn module providing utilities for RNNs.
"""

import warnings
from typing import List, Optional, Tuple, Union

from ...tensor import Tensor


class PackedSequence:
    """
    Holds the data and list of batch_sizes of a packed sequence.

    Note: PackedSequence is a stub in MLX. MLX handles variable-length
    sequences differently.
    """

    def __init__(
        self,
        data: Tensor,
        batch_sizes: Optional[Tensor] = None,
        sorted_indices: Optional[Tensor] = None,
        unsorted_indices: Optional[Tensor] = None,
    ):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """
    Packs a Tensor containing padded sequences of variable length.

    Note: This is a stub in MLX. Returns a PackedSequence with the input data.

    Args:
        input: Padded batch of variable length sequences
        lengths: List of sequence lengths
        batch_first: If True, input is (batch, seq, features)
        enforce_sorted: If True, sequences must be sorted by length

    Returns:
        A PackedSequence object
    """
    warnings.warn(
        "pack_padded_sequence is a stub in MLX - packed sequences work differently",
        UserWarning
    )
    return PackedSequence(input)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Pads a packed batch of variable length sequences.

    Note: This is a stub in MLX. Returns the data from the PackedSequence.

    Args:
        sequence: A PackedSequence
        batch_first: If True, output is (batch, seq, features)
        padding_value: Value for padded elements
        total_length: If provided, pad to this length

    Returns:
        Tuple of padded tensor and lengths tensor
    """
    warnings.warn(
        "pad_packed_sequence is a stub in MLX - packed sequences work differently",
        UserWarning
    )
    import mlx.core as mx
    return sequence.data, Tensor(mx.array([sequence.data.shape[0]]))


def pad_sequence(
    sequences: List[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = 'right',
) -> Tensor:
    """
    Pad a list of variable length Tensors with padding_value.

    Args:
        sequences: List of Tensors
        batch_first: If True, output is (batch, seq, features)
        padding_value: Value for padded elements
        padding_side: Side to pad ('right' or 'left')

    Returns:
        Padded tensor
    """
    import mlx.core as mx

    if padding_side not in ('left', 'right'):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    max_len = max(s.shape[0] for s in sequences)
    trailing_dims = sequences[0].shape[1:]

    # Get MLX dtype from first sequence
    first_seq = sequences[0]
    mlx_dtype = first_seq._mlx_array.dtype

    # Simplified implementation - just stack and pad
    padded = []
    for seq in sequences:
        length = seq.shape[0]
        if length < max_len:
            pad_shape = (max_len - length,) + trailing_dims
            padding = mx.full(pad_shape, padding_value, dtype=mlx_dtype)
            if padding_side == 'right':
                padded.append(mx.concatenate([seq._mlx_array, padding], axis=0))
            else:  # left padding
                padded.append(mx.concatenate([padding, seq._mlx_array], axis=0))
        else:
            padded.append(seq._mlx_array)

    stacked = mx.stack(padded, axis=0 if batch_first else 1)
    return Tensor(stacked)


def pack_sequence(
    sequences: List[Tensor],
    enforce_sorted: bool = True,
) -> PackedSequence:
    """
    Packs a list of variable length Tensors.

    Note: This is a stub in MLX.

    Args:
        sequences: List of Tensors sorted by length (longest first)
        enforce_sorted: If True, sequences must be sorted

    Returns:
        A PackedSequence object
    """
    warnings.warn(
        "pack_sequence is a stub in MLX - packed sequences work differently",
        UserWarning
    )
    import mlx.core as mx
    # Just concatenate for now
    return PackedSequence(Tensor(mx.concatenate([s._mlx_array for s in sequences])))


def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]:
    """
    Unpacks a PackedSequence into a list of variable length Tensors.

    Note: This is a stub in MLX.

    Args:
        packed_sequences: A PackedSequence

    Returns:
        List of Tensors
    """
    warnings.warn(
        "unpack_sequence is a stub in MLX - packed sequences work differently",
        UserWarning
    )
    return [packed_sequences.data]


def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
) -> List[Tensor]:
    """
    Unpad a tensor of padded sequences.

    Args:
        padded_sequences: Padded tensor
        lengths: Length of each sequence
        batch_first: If True, input is (batch, seq, features)

    Returns:
        List of unpadded Tensors
    """
    result = []
    lengths_list = lengths.tolist() if hasattr(lengths, 'tolist') else list(lengths)

    for i, length in enumerate(lengths_list):
        length = int(length)
        if batch_first:
            result.append(Tensor(padded_sequences._mlx_array[i, :length]))
        else:
            result.append(Tensor(padded_sequences._mlx_array[:length, i]))

    return result


def invert_permutation(permutation: Tensor) -> Tensor:
    """
    Invert a permutation.

    Given a permutation, return its inverse such that
    inverse[permutation[i]] = i for all i.

    Args:
        permutation: A 1D tensor representing a permutation

    Returns:
        The inverse permutation
    """
    import mlx.core as mx

    # Handle both Tensor and raw mlx array
    if hasattr(permutation, '_mlx_array'):
        data = permutation._mlx_array
    elif hasattr(permutation, '_data'):
        data = permutation._data
    else:
        data = permutation

    n = int(data.shape[0])
    inverse = mx.zeros((n,), dtype=mx.int64)

    # Create inverse permutation
    # MLX doesn't have scatter, so we use a different approach
    inverse = mx.zeros((n,), dtype=mx.int64)
    for i in range(n):
        # Convert MLX scalar to Python int
        idx = int(data[i].item()) if hasattr(data[i], 'item') else int(data[i])
        inverse = mx.concatenate([
            inverse[:idx],
            mx.array([i]),
            inverse[idx + 1:]
        ])

    return Tensor._from_mlx_array(inverse)


__all__ = [
    'PackedSequence',
    'pack_padded_sequence',
    'pad_packed_sequence',
    'pad_sequence',
    'pack_sequence',
    'unpack_sequence',
    'unpad_sequence',
    'invert_permutation',
]
