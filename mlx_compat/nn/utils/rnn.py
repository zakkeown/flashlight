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

    A packed sequence is a more efficient representation of variable-length
    sequences for RNN processing. It stores:
    - data: All time steps concatenated together (non-padding elements only)
    - batch_sizes: Number of sequences at each time step
    - sorted_indices: Indices to sort sequences by length (longest first)
    - unsorted_indices: Indices to restore original order

    The data is organized such that all elements from time step 0 come first
    (for all sequences), then all elements from time step 1, etc.
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

    def __repr__(self):
        return (f"PackedSequence(data={self.data.shape}, "
                f"batch_sizes={self.batch_sizes.shape if self.batch_sizes is not None else None})")


def pack_padded_sequence(
    input: Tensor,
    lengths: Union[Tensor, List[int]],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """
    Packs a Tensor containing padded sequences of variable length.

    The packed representation allows RNNs to process only the non-padding
    elements, improving efficiency for variable-length sequences.

    Args:
        input: Padded batch of variable length sequences.
               Shape: [seq_len, batch, *] or [batch, seq_len, *] if batch_first
        lengths: List or tensor of sequence lengths for each batch element
        batch_first: If True, input is (batch, seq, features)
        enforce_sorted: If True, sequences must be sorted by length (descending).
                       If False, will sort internally and track the permutation.

    Returns:
        A PackedSequence object containing:
        - data: Packed tensor with shape [total_elements, *]
        - batch_sizes: Number of sequences at each time step
        - sorted_indices: Permutation to sort by length
        - unsorted_indices: Permutation to restore original order

    Example:
        >>> seq1 = tensor([[1, 2], [3, 4], [5, 6]])  # length 3
        >>> seq2 = tensor([[7, 8], [9, 10], [0, 0]]) # length 2 (padded)
        >>> padded = stack([seq1, seq2])  # [2, 3, 2]
        >>> lengths = tensor([3, 2])
        >>> packed = pack_padded_sequence(padded, lengths, batch_first=True)
    """
    import mlx.core as mx

    # Convert lengths to list for processing
    if isinstance(lengths, Tensor):
        lengths_list = lengths.tolist()
    elif hasattr(lengths, 'tolist'):
        lengths_list = lengths.tolist()
    else:
        lengths_list = list(lengths)
    lengths_list = [int(l) for l in lengths_list]

    # Transpose to [seq_len, batch, *] if needed
    if batch_first:
        # [batch, seq, *] -> [seq, batch, *]
        input_data = mx.transpose(input._mlx_array, [1, 0] + list(range(2, input.ndim)))
    else:
        input_data = input._mlx_array

    seq_len, batch_size = input_data.shape[:2]
    feature_shape = input_data.shape[2:]

    # Handle sorting
    lengths_array = mx.array(lengths_list, dtype=mx.int32)

    if enforce_sorted:
        # Verify sequences are sorted by length (descending)
        for i in range(len(lengths_list) - 1):
            if lengths_list[i] < lengths_list[i + 1]:
                raise RuntimeError(
                    f"sequences are not sorted by length (got lengths {lengths_list}). "
                    "Set enforce_sorted=False to sort them internally."
                )
        sorted_indices = mx.arange(batch_size, dtype=mx.int64)
        unsorted_indices = mx.arange(batch_size, dtype=mx.int64)
        sorted_lengths = lengths_list
    else:
        # Sort by length (descending)
        sorted_idx = mx.argsort(-lengths_array)
        sorted_indices = sorted_idx.astype(mx.int64)
        # Compute unsorted indices (inverse permutation)
        unsorted_indices = mx.zeros((batch_size,), dtype=mx.int64)
        for i in range(batch_size):
            idx = int(sorted_indices[i].item())
            unsorted_indices = mx.concatenate([
                unsorted_indices[:idx],
                mx.array([i], dtype=mx.int64),
                unsorted_indices[idx + 1:]
            ])
        sorted_lengths = [lengths_list[int(i)] for i in sorted_indices.tolist()]
        # Reorder input according to sorted indices
        input_data = input_data[:, sorted_indices.astype(mx.int32)]

    # Compute batch_sizes: number of sequences still active at each time step
    max_len = max(sorted_lengths)
    batch_sizes = []
    for t in range(max_len):
        count = sum(1 for l in sorted_lengths if l > t)
        batch_sizes.append(count)

    # Pack the data: extract non-padding elements
    packed_data = []
    for t in range(max_len):
        # Number of active sequences at time t
        n_active = batch_sizes[t]
        # Extract elements for active sequences
        packed_data.append(input_data[t, :n_active])

    # Concatenate all time steps
    packed_tensor = mx.concatenate(packed_data, axis=0)

    return PackedSequence(
        data=Tensor._from_mlx_array(packed_tensor),
        batch_sizes=Tensor._from_mlx_array(mx.array(batch_sizes, dtype=mx.int64)),
        sorted_indices=Tensor._from_mlx_array(sorted_indices),
        unsorted_indices=Tensor._from_mlx_array(unsorted_indices),
    )


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Pads a packed batch of variable length sequences.

    This is the inverse operation of pack_padded_sequence. It takes a
    PackedSequence and returns a padded tensor along with the sequence lengths.

    Args:
        sequence: A PackedSequence to unpack
        batch_first: If True, output is (batch, seq, features)
        padding_value: Value to use for padded positions
        total_length: If provided, pad to this length instead of the maximum
                     sequence length in the batch

    Returns:
        Tuple of:
        - padded tensor: [seq_len, batch, *] or [batch, seq_len, *] if batch_first
        - lengths tensor: Length of each sequence

    Example:
        >>> packed = pack_padded_sequence(padded_input, lengths, batch_first=True)
        >>> # ... process with RNN ...
        >>> output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
    """
    import mlx.core as mx

    if sequence.batch_sizes is None:
        raise ValueError("PackedSequence must have batch_sizes to unpack")

    batch_sizes = sequence.batch_sizes._mlx_array.tolist()
    batch_sizes = [int(b) for b in batch_sizes]

    data = sequence.data._mlx_array
    feature_shape = data.shape[1:]

    # Determine dimensions
    batch_size = batch_sizes[0]  # First time step has all sequences
    seq_len = len(batch_sizes)

    if total_length is not None:
        if total_length < seq_len:
            raise ValueError(
                f"total_length ({total_length}) must be >= max sequence length ({seq_len})"
            )
        output_seq_len = total_length
    else:
        output_seq_len = seq_len

    # Compute lengths for each sequence (in sorted order)
    # Length of sequence i = number of time steps where batch_size > i
    lengths = []
    for i in range(batch_size):
        length = sum(1 for bs in batch_sizes if bs > i)
        lengths.append(length)

    # Create output tensor filled with padding value
    output_shape = (output_seq_len, batch_size) + feature_shape
    output = mx.full(output_shape, padding_value, dtype=data.dtype)

    # Unpack the data
    data_offset = 0
    for t in range(seq_len):
        n_active = batch_sizes[t]
        # Get the data for this time step
        step_data = data[data_offset:data_offset + n_active]
        # Place it in the output tensor
        output = mx.concatenate([
            output[:t],
            mx.expand_dims(
                mx.concatenate([step_data, output[t, n_active:]], axis=0),
                axis=0
            ),
            output[t + 1:]
        ], axis=0)
        data_offset += n_active

    # Restore original order if unsorted_indices is available
    if sequence.unsorted_indices is not None:
        unsorted_idx = sequence.unsorted_indices._mlx_array.astype(mx.int32)
        output = output[:, unsorted_idx]
        # Reorder lengths too
        lengths_array = mx.array(lengths, dtype=mx.int64)
        lengths = [lengths[int(i)] for i in unsorted_idx.tolist()]

    lengths_tensor = Tensor._from_mlx_array(mx.array(lengths, dtype=mx.int64))

    if batch_first:
        # [seq, batch, *] -> [batch, seq, *]
        output = mx.transpose(output, [1, 0] + list(range(2, output.ndim)))

    return Tensor._from_mlx_array(output), lengths_tensor


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

    This is a convenience function that pads the sequences and then packs them.
    It is equivalent to calling pad_sequence followed by pack_padded_sequence.

    Args:
        sequences: List of Tensors. Each tensor should have shape [length, *]
                  where length can vary across tensors.
        enforce_sorted: If True, sequences must be sorted by length (descending).
                       If False, will sort internally and track the permutation.

    Returns:
        A PackedSequence object

    Example:
        >>> seq1 = tensor([[1, 2], [3, 4], [5, 6]])  # length 3
        >>> seq2 = tensor([[7, 8], [9, 10]])         # length 2
        >>> packed = pack_sequence([seq1, seq2])
    """
    import mlx.core as mx

    if len(sequences) == 0:
        raise ValueError("pack_sequence requires at least one sequence")

    # Get lengths
    lengths = [seq.shape[0] for seq in sequences]

    # Pad the sequences (batch_first=False by default in pad_sequence)
    # pad_sequence returns [seq_len, batch, *]
    padded = pad_sequence(sequences, batch_first=False)

    # Pack with batch_first=False (since pad_sequence returns [seq, batch, *])
    return pack_padded_sequence(
        padded,
        lengths,
        batch_first=False,
        enforce_sorted=enforce_sorted
    )


def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]:
    """
    Unpacks a PackedSequence into a list of variable length Tensors.

    This is the inverse operation of pack_sequence. It returns a list of
    tensors, each with its original (unpadded) length.

    Args:
        packed_sequences: A PackedSequence created by pack_sequence or
                         pack_padded_sequence

    Returns:
        List of Tensors, each with shape [length_i, *] where length_i
        is the original length of the i-th sequence

    Example:
        >>> packed = pack_sequence([seq1, seq2])
        >>> sequences = unpack_sequence(packed)
        >>> len(sequences)  # 2
    """
    import mlx.core as mx

    # First unpad to get [seq, batch, *]
    padded, lengths = pad_packed_sequence(packed_sequences, batch_first=False)

    # Then unpad each sequence
    return unpad_sequence(padded, lengths, batch_first=False)


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
