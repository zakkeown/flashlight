"""
Embedding Layers

Implements embedding lookup tables for NLP and other sequence models.
"""

import math
from typing import Optional

from ...tensor import Tensor
from ..module import Module
from ..parameter import Parameter


class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output is
    the corresponding word embeddings.

    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        padding_idx: If specified, the entries at padding_idx do not contribute
                     to the gradient; therefore, the embedding vector at padding_idx
                     is not updated during training. (default: None)
        max_norm: If given, each embedding vector with norm larger than max_norm
                  is renormalized to have norm max_norm. (default: None)
        norm_type: The p of the p-norm to compute for the max_norm option.
                   (default: 2.0)
        scale_grad_by_freq: If True, scale gradients by the inverse of frequency
                            of the words in the mini-batch. (default: False)
        sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor.
                Note: this is not currently supported in MLX. (default: False)

    Shape:
        - Input: (*) containing indices in range [0, num_embeddings)
        - Output: (*, embedding_dim)

    Attributes:
        weight: The learnable weights of shape (num_embeddings, embedding_dim)

    Example:
        >>> embedding = nn.Embedding(10, 3)  # 10 words, 3-dim embeddings
        >>> input = mlx_compat.tensor([1, 2, 4, 5])
        >>> output = embedding(input)
        >>> output.shape
        (4, 3)

        >>> # With padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = mlx_compat.tensor([[0, 2, 0, 5]])
        >>> output = embedding(input)
        >>> output.shape
        (1, 4, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Initialize Embedding layer.

        Args:
            num_embeddings: Dictionary size (vocabulary size)
            embedding_dim: Embedding dimension
            padding_idx: Index for padding token (optional)
            max_norm: Max norm for embeddings (optional)
            norm_type: Norm type for max_norm (default: 2.0)
            scale_grad_by_freq: Scale gradients by frequency (default: False)
            sparse: Use sparse gradients (not supported, default: False)
            _weight: Optional pre-initialized weights
        """
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._freeze = _freeze

        if sparse:
            import warnings
            warnings.warn(
                "Sparse embeddings are not currently supported in MLX. "
                "Using dense embeddings instead."
            )

        if _weight is None:
            # Initialize embedding weight
            from ... import randn
            self.weight = Parameter(randn(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            # Use provided weight
            if _weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    f"Shape of _weight ({_weight.shape}) doesn't match "
                    f"(num_embeddings={num_embeddings}, embedding_dim={embedding_dim})"
                )
            self.weight = Parameter(_weight)

        # Zero out padding_idx embedding if specified
        if self.padding_idx is not None:
            self._zero_padding_idx()

    def reset_parameters(self) -> None:
        """
        Initialize parameters using normal distribution.

        Following PyTorch's default: N(0, 1)
        """
        import mlx.core as mx

        # Initialize from standard normal
        weight_data = mx.random.normal(
            shape=(self.num_embeddings, self.embedding_dim)
        )
        self.weight.data = Tensor._from_mlx_array(weight_data)

        if self.padding_idx is not None:
            self._zero_padding_idx()

    def _zero_padding_idx(self) -> None:
        """Zero out the embedding at padding_idx."""
        import mlx.core as mx

        if self.padding_idx is not None:
            # Get weight data
            weight_mlx = self.weight._mlx_array

            # Create a mask to zero out padding_idx row
            # This is a bit tricky with MLX's immutability
            # We'll reconstruct the weight with zeros at padding_idx
            zeros_row = mx.zeros((1, self.embedding_dim))

            # Split, replace, and concatenate
            if self.padding_idx == 0:
                new_weight = mx.concatenate([
                    zeros_row,
                    weight_mlx[1:]
                ], axis=0)
            elif self.padding_idx == self.num_embeddings - 1:
                new_weight = mx.concatenate([
                    weight_mlx[:-1],
                    zeros_row
                ], axis=0)
            else:
                new_weight = mx.concatenate([
                    weight_mlx[:self.padding_idx],
                    zeros_row,
                    weight_mlx[self.padding_idx + 1:]
                ], axis=0)

            self.weight.data = Tensor._from_mlx_array(new_weight)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of embedding lookup.

        Args:
            input: Tensor of indices with dtype int32/int64

        Returns:
            Tensor of embeddings with shape (*input.shape, embedding_dim)
        """
        import mlx.core as mx

        # Get the input indices as MLX array
        indices = input._mlx_array

        # Ensure indices are integers
        if indices.dtype not in [mx.int32, mx.int64, mx.uint32, mx.uint64]:
            indices = indices.astype(mx.int32)

        # Get weight as MLX array
        weight = self.weight._mlx_array

        # Apply max_norm if specified
        if self.max_norm is not None:
            weight = self._apply_max_norm(weight)

        # Perform embedding lookup using advanced indexing
        # MLX supports this via take operation
        # Flatten indices, lookup, then reshape
        original_shape = indices.shape
        flat_indices = indices.flatten()

        # Lookup embeddings
        embeddings = mx.take(weight, flat_indices, axis=0)

        # Reshape to original shape + embedding_dim
        output_shape = original_shape + (self.embedding_dim,)
        embeddings = embeddings.reshape(output_shape)

        # Wrap in Tensor
        result = Tensor._from_mlx_array(embeddings)

        # Preserve gradient tracking
        if input.requires_grad or self.weight.requires_grad:
            result.requires_grad = True

        return result

    def _apply_max_norm(self, weight):
        """Apply max_norm constraint to embeddings."""
        import mlx.core as mx

        # Compute norms
        norms = mx.linalg.norm(weight, ord=self.norm_type, axis=1, keepdims=True)

        # Clamp norms to max_norm
        scale = mx.minimum(self.max_norm / (norms + 1e-8), mx.array(1.0))

        return weight * scale

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        if self.max_norm is not None:
            s += f', max_norm={self.max_norm}'
        if self.norm_type != 2.0:
            s += f', norm_type={self.norm_type}'
        if self.scale_grad_by_freq:
            s += ', scale_grad_by_freq=True'
        if self.sparse:
            s += ', sparse=True'
        return s

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> 'Embedding':
        """
        Create an Embedding instance from pre-trained embeddings.

        Args:
            embeddings: Tensor containing the pre-trained weights.
                       Shape: (num_embeddings, embedding_dim)
            freeze: If True, the tensor weights are not updated during training.
                   (default: True)
            padding_idx: See Embedding init.
            max_norm: See Embedding init.
            norm_type: See Embedding init.
            scale_grad_by_freq: See Embedding init.
            sparse: See Embedding init.

        Returns:
            Embedding instance with pre-trained weights.

        Example:
            >>> pretrained = mlx_compat.randn(1000, 300)  # 1000 words, 300-dim
            >>> embedding = nn.Embedding.from_pretrained(pretrained)
            >>> embedding.weight.requires_grad
            False
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D, got {embeddings.ndim}D"
            )

        num_embeddings, embedding_dim = embeddings.shape

        embedding = cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=embeddings,
        )

        if freeze:
            embedding.weight.requires_grad = False

        return embedding


class EmbeddingBag(Module):
    """
    Computes sums or means of 'bags' of embeddings, without instantiating
    the intermediate embeddings.

    This is more efficient than using Embedding followed by a reduction
    operation, especially for variable-length inputs.

    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        max_norm: See Embedding.
        norm_type: See Embedding.
        scale_grad_by_freq: See Embedding.
        mode: 'sum', 'mean', or 'max' (default: 'mean')
        sparse: See Embedding.
        include_last_offset: If True, offsets has one additional element,
                            where the last element is equivalent to the size
                            of indices. (default: False)
        padding_idx: See Embedding.

    Shape:
        - input: (N) where N is the total number of indices
        - offsets: (B) where B is the number of bags. offsets[i] is the
                   starting index for bag i.
        - Output: (B, embedding_dim)

    Example:
        >>> embedding_bag = nn.EmbeddingBag(10, 3, mode='sum')
        >>> input = mlx_compat.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        >>> offsets = mlx_compat.tensor([0, 4])  # 2 bags: [1,2,4,5] and [4,3,2,9]
        >>> output = embedding_bag(input, offsets)
        >>> output.shape
        (2, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = 'mean',
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        """Initialize EmbeddingBag."""
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError(f"mode must be 'sum', 'mean', or 'max', got '{mode}'")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset
        self.padding_idx = padding_idx

        # Initialize weight
        if _weight is None:
            from ... import randn
            self.weight = Parameter(randn(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            self.weight = Parameter(_weight)

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        import mlx.core as mx

        weight_data = mx.random.normal(
            shape=(self.num_embeddings, self.embedding_dim)
        )
        self.weight.data = Tensor._from_mlx_array(weight_data)

        if self.padding_idx is not None:
            # Zero out padding_idx
            zeros_row = mx.zeros((1, self.embedding_dim))
            weight_mlx = self.weight._mlx_array

            if self.padding_idx == 0:
                new_weight = mx.concatenate([zeros_row, weight_mlx[1:]], axis=0)
            elif self.padding_idx == self.num_embeddings - 1:
                new_weight = mx.concatenate([weight_mlx[:-1], zeros_row], axis=0)
            else:
                new_weight = mx.concatenate([
                    weight_mlx[:self.padding_idx],
                    zeros_row,
                    weight_mlx[self.padding_idx + 1:]
                ], axis=0)
            self.weight.data = Tensor._from_mlx_array(new_weight)

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of EmbeddingBag.

        Args:
            input: Tensor of indices.
            offsets: Only used when input is 1D. offsets[i] is the starting
                    position for bag i.
            per_sample_weights: Optional weights for weighted aggregation.

        Returns:
            Aggregated embeddings.
        """
        import mlx.core as mx

        indices = input._mlx_array
        weight = self.weight._mlx_array

        # Handle max_norm
        if self.max_norm is not None:
            norms = mx.linalg.norm(weight, ord=self.norm_type, axis=1, keepdims=True)
            scale = mx.minimum(self.max_norm / (norms + 1e-8), mx.array(1.0))
            weight = weight * scale

        # If input is 2D, treat each row as a bag
        if indices.ndim == 2:
            batch_size, bag_size = indices.shape

            # Lookup all embeddings
            flat_indices = indices.flatten()
            embeddings = mx.take(weight, flat_indices, axis=0)
            embeddings = embeddings.reshape((batch_size, bag_size, self.embedding_dim))

            # Apply per_sample_weights if provided
            if per_sample_weights is not None:
                psw = per_sample_weights._mlx_array
                embeddings = embeddings * psw.reshape((batch_size, bag_size, 1))

            # Aggregate
            if self.mode == 'sum':
                result = mx.sum(embeddings, axis=1)
            elif self.mode == 'mean':
                result = mx.mean(embeddings, axis=1)
            else:  # max
                result = mx.max(embeddings, axis=1)

        else:
            # 1D input with offsets
            if offsets is None:
                raise ValueError("offsets is required when input is 1D")

            offsets_mlx = offsets._mlx_array.astype(mx.int32)

            # Determine bag boundaries
            if self.include_last_offset:
                bag_boundaries = offsets_mlx
            else:
                # Add length of input as final boundary
                bag_boundaries = mx.concatenate([
                    offsets_mlx,
                    mx.array([len(indices)])
                ])

            num_bags = len(offsets_mlx)

            # Process each bag
            results = []
            for i in range(num_bags):
                start = int(bag_boundaries[i])
                end = int(bag_boundaries[i + 1])

                if start >= end:
                    # Empty bag
                    results.append(mx.zeros((self.embedding_dim,)))
                else:
                    bag_indices = indices[start:end]
                    bag_embeddings = mx.take(weight, bag_indices, axis=0)

                    # Apply per_sample_weights if provided
                    if per_sample_weights is not None:
                        psw = per_sample_weights._mlx_array[start:end]
                        bag_embeddings = bag_embeddings * psw.reshape((-1, 1))

                    # Aggregate
                    if self.mode == 'sum':
                        results.append(mx.sum(bag_embeddings, axis=0))
                    elif self.mode == 'mean':
                        results.append(mx.mean(bag_embeddings, axis=0))
                    else:  # max
                        results.append(mx.max(bag_embeddings, axis=0))

            result = mx.stack(results, axis=0)

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.max_norm is not None:
            s += f', max_norm={self.max_norm}'
        if self.norm_type != 2.0:
            s += f', norm_type={self.norm_type}'
        if self.scale_grad_by_freq:
            s += ', scale_grad_by_freq=True'
        s += f', mode={self.mode}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        return s


__all__ = ['Embedding', 'EmbeddingBag']
