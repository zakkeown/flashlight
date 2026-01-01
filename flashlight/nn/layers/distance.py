"""
Distance Modules

Implements nn.CosineSimilarity and nn.PairwiseDistance.
"""

from typing import Optional

from ...tensor import Tensor
from ..functional import cosine_similarity, pairwise_distance
from ..module import Module


class CosineSimilarity(Module):
    """
    Compute cosine similarity between two tensors along a dimension.

    Args:
        dim: Dimension along which to compute cosine similarity (default: 1)
        eps: Small constant for numerical stability (default: 1e-8)

    Shape:
        - Input1: (*, D, *) where D is at position `dim`
        - Input2: (*, D, *), same shape as Input1
        - Output: (*, *) with dimension D removed

    Example:
        >>> cos = nn.CosineSimilarity(dim=1)
        >>> x1 = torch.randn(100, 128)
        >>> x2 = torch.randn(100, 128)
        >>> output = cos(x1, x2)
        >>> output.shape
        torch.Size([100])
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class PairwiseDistance(Module):
    """
    Compute pairwise distance between two vectors using the p-norm.

    The distance is computed as:
        d(x, y) = ||x - y||_p

    Args:
        p: Norm degree (default: 2)
        eps: Small constant to avoid division by zero (default: 1e-6)
        keepdim: Whether to keep the output dimension (default: False)

    Shape:
        - Input1: (N, D)
        - Input2: (N, D)
        - Output: (N,) or (N, 1) if keepdim=True

    Example:
        >>> pdist = nn.PairwiseDistance(p=2)
        >>> x1 = torch.randn(100, 128)
        >>> x2 = torch.randn(100, 128)
        >>> output = pdist(x1, x2)
        >>> output.shape
        torch.Size([100])
    """

    def __init__(self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False):
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pairwise_distance(x1, x2, p=self.p, eps=self.eps, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return f"p={self.p}, eps={self.eps}, keepdim={self.keepdim}"


__all__ = ["CosineSimilarity", "PairwiseDistance"]
