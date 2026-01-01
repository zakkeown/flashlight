"""
Common Type Definitions

PyTorch-compatible torch.nn.common_types module with type aliases
used for type hints in neural network modules.
"""

from typing import List, Sequence, Tuple, Union

# Size type aliases for different dimensionalities
# These are used extensively in convolution and pooling layer signatures

_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]
_size_4_t = Union[int, Tuple[int, int, int, int]]
_size_5_t = Union[int, Tuple[int, int, int, int, int]]
_size_6_t = Union[int, Tuple[int, int, int, int, int, int]]

# Any size
_size_any_t = Union[int, Tuple[int, ...]]

# Optional size types
_size_any_opt_t = Union[int, Tuple[int, ...], None]

# Ratio types for things like aspect ratios
_ratio_2_t = Union[float, Tuple[float, float]]
_ratio_3_t = Union[float, Tuple[float, float, float]]
_ratio_any_t = Union[float, Tuple[float, ...]]

# Boolean types for different dimensionalities
_bool_t = bool
_bool_1_t = Union[bool, Tuple[bool]]
_bool_2_t = Union[bool, Tuple[bool, bool]]
_bool_3_t = Union[bool, Tuple[bool, bool, bool]]

# Reverse types (for transposed convolutions)
_reverse_repeat_tuple = Tuple

# Type for list of integers
_list_int_t = List[int]

__all__ = [
    "_size_1_t",
    "_size_2_t",
    "_size_3_t",
    "_size_4_t",
    "_size_5_t",
    "_size_6_t",
    "_size_any_t",
    "_size_any_opt_t",
    "_ratio_2_t",
    "_ratio_3_t",
    "_ratio_any_t",
    "_bool_t",
    "_bool_1_t",
    "_bool_2_t",
    "_bool_3_t",
    "_reverse_repeat_tuple",
    "_list_int_t",
]
