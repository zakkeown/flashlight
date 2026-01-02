"""
Random Number Generation Module

Provides PyTorch-compatible random number generation using the Philox algorithm.
"""

from .philox import PhiloxEngine
from .generator import Generator, default_generator

__all__ = [
    "PhiloxEngine",
    "Generator",
    "default_generator",
]
