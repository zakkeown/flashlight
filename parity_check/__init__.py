"""
PyTorch API Parity Validation System for flashlight.

This module provides tools to automatically detect PyTorch's public API
and validate that flashlight implements matching functions with compatible signatures.

Usage:
    python -m parity_check              # Console output
    python -m parity_check --strict     # Exit 1 if APIs missing
    python -m parity_check -f json      # JSON output
"""

from .runner import ParityRunner
from .config import MODULE_MAPPING, PYTORCH_MODULES

__all__ = ["ParityRunner", "MODULE_MAPPING", "PYTORCH_MODULES"]
