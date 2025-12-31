"""Output formatters for benchmark results."""

from benchmarks.output.console import ConsoleFormatter
from benchmarks.output.json_output import JSONFormatter

__all__ = [
    "ConsoleFormatter",
    "JSONFormatter",
]
