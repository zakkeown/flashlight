"""
Comparison display utilities for benchmark results.

Provides formatted table output for comparing measurements.
"""

import collections
import enum
from typing import List, Optional

from .common import Measurement, select_unit, ordered_unique, trim_sigfig

__all__ = ["Compare", "Colorize"]

# ANSI color codes
_BEST = "\033[92m"  # Green
_GOOD = "\033[34m"  # Blue
_BAD = "\033[2m\033[91m"  # Dim red
_VERY_BAD = "\033[31m"  # Red
_BOLD = "\033[1m"
_TERMINATE = "\033[0m"


class Colorize(enum.Enum):
    """
    Colorization modes for Compare output.

    Attributes:
        NONE: No colorization.
        COLUMNWISE: Color based on column values.
        ROWWISE: Color based on row values.
    """

    NONE = "none"
    COLUMNWISE = "columnwise"
    ROWWISE = "rowwise"


class Compare:
    """
    Helper class for displaying benchmark results in formatted tables.

    Collects Measurement objects and displays them in a nicely formatted
    table with optional colorization and statistical analysis.

    Args:
        results: List of Measurements to display.

    Example:
        >>> timer1 = Timer(stmt="x + y", label="addition")
        >>> timer2 = Timer(stmt="x * y", label="multiplication")
        >>> m1 = timer1.blocked_autorange()
        >>> m2 = timer2.blocked_autorange()
        >>> compare = Compare([m1, m2])
        >>> compare.print()
    """

    def __init__(self, results: List[Measurement]) -> None:
        self._results: List[Measurement] = []
        self.extend_results(results)
        self._trim_significant_figures = False
        self._colorize = Colorize.NONE
        self._highlight_warnings = False

    def __str__(self) -> str:
        """Return formatted table as string."""
        return "\n".join(self._render())

    def extend_results(self, results: List[Measurement]) -> None:
        """
        Append results to stored results.

        Args:
            results: List of Measurements to add.

        Raises:
            ValueError: If any item is not a Measurement.
        """
        for r in results:
            if not isinstance(r, Measurement):
                raise ValueError(f"Expected Measurement, got {type(r)}")
        self._results.extend(results)

    def trim_significant_figures(self) -> "Compare":
        """
        Enable trimming of significant figures in output.

        Returns:
            self for method chaining.
        """
        self._trim_significant_figures = True
        return self

    def colorize(self, rowwise: bool = False) -> "Compare":
        """
        Enable colorization of output.

        Args:
            rowwise: If True, color by row; if False, color by column.

        Returns:
            self for method chaining.
        """
        self._colorize = Colorize.ROWWISE if rowwise else Colorize.COLUMNWISE
        return self

    def highlight_warnings(self) -> "Compare":
        """
        Enable warning highlighting in output.

        Returns:
            self for method chaining.
        """
        self._highlight_warnings = True
        return self

    def print(self) -> None:
        """Print formatted table to stdout."""
        print(str(self))

    def _render(self) -> List[str]:
        """Render all result groups as formatted strings."""
        results = Measurement.merge(self._results)
        grouped = self._group_by_label(results)
        return [self._layout(group) for group in grouped.values()]

    def _group_by_label(self, results: List[Measurement]):
        """Group results by their label."""
        grouped = collections.defaultdict(list)
        for r in results:
            grouped[r.label].append(r)
        return grouped

    def _layout(self, results: List[Measurement]) -> str:
        """Format a group of results as a table."""
        if not results:
            return ""

        # Determine time unit from minimum time
        min_time = min(r.median for r in results)
        time_unit, time_scale = select_unit(min_time)

        # Build table structure
        descriptions = ordered_unique([r.description for r in results])
        rows = ordered_unique([(r.sub_label or r.stmt) for r in results])

        # Handle empty descriptions
        if descriptions == [None]:
            descriptions = [""]

        # Create header
        header = [""] + [d or "" for d in descriptions]

        # Create data rows with values
        data_rows = []
        row_values = []  # Store numeric values for colorization

        for row_name in rows:
            row = [row_name]
            row_nums = []
            for desc in descriptions:
                m = next(
                    (
                        r
                        for r in results
                        if (r.sub_label or r.stmt) == row_name and r.description == desc
                    ),
                    None,
                )
                if m:
                    val = m.median / time_scale
                    if self._trim_significant_figures:
                        val = trim_sigfig(val, m.significant_figures)
                    row.append(f"{val:.2f}")
                    row_nums.append(val)
                else:
                    row.append("")
                    row_nums.append(None)
            data_rows.append(row)
            row_values.append(row_nums)

        # Apply colorization if enabled
        if self._colorize != Colorize.NONE:
            data_rows = self._apply_colorization(data_rows, row_values)

        # Calculate column widths
        all_rows = [header] + data_rows
        num_cols = len(header)
        col_widths = []
        for i in range(num_cols):
            width = max(
                len(self._strip_ansi(str(row[i]))) for row in all_rows if i < len(row)
            )
            col_widths.append(width)

        # Format output
        lines = []

        # Label header
        label = results[0].label or "Benchmark"
        lines.append(f"[{label}]")

        # Column header
        header_line = "  |  ".join(
            h.center(w) for h, w in zip(header, col_widths)
        )
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Data rows
        for row in data_rows:
            row_parts = []
            for i, (c, w) in enumerate(zip(row, col_widths)):
                # Center the display, accounting for ANSI codes
                visible_len = len(self._strip_ansi(str(c)))
                padding = w - visible_len
                left_pad = padding // 2
                right_pad = padding - left_pad
                row_parts.append(" " * left_pad + str(c) + " " * right_pad)
            lines.append("  |  ".join(row_parts))

        lines.append(f"\nTimes are in {time_unit}s.")

        return "\n".join(lines)

    def _strip_ansi(self, s: str) -> str:
        """Remove ANSI escape codes from string."""
        import re

        return re.sub(r"\033\[[0-9;]*m", "", s)

    def _apply_colorization(
        self, data_rows: List[List[str]], row_values: List[List[Optional[float]]]
    ) -> List[List[str]]:
        """Apply colorization to data rows based on values."""
        if not row_values:
            return data_rows

        if self._colorize == Colorize.ROWWISE:
            # Color within each row
            for i, (row, values) in enumerate(zip(data_rows, row_values)):
                valid_values = [v for v in values if v is not None]
                if not valid_values:
                    continue
                min_val = min(valid_values)
                max_val = max(valid_values)
                for j in range(1, len(row)):  # Skip row label
                    if values[j - 1] is not None:
                        row[j] = self._colorize_value(
                            row[j], values[j - 1], min_val, max_val
                        )

        elif self._colorize == Colorize.COLUMNWISE:
            # Color within each column
            num_cols = len(row_values[0]) if row_values else 0
            for col in range(num_cols):
                col_values = [rv[col] for rv in row_values if rv[col] is not None]
                if not col_values:
                    continue
                min_val = min(col_values)
                max_val = max(col_values)
                for i, row in enumerate(data_rows):
                    if row_values[i][col] is not None:
                        row[col + 1] = self._colorize_value(
                            row[col + 1], row_values[i][col], min_val, max_val
                        )

        return data_rows

    def _colorize_value(
        self, text: str, value: float, min_val: float, max_val: float
    ) -> str:
        """Apply color to a single value based on its position in range."""
        if min_val == max_val:
            return text

        # Lower is better for timing
        position = (value - min_val) / (max_val - min_val)

        if position < 0.1:
            return _BEST + text + _TERMINATE
        elif position < 0.5:
            return _GOOD + text + _TERMINATE
        elif position < 0.9:
            return _BAD + text + _TERMINATE
        else:
            return _VERY_BAD + text + _TERMINATE
