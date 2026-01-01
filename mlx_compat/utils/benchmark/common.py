"""
Base classes and utilities for benchmarking.

Provides TaskSpec, Measurement, and utility functions for the benchmark module.
"""

import collections
import dataclasses
import math
import statistics
from typing import Any, List, Dict, Optional, Iterable, Tuple


__all__ = [
    "TaskSpec",
    "Measurement",
    "select_unit",
    "unit_to_english",
    "trim_sigfig",
    "ordered_unique",
]

_MAX_SIGNIFICANT_FIGURES = 4
_MIN_CONFIDENCE_INTERVAL = 25e-9  # 25 ns
_IQR_WARN_THRESHOLD = 0.1
_IQR_GROSS_WARN_THRESHOLD = 0.25


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """
    Container for Timer task specification.

    Attributes:
        stmt: The statement to be timed.
        setup: Setup code to be run once before timing.
        global_setup: Global setup code.
        label: Main label for the task.
        sub_label: Sub-label for additional categorization.
        description: Description for display in tables.
        env: Environment tag for A/B testing.
        num_threads: Thread count (always 1 for MLX).
    """

    stmt: str
    setup: str = "pass"
    global_setup: str = ""
    label: Optional[str] = None
    sub_label: Optional[str] = None
    description: Optional[str] = None
    env: Optional[str] = None
    num_threads: int = 1

    @property
    def title(self) -> str:
        """Return a display title for this task."""
        if self.label is not None:
            return self.label + (f": {self.sub_label}" if self.sub_label else "")
        elif "\n" not in self.stmt:
            return self.stmt + (f": {self.sub_label}" if self.sub_label else "")
        return f"stmt:{f' ({self.sub_label})' if self.sub_label else ''}"

    def summarize(self) -> str:
        """Return a summary of this task."""
        sections = [self.title, self.description or ""]
        return "\n".join([s for s in sections if s])


@dataclasses.dataclass
class Measurement:
    """
    The result of a Timer measurement.

    Attributes:
        number_per_run: Number of times the statement was executed per timing run.
        raw_times: List of raw timing results (total time for number_per_run executions).
        task_spec: The TaskSpec describing what was measured.
        metadata: Optional additional metadata.
    """

    number_per_run: int
    raw_times: List[float]
    task_spec: TaskSpec
    metadata: Optional[Dict[Any, Any]] = None

    def __post_init__(self) -> None:
        self._sorted_times: Tuple[float, ...] = ()
        self._warnings: Tuple[str, ...] = ()
        self._median: float = -1.0
        self._mean: float = -1.0
        self._p25: float = -1.0
        self._p75: float = -1.0

    @property
    def times(self) -> List[float]:
        """Return per-iteration times (raw_times / number_per_run)."""
        return [t / self.number_per_run for t in self.raw_times]

    @property
    def median(self) -> float:
        """Return the median time per iteration."""
        self._lazy_init()
        return self._median

    @property
    def mean(self) -> float:
        """Return the mean time per iteration."""
        self._lazy_init()
        return self._mean

    @property
    def iqr(self) -> float:
        """Return the interquartile range of times."""
        self._lazy_init()
        return self._p75 - self._p25

    @property
    def significant_figures(self) -> int:
        """Estimate the number of significant figures in the measurement."""
        self._lazy_init()
        n_total = len(self._sorted_times)
        lower = int(n_total // 4)
        upper = int(math.ceil(3 * n_total / 4))
        iqr_points = self._sorted_times[lower:upper]

        if len(iqr_points) < 2:
            return 1

        std = statistics.stdev(iqr_points) if len(iqr_points) > 1 else 0
        sqrt_n = math.sqrt(len(iqr_points))
        ci = max(1.645 * std / sqrt_n, _MIN_CONFIDENCE_INTERVAL)
        relative_ci = math.log10(self._median / ci) if ci > 0 and self._median > 0 else 0
        return min(max(int(relative_ci), 1), _MAX_SIGNIFICANT_FIGURES)

    @property
    def has_warnings(self) -> bool:
        """Return True if this measurement has any warnings."""
        self._lazy_init()
        return bool(self._warnings)

    def _lazy_init(self) -> None:
        """Lazily compute statistics from raw times."""
        if self.raw_times and not self._sorted_times:
            self._sorted_times = tuple(sorted(self.times))
            self._median = statistics.median(self._sorted_times)
            self._mean = statistics.mean(self._sorted_times)
            n = len(self._sorted_times)
            self._p25 = self._sorted_times[n // 4] if n >= 4 else self._sorted_times[0]
            self._p75 = (
                self._sorted_times[3 * n // 4] if n >= 4 else self._sorted_times[-1]
            )

            if not self.meets_confidence(_IQR_GROSS_WARN_THRESHOLD):
                self._warnings += ("High variance in measurements",)

    def meets_confidence(self, threshold: float = _IQR_WARN_THRESHOLD) -> bool:
        """Check if IQR/median is below the threshold."""
        return self.iqr / self.median < threshold if self.median > 0 else True

    @property
    def label(self) -> Optional[str]:
        """Return the task label."""
        return self.task_spec.label

    @property
    def sub_label(self) -> Optional[str]:
        """Return the task sub-label."""
        return self.task_spec.sub_label

    @property
    def description(self) -> Optional[str]:
        """Return the task description."""
        return self.task_spec.description

    @property
    def env(self) -> Optional[str]:
        """Return the environment tag."""
        return self.task_spec.env

    @property
    def num_threads(self) -> int:
        """Return the thread count."""
        return self.task_spec.num_threads

    @property
    def stmt(self) -> str:
        """Return the timed statement."""
        return self.task_spec.stmt

    @property
    def as_row_name(self) -> str:
        """Return a name suitable for use as a table row."""
        return self.sub_label or self.stmt or "[Unknown]"

    def __repr__(self) -> str:
        self._lazy_init()
        time_unit, time_scale = select_unit(self._median)
        n = len(self._sorted_times)
        return (
            f"<Measurement: {self._median / time_scale:.2f} {time_unit} "
            f"({n} runs, {self.number_per_run} per run)>"
        )

    @staticmethod
    def merge(measurements: Iterable["Measurement"]) -> List["Measurement"]:
        """
        Merge measurements with the same TaskSpec.

        Args:
            measurements: Iterable of Measurement objects.

        Returns:
            List of merged Measurements.
        """
        grouped: Dict[TaskSpec, List[Measurement]] = collections.defaultdict(list)
        for m in measurements:
            grouped[m.task_spec].append(m)

        result = []
        for task_spec, group in grouped.items():
            times = []
            for m in group:
                times.extend(m.times)
            result.append(
                Measurement(
                    number_per_run=1,
                    raw_times=times,
                    task_spec=task_spec,
                )
            )
        return result


def select_unit(t: float) -> Tuple[str, float]:
    """
    Select appropriate time unit for display.

    Args:
        t: Time in seconds.

    Returns:
        Tuple of (unit_string, scale_factor).
    """
    if t < 1e-6:
        return "ns", 1e-9
    elif t < 1e-3:
        return "us", 1e-6
    elif t < 1.0:
        return "ms", 1e-3
    else:
        return "s", 1.0


def unit_to_english(u: str) -> str:
    """
    Convert unit abbreviation to English word.

    Args:
        u: Unit abbreviation ('ns', 'us', 'ms', or 's').

    Returns:
        English word for the unit.
    """
    return {
        "ns": "nanosecond",
        "us": "microsecond",
        "ms": "millisecond",
        "s": "second",
    }[u]


def trim_sigfig(x: float, n: int) -> float:
    """
    Trim a number to n significant figures.

    Args:
        x: The number to trim.
        n: Number of significant figures.

    Returns:
        The trimmed number.
    """
    if x == 0:
        return 0.0
    magnitude = int(math.ceil(math.log10(abs(x))))
    scale = 10 ** (magnitude - n)
    return round(x / scale) * scale


def ordered_unique(elements: Iterable[Any]) -> List[Any]:
    """
    Return unique elements while preserving order.

    Args:
        elements: Iterable of elements.

    Returns:
        List of unique elements in original order.
    """
    return list(collections.OrderedDict.fromkeys(elements))
