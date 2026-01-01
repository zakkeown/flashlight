"""
Timer class for benchmarking MLX operations.

Provides accurate timing with MLX synchronization for lazy evaluation.
"""

import timeit
import textwrap
import time
from typing import Any, Callable, Optional, List, Dict

from .common import TaskSpec, Measurement

__all__ = ["Timer", "timer"]


def timer() -> float:
    """
    Return current time, with MLX synchronization.

    This ensures all pending MLX operations are complete before
    taking the timestamp, providing accurate timing for lazy operations.

    Returns:
        Current time in seconds.
    """
    try:
        import mlx.core as mx

        mx.synchronize()
    except (ImportError, AttributeError):
        pass
    return timeit.default_timer()


class Timer:
    """
    Helper class for measuring execution time of MLX statements.

    Based on PyTorch's Timer, optimized for MLX operations with
    proper lazy evaluation handling and synchronization.

    Args:
        stmt: Code snippet to be run and timed.
        setup: Optional setup code to run before timing.
        timer: Callable returning current time (default uses MLX-aware timer).
        globals: Dict of global variables for stmt execution.
        label: String summarizing stmt for display.
        sub_label: Supplemental disambiguation info.
        description: Column description for Compare tables.
        env: Environment tag for A/B testing.
        num_threads: Thread count (compatibility only, always 1 for MLX).

    Example:
        >>> t = Timer(
        ...     stmt="mx.matmul(a, b)",
        ...     setup="import mlx.core as mx; a = mx.random.normal((100, 100)); b = mx.random.normal((100, 100))",
        ...     label="matmul",
        ... )
        >>> print(t.blocked_autorange())
    """

    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        timer: Callable[[], float] = timer,
        globals: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads: int = 1,
    ) -> None:
        self._globals = dict(globals or {})

        # Add mlx_compat to globals by default
        try:
            import mlx_compat

            self._globals.setdefault("mlx_compat", mlx_compat)
            self._globals.setdefault("mx", mlx_compat)  # Convenience alias
        except ImportError:
            pass

        # Add mlx.core as well
        try:
            import mlx.core as mx

            self._globals.setdefault("mlx", mx)
        except ImportError:
            pass

        # Clean up statements
        stmt = textwrap.dedent(stmt).strip()
        setup = textwrap.dedent(setup).strip()

        self._timer_fn = timer
        self._timer = timeit.Timer(
            stmt=stmt,
            setup=setup,
            timer=timer,
            globals=self._globals,
        )

        self._task_spec = TaskSpec(
            stmt=stmt,
            setup=setup,
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
            num_threads=num_threads,
        )

    def _sync_and_time(self, number: int) -> float:
        """
        Time with MLX synchronization.

        Args:
            number: Number of times to execute the statement.

        Returns:
            Total elapsed time in seconds.
        """
        try:
            import mlx.core as mx
        except ImportError:
            return max(self._timer.timeit(number), 1e-9)

        # Warmup and sync
        mx.synchronize()

        # Time
        start = time.perf_counter()
        self._timer.timeit(number)
        mx.synchronize()
        elapsed = time.perf_counter() - start

        return max(elapsed, 1e-9)

    def timeit(self, number: int = 1000000) -> Measurement:
        """
        Execute stmt `number` times and return measurement.

        Args:
            number: Number of times to execute stmt.

        Returns:
            Measurement with timing results.

        Example:
            >>> t = Timer(stmt="x = 1 + 1")
            >>> m = t.timeit(1000)
            >>> print(m.median)
        """
        # Warmup
        self._sync_and_time(max(number // 100, 2))

        return Measurement(
            number_per_run=number,
            raw_times=[self._sync_and_time(number)],
            task_spec=self._task_spec,
        )

    def repeat(self, repeat: int = 5, number: int = 1000000) -> Measurement:
        """
        Execute stmt `number` times, repeating `repeat` times.

        Args:
            repeat: Number of times to repeat the timing.
            number: Number of statement executions per repeat.

        Returns:
            Measurement with all timing results.

        Example:
            >>> t = Timer(stmt="x = 1 + 1")
            >>> m = t.repeat(repeat=5, number=1000)
            >>> print(f"Median: {m.median:.6f}s")
        """
        # Warmup
        self._sync_and_time(max(number // 100, 2))

        times = [self._sync_and_time(number) for _ in range(repeat)]

        return Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec,
        )

    def blocked_autorange(
        self,
        callback: Optional[Callable[[int, float], None]] = None,
        min_run_time: float = 0.2,
    ) -> Measurement:
        """
        Measure many replicates while minimizing timer overhead.

        This method automatically determines an appropriate number of
        iterations and collects enough measurements to meet the minimum
        run time.

        Args:
            callback: Called after each measurement with (number, time).
            min_run_time: Minimum total measurement time in seconds.

        Returns:
            Measurement with timing statistics.

        Example:
            >>> t = Timer(stmt="mx.matmul(a, b)", setup="...")
            >>> m = t.blocked_autorange(min_run_time=1.0)
            >>> print(m)
        """
        # Estimate block size
        number = self._estimate_block_size(min_run_time)

        # Collect measurements
        times: List[float] = []
        total_time = 0.0

        while total_time < min_run_time:
            t = self._sync_and_time(number)
            times.append(t)
            total_time += t
            if callback:
                callback(number, t)

        return Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec,
        )

    def _estimate_block_size(self, min_run_time: float) -> int:
        """
        Estimate appropriate block size for measurement.

        Args:
            min_run_time: Target minimum run time.

        Returns:
            Number of iterations per block.
        """
        number = 1
        while True:
            t = self._sync_and_time(number)
            if t >= min_run_time / 1000 or number >= 10_000_000:
                break
            number *= 10
        return number

    def adaptive_autorange(
        self,
        threshold: float = 0.1,
        *,
        min_run_time: float = 0.01,
        max_run_time: float = 10.0,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Measurement:
        """
        Similar to blocked_autorange but adapts until variance is acceptable.

        Continues measuring until the IQR/median ratio falls below the
        threshold or max_run_time is reached.

        Args:
            threshold: IQR/median threshold for stopping (default 0.1 = 10%).
            min_run_time: Minimum total run time before checking threshold.
            max_run_time: Maximum total run time.
            callback: Called after each measurement.

        Returns:
            Measurement with timing statistics.

        Example:
            >>> t = Timer(stmt="mx.matmul(a, b)", setup="...")
            >>> m = t.adaptive_autorange(threshold=0.05)
            >>> print(f"Confident measurement: {m.median:.6f}s")
        """
        number = self._estimate_block_size(0.05)
        times: List[float] = []
        total_time = 0.0

        while total_time < max_run_time:
            t = self._sync_and_time(number)
            times.append(t)
            total_time += t

            if callback:
                callback(number, t)

            if len(times) > 3 and total_time >= min_run_time:
                m = Measurement(
                    number_per_run=number,
                    raw_times=times,
                    task_spec=self._task_spec,
                )
                if m.meets_confidence(threshold):
                    break

        return Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec,
        )

    def autorange(self, callback: Optional[Callable[[int, float], None]] = None) -> Measurement:
        """
        Automatically determine number of iterations and time.

        Similar to timeit.Timer.autorange(), finds the smallest number
        that takes at least 0.2 seconds and returns a single measurement.

        Args:
            callback: Called after the measurement with (number, time).

        Returns:
            Measurement with timing result.
        """
        number = self._estimate_block_size(0.2)
        t = self._sync_and_time(number)

        if callback:
            callback(number, t)

        return Measurement(
            number_per_run=number,
            raw_times=[t],
            task_spec=self._task_spec,
        )
