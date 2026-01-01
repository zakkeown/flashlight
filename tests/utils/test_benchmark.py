"""
Tests for mlx_compat.utils.benchmark module.

Tests Timer, Measurement, Compare, and utility functions.
"""

import unittest
import time

from mlx_compat.utils.benchmark import (
    Timer,
    Measurement,
    Compare,
    TaskSpec,
    Colorize,
    timer,
    select_unit,
    unit_to_english,
    trim_sigfig,
    ordered_unique,
)


class TestTaskSpec(unittest.TestCase):
    """Test TaskSpec dataclass."""

    def test_basic_creation(self):
        """Test basic TaskSpec creation."""
        spec = TaskSpec(stmt="x + y")
        self.assertEqual(spec.stmt, "x + y")
        self.assertEqual(spec.setup, "pass")
        self.assertIsNone(spec.label)

    def test_title_from_label(self):
        """Test title generation from label."""
        spec = TaskSpec(stmt="x + y", label="addition")
        self.assertEqual(spec.title, "addition")

    def test_title_with_sublabel(self):
        """Test title generation with sub_label."""
        spec = TaskSpec(stmt="x + y", label="addition", sub_label="100x100")
        self.assertEqual(spec.title, "addition: 100x100")

    def test_title_from_stmt(self):
        """Test title generation from statement when no label."""
        spec = TaskSpec(stmt="x + y")
        self.assertEqual(spec.title, "x + y")

    def test_summarize(self):
        """Test summarize method."""
        spec = TaskSpec(stmt="x + y", label="test", description="A test operation")
        summary = spec.summarize()
        self.assertIn("test", summary)
        self.assertIn("A test operation", summary)


class TestMeasurement(unittest.TestCase):
    """Test Measurement class."""

    def test_basic_creation(self):
        """Test basic Measurement creation."""
        spec = TaskSpec(stmt="pass")
        m = Measurement(
            number_per_run=1000,
            raw_times=[0.1, 0.11, 0.09],
            task_spec=spec,
        )

        self.assertEqual(m.number_per_run, 1000)
        self.assertEqual(len(m.raw_times), 3)

    def test_times_property(self):
        """Test times property divides by number_per_run."""
        spec = TaskSpec(stmt="pass")
        m = Measurement(
            number_per_run=100,
            raw_times=[1.0, 2.0],
            task_spec=spec,
        )

        times = m.times
        self.assertAlmostEqual(times[0], 0.01)
        self.assertAlmostEqual(times[1], 0.02)

    def test_median(self):
        """Test median calculation."""
        spec = TaskSpec(stmt="pass")
        m = Measurement(
            number_per_run=1,
            raw_times=[0.1, 0.2, 0.3, 0.4, 0.5],
            task_spec=spec,
        )

        self.assertAlmostEqual(m.median, 0.3)

    def test_mean(self):
        """Test mean calculation."""
        spec = TaskSpec(stmt="pass")
        m = Measurement(
            number_per_run=1,
            raw_times=[0.1, 0.2, 0.3],
            task_spec=spec,
        )

        self.assertAlmostEqual(m.mean, 0.2)

    def test_iqr(self):
        """Test interquartile range calculation."""
        spec = TaskSpec(stmt="pass")
        m = Measurement(
            number_per_run=1,
            raw_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            task_spec=spec,
        )

        # IQR should be reasonable
        self.assertGreater(m.iqr, 0)

    def test_as_row_name(self):
        """Test as_row_name property."""
        spec = TaskSpec(stmt="test_stmt", sub_label="my_label")
        m = Measurement(number_per_run=1, raw_times=[0.1], task_spec=spec)

        self.assertEqual(m.as_row_name, "my_label")

    def test_repr(self):
        """Test string representation."""
        spec = TaskSpec(stmt="pass")
        m = Measurement(
            number_per_run=100,
            raw_times=[0.001, 0.001],
            task_spec=spec,
        )

        repr_str = repr(m)
        self.assertIn("Measurement", repr_str)
        self.assertIn("2 runs", repr_str)
        self.assertIn("100 per run", repr_str)

    def test_merge(self):
        """Test merging measurements with same TaskSpec."""
        spec = TaskSpec(stmt="pass", label="test")
        m1 = Measurement(number_per_run=1, raw_times=[0.1, 0.2], task_spec=spec)
        m2 = Measurement(number_per_run=1, raw_times=[0.15, 0.25], task_spec=spec)

        merged = Measurement.merge([m1, m2])

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].times), 4)


class TestTimer(unittest.TestCase):
    """Test Timer class."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        t = Timer(stmt="x = 1 + 1")
        m = t.timeit(number=100)

        self.assertIsInstance(m, Measurement)
        self.assertEqual(m.number_per_run, 100)
        self.assertGreater(m.median, 0)

    def test_setup_code(self):
        """Test that setup code runs before timing."""
        t = Timer(
            stmt="y = x * 2",
            setup="x = 5",
        )
        m = t.timeit(number=100)

        self.assertIsInstance(m, Measurement)
        self.assertGreater(m.median, 0)

    def test_repeat(self):
        """Test repeat method."""
        t = Timer(stmt="x = 1 + 1")
        m = t.repeat(repeat=5, number=100)

        self.assertEqual(len(m.raw_times), 5)
        self.assertEqual(m.number_per_run, 100)

    def test_blocked_autorange(self):
        """Test blocked_autorange method."""
        t = Timer(stmt="x = 1 + 1")
        m = t.blocked_autorange(min_run_time=0.05)

        self.assertIsInstance(m, Measurement)
        self.assertGreater(len(m.raw_times), 0)

    def test_adaptive_autorange(self):
        """Test adaptive_autorange method."""
        t = Timer(stmt="x = 1 + 1")
        m = t.adaptive_autorange(
            threshold=0.5,  # Relaxed threshold for testing
            min_run_time=0.01,
            max_run_time=0.5,
        )

        self.assertIsInstance(m, Measurement)

    def test_autorange(self):
        """Test autorange method."""
        t = Timer(stmt="x = 1 + 1")
        m = t.autorange()

        self.assertIsInstance(m, Measurement)
        self.assertEqual(len(m.raw_times), 1)

    def test_callback(self):
        """Test callback functionality."""
        callback_calls = []

        def my_callback(number, time):
            callback_calls.append((number, time))

        t = Timer(stmt="x = 1 + 1")
        t.autorange(callback=my_callback)

        self.assertEqual(len(callback_calls), 1)
        self.assertIsInstance(callback_calls[0][0], int)
        self.assertIsInstance(callback_calls[0][1], float)

    def test_label_and_sublabel(self):
        """Test label and sub_label are preserved."""
        t = Timer(
            stmt="x = 1 + 1",
            label="arithmetic",
            sub_label="addition",
        )
        m = t.timeit(100)

        self.assertEqual(m.label, "arithmetic")
        self.assertEqual(m.sub_label, "addition")

    def test_mlx_operations(self):
        """Test timing MLX operations."""
        t = Timer(
            stmt="mx.add(a, b)",
            setup="import mlx.core as mx; a = mx.ones((10, 10)); b = mx.ones((10, 10))",
            label="mlx_add",
        )
        m = t.timeit(100)

        self.assertIsInstance(m, Measurement)
        self.assertGreater(m.median, 0)


class TestCompare(unittest.TestCase):
    """Test Compare class."""

    def test_basic_creation(self):
        """Test basic Compare creation."""
        spec = TaskSpec(stmt="pass", label="test")
        m = Measurement(number_per_run=1, raw_times=[0.1], task_spec=spec)

        compare = Compare([m])
        self.assertIsNotNone(compare)

    def test_str_output(self):
        """Test string output."""
        spec = TaskSpec(stmt="pass", label="test")
        m = Measurement(number_per_run=1, raw_times=[0.001], task_spec=spec)

        compare = Compare([m])
        output = str(compare)

        self.assertIn("test", output)
        self.assertIn("Times are in", output)

    def test_extend_results(self):
        """Test extending results."""
        spec1 = TaskSpec(stmt="a", label="test")
        spec2 = TaskSpec(stmt="b", label="test")
        m1 = Measurement(number_per_run=1, raw_times=[0.1], task_spec=spec1)
        m2 = Measurement(number_per_run=1, raw_times=[0.2], task_spec=spec2)

        compare = Compare([m1])
        compare.extend_results([m2])

        output = str(compare)
        self.assertIn("a", output)
        self.assertIn("b", output)

    def test_trim_significant_figures(self):
        """Test trim_significant_figures method."""
        spec = TaskSpec(stmt="pass", label="test")
        m = Measurement(number_per_run=1, raw_times=[0.001], task_spec=spec)

        compare = Compare([m])
        result = compare.trim_significant_figures()

        self.assertIs(result, compare)  # Returns self for chaining

    def test_colorize(self):
        """Test colorize method."""
        spec = TaskSpec(stmt="pass", label="test")
        m = Measurement(number_per_run=1, raw_times=[0.001], task_spec=spec)

        compare = Compare([m])
        result = compare.colorize()

        self.assertIs(result, compare)

    def test_colorize_rowwise(self):
        """Test rowwise colorization."""
        spec = TaskSpec(stmt="pass", label="test")
        m = Measurement(number_per_run=1, raw_times=[0.001], task_spec=spec)

        compare = Compare([m])
        compare.colorize(rowwise=True)

        self.assertEqual(compare._colorize, Colorize.ROWWISE)

    def test_invalid_result_type(self):
        """Test that invalid result types raise ValueError."""
        with self.assertRaises(ValueError):
            Compare(["not a measurement"])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_timer_function(self):
        """Test timer() function returns time."""
        t1 = timer()
        time.sleep(0.01)
        t2 = timer()

        self.assertGreater(t2, t1)

    def test_select_unit_nanoseconds(self):
        """Test unit selection for very small times."""
        unit, scale = select_unit(1e-9)
        self.assertEqual(unit, "ns")
        self.assertEqual(scale, 1e-9)

    def test_select_unit_microseconds(self):
        """Test unit selection for microseconds."""
        unit, scale = select_unit(1e-6)
        self.assertEqual(unit, "us")
        self.assertEqual(scale, 1e-6)

    def test_select_unit_milliseconds(self):
        """Test unit selection for milliseconds."""
        unit, scale = select_unit(0.001)
        self.assertEqual(unit, "ms")
        self.assertEqual(scale, 1e-3)

    def test_select_unit_seconds(self):
        """Test unit selection for seconds."""
        unit, scale = select_unit(1.0)
        self.assertEqual(unit, "s")
        self.assertEqual(scale, 1.0)

    def test_unit_to_english(self):
        """Test unit abbreviation to English conversion."""
        self.assertEqual(unit_to_english("ns"), "nanosecond")
        self.assertEqual(unit_to_english("us"), "microsecond")
        self.assertEqual(unit_to_english("ms"), "millisecond")
        self.assertEqual(unit_to_english("s"), "second")

    def test_trim_sigfig(self):
        """Test significant figure trimming."""
        self.assertAlmostEqual(trim_sigfig(123.456, 3), 123.0)
        self.assertAlmostEqual(trim_sigfig(0.00123, 2), 0.0012)
        self.assertEqual(trim_sigfig(0, 3), 0.0)

    def test_ordered_unique(self):
        """Test ordered unique function."""
        result = ordered_unique([1, 2, 2, 3, 1, 4])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_ordered_unique_preserves_order(self):
        """Test that first occurrence order is preserved."""
        result = ordered_unique(["b", "a", "c", "a", "b"])
        self.assertEqual(result, ["b", "a", "c"])


if __name__ == "__main__":
    unittest.main()
