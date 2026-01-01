#!/usr/bin/env python3
"""Compare benchmark results and detect performance regressions.

Usage:
    python scripts/compare_benchmarks.py --current results.json --baseline baseline.json
    python scripts/compare_benchmarks.py --current results.json --baseline baseline.json --threshold 0.15
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(filepath: Path) -> dict[str, Any]:
    """Load JSON file, return empty dict if not found."""
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        return json.load(f)


def compare_benchmarks(
    current: dict[str, Any],
    baseline: dict[str, Any],
    threshold: float = 0.1,
) -> tuple[list[tuple[str, float, float, float]], list[tuple[str, float, float, float]], list[tuple[str, float]]]:
    """Compare benchmark results.

    Returns:
        Tuple of (regressions, improvements, new_benchmarks)
        Each regression/improvement is (name, current_time, baseline_time, change_pct)
        Each new benchmark is (name, time)
    """
    regressions = []
    improvements = []
    new_benchmarks = []

    current_benchmarks = current.get("benchmarks", current.get("results", {}))
    baseline_benchmarks = baseline.get("benchmarks", baseline.get("results", {}))

    for name, result in current_benchmarks.items():
        # Handle different result formats
        if isinstance(result, dict):
            curr_time = result.get("mean_time_ms", result.get("mean", result.get("time_ms", 0)))
        else:
            curr_time = float(result)

        if name in baseline_benchmarks:
            base_result = baseline_benchmarks[name]
            if isinstance(base_result, dict):
                base_time = base_result.get("mean_time_ms", base_result.get("mean", base_result.get("time_ms", 0)))
            else:
                base_time = float(base_result)

            if base_time > 0:
                change = (curr_time - base_time) / base_time

                if change > threshold:
                    regressions.append((name, curr_time, base_time, change))
                elif change < -threshold:
                    improvements.append((name, curr_time, base_time, change))
        else:
            new_benchmarks.append((name, curr_time))

    return regressions, improvements, new_benchmarks


def format_markdown(
    regressions: list[tuple[str, float, float, float]],
    improvements: list[tuple[str, float, float, float]],
    new_benchmarks: list[tuple[str, float]],
    threshold: float,
) -> str:
    """Format comparison results as markdown."""
    lines = ["## Benchmark Results\n"]

    # Summary
    if regressions:
        lines.append(f"⚠️ **{len(regressions)} performance regression(s) detected** (>{threshold:.0%} slower)\n")
    else:
        lines.append("✅ **No performance regressions detected**\n")

    if improvements:
        lines.append(f"🚀 **{len(improvements)} performance improvement(s)**\n")

    lines.append("")

    # Regressions table
    if regressions:
        lines.append("### Regressions\n")
        lines.append("| Benchmark | Current (ms) | Baseline (ms) | Change |")
        lines.append("|-----------|-------------|---------------|--------|")
        for name, curr, base, change in sorted(regressions, key=lambda x: -x[3]):
            lines.append(f"| {name} | {curr:.3f} | {base:.3f} | +{change:.1%} 🔴 |")
        lines.append("")

    # Improvements table
    if improvements:
        lines.append("### Improvements\n")
        lines.append("| Benchmark | Current (ms) | Baseline (ms) | Change |")
        lines.append("|-----------|-------------|---------------|--------|")
        for name, curr, base, change in sorted(improvements, key=lambda x: x[3]):
            lines.append(f"| {name} | {curr:.3f} | {base:.3f} | {change:.1%} 🟢 |")
        lines.append("")

    # New benchmarks
    if new_benchmarks:
        lines.append("### New Benchmarks\n")
        lines.append("| Benchmark | Time (ms) |")
        lines.append("|-----------|-----------|")
        for name, time in sorted(new_benchmarks):
            lines.append(f"| {name} | {time:.3f} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--current", "-c", required=True, help="Current benchmark results JSON")
    parser.add_argument("--baseline", "-b", required=True, help="Baseline benchmark results JSON")
    parser.add_argument("--threshold", "-t", type=float, default=0.1, help="Regression threshold (default: 0.1 = 10%%)")
    parser.add_argument("--output", "-o", help="Output markdown file")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with error if regressions found")

    args = parser.parse_args()

    current = load_json(Path(args.current))
    baseline = load_json(Path(args.baseline))

    if not current:
        print(f"Error: Could not load current results from {args.current}", file=sys.stderr)
        return 1

    if not baseline:
        print(f"Warning: No baseline found at {args.baseline}, skipping comparison", file=sys.stderr)
        # Generate markdown with just current results
        markdown = "## Benchmark Results\n\n"
        markdown += "ℹ️ **No baseline available for comparison**\n\n"
        markdown += "This is the first benchmark run. Results will be used as baseline for future comparisons.\n"

        if args.output:
            Path(args.output).write_text(markdown)
            print(f"Report written to {args.output}")
        else:
            print(markdown)
        return 0

    regressions, improvements, new_benchmarks = compare_benchmarks(
        current, baseline, args.threshold
    )

    markdown = format_markdown(regressions, improvements, new_benchmarks, args.threshold)

    if args.output:
        Path(args.output).write_text(markdown)
        print(f"Report written to {args.output}")
    else:
        print(markdown)

    # Print summary to stderr for CI logs
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Regressions: {len(regressions)}", file=sys.stderr)
    print(f"  Improvements: {len(improvements)}", file=sys.stderr)
    print(f"  New benchmarks: {len(new_benchmarks)}", file=sys.stderr)

    if args.fail_on_regression and regressions:
        print(f"\nFailing due to {len(regressions)} regression(s)", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
