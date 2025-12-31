"""
Before/after comparison framework for measuring performance improvements.

Provides utilities for:
- Saving benchmark results as baselines
- Comparing current results against baselines
- Statistical significance testing
- Report generation
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import datetime


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""
    benchmark_name: str
    config: Dict[str, Any]
    before_mean_ms: float
    before_std_ms: float
    after_mean_ms: float
    after_std_ms: float
    speedup: float  # before/after (>1 = faster)
    improvement_pct: float  # (before - after) / before * 100
    statistically_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "config": self.config,
            "before_mean_ms": self.before_mean_ms,
            "before_std_ms": self.before_std_ms,
            "after_mean_ms": self.after_mean_ms,
            "after_std_ms": self.after_std_ms,
            "speedup": self.speedup,
            "improvement_pct": self.improvement_pct,
            "statistically_significant": self.statistically_significant,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
        }


class BeforeAfterComparator:
    """Compare benchmark results before and after optimization."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def save_baseline(
        self,
        results: List[Dict[str, Any]],
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save benchmark results as baseline for future comparison."""
        baseline = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "1.0",
                **(metadata or {}),
            },
            "results": results,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)

    def load_baseline(self, filepath: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Load baseline results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        return data.get("metadata", {}), data.get("results", [])

    def compare_runs(
        self,
        before_results: List[Dict[str, Any]],
        after_results: List[Dict[str, Any]],
    ) -> List[ComparisonResult]:
        """
        Compare before/after results with statistical significance testing.

        Args:
            before_results: Baseline benchmark results
            after_results: Current benchmark results

        Returns:
            List of ComparisonResult objects
        """
        # Index results by (name, config) for matching
        before_index = {}
        for r in before_results:
            key = self._make_key(r)
            before_index[key] = r

        comparisons = []
        for after in after_results:
            key = self._make_key(after)
            if key not in before_index:
                continue

            before = before_index[key]

            # Extract timing stats
            before_mean = before.get("mlx_compat", {}).get("mean_ms", 0)
            before_std = before.get("mlx_compat", {}).get("std_ms", 0)
            after_mean = after.get("mlx_compat", {}).get("mean_ms", 0)
            after_std = after.get("mlx_compat", {}).get("std_ms", 0)

            if after_mean <= 0 or before_mean <= 0:
                continue

            # Calculate speedup
            speedup = before_mean / after_mean
            improvement_pct = (before_mean - after_mean) / before_mean * 100

            # Statistical significance testing (Welch's t-test approximation)
            p_value, significant = self._welch_ttest_approx(
                before_mean, before_std,
                after_mean, after_std,
                n_samples=100,  # Assume 100 samples
            )

            # Confidence interval for improvement
            ci = self._confidence_interval(
                before_mean, before_std,
                after_mean, after_std,
                n_samples=100,
            )

            comparisons.append(ComparisonResult(
                benchmark_name=after.get("name", "unknown"),
                config=after.get("input_config", {}),
                before_mean_ms=before_mean,
                before_std_ms=before_std,
                after_mean_ms=after_mean,
                after_std_ms=after_std,
                speedup=speedup,
                improvement_pct=improvement_pct,
                statistically_significant=significant,
                p_value=p_value,
                confidence_interval=ci,
            ))

        return comparisons

    def _make_key(self, result: Dict[str, Any]) -> str:
        """Create a unique key for matching results."""
        name = result.get("name", "")
        config = result.get("input_config", {})
        config_str = json.dumps(config, sort_keys=True)
        return f"{name}:{config_str}"

    def _welch_ttest_approx(
        self,
        mean1: float, std1: float,
        mean2: float, std2: float,
        n_samples: int,
    ) -> Tuple[float, bool]:
        """
        Approximate Welch's t-test using summary statistics.

        Returns (p_value, is_significant).
        """
        if std1 <= 0 or std2 <= 0:
            return 1.0, False

        # Standard error of difference
        se_diff = math.sqrt((std1**2 / n_samples) + (std2**2 / n_samples))

        if se_diff <= 0:
            return 1.0, False

        # t-statistic
        t_stat = (mean1 - mean2) / se_diff

        # Approximate degrees of freedom (Welch-Satterthwaite)
        v1 = std1**2 / n_samples
        v2 = std2**2 / n_samples
        dof = (v1 + v2)**2 / (v1**2 / (n_samples - 1) + v2**2 / (n_samples - 1))

        # Approximate p-value using normal distribution for large dof
        # For a two-tailed test
        z = abs(t_stat)

        # Simple normal approximation for p-value
        # p = 2 * (1 - Phi(|z|)) where Phi is standard normal CDF
        # Using approximation: p ≈ 2 * exp(-0.5 * z^2) / sqrt(2*pi) for large z
        if z > 6:
            p_value = 0.0
        else:
            # More accurate approximation
            p_value = 2 * (1 - self._normal_cdf(z))

        return p_value, p_value < self.significance_level

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        # Abramowitz and Stegun approximation
        if x < 0:
            return 1 - self._normal_cdf(-x)

        t = 1.0 / (1.0 + 0.2316419 * x)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        poly = (0.319381530 * t
                - 0.356563782 * t**2
                + 1.781477937 * t**3
                - 1.821255978 * t**4
                + 1.330274429 * t**5)
        return 1 - d * math.exp(-0.5 * x**2) * poly

    def _confidence_interval(
        self,
        mean1: float, std1: float,
        mean2: float, std2: float,
        n_samples: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        diff_mean = mean1 - mean2

        se_diff = math.sqrt((std1**2 / n_samples) + (std2**2 / n_samples))

        # z-value for 95% CI
        z = 1.96 if confidence == 0.95 else 1.645

        margin = z * se_diff
        return (diff_mean - margin, diff_mean + margin)

    def generate_report(
        self,
        comparisons: List[ComparisonResult],
        output_format: str = "console",
    ) -> str:
        """Generate before/after comparison report."""
        if output_format == "json":
            return self._generate_json_report(comparisons)
        else:
            return self._generate_console_report(comparisons)

    def _generate_console_report(self, comparisons: List[ComparisonResult]) -> str:
        """Generate console-friendly report."""
        lines = [
            "=" * 80,
            "           mlx_compat Performance Comparison Report",
            "=" * 80,
            "",
        ]

        # Summary stats
        improved = [c for c in comparisons if c.speedup > 1.0]
        improved_sig = [c for c in improved if c.statistically_significant]
        regressed = [c for c in comparisons if c.speedup < 1.0]
        unchanged = [c for c in comparisons if c.speedup == 1.0]

        # Geometric mean speedup
        if comparisons:
            log_sum = sum(math.log(c.speedup) for c in comparisons if c.speedup > 0)
            geo_mean = math.exp(log_sum / len(comparisons))
        else:
            geo_mean = 1.0

        # Table header
        lines.extend([
            f"{'Benchmark':<30} {'Before (ms)':<12} {'After (ms)':<12} {'Speedup':<10} {'Status'}",
            "-" * 80,
        ])

        # Sort by speedup (best improvements first)
        for comp in sorted(comparisons, key=lambda c: -c.speedup):
            status = ""
            if comp.statistically_significant and comp.speedup > 1.0:
                status = "**improved**"
            elif comp.speedup > 1.0:
                status = "improved"
            elif comp.speedup < 1.0:
                status = "regressed"
            else:
                status = "unchanged"

            name = comp.benchmark_name[:28]
            lines.append(
                f"{name:<30} {comp.before_mean_ms:<12.3f} {comp.after_mean_ms:<12.3f} "
                f"{comp.speedup:<10.2f}x {status}"
            )

        # Summary
        lines.extend([
            "",
            "=" * 80,
            "SUMMARY",
            "=" * 80,
            f"Total benchmarks: {len(comparisons)}",
            f"Improved:        {len(improved)} ({len(improved)/len(comparisons)*100:.1f}%)" if comparisons else "",
            f"  Statistically significant: {len(improved_sig)} (**marked)",
            f"Regressed:       {len(regressed)} ({len(regressed)/len(comparisons)*100:.1f}%)" if comparisons else "",
            "",
            f"Geometric mean speedup: {geo_mean:.2f}x",
            "=" * 80,
        ])

        return "\n".join(lines)

    def _generate_json_report(self, comparisons: List[ComparisonResult]) -> str:
        """Generate JSON report."""
        improved = [c for c in comparisons if c.speedup > 1.0]
        improved_sig = [c for c in improved if c.statistically_significant]
        regressed = [c for c in comparisons if c.speedup < 1.0]

        if comparisons:
            log_sum = sum(math.log(c.speedup) for c in comparisons if c.speedup > 0)
            geo_mean = math.exp(log_sum / len(comparisons))
        else:
            geo_mean = 1.0

        report = {
            "metadata": {
                "comparison_date": datetime.datetime.now().isoformat(),
                "significance_level": self.significance_level,
            },
            "comparisons": [c.to_dict() for c in comparisons],
            "summary": {
                "total": len(comparisons),
                "improved": len(improved),
                "improved_significant": len(improved_sig),
                "regressed": len(regressed),
                "geometric_mean_speedup": geo_mean,
            },
        }

        return json.dumps(report, indent=2)
