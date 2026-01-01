"""
CLI entry point for parity checking.

Usage:
    python -m parity_check              # Console output
    python -m parity_check --strict     # Exit 1 if APIs missing
    python -m parity_check -f json      # JSON output
    python -m parity_check -f markdown  # Markdown output
"""

import argparse
import sys
from typing import List, Optional

from .config import PYTORCH_MODULES
from .reports import ReportFormatter
from .runner import ParityRunner


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check flashlight API parity with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m parity_check                    # Console output
    python -m parity_check --strict           # Exit 1 if missing APIs
    python -m parity_check -f json -o report.json
    python -m parity_check --modules torch torch.nn
        """,
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Exit with error code 1 if any APIs are missing",
    )

    parser.add_argument(
        "--signature-strict",
        action="store_true",
        help="Exit with error code 1 if any signatures mismatch",
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        default=PYTORCH_MODULES,
        help=f"PyTorch modules to check (default: {', '.join(PYTORCH_MODULES)})",
    )

    parser.add_argument(
        "--exclude-file",
        type=str,
        help="Additional exclusion YAML file",
    )

    parser.add_argument(
        "--no-strict-defaults",
        action="store_true",
        help="Don't strictly check default values",
    )

    parser.add_argument(
        "--strict-annotations",
        action="store_true",
        help="Strictly check type annotations",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output errors (for CI)",
    )

    # Strict PyTorch mode (disable MLX-aware defaults)
    parser.add_argument(
        "--strict-pytorch",
        action="store_true",
        help="Disable MLX-aware comparison (check 'out=', 'layout', etc.)",
    )

    # Numerical parity options
    parser.add_argument(
        "--numerical",
        action="store_true",
        help="Run numerical parity testing (compare actual outputs)",
    )

    parser.add_argument(
        "--numerical-strict",
        action="store_true",
        help="Exit with error code 1 if any numerical mismatches",
    )

    parser.add_argument(
        "--numerical-rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for numerical comparison (default: 1e-5)",
    )

    parser.add_argument(
        "--numerical-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numerical comparison (default: 1e-6)",
    )

    # Behavioral parity options
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Run behavioral parity testing",
    )

    parser.add_argument(
        "--behavioral-strict",
        action="store_true",
        help="Exit with error code 1 if any behavioral tests fail",
    )

    parser.add_argument(
        "--behavioral-categories",
        nargs="+",
        default=None,
        help="Categories to test: context_manager, module_state, container, optimizer, layer_mode, distribution, edge_cases",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point.

    Args:
        args: Command-line arguments (for testing)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parsed = parse_args(args)

    # MLX-aware comparison is default; --strict-pytorch disables it
    mlx_aware = not parsed.strict_pytorch

    # Run validation
    runner = ParityRunner(
        modules=parsed.modules,
        extra_exclusions_file=parsed.exclude_file,
        strict_defaults=not parsed.no_strict_defaults,
        strict_annotations=parsed.strict_annotations,
        ignore_out_param=mlx_aware,
        ignore_layout_params=mlx_aware,
        normalize_param_names=mlx_aware,
        # Numerical parity options
        run_numerical=parsed.numerical,
        numerical_rtol=parsed.numerical_rtol,
        numerical_atol=parsed.numerical_atol,
        # Behavioral parity options
        run_behavioral=parsed.behavioral,
        behavioral_categories=parsed.behavioral_categories,
    )

    try:
        report = runner.run()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Make sure PyTorch and flashlight are installed.", file=sys.stderr)
        return 1

    # Format output
    formatter = ReportFormatter()

    if parsed.format == "console":
        output = formatter.to_console(report)
    elif parsed.format == "json":
        output = formatter.to_json(report)
    elif parsed.format == "markdown":
        output = formatter.to_markdown(report)
    else:
        output = formatter.to_console(report)

    # Write output
    if not parsed.quiet:
        if parsed.output:
            with open(parsed.output, "w") as f:
                f.write(output)
            print(f"Report written to {parsed.output}")
        else:
            print(output)

    # Determine exit code
    exit_code = 0

    if parsed.strict and report.missing_apis > 0:
        if not parsed.quiet:
            print(
                f"\nStrict mode: {report.missing_apis} missing APIs",
                file=sys.stderr,
            )
        exit_code = 1

    if parsed.signature_strict and report.signature_mismatches > 0:
        if not parsed.quiet:
            print(
                f"\nSignature strict mode: {report.signature_mismatches} mismatches",
                file=sys.stderr,
            )
        exit_code = 1

    if parsed.numerical_strict and report.numerical_mismatches > 0:
        if not parsed.quiet:
            print(
                f"\nNumerical strict mode: {report.numerical_mismatches} mismatches",
                file=sys.stderr,
            )
        exit_code = 1

    if parsed.behavioral_strict and report.behavioral_failed > 0:
        if not parsed.quiet:
            print(
                f"\nBehavioral strict mode: {report.behavioral_failed} failed tests",
                file=sys.stderr,
            )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
