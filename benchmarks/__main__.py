"""
flashlight Benchmarking Suite CLI

Usage:
    python -m benchmarks [OPTIONS]

Examples:
    python -m benchmarks                          # Run all benchmarks
    python -m benchmarks --level operator         # Operator benchmarks only
    python -m benchmarks --level layer            # Layer benchmarks only
    python -m benchmarks --level model            # Model benchmarks only
    python -m benchmarks --filter "matmul*"       # Filter by name pattern
    python -m benchmarks --compare-pytorch        # Include PyTorch comparison
    python -m benchmarks -o json -f results.json  # Output to JSON file
    python -m benchmarks --memory                 # Include memory profiling
    python -m benchmarks --quick                  # Quick mode (fewer iterations)
    python -m benchmarks --list                   # List available benchmarks
"""

import argparse
import sys
from typing import List, Optional

from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BenchmarkRunner, BaseBenchmark
from benchmarks.output.console import ConsoleFormatter
from benchmarks.output.json_output import JSONFormatter


def get_all_benchmarks(config: BenchmarkConfig) -> List[BaseBenchmark]:
    """Load all benchmark classes and instantiate them."""
    from benchmarks.operators.arithmetic import ARITHMETIC_BENCHMARKS
    from benchmarks.operators.activations import ACTIVATION_BENCHMARKS
    from benchmarks.operators.reductions import REDUCTION_BENCHMARKS
    from benchmarks.operators.convolution import CONVOLUTION_BENCHMARKS
    from benchmarks.operators.pooling import POOLING_BENCHMARKS
    from benchmarks.operators.linalg import LINALG_BENCHMARKS
    from benchmarks.operators.comparison import COMPARISON_BENCHMARKS
    from benchmarks.operators.math_funcs import MATH_FUNCS_BENCHMARKS
    from benchmarks.layers.linear import LINEAR_BENCHMARKS
    from benchmarks.layers.conv import CONV_BENCHMARKS
    from benchmarks.layers.normalization import NORMALIZATION_BENCHMARKS
    from benchmarks.layers.attention import ATTENTION_BENCHMARKS
    from benchmarks.layers.pooling import POOLING_BENCHMARKS as LAYER_POOLING_BENCHMARKS
    from benchmarks.layers.activation import ACTIVATION_BENCHMARKS as LAYER_ACTIVATION_BENCHMARKS
    from benchmarks.layers.embedding import EMBEDDING_BENCHMARKS
    from benchmarks.layers.rnn import RNN_BENCHMARKS
    from benchmarks.layers.losses import LOSS_BENCHMARKS
    from benchmarks.models.mlp import MLP_BENCHMARKS
    from benchmarks.models.cnn import CNN_BENCHMARKS
    from benchmarks.models.resnet import RESNET_BENCHMARKS
    from benchmarks.models.transformer import TRANSFORMER_BENCHMARKS

    all_classes = (
        ARITHMETIC_BENCHMARKS +
        ACTIVATION_BENCHMARKS +
        REDUCTION_BENCHMARKS +
        CONVOLUTION_BENCHMARKS +
        POOLING_BENCHMARKS +
        LINALG_BENCHMARKS +
        COMPARISON_BENCHMARKS +
        MATH_FUNCS_BENCHMARKS +
        LINEAR_BENCHMARKS +
        CONV_BENCHMARKS +
        NORMALIZATION_BENCHMARKS +
        ATTENTION_BENCHMARKS +
        LAYER_POOLING_BENCHMARKS +
        LAYER_ACTIVATION_BENCHMARKS +
        EMBEDDING_BENCHMARKS +
        RNN_BENCHMARKS +
        LOSS_BENCHMARKS +
        MLP_BENCHMARKS +
        CNN_BENCHMARKS +
        RESNET_BENCHMARKS +
        TRANSFORMER_BENCHMARKS
    )

    return [cls(config) for cls in all_classes]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="flashlight Benchmarking Suite - Performance comparison with PyTorch MPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks                          Run all benchmarks (console output)
  python -m benchmarks --level operator         Run operator-level benchmarks only
  python -m benchmarks --level layer            Run layer-level benchmarks only
  python -m benchmarks --level model            Run model-level benchmarks only
  python -m benchmarks --filter "matmul*"       Run benchmarks matching pattern
  python -m benchmarks --compare-pytorch        Include PyTorch MPS comparison
  python -m benchmarks -o json -f results.json  Output to JSON file
  python -m benchmarks --memory                 Include memory profiling
  python -m benchmarks --quick                  Quick mode (fewer iterations)
  python -m benchmarks --list                   List available benchmarks
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--level", "-l",
        choices=["operator", "layer", "model", "all"],
        default="all",
        help="Benchmark level to run (default: all)",
    )

    parser.add_argument(
        "--filter", "-F",
        type=str,
        default=None,
        help="Filter benchmarks by name pattern (glob-style)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks without running",
    )

    # PyTorch comparison
    parser.add_argument(
        "--compare-pytorch", "--compare", "-c",
        action="store_true",
        default=True,
        help="Compare against PyTorch (default: enabled)",
    )

    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Disable PyTorch comparison",
    )

    parser.add_argument(
        "--pytorch-device",
        choices=["mps", "cpu", "cuda"],
        default="mps",
        help="PyTorch device for comparison (default: mps)",
    )

    # Execution settings
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )

    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )

    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=5,
        help="Number of trial runs (default: 5)",
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: warmup=5, iterations=20, trials=3",
    )

    # Memory profiling
    parser.add_argument(
        "--memory", "-m",
        action="store_true",
        help="Enable memory profiling",
    )

    # Output settings
    parser.add_argument(
        "--output", "-o",
        choices=["console", "json", "both"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--output-file", "-f",
        type=str,
        default=None,
        help="Output file path (for json output)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with additional statistics",
    )

    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress header in console output",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress during benchmarking (default: enabled)",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Quick mode overrides
    if args.quick:
        args.warmup = 5
        args.iterations = 20
        args.trials = 3

    # Handle comparison flag
    compare_pytorch = args.compare_pytorch and not args.no_compare

    # Handle progress flag
    show_progress = args.progress and not args.no_progress

    # Map level string to enum
    level_map = {
        "operator": BenchmarkLevel.OPERATOR,
        "layer": BenchmarkLevel.LAYER,
        "model": BenchmarkLevel.MODEL,
        "all": BenchmarkLevel.ALL,
    }
    level = level_map[args.level]

    # Create configuration
    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        num_trials=args.trials,
        compare_pytorch=compare_pytorch,
        pytorch_device=args.pytorch_device,
        track_memory=args.memory,
        output_format=args.output,
        output_file=args.output_file,
        verbose=args.verbose,
    )

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Load benchmarks
    try:
        all_benchmarks = get_all_benchmarks(config)
    except ImportError as e:
        print(f"Error loading benchmarks: {e}", file=sys.stderr)
        return 1

    # Filter by level
    if level != BenchmarkLevel.ALL:
        all_benchmarks = [b for b in all_benchmarks if b.level == level]

    # Filter by name pattern
    if args.filter:
        import fnmatch
        all_benchmarks = [
            b for b in all_benchmarks
            if fnmatch.fnmatch(b.name.lower(), args.filter.lower())
        ]

    # List mode
    if args.list:
        print("\nAvailable benchmarks:")
        print("-" * 50)

        operators = [b for b in all_benchmarks if b.level == BenchmarkLevel.OPERATOR]
        layers = [b for b in all_benchmarks if b.level == BenchmarkLevel.LAYER]
        models = [b for b in all_benchmarks if b.level == BenchmarkLevel.MODEL]

        if operators:
            print("\nOperators:")
            for b in operators:
                print(f"  - {b.name}")

        if layers:
            print("\nLayers:")
            for b in layers:
                print(f"  - {b.name}")

        if models:
            print("\nModels:")
            for b in models:
                print(f"  - {b.name}")

        print(f"\nTotal: {len(all_benchmarks)} benchmarks")
        return 0

    if not all_benchmarks:
        print("No benchmarks match the specified filters.", file=sys.stderr)
        return 1

    # Create runner
    runner = BenchmarkRunner(config)
    runner.add_benchmarks(all_benchmarks)

    # Progress callback
    console = ConsoleFormatter(verbose=args.verbose, show_header=not args.no_header)

    def progress_callback(name: str, current: int, total: int) -> None:
        if show_progress:
            console.print_progress(name, current, total)

    # Run benchmarks
    print("\nRunning benchmarks...")
    print("=" * 50)

    try:
        results = runner.run(
            level=level,
            filter_pattern=args.filter,
            progress_callback=progress_callback if show_progress else None,
        )
    except KeyboardInterrupt:
        print("\n\nBenchmarking interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during benchmarking: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Generate summary
    summary = runner.summarize(results)

    # Output results
    if args.output in ["console", "both"]:
        console.print_results(results, summary)

    if args.output in ["json", "both"]:
        json_formatter = JSONFormatter()

        if args.output_file:
            json_formatter.save_results(results, args.output_file, summary)
            print(f"\nResults saved to: {args.output_file}")
        else:
            print(json_formatter.format_results(results, summary))

    return 0


if __name__ == "__main__":
    sys.exit(main())
