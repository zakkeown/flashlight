#!/usr/bin/env python3
"""
Run optimizer benchmarks and save results.

Usage:
    python benchmarks/optim/run_benchmarks.py
    python benchmarks/optim/run_benchmarks.py --output results.md
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from datetime import datetime
from benchmark_optimizers import run_all_benchmarks, format_results_as_markdown


def main():
    parser = argparse.ArgumentParser(description='Run optimizer benchmarks')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for results (default: print to stdout)')
    args = parser.parse_args()

    print("Running optimizer benchmarks...")
    print()

    results = run_all_benchmarks()
    markdown = format_results_as_markdown(results)

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown += f"\n\n*Benchmark run on {timestamp}*\n"

    if args.output:
        with open(args.output, 'w') as f:
            f.write(markdown)
        print(f"\nResults saved to {args.output}")
    else:
        print()
        print(markdown)


if __name__ == '__main__':
    main()
