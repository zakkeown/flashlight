#!/usr/bin/env python3
"""
Run all data loading benchmarks.

Usage:
    python benchmarks/data/run_benchmarks.py [--quick]

Options:
    --quick    Run a quick subset of benchmarks
"""

import sys
import argparse

sys.path.insert(0, '../..')

from benchmark_dataloader import run_dataloader_benchmarks
from benchmark_samplers import run_sampler_benchmarks
from benchmark_collate import run_collate_benchmarks


def main():
    parser = argparse.ArgumentParser(description='Run data loading benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks only')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MLX Compat Data Loading Benchmarks")
    print("=" * 70)
    print("\nComparing MLX data loading against PyTorch...")
    print("All benchmarks use num_workers=0 for fair comparison.\n")

    try:
        # Run all benchmark suites
        run_dataloader_benchmarks()
        run_sampler_benchmarks()
        run_collate_benchmarks()

        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print("\nNotes:")
        print("  - MLX uses unified memory (no CPU/GPU transfer overhead)")
        print("  - PyTorch benchmarks run with num_workers=0 (single process)")
        print("  - Shuffle performance depends on MLX random vs Python random")
        print("  - Collate performance depends on stack/concatenation speed")
        print("\n" + "=" * 70)
        print("Benchmarks complete!")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nBenchmarks interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmarks: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
