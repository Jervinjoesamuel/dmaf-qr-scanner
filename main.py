#!/usr/bin/env python3
"""
CLI entry point for the DMAF reference implementation.

Usage:
    python main.py                     # run with defaults (4000 synthetic samples)
    python main.py --n-samples 8000    # larger synthetic dataset
    python main.py --out report.json   # also save the results report to a file
"""
import argparse
import json

from dmaf.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run the DMAF malicious-QR detection pipeline.")
    parser.add_argument("--n-samples", type=int, default=4000, help="number of synthetic samples to generate")
    parser.add_argument("--test-size", type=float, default=0.25, help="held-out test fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default=None, help="optional path to save the JSON report")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logging")
    args = parser.parse_args()

    report = run(n_samples=args.n_samples, test_size=args.test_size,
                 seed=args.seed, verbose=not args.quiet)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
