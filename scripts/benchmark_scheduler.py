#!/usr/bin/env python3
"""
benchmark_scheduler.py
Automates chunk size sweep and thread scaling benchmarks for the scheduler runtime.
Logs load imbalance ratios and execution times.
"""

import subprocess
import sys
import os
import re

def run_benchmark(exe_path):
    if not os.path.exists(exe_path):
        print(f"Error: executable {exe_path} not found. Build it first.")
        sys.exit(1)

    print(f"Executing: {exe_path}")
    result = subprocess.run([exe_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(base_dir, "bin", "benchmark_scheduler.exe")
    if not os.path.exists(exe):
        exe = os.path.join(base_dir, "bin", "benchmark_scheduler")
    run_benchmark(exe)
