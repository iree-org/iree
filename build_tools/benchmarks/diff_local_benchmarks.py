#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Diffs two local benchmark result JSON files.

Example usage:
  python3 diff_local_benchmarks.py --base=/path/to/base_benchmarks.json
                                   --target=/path/to/target_benchmarks.json
"""

import pathlib
import sys

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse

from typing import Optional

from common.benchmark_presentation import *


def get_benchmark_result_markdown(
    base_benchmark_file: Optional[pathlib.Path],
    target_benchmark_file: Optional[pathlib.Path],
    base_compile_stats_file: Optional[pathlib.Path],
    target_compile_stats_file: Optional[pathlib.Path],
    verbose: bool = False) -> str:
  """Gets the full markdown summary of all benchmarks in files."""
  base_benchmarks = {}
  target_benchmarks = {}
  base_compilation_metrics = {}
  target_compilation_metrics = {}
  if base_benchmark_file and target_benchmark_file:
    base_benchmarks = aggregate_all_benchmarks([base_benchmark_file])
    target_benchmarks = aggregate_all_benchmarks([target_benchmark_file])
  if base_compile_stats_file and target_compile_stats_file:
    base_compilation_metrics = collect_all_compilation_metrics(
        [base_compile_stats_file])
    target_compilation_metrics = collect_all_compilation_metrics(
        [target_compile_stats_file])

  # Update the target benchmarks with their corresponding base numbers.
  for bench in base_benchmarks:
    if bench in target_benchmarks:
      target_benchmarks[bench].base_mean_time = base_benchmarks[bench].mean_time

  for target_name, base_metrics in base_compilation_metrics.items():
    updated_metrics = base_metrics
    for mapper in COMPILATION_METRICS_TO_TABLE_MAPPERS:
      metric_key = mapper.get_series_name(target_name)
      base_value, _ = mapper.get_current_and_base_value(base_metrics)
      updated_metrics = mapper.update_base_value(updated_metrics, base_value)
    target_compilation_metrics[target_name] = updated_metrics

  # Compose the full benchmark tables.
  full_table = [md.header("Full Benchmark Summary", 2)]
  full_table.append(categorize_benchmarks_into_tables(target_benchmarks))

  # Compose the full compilation metrics tables.
  full_table.append(
      categorize_compilation_metrics_into_tables(target_compilation_metrics))

  return "\n\n".join(full_table)


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    path = pathlib.Path(path)
    if path.is_file():
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("--base",
                      type=check_file_path,
                      help="Base benchmark results")
  parser.add_argument("--target",
                      type=check_file_path,
                      help="Target benchmark results")
  parser.add_argument("--base-compile-stats",
                      type=check_file_path,
                      help="Base compilation statistics")
  parser.add_argument("--target-compile-stats",
                      type=check_file_path,
                      help="Target compilation statistics")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  args = parser.parse_args()

  return args


if __name__ == "__main__":
  args = parse_arguments()
  if args.base or args.target:
    if not args.base or not args.target:
      raise ValueError("--base and --target must be used together.")
  if args.base_compile_stats or args.target_compile_stats:
    if not args.base_compile_stats or not args.target_compile_stats:
      raise ValueError("--base-compile-stats and --target-compile-stats must "
                       "be used together.")

  print(
      get_benchmark_result_markdown(args.base,
                                    args.target,
                                    args.base_compile_stats,
                                    args.target_compile_stats,
                                    verbose=args.verbose))
