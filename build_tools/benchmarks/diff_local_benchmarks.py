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

import argparse
import json
import os
import requests

from common.benchmark_presentation import *


def get_benchmark_result_markdown(base_benchmark_file: str,
                                  target_benchmark_file: str,
                                  verbose: bool = False) -> str:
  """Gets the full markdown summary of all benchmarks in files."""
  base_benchmarks = aggregate_all_benchmarks([base_benchmark_file])
  target_benchmarks = aggregate_all_benchmarks([target_benchmark_file])

  # Update the target benchmarks with their corresponding base numbers.
  for bench in base_benchmarks:
    if bench in target_benchmarks:
      target_benchmarks[bench].base_mean_time = base_benchmarks[bench].mean_time

  # Compose the full benchmark tables.
  full_table = [md.header("Full Benchmark Summary", 2)]
  full_table.append(categorize_benchmarks_into_tables(target_benchmarks))

  return "\n\n".join(full_table)


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("--base",
                      type=check_file_path,
                      required=True,
                      help="Base benchmark results")
  parser.add_argument("--target",
                      type=check_file_path,
                      required=True,
                      help="Target benchmark results")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  args = parser.parse_args()

  return args


if __name__ == "__main__":
  args = parse_arguments()
  print(
      get_benchmark_result_markdown(args.base,
                                    args.target,
                                    verbose=args.verbose))
