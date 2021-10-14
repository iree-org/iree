# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import urllib.parse
import markdown_strings as md

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .benchmark_definition import BenchmarkResults
from .benchmark_thresholds import BENCHMARK_THRESHOLDS, ThresholdUnit

PERFBOARD_SERIES_PREFIX = "https://perf.iree.dev/serie?IREE?"


@dataclass
class AggregateBenchmarkLatency:
  """An object for describing aggregate latency numbers for a benchmark."""
  mean_time: int
  median_time: int
  stddev_time: int
  # The average latency time for the base commit to compare against.
  base_mean_time: Optional[int] = None


def aggregate_all_benchmarks(
    benchmark_files: Sequence[str],
    expected_pr_commit: Optional[str] = None,
    verbose: bool = False) -> Dict[str, AggregateBenchmarkLatency]:
  """Aggregates all benchmarks in the given files.

  Args:
  - benchmark_files: A list of JSON files, each can be decoded as a
    BenchmarkResults.
  - expected_pr_commit: An optional Git commit SHA to match against.

  Returns:
  - A dict of benchmark names to AggregateBenchmarkLatency numbers.
  """

  aggregate_results = {}

  for benchmark_file in benchmark_files:
    with open(benchmark_file) as f:
      content = f.read()
    file_results = BenchmarkResults.from_json_str(content)

    if (expected_pr_commit is not None) and \
            (file_results.commit != expected_pr_commit):
      raise ValueError("Inconsistent pull request commit")

    for benchmark_index in range(len(file_results.benchmarks)):
      benchmark_case = file_results.benchmarks[benchmark_index]

      # Make sure each benchmark has a unique name.
      name = str(benchmark_case["benchmark"])
      if name in aggregate_results:
        raise ValueError(f"Duplicated benchmarks: {name}")

      # Now scan all benchmark iterations and find the aggregate results.
      mean_time = file_results.get_aggregate_time(benchmark_index, "mean")
      median_time = file_results.get_aggregate_time(benchmark_index, "median")
      stddev_time = file_results.get_aggregate_time(benchmark_index, "stddev")

      aggregate_results[name] = AggregateBenchmarkLatency(
          mean_time, median_time, stddev_time)

  return aggregate_results


def _make_benchmark_clickable(name: str) -> str:
  """Add link to the given benchmark name."""
  url = PERFBOARD_SERIES_PREFIX + urllib.parse.quote(name, safe="()[]@,")
  return md.link(name, url)


def _add_header_and_get_markdown_table(names: Tuple[str],
                                       means: Tuple[Any],
                                       medians: Tuple[int],
                                       stddevs: Tuple[int],
                                       size_cut: Optional[int] = None) -> str:
  """Generates a markdown table with proper headers for benchmarks.

  Args:
  - size_cut: If not None, only show the top N results for each table.
  """
  total_size = len(names)
  if size_cut is not None:
    names = names[0:size_cut]
    means = means[0:size_cut]
    medians = medians[0:size_cut]
    stddevs = stddevs[0:size_cut]

  names = tuple([_make_benchmark_clickable(name) for name in names])
  names = ("Benchmark Name",) + names
  means = ("Average Latency (ms)",) + means
  medians = ("Median Latency (ms)",) + medians
  stddevs = ("Latency Standard Deviation (ms)",) + stddevs

  table_str = md.table([names, means, medians, stddevs])
  if size_cut is not None and size_cut < total_size:
    table_str += "\n\n"
    table_str += md.italics(
        f"[Top {size_cut} out of {total_size} benchmark results showed]")
  return table_str


def _sort_benchmarks_and_get_table(benchmarks: Dict[str,
                                                    AggregateBenchmarkLatency],
                                   size_cut: Optional[int] = None):
  """Sorts all benchmarks according to the improvement/regression ratio and
  returns a markdown table for it.

  Args:
  - size_cut: If not None, only show the top N results for each table.
  """
  sorted_benchmarks = []
  for k, v in benchmarks.items():
    ratio = abs(v.mean_time - v.base_mean_time) / v.base_mean_time
    sorted_benchmarks.append((k, (v.mean_time, v.base_mean_time, ratio),
                              v.median_time, v.stddev_time))
  # Sort according to ratio in the reverse order.
  sorted_benchmarks.sort(key=lambda benchmark: benchmark[1][2], reverse=True)

  # Split each field into its own tuple in prepration for markdown table.
  names, means, medians, stddevs = zip(*sorted_benchmarks)

  # Turn the tuple about means into a string representation.
  str_means = []
  for pr, base, ratio in means:
    direction = "↑" if pr > base else ("↓" if pr < base else "")
    str_means.append(f"{pr} (vs. {base}, {ratio:.2%}{direction})")
  str_means = tuple(str_means)

  return _add_header_and_get_markdown_table(names, str_means, medians, stddevs,
                                            size_cut)


def categorize_benchmarks_into_tables(benchmarks: Dict[
    str, AggregateBenchmarkLatency],
                                      size_cut: Optional[int] = None) -> str:
  """Splits benchmarks into regressed/improved/similar/raw categories and
  returns their markdown tables.

    Args:
    - benchmarks: A dictionary of benchmark names to its aggregate info.
    - size_cut: If not None, only show the top N results for each table.
    """
  regressed, improved, similar, raw = {}, {}, {}, {}

  for name, results in benchmarks.items():
    # If no informatio about the base result. Then we cannot analyze.
    if results.base_mean_time is None:
      raw[name] = results
      continue

    similar_threshold = None
    for threshold in BENCHMARK_THRESHOLDS:
      if threshold.regex.match(name):
        similar_threshold = threshold
        break
    if similar_threshold is None:
      raise ValueError(f"no matched threshold setting for benchmark: {name}")

    current = results.mean_time
    base = results.base_mean_time
    if similar_threshold.unit == ThresholdUnit.PERCENTAGE:
      ratio = abs(current - base) / base * 100
    else:
      ratio = abs(current - base)

    if ratio <= similar_threshold.threshold:
      similar[name] = results
    elif current > base:
      regressed[name] = results
    else:
      improved[name] = results

  tables = []
  if regressed:
    tables.append(md.header("Regressed Benchmarks 🚩", 3))
    tables.append(_sort_benchmarks_and_get_table(regressed, size_cut))
  if improved:
    tables.append(md.header("Improved Benchmarks 🎉", 3))
    tables.append(_sort_benchmarks_and_get_table(improved, size_cut))
  # If we want to abbreviate, similar results won't be interesting.
  if similar and size_cut is None:
    tables.append(md.header("Similar Benchmarks", 3))
    tables.append(_sort_benchmarks_and_get_table(similar, size_cut))
  if raw:
    tables.append(md.header("Raw Benchmarks", 3))
    raw_list = [
        (k, v.mean_time, v.median_time, v.stddev_time) for k, v in raw.items()
    ]
    names, means, medians, stddevs = zip(*raw_list)
    tables.append(
        _add_header_and_get_markdown_table(names=names,
                                           means=means,
                                           medians=medians,
                                           stddevs=stddevs,
                                           size_cut=size_cut))
  return "\n\n".join(tables)
