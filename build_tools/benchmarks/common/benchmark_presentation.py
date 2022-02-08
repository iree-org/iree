# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import urllib.parse
import markdown_strings as md

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .benchmark_definition import BenchmarkResults, StatisticResults
from .benchmark_thresholds import BENCHMARK_THRESHOLDS, ThresholdUnit

PERFBOARD_SERIES_PREFIX = "https://perf.iree.dev/serie?IREE?"


@dataclass
class BenchmarkData:
  """An object for describing latency numbers for a benchmark."""
  mean_time: int
  median_time: int
  stddev_time: int
  # The average latency time for the base commit to compare against.
  base_mean_time: Optional[int] = None


@dataclass
class StatisticData:
  """An object for describing values for a statistic."""
  value: int
  # The base value to compare against.
  base_value: Optional[int] = None


def aggregate_all_benchmarks_and_statistics(
    result_files: Sequence[str],
    expected_pr_commit: Optional[str] = None,
    verbose: bool = False
) -> Tuple[Dict[str, BenchmarkData], Dict[str, StatisticData]]:
  """Aggregates all benchmarks and statistics in the given files.

  Args:
  - result_files: A list of JSON files, each can be decoded as a
    BenchmarkResults or StatisticResults.
  - expected_pr_commit: An optional Git commit SHA to match against.

  Returns:
  - A tuple containing two dicts from names to BenchmarkData/StatisticData numbers.
  """

  benchmark_results = {}
  statistic_results = {}

  for benchmark_file in result_files:
    with open(benchmark_file) as f:
      content = f.read()
    is_statistics = "statistics" in json.loads(content)

    if is_statistics:
      file_results = StatisticResults.from_json_str(content)
    else:
      file_results = BenchmarkResults.from_json_str(content)

    if (expected_pr_commit is not None) and \
            (file_results.commit != expected_pr_commit):
      raise ValueError("Inconsistent pull request commit")

    if is_statistics:
      for statistic_index in range(len(file_results.statistics)):
        statistic_case = file_results.statistics[statistic_index]

        # Make sure each statistic has a unique name.
        name = str(statistic_case.statistic_info)
        if name in statistic_results:
          raise ValueError(f"Duplicated statistics: {name}")

        statistic_results[name] = StatisticData(statistic_case.value)
    else:
      for benchmark_index in range(len(file_results.benchmarks)):
        benchmark_case = file_results.benchmarks[benchmark_index]

        # Make sure each benchmark has a unique name.
        name = str(benchmark_case.benchmark_info)
        if name in benchmark_results:
          raise ValueError(f"Duplicated benchmarks: {name}")

        # Now scan all benchmark iterations and find the aggregate results.
        mean_time = file_results.get_aggregate_time(benchmark_index, "mean")
        median_time = file_results.get_aggregate_time(benchmark_index, "median")
        stddev_time = file_results.get_aggregate_time(benchmark_index, "stddev")

        benchmark_results[name] = BenchmarkData(mean_time, median_time,
                                                stddev_time)

  return (benchmark_results, statistic_results)


def _make_benchmark_clickable(name: str) -> str:
  """Add link to the given benchmark name."""
  url = PERFBOARD_SERIES_PREFIX + urllib.parse.quote(name, safe="()[]@,")
  return md.link(name, url)


def _add_header_and_get_markdown_table_for_benchmarks(
    names: Tuple[str],
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


def _sort_benchmarks_and_get_table(benchmarks: Dict[str, BenchmarkData],
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

  return _add_header_and_get_markdown_table_for_benchmarks(
      names, str_means, medians, stddevs, size_cut)


def categorize_benchmarks_into_tables(benchmarks: Dict[str, BenchmarkData],
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
        _add_header_and_get_markdown_table_for_benchmarks(names=names,
                                                          means=means,
                                                          medians=medians,
                                                          stddevs=stddevs,
                                                          size_cut=size_cut))
  return "\n\n".join(tables)


def _add_header_and_get_markdown_table_for_statistics(
    names: Tuple[str],
    values: Tuple[Any],
    size_cut: Optional[int] = None) -> str:
  """Generates a markdown table with proper headers for statistics.

  Args:
  - size_cut: If not None, only show the top N results for each table.
  """
  total_size = len(names)
  if size_cut is not None:
    names = names[0:size_cut]
    values = values[0:size_cut]

  names = tuple([_make_benchmark_clickable(name) for name in names])
  names = ("Statistic Name",) + names
  values = ("Value",) + values

  table_str = md.table([names, values])
  if size_cut is not None and size_cut < total_size:
    table_str += "\n\n"
    table_str += md.italics(
        f"[Top {size_cut} out of {total_size} statistic results showed]")
  return table_str


def _sort_statistics_and_get_table(statistics: Dict[str, StatisticData],
                                   size_cut: Optional[int] = None):
  """Sorts all statistics according to the improvement/regression ratio and
  returns a markdown table for it.

  Args:
  - size_cut: If not None, only show the top N results for each table.
  """
  sorted_statistics = []
  for k, v in statistics.items():
    ratio = abs(v.value - v.base_value) / v.base_value
    sorted_statistics.append((k, (v.value, v.base_value, ratio)))
  # Sort according to ratio in the reverse order.
  sorted_statistics.sort(key=lambda statistic: statistic[1][2], reverse=True)

  # Split each field into its own tuple in prepration for markdown table.
  names, values = zip(*sorted_statistics)

  # Turn the tuple about values into a string representation.
  str_values = []
  for pr, base, ratio in values:
    direction = "↑" if pr > base else ("↓" if pr < base else "")
    str_values.append(f"{pr} (vs. {base}, {ratio:.2%}{direction})")
  str_values = tuple(str_values)

  return _add_header_and_get_markdown_table_for_statistics(names, str_values)


def categorize_statistics_into_tables(statistics: Dict[str, StatisticData],
                                      size_cut: Optional[int] = None) -> str:
  """Splits statistics into regressed/improved/similar/raw categories and
  returns their markdown tables.

    Args:
    - statistics: A dictionary of statistic names to its aggregate info.
    - size_cut: If not None, only show the top N results for each table.
    """
  regressed, improved, similar, raw = {}, {}, {}, {}

  for name, results in statistics.items():
    # If no informatio about the base result. Then we cannot analyze.
    if results.base_value is None:
      raw[name] = results
      continue

    similar_threshold = None
    for threshold in BENCHMARK_THRESHOLDS:
      if threshold.regex.match(name):
        similar_threshold = threshold
        break
    if similar_threshold is None:
      raise ValueError(f"no matched threshold setting for statistic: {name}")

    current = results.value
    base = results.base_value
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
    tables.append(md.header("Regressed Statistics 🚩", 3))
    tables.append(_sort_statistics_and_get_table(regressed, size_cut))
  if improved:
    tables.append(md.header("Improved Statistics 🎉", 3))
    tables.append(_sort_statistics_and_get_table(improved, size_cut))
  # If we want to abbreviate, similar results won't be interesting.
  if similar and size_cut is None:
    tables.append(md.header("Similar Statistics", 3))
    tables.append(_sort_statistics_and_get_table(similar, size_cut))
  if raw:
    tables.append(md.header("Raw Statistics", 3))
    raw_list = [(k, v.value) for k, v in raw.items()]
    names, values = zip(*raw_list)
    tables.append(
        _add_header_and_get_markdown_table_for_statistics(names=names,
                                                          values=values,
                                                          size_cut=size_cut))
  return "\n\n".join(tables)
