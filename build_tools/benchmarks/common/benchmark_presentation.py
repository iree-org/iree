# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import urllib.parse
import markdown_strings as md

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

from common.benchmark_definition import BenchmarkResults
from common.benchmark_thresholds import BENCHMARK_THRESHOLDS, BenchmarkThreshold, ThresholdUnit

GetMetricFunc = Callable[[Any], Tuple[int, Optional[int]]]
GetTableRowFunc = Callable[[str, Any], Tuple]

PERFBOARD_SERIES_PREFIX = "https://perf.iree.dev/serie?IREE?"
BENCHMARK_RESULTS_HEADERS = [
    "Benchmark Name",
    "Average Latency (ms)",
    "Median Latency (ms)",
    "Latency Standard Deviation (ms)",
]


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
      name = str(benchmark_case.benchmark_info)
      if name in aggregate_results:
        raise ValueError(f"Duplicated benchmarks: {name}")

      # Now scan all benchmark iterations and find the aggregate results.
      mean_time = file_results.get_aggregate_time(benchmark_index, "mean")
      median_time = file_results.get_aggregate_time(benchmark_index, "median")
      stddev_time = file_results.get_aggregate_time(benchmark_index, "stddev")

      aggregate_results[name] = AggregateBenchmarkLatency(
          mean_time, median_time, stddev_time)

  return aggregate_results


def _make_series_link(name: str, series: Optional[str] = None) -> str:
  """Add link to the given benchmark name.

    Args:
      name: the text to show on the link.
      series: the dashboard series name. Use name if None.
  """
  if series is None:
    series = name
  url = PERFBOARD_SERIES_PREFIX + urllib.parse.quote(series, safe="()[]@,")
  return md.link(name, url)


def _add_header_and_get_markdown_table(headers: Sequence[str],
                                       rows: Sequence[Tuple],
                                       size_cut: Optional[int] = None) -> str:
  """Generates a markdown table with headers.

  Args:
    headers: list of table headers.
    rows: list of rows. Each row is a tuple with the same length as headers.
    size_cut: If not None, only show the top N results for each table.
  """

  total_size = len(rows)
  if size_cut is not None:
    rows = rows[0:size_cut]

  columns = [[header] for header in headers]
  for row in rows:
    for column, item in zip(columns, row):
      column.append(item)

  table_str = md.table(columns)
  if size_cut is not None and size_cut < total_size:
    table_str += "\n\n"
    table_str += md.italics(
        f"[Top {size_cut} out of {total_size} results showed]")
  return table_str


T = TypeVar("T")


def _categorize_on_single_metric(
    metrics_map: Dict[str, T],
    metric_func: GetMetricFunc,
    thresholds: Sequence[BenchmarkThreshold],
) -> Tuple[Dict[str, T], Dict[str, T], Dict[str, T], Dict[str, T]]:
  """Categorize the metrics object into regressed, improved, similar, and the
    raw group (the group with no base to compare to).

    Args:
      metrics_map: map of (name, metrics object).
      metric_func: the function returns current and base value of the metric.
      thresholds: list of threshold settings to match for categorizing.
    Returns: 
      A tuple of (regressed, improved, similar, raw) groups.
  """

  regressed_map = {}
  improved_map = {}
  similar_map = {}
  raw_map = {}
  for name, metrics_obj in metrics_map.items():
    current, base = metric_func(metrics_obj)
    if base is None:
      raw_map[name] = metrics_obj
      continue

    similar_threshold = None
    for threshold in thresholds:
      if threshold.regex.match(name):
        similar_threshold = threshold
        break
    if similar_threshold is None:
      raise ValueError(f"No matched threshold setting for: {name}")

    if similar_threshold.unit == ThresholdUnit.PERCENTAGE:
      ratio = abs(current - base) / base * 100
    else:
      ratio = abs(current - base)

    if ratio <= similar_threshold.threshold:
      similar_map[name] = metrics_obj
    elif current > base:
      regressed_map[name] = metrics_obj
    else:
      improved_map[name] = metrics_obj

  return (regressed_map, improved_map, similar_map, raw_map)


def _get_compare_text(current: int, base: Optional[int]) -> str:
  """Generates the text of comparison between current and base value. Returns
    the current value if the base value is None.
  """
  # If base is None, don't need to do compare.
  if base is None:
    return f"{current}"

  ratio = abs(current - base) / base
  direction = "↑" if current > base else ("↓" if current < base else "")
  return f"{current} (vs. {base}, {ratio:.2%}{direction})"


def _sort_benchmarks_and_get_table(benchmarks: Dict[str,
                                                    AggregateBenchmarkLatency],
                                   size_cut: Optional[int] = None) -> str:
  """Sorts all benchmarks according to the improvement/regression ratio and
    returns a markdown table for it.

    Args:
      benchmarks_map: map of (name, benchmark object).
      size_cut: If not None, only show the top N results for each table.
  """
  sorted_rows = []
  for name, benchmark in benchmarks.items():
    current = benchmark.mean_time
    base = benchmark.base_mean_time
    ratio = abs(current - base) / base
    str_mean = _get_compare_text(current, base)
    clickable_name = _make_series_link(name)
    sorted_rows.append((ratio, (clickable_name, str_mean, benchmark.median_time,
                                benchmark.stddev_time)))
  sorted_rows.sort(key=lambda row: row[0], reverse=True)

  return _add_header_and_get_markdown_table(
      headers=BENCHMARK_RESULTS_HEADERS,
      rows=[row[1] for row in sorted_rows],
      size_cut=size_cut)


def categorize_benchmarks_into_tables(benchmarks: Dict[
    str, AggregateBenchmarkLatency],
                                      size_cut: Optional[int] = None) -> str:
  """Splits benchmarks into regressed/improved/similar/raw categories and
    returns their markdown tables.

    Args:
      benchmarks: A dictionary of benchmark names to its aggregate info.
      size_cut: If not None, only show the top N results for each table.
  """
  regressed, improved, similar, raw = _categorize_on_single_metric(
      benchmarks, lambda results: (results.mean_time, results.base_mean_time),
      BENCHMARK_THRESHOLDS)

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
    raw_list = [(_make_series_link(k), v.mean_time, v.median_time,
                 v.stddev_time) for k, v in raw.items()]
    tables.append(
        _add_header_and_get_markdown_table(BENCHMARK_RESULTS_HEADERS,
                                           raw_list,
                                           size_cut=size_cut))
  return "\n\n".join(tables)
