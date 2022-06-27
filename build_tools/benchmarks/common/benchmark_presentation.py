# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
import json
import urllib.parse
import markdown_strings as md

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from common.benchmark_definition import BenchmarkResults, CompilationInfo, CompilationResults
from common.benchmark_thresholds import BENCHMARK_THRESHOLDS, COMPILATION_TIME_THRESHOLDS, TOTAL_DISPATCH_SIZE_THRESHOLDS, BenchmarkThreshold, ThresholdUnit

GetMetricFunc = Callable[[Any], Tuple[int, Optional[int]]]
GetTableRowFunc = Callable[[str, Any], Tuple]

PERFBOARD_SERIES_PREFIX = "https://perf.iree.dev/serie?IREE?"
BENCHMARK_RESULTS_HEADERS = [
    "Benchmark Name",
    "Average Latency (ms)",
    "Median Latency (ms)",
    "Latency Standard Deviation (ms)",
]
COMPILATION_TIME_SERIES_SUFFIX = "compilation:module:compilation-time"
TOTAL_DISPATCH_SIZE_SERIES_SUFFIX = "compilation:module:component-size:total-dispatch-size"


@dataclass
class AggregateBenchmarkLatency:
  """An object for describing aggregate latency numbers for a benchmark."""
  mean_time: int
  median_time: int
  stddev_time: int
  # The average latency time for the base commit to compare against.
  base_mean_time: Optional[int] = None


@dataclass
class CompilationMetrics:
  """An object for describing the summary of statistics and the reference."""
  compilation_info: CompilationInfo
  compilation_time: int
  total_dispatch_component_size: int
  base_compilation_time: Optional[int] = None
  base_total_dispatch_component_size: Optional[int] = None


T = TypeVar("T")


class MetricsToTableMapper(ABC, Generic[T]):
  """Abstract class to help map benchmark metrics to table.
  
    It contains a set of methods to help table generator get the required
    information for a metric. For example, extract the current and base metric
    value, the metric thresholds, the table header of the metrics, ...
  """

  @abstractmethod
  def get_current_and_base_value(self, obj: T) -> Tuple[int, Optional[int]]:
    """Returns the current and base (can be None) value."""
    raise NotImplementedError()

  @abstractmethod
  def get_series_name(self, name: str) -> str:
    """Returns the dashboard series name."""
    raise NotImplementedError()

  @staticmethod
  @abstractmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    raise NotImplementedError()

  @staticmethod
  @abstractmethod
  def get_table_title() -> str:
    raise NotImplementedError()

  @staticmethod
  @abstractmethod
  def get_table_header() -> str:
    raise NotImplementedError()


class CompilationTimeToTable(MetricsToTableMapper[CompilationMetrics]):
  """Helper to map CompilationMetrics to compilation time column."""

  def get_current_and_base_value(
      self, compile_metrics: CompilationMetrics) -> Tuple[int, Optional[int]]:
    return (compile_metrics.compilation_time,
            compile_metrics.base_compilation_time)

  def get_series_name(self, name: str) -> str:
    return f"{name} [{COMPILATION_TIME_SERIES_SUFFIX}]"

  @staticmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    return COMPILATION_TIME_THRESHOLDS

  @staticmethod
  def get_table_title() -> str:
    return "Compilation Times"

  @staticmethod
  def get_table_header() -> str:
    return "Compilation Time (ms)"


class TotalDispatchSizeToTable(MetricsToTableMapper[CompilationMetrics]):
  """Helper to map CompilationMetrics to total dispatch size column."""

  def get_current_and_base_value(
      self, compile_metrics: CompilationMetrics) -> Tuple[int, Optional[int]]:
    return (compile_metrics.total_dispatch_component_size,
            compile_metrics.base_total_dispatch_component_size)

  def get_series_name(self, name: str) -> str:
    return f"{name} [{TOTAL_DISPATCH_SIZE_SERIES_SUFFIX}]"

  @staticmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    return TOTAL_DISPATCH_SIZE_THRESHOLDS

  @staticmethod
  def get_table_title() -> str:
    return "Total Dispatch Sizes"

  @staticmethod
  def get_table_header() -> str:
    return "Total Dispatch Size (bytes)"


COMPILATION_METRICS_TO_TABLE_MAPPERS: List[
    MetricsToTableMapper[CompilationMetrics]] = [
        CompilationTimeToTable(),
        TotalDispatchSizeToTable(),
    ]


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

    if ((expected_pr_commit is not None) and
        (file_results.commit != expected_pr_commit)):
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


def collect_all_compilation_metrics(
    compile_stats_files: Sequence[str],
    expected_pr_commit: Optional[str] = None) -> Dict[str, CompilationMetrics]:
  """Collects all compilation statistics in the given files.

    Args:
      compile_stats_files: A list of JSON files, each can be decoded as a
        CompilationResults.
      expected_pr_commit: An optional Git commit SHA to match against.

    Returns:
      A dict of benchmark names to CompilationMetrics.
  """
  compile_metrics = {}

  for compile_stats_file in compile_stats_files:
    with open(compile_stats_file) as f:
      file_results = CompilationResults.from_json_object(json.load(f))

    if ((expected_pr_commit is not None) and
        (file_results.commit != expected_pr_commit)):
      raise ValueError("Inconsistent pull request commit")

    for compile_stats in file_results.compilation_statistics:
      component_sizes = compile_stats.module_component_sizes
      name = str(compile_stats.compilation_info)
      compile_metrics[name] = CompilationMetrics(
          compilation_info=compile_stats.compilation_info,
          compilation_time=compile_stats.compilation_time,
          total_dispatch_component_size=component_sizes.
          total_dispatch_component_size)

  return compile_metrics


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

    If size_cut is None, the table includes regressed/improved/similar/raw
    categories; otherwise, the table includes regressed/improved/raw categories.

    Args:
      benchmarks: A dictionary of benchmark names to its aggregate info.
      size_cut: If not None, only show the top N results for each table.
  """
  regressed, improved, similar, raw = _categorize_on_single_metric(
      benchmarks, lambda results: (results.mean_time, results.base_mean_time),
      BENCHMARK_THRESHOLDS)

  tables = []
  if regressed:
    tables.append(md.header("Regressed Latencies 🚩", 3))
    tables.append(_sort_benchmarks_and_get_table(regressed, size_cut))
  if improved:
    tables.append(md.header("Improved Latencies 🎉", 3))
    tables.append(_sort_benchmarks_and_get_table(improved, size_cut))
  # If we want to abbreviate, similar results won't be interesting.
  if similar and size_cut is None:
    tables.append(md.header("Similar Latencies", 3))
    tables.append(_sort_benchmarks_and_get_table(similar, size_cut))
  if raw:
    tables.append(md.header("Raw Latencies", 3))
    raw_list = [(_make_series_link(k), v.mean_time, v.median_time,
                 v.stddev_time) for k, v in raw.items()]
    tables.append(
        _add_header_and_get_markdown_table(BENCHMARK_RESULTS_HEADERS,
                                           raw_list,
                                           size_cut=size_cut))
  return "\n\n".join(tables)


def _sort_metrics_objects_and_get_table(metrics_objs: Dict[str, T],
                                        mapper: MetricsToTableMapper[T],
                                        headers: Sequence[str],
                                        size_cut: Optional[int] = None) -> str:
  """Sorts all metrics objects according to the improvement/regression ratio and
    returns a markdown table for it.

    Args:
      metrics_objs: map of (name, metrics object). All objects must contain base
        value.
      mapper: MetricsToTableMapper for metrics_objs.
      headers: list of table headers.
      size_cut: If not None, only show the top N results for each table.
  """
  sorted_rows = []
  for name, metrics_obj in metrics_objs.items():
    current, base = mapper.get_current_and_base_value(metrics_obj)
    if base is None:
      raise AssertionError("Base can't be None for sorting.")
    ratio = abs(current - base) / base
    sorted_rows.append((ratio, (
        _make_series_link(name, mapper.get_series_name(name)),
        _get_compare_text(current, base),
    )))
  sorted_rows.sort(key=lambda row: row[0], reverse=True)

  return _add_header_and_get_markdown_table(
      headers=headers, rows=[row[1] for row in sorted_rows], size_cut=size_cut)


def categorize_compilation_metrics_into_tables(
    compile_metrics_map: Dict[str, CompilationMetrics],
    size_cut: Optional[int] = None) -> str:
  """Splits compilation metrics into regressed/improved/all categories
    and returns their markdown tables.

    If size_cut is None, the table includes regressed/improved/all categories;
    otherwise, the table includes regressed/improved categories.

    Args:
      compile_metrics_map: A dictionary of benchmark names to its compilation
        metrics.
      size_cut: If not None, only show the top N results for each table.
  """

  tables = []
  for mapper in COMPILATION_METRICS_TO_TABLE_MAPPERS:
    regressed, improved, _, _ = _categorize_on_single_metric(
        compile_metrics_map, mapper.get_current_and_base_value,
        mapper.get_metric_thresholds())

    table_title = mapper.get_table_title()
    table_header = mapper.get_table_header()
    if regressed:
      tables.append(md.header(f"Regressed {table_title} 🚩", 3))
      tables.append(
          _sort_metrics_objects_and_get_table(regressed, mapper,
                                              ["Benchmark Name", table_header],
                                              size_cut))
    if improved:
      tables.append(md.header(f"Improved {table_title} 🎉", 3))
      tables.append(
          _sort_metrics_objects_and_get_table(improved, mapper,
                                              ["Benchmark Name", table_header],
                                              size_cut))

  # If we want to abbreviate, similar results won't be interesting.
  if size_cut is None and compile_metrics_map:
    tables.append(md.header("All Compilation Metrics", 3))
    headers = ["Benchmark Name"] + [
        mapper.get_table_header()
        for mapper in COMPILATION_METRICS_TO_TABLE_MAPPERS
    ]
    rows = []
    for name, metrics in compile_metrics_map.items():
      row = [name]
      for mapper in COMPILATION_METRICS_TO_TABLE_MAPPERS:
        current, base = mapper.get_current_and_base_value(metrics)
        row.append(
            _make_series_link(_get_compare_text(current, base),
                              mapper.get_series_name(name)))
      rows.append(tuple(row))

    tables.append(
        _add_header_and_get_markdown_table(headers, rows, size_cut=size_cut))

  return "\n\n".join(tables)
