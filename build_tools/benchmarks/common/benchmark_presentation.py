# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generic, List, Optional, Sequence,
                    Tuple, TypeVar, Union)
import pathlib
import dataclasses
import json
import urllib.parse
import markdown_strings as md
import math

from common import benchmark_definition, benchmark_thresholds
from common.benchmark_thresholds import (BENCHMARK_THRESHOLDS,
                                         COMPILATION_TIME_THRESHOLDS,
                                         TOTAL_ARTIFACT_SIZE_THRESHOLDS,
                                         TOTAL_DISPATCH_SIZE_THRESHOLDS,
                                         BenchmarkThreshold, ThresholdUnit)

GetMetricFunc = Callable[[Any], Tuple[int, Optional[int]]]

PERFBOARD_SERIES_PREFIX = "https://perf.iree.dev/serie?IREE?"
BENCHMARK_RESULTS_HEADERS = [
    "Benchmark Name",
    "Average Latency (ms)",
    "Median Latency (ms)",
    "Latency Standard Deviation (ms)",
]
# Since We don't have a structural way to store metric data yet, each metric is
# assigned with a fixed id generated from uuid.uuid4(), to identify the series.
COMPILATION_TIME_METRIC_ID = "e54cd682-c079-4c42-b4ad-d92c4bedea13"
COMPILATION_TIME_SERIES_SUFFIX = "compilation:module:compilation-time"
TOTAL_DISPATCH_SIZE_METRIC_ID = "9e15f7e6-383c-47ec-bd38-ecba55a5f10a"
TOTAL_DISPATCH_SIZE_SERIES_SUFFIX = "compilation:module:component-size:total-dispatch-size"
TOTAL_ARTIFACT_SIZE_METRIC_ID = "2c8a9198-c01c-45b9-a7da-69c82cf749f7"
TOTAL_ARTIFACT_SIZE_SERIES_SUFFIX = "compilation:module:total-artifact-size"
STREAM_IR_DISPATCH_COUNT_METRIC_ID = "7b72cd9e-43ed-4078-b6d3-20b810f9e4ad"
STREAM_IR_DISPATCH_COUNT_SERIES_SUFFIX = "compilation:ir:stream-dispatch-count"


@dataclass
class AggregateBenchmarkLatency:
  """An object for describing aggregate latency numbers for a benchmark."""
  name: str
  benchmark_info: benchmark_definition.BenchmarkInfo
  mean_time: int
  median_time: int
  stddev_time: int
  # The average latency time for the base commit to compare against.
  base_mean_time: Optional[int] = None

  def __str__(self) -> str:
    return self.name


@dataclass(frozen=True)
class CompilationMetrics:
  """An object for describing the summary of statistics and the reference."""
  name: str
  compilation_info: benchmark_definition.CompilationInfo
  compilation_time_ms: int
  total_dispatch_component_bytes: int
  total_artifact_bytes: int
  stream_ir_dispatch_count: int
  base_compilation_time_ms: Optional[int] = None
  base_total_artifact_bytes: Optional[int] = None
  base_total_dispatch_component_bytes: Optional[int] = None
  base_stream_ir_dispatch_count: Optional[int] = None

  def __str__(self) -> str:
    return self.name


T = TypeVar("T")


class MetricsToTableMapper(ABC, Generic[T]):
  """Abstract class to help map benchmark metrics to table.

    It contains a set of methods to help table generator get the required
    information for a metric. For example, extract the current and base metric
    value, the metric thresholds, the table header of the metrics, ...
  """

  @abstractmethod
  def update_base_value(self, obj: T, base_value: Any) -> T:
    """Sets the base value and returns the updated metric object."""
    raise NotImplementedError()

  @abstractmethod
  def get_current_and_base_value(self, obj: T) -> Tuple[int, Optional[int]]:
    """Returns the current and base (can be None) value."""
    raise NotImplementedError()

  def get_series_id(self, benchmark_id: str) -> str:
    """Returns the dashboard series id."""
    # TODO(#11076): Remove legacy path.
    # Whitespace is used in the legacy benchmark id as the delimiter, while not
    # used in the new benchmark id. This is a temporary solution to generate
    # both ids during the migration.
    if " " in benchmark_id:
      return self.get_series_name(benchmark_id)
    return f"{benchmark_id}-{self.get_metric_id()}"

  @abstractmethod
  def get_metric_id(self) -> str:
    """Returns the dashboard series id."""
    raise NotImplementedError()

  @abstractmethod
  def get_series_name(self, name: str) -> str:
    """Returns the dashboard series name."""
    raise NotImplementedError()

  @abstractmethod
  def get_unit(self) -> str:
    """Returns the unit of the metric value."""
    raise NotImplementedError()

  @abstractmethod
  def get_table_header(self) -> str:
    """Returns the header of the table."""
    raise NotImplementedError()

  @staticmethod
  @abstractmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    raise NotImplementedError()

  @staticmethod
  @abstractmethod
  def get_table_title() -> str:
    raise NotImplementedError()


class CompilationTimeToTable(MetricsToTableMapper[CompilationMetrics]):
  """Helper to map CompilationMetrics to compilation time column."""

  def update_base_value(self, compile_metrics: CompilationMetrics,
                        base_value: Any) -> CompilationMetrics:
    return dataclasses.replace(compile_metrics,
                               base_compilation_time_ms=base_value)

  def get_current_and_base_value(
      self, compile_metrics: CompilationMetrics) -> Tuple[int, Optional[int]]:
    return (compile_metrics.compilation_time_ms,
            compile_metrics.base_compilation_time_ms)

  def get_metric_id(self) -> str:
    return COMPILATION_TIME_METRIC_ID

  def get_series_name(self, name: str) -> str:
    return f"{name} [{COMPILATION_TIME_SERIES_SUFFIX}]"

  def get_unit(self) -> str:
    return "ms"

  def get_table_header(self) -> str:
    return f"Compilation Time ({self.get_unit()})"

  @staticmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    return COMPILATION_TIME_THRESHOLDS

  @staticmethod
  def get_table_title() -> str:
    return "Compilation Times"


class TotalDispatchSizeToTable(MetricsToTableMapper[CompilationMetrics]):
  """Helper to map CompilationMetrics to total dispatch size column."""

  def update_base_value(self, compile_metrics: CompilationMetrics,
                        base_value: Any) -> CompilationMetrics:
    return dataclasses.replace(compile_metrics,
                               base_total_dispatch_component_bytes=base_value)

  def get_current_and_base_value(
      self, compile_metrics: CompilationMetrics) -> Tuple[int, Optional[int]]:
    return (compile_metrics.total_dispatch_component_bytes,
            compile_metrics.base_total_dispatch_component_bytes)

  def get_metric_id(self) -> str:
    return TOTAL_DISPATCH_SIZE_METRIC_ID

  def get_series_name(self, name: str) -> str:
    return f"{name} [{TOTAL_DISPATCH_SIZE_SERIES_SUFFIX}]"

  def get_unit(self) -> str:
    return "bytes"

  def get_table_header(self) -> str:
    return f"Total Dispatch Size ({self.get_unit()})"

  @staticmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    return TOTAL_DISPATCH_SIZE_THRESHOLDS

  @staticmethod
  def get_table_title() -> str:
    return "Total Dispatch Sizes"


class TotalArtifactSizeToTable(MetricsToTableMapper[CompilationMetrics]):
  """Helper to map CompilationMetrics to total artifact size column."""

  def update_base_value(self, compile_metrics: CompilationMetrics,
                        base_value: Any) -> CompilationMetrics:
    return dataclasses.replace(compile_metrics,
                               base_total_artifact_bytes=base_value)

  def get_current_and_base_value(
      self, compile_metrics: CompilationMetrics) -> Tuple[int, Optional[int]]:
    return (compile_metrics.total_artifact_bytes,
            compile_metrics.base_total_artifact_bytes)

  def get_metric_id(self) -> str:
    return TOTAL_ARTIFACT_SIZE_METRIC_ID

  def get_series_name(self, name: str) -> str:
    return f"{name} [{TOTAL_ARTIFACT_SIZE_SERIES_SUFFIX}]"

  def get_unit(self) -> str:
    return "bytes"

  def get_table_header(self) -> str:
    return f"Total Artifact Size ({self.get_unit()})"

  @staticmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    return TOTAL_ARTIFACT_SIZE_THRESHOLDS

  @staticmethod
  def get_table_title() -> str:
    return "Total Artifact Sizes"


class StreamIRDispatchCountToTable(MetricsToTableMapper[CompilationMetrics]):
  """Helper to map CompilationMetrics to Stream IR Dispatch Count column."""

  def update_base_value(self, compile_metrics: CompilationMetrics,
                        base_value: Any) -> CompilationMetrics:
    return dataclasses.replace(compile_metrics,
                               base_stream_ir_dispatch_count=base_value)

  def get_current_and_base_value(
      self, compile_metrics: CompilationMetrics) -> Tuple[int, Optional[int]]:
    return (compile_metrics.stream_ir_dispatch_count,
            compile_metrics.base_stream_ir_dispatch_count)

  def get_metric_id(self) -> str:
    return STREAM_IR_DISPATCH_COUNT_METRIC_ID

  def get_series_name(self, name: str) -> str:
    return f"{name} [{STREAM_IR_DISPATCH_COUNT_SERIES_SUFFIX}]"

  def get_unit(self) -> str:
    return "number"

  def get_table_header(self) -> str:
    return f"Stream IR Dispatch Count (# of cmd.dispatch ops)"

  @staticmethod
  def get_metric_thresholds() -> Sequence[BenchmarkThreshold]:
    return benchmark_thresholds.STREAM_IR_DISPATCH_COUNT_THRESHOLDS

  @staticmethod
  def get_table_title() -> str:
    return "Stream IR Dispatch Count (# of cmd.dispatch ops)"


COMPILATION_METRICS_TO_TABLE_MAPPERS: List[
    MetricsToTableMapper[CompilationMetrics]] = [
        CompilationTimeToTable(),
        TotalDispatchSizeToTable(),
        TotalArtifactSizeToTable(),
        StreamIRDispatchCountToTable(),
    ]


def aggregate_all_benchmarks(
    benchmark_files: Sequence[pathlib.Path],
    expected_pr_commit: Optional[str] = None
) -> Dict[str, AggregateBenchmarkLatency]:
  """Aggregates all benchmarks in the given files.

  Args:
  - benchmark_files: A list of JSON files, each can be decoded as a
    BenchmarkResults.
  - expected_pr_commit: An optional Git commit SHA to match against.

  Returns:
  - A dict of benchmark names to AggregateBenchmarkLatency numbers.
  """

  aggregate_results = {}
  benchmark_names = set()
  for benchmark_file in benchmark_files:
    file_results = benchmark_definition.BenchmarkResults.from_json_str(
        benchmark_file.read_text())

    if ((expected_pr_commit is not None) and
        (file_results.commit != expected_pr_commit)):
      raise ValueError("Inconsistent pull request commit")

    for benchmark_index in range(len(file_results.benchmarks)):
      benchmark_run = file_results.benchmarks[benchmark_index]

      series_name = str(benchmark_run.info)
      # Make sure each benchmark has a unique name.
      if series_name in benchmark_names:
        raise ValueError(f"Duplicated benchmark name: {series_name}")
      benchmark_names.add(series_name)

      # TODO(#11076): Remove legacy path.
      series_id = benchmark_run.info.run_config_id
      if series_id is None:
        series_id = series_name

      if series_id in aggregate_results:
        raise ValueError(f"Duplicated benchmark id: {series_id}")

      aggregate_results[series_id] = AggregateBenchmarkLatency(
          name=series_name,
          benchmark_info=benchmark_run.info,
          mean_time=benchmark_run.metrics.real_time.mean,
          median_time=benchmark_run.metrics.real_time.median,
          stddev_time=benchmark_run.metrics.real_time.stddev)

  return aggregate_results


def collect_all_compilation_metrics(
    compile_stats_files: Sequence[pathlib.Path],
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
  target_names = set()
  for compile_stats_file in compile_stats_files:
    with compile_stats_file.open("r") as f:
      file_results = benchmark_definition.CompilationResults.from_json_object(
          json.load(f))

    if ((expected_pr_commit is not None) and
        (file_results.commit != expected_pr_commit)):
      raise ValueError("Inconsistent pull request commit")

    for compile_stats in file_results.compilation_statistics:
      component_sizes = compile_stats.module_component_sizes
      stream_dispatch_count = compile_stats.ir_stats.stream_dispatch_count

      target_name = str(compile_stats.compilation_info)
      if target_name in target_names:
        raise ValueError(f"Duplicated target name: {target_name}")
      target_names.add(target_name)

      target_id = compile_stats.compilation_info.gen_config_id
      # TODO(#11076): Remove legacy path.
      if target_id is None:
        target_id = target_name

      if target_id in compile_metrics:
        raise ValueError(f"Duplicated target id: {target_id}")

      compile_metrics[target_id] = CompilationMetrics(
          name=target_name,
          compilation_info=compile_stats.compilation_info,
          compilation_time_ms=compile_stats.compilation_time_ms,
          total_artifact_bytes=component_sizes.file_bytes,
          total_dispatch_component_bytes=component_sizes.
          total_dispatch_component_bytes,
          stream_ir_dispatch_count=stream_dispatch_count)

  return compile_metrics


def _make_series_link(name: str, series_id: str) -> str:
  """Add link to the given benchmark name.

    Args:
      name: the text to show on the link.
      series_id: the dashboard series id.
  """
  url = PERFBOARD_SERIES_PREFIX + urllib.parse.quote(series_id, safe="()[]@,")
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
    metric_unit: str,
) -> Tuple[Dict[str, T], Dict[str, T], Dict[str, T], Dict[str, T]]:
  """Categorize the metrics object into regressed, improved, similar, and the
    raw group (the group with no base to compare to).

    Args:
      metrics_map: map of (series_id, metrics object).
      metric_func: the function returns current and base value of the metric.
      thresholds: list of threshold settings to match for categorizing.
    Returns:
      A tuple of (regressed, improved, similar, raw) groups.
  """

  regressed_map = {}
  improved_map = {}
  similar_map = {}
  raw_map = {}
  for series_id, metrics_obj in metrics_map.items():
    current, base = metric_func(metrics_obj)
    if base is None:
      raw_map[series_id] = metrics_obj
      continue

    series_name = str(metrics_obj)
    similar_threshold = None
    for threshold in thresholds:
      if threshold.regex.match(series_name):
        similar_threshold = threshold
        break
    if similar_threshold is None:
      raise ValueError(f"No matched threshold setting for: {series_name}")

    if similar_threshold.unit == ThresholdUnit.PERCENTAGE:
      ratio = abs(current - base) / base * 100
    elif similar_threshold.unit.value == metric_unit:
      ratio = abs(current - base)
    else:
      raise ValueError(
          f"Mismatch between metric unit '{metric_unit}' and threshold unit '{similar_threshold.unit.value}'"
      )

    if ratio <= similar_threshold.threshold:
      similar_map[series_id] = metrics_obj
    elif current > base:
      regressed_map[series_id] = metrics_obj
    else:
      improved_map[series_id] = metrics_obj

  return (regressed_map, improved_map, similar_map, raw_map)


def _get_fixed_point_str(value: Union[int, float], digits=3) -> str:
  if isinstance(value, int) or value.is_integer():
    return str(math.floor(value))
  return f"{{:.{digits}f}}".format(value)


def _get_compare_text(current: float, base: Optional[int]) -> str:
  """Generates the text of comparison between current and base value. Returns
    the current value if the base value is None.
  """
  # If base is None, don't need to do compare.
  if base is None:
    return f"{_get_fixed_point_str(current)}"

  ratio = abs(current - base) / base
  direction = "↑" if current > base else ("↓" if current < base else "")
  return f"{_get_fixed_point_str(current)} (vs. {_get_fixed_point_str(base)}, {ratio:.2%}{direction})"


def _sort_benchmarks_and_get_table(benchmarks: Dict[str,
                                                    AggregateBenchmarkLatency],
                                   size_cut: Optional[int] = None) -> str:
  """Sorts all benchmarks according to the improvement/regression ratio and
    returns a markdown table for it.

    Args:
      benchmarks_map: map of (series_id, benchmark object).
      size_cut: If not None, only show the top N results for each table.
  """
  sorted_rows = []
  for series_id, benchmark in benchmarks.items():
    current = benchmark.mean_time / 1e6
    base = benchmark.base_mean_time / 1e6
    ratio = abs(current - base) / base
    str_mean = _get_compare_text(current, base)
    clickable_name = _make_series_link(benchmark.name, series_id)
    sorted_rows.append(
        (ratio, (clickable_name, str_mean,
                 f"{_get_fixed_point_str(benchmark.median_time / 1e6)}",
                 f"{_get_fixed_point_str(benchmark.stddev_time / 1e6)}")))
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
      BENCHMARK_THRESHOLDS, "ns")

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
    raw_list = [(_make_series_link(name=v.name, series_id=k),
                 f"{_get_fixed_point_str(v.mean_time / 1e6)}",
                 f"{_get_fixed_point_str(v.median_time / 1e6)}",
                 f"{_get_fixed_point_str(v.stddev_time / 1e6)}")
                for k, v in raw.items()]
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
      metrics_objs: map of (target_id, CompilationMetrics). All objects must
        contain base value.
      mapper: MetricsToTableMapper for metrics_objs.
      headers: list of table headers.
      size_cut: If not None, only show the top N results for each table.
  """
  sorted_rows = []
  for target_id, metrics_obj in metrics_objs.items():
    current, base = mapper.get_current_and_base_value(metrics_obj)
    if base is None:
      raise AssertionError("Base can't be None for sorting.")
    ratio = abs(current - base) / base
    sorted_rows.append((ratio, (
        _make_series_link(str(metrics_obj), mapper.get_series_id(target_id)),
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
        mapper.get_metric_thresholds(), mapper.get_unit())

    table_title = mapper.get_table_title()
    table_header = mapper.get_table_header()
    if regressed:
      tables.append(md.header(f"Regressed {table_title} 🚩", 3))
      tables.append(
          _sort_metrics_objects_and_get_table(
              metrics_objs=regressed,
              mapper=mapper,
              headers=["Benchmark Name", table_header],
              size_cut=size_cut))
    if improved:
      tables.append(md.header(f"Improved {table_title} 🎉", 3))
      tables.append(
          _sort_metrics_objects_and_get_table(
              metrics_objs=improved,
              mapper=mapper,
              headers=["Benchmark Name", table_header],
              size_cut=size_cut))

  # If we want to abbreviate, similar results won't be interesting.
  if size_cut is None and compile_metrics_map:
    tables.append(md.header("All Compilation Metrics", 3))
    headers = ["Benchmark Name"] + [
        mapper.get_table_header()
        for mapper in COMPILATION_METRICS_TO_TABLE_MAPPERS
    ]
    rows = []
    for target_id, metrics in compile_metrics_map.items():
      row = [metrics.name]
      for mapper in COMPILATION_METRICS_TO_TABLE_MAPPERS:
        current, base = mapper.get_current_and_base_value(metrics)
        row.append(
            _make_series_link(_get_compare_text(current, base),
                              mapper.get_series_id(target_id)))
      rows.append(tuple(row))

    tables.append(
        _add_header_and_get_markdown_table(headers, rows, size_cut=size_cut))

  return "\n\n".join(tables)
