# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# All benchmarks' relative path against root build directory.

import os

from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

BENCHMARK_SUITE_REL_PATH = "benchmark_suites"
BENCHMARK_RESULTS_REL_PATH = "benchmark-results"
CAPTURES_REL_PATH = "captures"


@dataclass
class TraceCaptureConfig:
  """Represents the settings for capturing traces during benchamrking.

    traced_benchmark_tool_dir: the path to the tracing-enabled benchmark tool
      directory.
    trace_capture_tool: the path to the tool for collecting captured traces.
    capture_tarball: the path of capture tar archive.
    capture_tmp_dir: the temporary directory to store captured traces.
  """

  traced_benchmark_tool_dir: str
  trace_capture_tool: str
  capture_tarball: str
  capture_tmp_dir: str


@dataclass
class BenchmarkConfig:
  """Represents the settings to run benchmarks.

    root_benchmark_dir: the root directory containing the built benchmark
      suites.
    benchmark_results_dir: the directory to store benchmark results files.
    normal_benchmark_tool_dir: the path to the non-traced benchmark tool
      directory.
    trace_capture_config: the config for capturing traces. Set if and only if
      the traces need to be captured.
    driver_filter: filter benchmarks to those whose driver matches this regex
      (or all if this is None).
    model_name_filter: filter benchmarks to those whose model name matches this
      regex (or all if this is None).
    mode_filter: filter benchmarks to those whose benchmarking mode matches this
      regex (or all if this is None).
    keep_going: whether to proceed if an individual run fails. Exceptions will
      logged and returned.
    benchmark_min_time: min number of seconds to run the benchmark for, if
      specified. Otherwise, the benchmark will be repeated a fixed number of
      times.
  """

  root_benchmark_dir: str
  benchmark_results_dir: str

  normal_benchmark_tool_dir: Optional[str] = None
  trace_capture_config: Optional[TraceCaptureConfig] = None

  driver_filter: Optional[str] = None
  model_name_filter: Optional[str] = None
  mode_filter: Optional[str] = None

  keep_going: bool = False
  benchmark_min_time: float = 0

  @staticmethod
  def build_from_args(args: Namespace, git_commit_hash: str):
    """Build config from command arguments.
    
    Args:
      args: the command arguments.
      git_commit_hash: the commit hash to separete different benchmark runs in
        the temporary directory.
    """

    def real_path_or_none(path: str) -> Optional[str]:
      return os.path.realpath(path) if path else None

    if not args.normal_benchmark_tool_dir and not args.traced_benchmark_tool_dir:
      raise ValueError(
          "At least one of --normal_benchmark_tool_dir or --traced_benchmark_tool_dir should be specified."
      )
    if not ((args.traced_benchmark_tool_dir is None) ==
            (args.trace_capture_tool is None) ==
            (args.capture_tarball is None)):
      raise ValueError(
          "The following 3 flags should be simultaneously all specified or all unspecified: --traced_benchmark_tool_dir, --trace_capture_tool, --capture_tarball"
      )

    per_commit_tmp_dir = os.path.realpath(
        os.path.join(args.tmp_dir, git_commit_hash))

    if args.traced_benchmark_tool_dir is None:
      trace_capture_config = None
    else:
      trace_capture_config = TraceCaptureConfig(
          traced_benchmark_tool_dir=real_path_or_none(
              args.traced_benchmark_tool_dir),
          trace_capture_tool=real_path_or_none(args.trace_capture_tool),
          capture_tarball=real_path_or_none(args.capture_tarball),
          capture_tmp_dir=os.path.join(per_commit_tmp_dir, CAPTURES_REL_PATH))

    build_dir = os.path.realpath(args.build_dir)
    return BenchmarkConfig(
        root_benchmark_dir=os.path.join(build_dir, BENCHMARK_SUITE_REL_PATH),
        benchmark_results_dir=os.path.join(per_commit_tmp_dir,
                                           BENCHMARK_RESULTS_REL_PATH),
        normal_benchmark_tool_dir=real_path_or_none(
            args.normal_benchmark_tool_dir),
        trace_capture_config=trace_capture_config,
        driver_filter=args.driver_filter_regex,
        model_name_filter=args.model_name_regex,
        mode_filter=args.mode_regex,
        keep_going=args.keep_going,
        benchmark_min_time=args.benchmark_min_time)
