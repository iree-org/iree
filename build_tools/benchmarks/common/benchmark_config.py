# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# All benchmarks' relative path against root build directory.

from argparse import Namespace
from dataclasses import dataclass
from typing import Optional, Union
import pathlib

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

    traced_benchmark_tool_dir: pathlib.Path
    trace_capture_tool: pathlib.Path
    capture_tarball: pathlib.Path
    capture_tmp_dir: pathlib.Path


@dataclass
class BenchmarkConfig:
    """Represents the settings to run benchmarks.

    tmp_dir: per-commit temporary directory.
    root_benchmark_dir: the root directory path/URL containing the built
      benchmark suites.
    benchmark_results_dir: the directory to store benchmark results files.
    git_commit_hash: the git commit hash.
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
    continue_from_previous: skip the benchmarks if their results are found in
      the benchmark_results_dir.
    verify: verify the output if model's expected output is available.
    """

    tmp_dir: pathlib.Path
    root_benchmark_dir: Union[str, pathlib.Path]
    benchmark_results_dir: pathlib.Path
    git_commit_hash: str

    normal_benchmark_tool_dir: Optional[pathlib.Path] = None
    trace_capture_config: Optional[TraceCaptureConfig] = None

    driver_filter: Optional[str] = None
    model_name_filter: Optional[str] = None
    mode_filter: Optional[str] = None
    use_compatible_filter: bool = False

    keep_going: bool = False
    benchmark_min_time: float = 0
    continue_from_previous: bool = False
    verify: bool = False

    @staticmethod
    def build_from_args(args: Namespace, git_commit_hash: str):
        """Build config from command arguments.

        Args:
          args: the command arguments.
          git_commit_hash: the git commit hash of IREE.
        """

        def real_path_or_none(path: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
            return path.resolve() if path else None

        if not args.normal_benchmark_tool_dir and not args.traced_benchmark_tool_dir:
            raise ValueError(
                "At least one of --normal_benchmark_tool_dir or --traced_benchmark_tool_dir should be specified."
            )
        if not (
            (args.traced_benchmark_tool_dir is None)
            == (args.trace_capture_tool is None)
            == (args.capture_tarball is None)
        ):
            raise ValueError(
                "The following 3 flags should be simultaneously all specified or all unspecified: --traced_benchmark_tool_dir, --trace_capture_tool, --capture_tarball"
            )

        per_commit_tmp_dir: pathlib.Path = (args.tmp_dir / git_commit_hash).resolve()

        if args.traced_benchmark_tool_dir is None:
            trace_capture_config = None
        else:
            trace_capture_config = TraceCaptureConfig(
                traced_benchmark_tool_dir=args.traced_benchmark_tool_dir.resolve(),
                trace_capture_tool=args.trace_capture_tool.resolve(),
                capture_tarball=args.capture_tarball.resolve(),
                capture_tmp_dir=per_commit_tmp_dir / CAPTURES_REL_PATH,
            )

        root_benchmark_dir = args.e2e_test_artifacts_dir
        if not str(root_benchmark_dir).startswith("https://"):
            root_benchmark_dir = pathlib.Path(root_benchmark_dir)

        return BenchmarkConfig(
            tmp_dir=per_commit_tmp_dir,
            root_benchmark_dir=root_benchmark_dir,
            benchmark_results_dir=per_commit_tmp_dir / BENCHMARK_RESULTS_REL_PATH,
            git_commit_hash=git_commit_hash,
            normal_benchmark_tool_dir=real_path_or_none(args.normal_benchmark_tool_dir),
            trace_capture_config=trace_capture_config,
            driver_filter=args.driver_filter_regex,
            model_name_filter=args.model_name_regex,
            mode_filter=args.mode_regex,
            use_compatible_filter=args.compatible_only,
            keep_going=args.keep_going,
            benchmark_min_time=args.benchmark_min_time,
            continue_from_previous=args.continue_from_previous,
            verify=args.verify,
        )
