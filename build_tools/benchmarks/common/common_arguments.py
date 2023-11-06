#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import os
import argparse
import pathlib
from typing import List, Optional, Sequence


def _check_dir_path(path):
    path = pathlib.Path(path)
    if path.is_dir():
        return path
    else:
        raise argparse.ArgumentTypeError(path)


def _check_file_path(path):
    path = pathlib.Path(path)
    if path.is_file():
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not found")


def _check_exe_path(path):
    path = pathlib.Path(path)
    if os.access(path, os.X_OK):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not an executable")


class Parser(argparse.ArgumentParser):
    """Argument parser that includes common arguments and does validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument(
            "--e2e_test_artifacts_dir",
            metavar="<e2e-test-artifacts-dir>",
            type=str,
            required=True,
            help="Path to the IREE e2e test artifacts directory.",
        )

        self.add_argument(
            "--normal_benchmark_tool_dir",
            "--normal-benchmark-tool-dir",
            type=_check_dir_path,
            default=None,
            help="Path to the normal (non-tracing) iree tool directory",
        )
        self.add_argument(
            "--traced_benchmark_tool_dir",
            "--traced-benchmark-tool-dir",
            type=_check_dir_path,
            default=None,
            help="Path to the tracing-enabled iree tool directory",
        )
        self.add_argument(
            "--trace_capture_tool",
            "--trace-capture-tool",
            type=_check_exe_path,
            default=None,
            help="Path to the tool for collecting captured traces",
        )
        self.add_argument(
            "--driver-filter-regex",
            "--driver_filter_regex",
            type=str,
            default=None,
            help="Only run benchmarks matching the given driver regex",
        )
        self.add_argument(
            "--model-name-regex",
            "--model_name_regex",
            type=str,
            default=None,
            help="Only run benchmarks matching the given model name regex",
        )
        self.add_argument(
            "--mode-regex",
            "--mode_regex",
            type=str,
            default=None,
            help="Only run benchmarks matching the given benchmarking mode regex",
        )
        self.add_argument(
            "--output",
            "-o",
            default=None,
            type=pathlib.Path,
            help="Path to the output file",
        )
        self.add_argument(
            "--capture_tarball",
            "--capture-tarball",
            default=None,
            type=pathlib.Path,
            help="Path to the tarball for captures",
        )
        self.add_argument(
            "--no-clean",
            action="store_true",
            help="Do not clean up the temporary directory used for "
            "benchmarking on the Android device",
        )
        self.add_argument(
            "--verbose",
            action="store_true",
            help="Print internal information during execution",
        )
        self.add_argument(
            "--pin-cpu-freq",
            "--pin_cpu_freq",
            action="store_true",
            help="Pin CPU frequency for all cores to the maximum. Requires root",
        )
        self.add_argument(
            "--pin-gpu-freq",
            "--pin_gpu_freq",
            action="store_true",
            help="Pin GPU frequency to the maximum. Requires root",
        )
        self.add_argument(
            "--keep_going",
            "--keep-going",
            action="store_true",
            help="Continue running after a failed benchmark. The overall exit status"
            " will still indicate failure and all errors will be reported at the end.",
        )
        self.add_argument(
            "--tmp_dir",
            "--tmp-dir",
            "--tmpdir",
            default=pathlib.Path("/tmp/iree-benchmarks"),
            type=_check_dir_path,
            help="Base directory in which to store temporary files. A subdirectory"
            " with a name matching the git commit hash will be created.",
        )
        self.add_argument(
            "--continue_from_previous",
            "--continue-from-previous",
            action="store_true",
            help="Previous benchmark and capture results will be used and not "
            "rerun if they are found in the benchmark results directory.",
        )
        self.add_argument(
            "--benchmark_min_time",
            "--benchmark-min-time",
            default=0,
            type=float,
            help="If specified, this will be passed as --benchmark_min_time to the"
            "iree-benchmark-module (minimum number of seconds to repeat running "
            "for). In that case, no --benchmark_repetitions flag will be passed."
            " If not specified, a --benchmark_repetitions will be passed "
            "instead.",
        )
        self.add_argument(
            "--compatible_only",
            "--compatible-only",
            action="store_true",
            help="Only run compatible benchmarks based on the detected device "
            "information",
        )
        self.add_argument(
            "--verify",
            action="store_true",
            help="Verify the output when the expected output is available",
        )
        self.add_argument(
            "--execution_benchmark_config",
            type=_check_file_path,
            required=True,
            help="JSON config for the execution benchmarks",
        )
        self.add_argument(
            "--target_device_name",
            type=str,
            required=True,
            help="Target device in benchmark config to run",
        )
        self.add_argument(
            "--shard_index",
            type=int,
            default=None,
            help="Shard in benchmark config to run",
        )


def expand_and_check_file_paths(paths: Sequence[str]) -> List[pathlib.Path]:
    """Expands the wildcards in the paths and check if they are files.
    Returns:
      List of expanded paths.
    """

    expanded_paths = []
    for path in paths:
        expanded_paths += [pathlib.Path(path) for path in glob.glob(path)]

    for path in expanded_paths:
        if not path.is_file():
            raise ValueError(f"{path} is not a file.")

    return expanded_paths
