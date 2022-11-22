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
from typing import List, Sequence


def build_common_argument_parser():
  """Returns an argument parser with common options."""

  def check_dir_path(path):
    path = pathlib.Path(path)
    if path.is_dir():
      return path
    else:
      raise argparse.ArgumentTypeError(path)

  def check_exe_path(path):
    path = pathlib.Path(path)
    if os.access(path, os.X_OK):
      return path
    else:
      raise argparse.ArgumentTypeError(f"'{path}' is not an executable")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites")
  parser.add_argument(
      "--normal_benchmark_tool_dir",
      "--normal-benchmark-tool-dir",
      type=check_dir_path,
      default=None,
      help="Path to the normal (non-tracing) iree tool directory")
  parser.add_argument("--traced_benchmark_tool_dir",
                      "--traced-benchmark-tool-dir",
                      type=check_dir_path,
                      default=None,
                      help="Path to the tracing-enabled iree tool directory")
  parser.add_argument("--trace_capture_tool",
                      "--trace-capture-tool",
                      type=check_exe_path,
                      default=None,
                      help="Path to the tool for collecting captured traces")
  parser.add_argument(
      "--driver-filter-regex",
      "--driver_filter_regex",
      type=str,
      default=None,
      help="Only run benchmarks matching the given driver regex")
  parser.add_argument(
      "--model-name-regex",
      "--model_name_regex",
      type=str,
      default=None,
      help="Only run benchmarks matching the given model name regex")
  parser.add_argument(
      "--mode-regex",
      "--mode_regex",
      type=str,
      default=None,
      help="Only run benchmarks matching the given benchmarking mode regex")
  parser.add_argument("--output",
                      "-o",
                      default=None,
                      type=pathlib.Path,
                      help="Path to the output file")
  parser.add_argument("--capture_tarball",
                      "--capture-tarball",
                      default=None,
                      type=pathlib.Path,
                      help="Path to the tarball for captures")
  parser.add_argument("--no-clean",
                      action="store_true",
                      help="Do not clean up the temporary directory used for "
                      "benchmarking on the Android device")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  parser.add_argument(
      "--pin-cpu-freq",
      "--pin_cpu_freq",
      action="store_true",
      help="Pin CPU frequency for all cores to the maximum. Requires root")
  parser.add_argument("--pin-gpu-freq",
                      "--pin_gpu_freq",
                      action="store_true",
                      help="Pin GPU frequency to the maximum. Requires root")
  parser.add_argument(
      "--keep_going",
      "--keep-going",
      action="store_true",
      help="Continue running after a failed benchmark. The overall exit status"
      " will still indicate failure and all errors will be reported at the end."
  )
  parser.add_argument(
      "--tmp_dir",
      "--tmp-dir",
      "--tmpdir",
      default=pathlib.Path("/tmp/iree-benchmarks"),
      type=check_dir_path,
      help="Base directory in which to store temporary files. A subdirectory"
      " with a name matching the git commit hash will be created.")
  parser.add_argument(
      "--continue_from_directory",
      "--continue-from-directory",
      default=None,
      type=check_dir_path,
      help="Path to directory with previous benchmark temporary files. This"
      " should be for the specific commit (not the general tmp-dir). Previous"
      " benchmark and capture results from here will not be rerun and will be"
      " combined with the new runs.")
  parser.add_argument(
      "--benchmark_min_time",
      "--benchmark-min-time",
      default=0,
      type=float,
      help="If specified, this will be passed as --benchmark_min_time to the"
      "iree-benchmark-module (minimum number of seconds to repeat running "
      "for). In that case, no --benchmark_repetitions flag will be passed."
      " If not specified, a --benchmark_repetitions will be passed "
      "instead.")

  return parser


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
