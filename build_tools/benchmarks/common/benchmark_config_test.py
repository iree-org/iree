#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import stat
import unittest
import tempfile
import os

from common.common_arguments import build_common_argument_parser
from common.benchmark_config import BenchmarkConfig, TraceCaptureConfig


class BenchmarkConfigTest(unittest.TestCase):

  def setUp(self):
    self.build_dir = tempfile.TemporaryDirectory()
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.normal_tool_dir = os.path.join(self.build_dir.name, "normal_tool")
    os.mkdir(self.normal_tool_dir)
    self.traced_tool_dir = os.path.join(self.build_dir.name, "traced_tool")
    os.mkdir(self.traced_tool_dir)
    self.trace_capture_tool = tempfile.NamedTemporaryFile()
    os.chmod(self.trace_capture_tool.name, stat.S_IEXEC)

  def tearDown(self):
    self.tmp_dir.cleanup()
    self.build_dir.cleanup()

  def test_build_from_args(self):
    args = build_common_argument_parser().parse_args([
        f"--tmp_dir={self.tmp_dir.name}",
        f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
        f"--traced_benchmark_tool_dir={self.traced_tool_dir}",
        f"--trace_capture_tool={self.trace_capture_tool.name}",
        f"--capture_tarball=capture.tar", f"--driver_filter_regex=a",
        f"--model_name_regex=b", f"--mode_regex=c", f"--keep_going",
        f"--benchmark_min_time=10", self.build_dir.name
    ])

    config = BenchmarkConfig.build_from_args(args=args, git_commit_hash="abcd")

    per_commit_tmp_dir = os.path.join(self.tmp_dir.name, "abcd")
    expected_trace_capture_config = TraceCaptureConfig(
        traced_benchmark_tool_dir=self.traced_tool_dir,
        trace_capture_tool=self.trace_capture_tool.name,
        capture_tarball=os.path.realpath("capture.tar"),
        capture_tmp_dir=os.path.join(per_commit_tmp_dir, "captures"))
    self.assertEqual(
        config,
        BenchmarkConfig(root_benchmark_dir=os.path.join(self.build_dir.name,
                                                        "benchmark_suites"),
                        benchmark_results_dir=os.path.join(
                            per_commit_tmp_dir, "benchmark-results"),
                        git_commit_hash="abcd",
                        normal_benchmark_tool_dir=self.normal_tool_dir,
                        trace_capture_config=expected_trace_capture_config,
                        driver_filter="a",
                        model_name_filter="b",
                        mode_filter="c",
                        keep_going=True,
                        benchmark_min_time=10))

  def test_build_from_args_benchmark_only(self):
    args = build_common_argument_parser().parse_args([
        f"--tmp_dir={self.tmp_dir.name}",
        f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
        self.build_dir.name
    ])

    config = BenchmarkConfig.build_from_args(args=args, git_commit_hash="abcd")

    self.assertIsNone(config.trace_capture_config)

  def test_build_from_args_invalid_capture_args(self):
    args = build_common_argument_parser().parse_args([
        f"--tmp_dir={self.tmp_dir.name}",
        f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
        f"--traced_benchmark_tool_dir={self.traced_tool_dir}",
        self.build_dir.name
    ])

    self.assertRaises(
        ValueError,
        lambda: BenchmarkConfig.build_from_args(args=args,
                                                git_commit_hash="abcd"))


if __name__ == "__main__":
  unittest.main()
