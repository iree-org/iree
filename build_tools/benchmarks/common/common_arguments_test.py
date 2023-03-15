#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import shutil
import tempfile
import unittest

from common import benchmark_config, common_arguments


class CommonArgumentsTest(unittest.TestCase):

  def test_parser(self):
    with tempfile.TemporaryDirectory() as tempdir:
      common_arguments.Parser().parse_args([
          "--normal_benchmark_tool_dir=" + tempdir,
          "--traced_benchmark_tool_dir=" + tempdir,
          "--trace_capture_tool=" + shutil.which("ls"), "."
      ])

  def test_parser_check_build_dir(self):
    arg_parser = common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["nonexistent"])

  def test_parser_check_normal_benchmark_tool(self):
    arg_parser = common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--normal_benchmark_tool_dir=nonexistent", "."])

  def test_parser_check_traced_benchmark_tool(self):
    arg_parser = common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--traced_benchmark_tool_dir=nonexistent", "."])

  def test_parser_check_trace_capture_tool(self):
    arg_parser = common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--trace_capture_tool=nonexistent", "."])

  def test_parser_incomplete_new_benchmark_suites_args(self):
    arg_parser = common_arguments.Parser()

    with self.assertRaises(SystemExit):
      arg_parser.parse_args([f"--e2e_test_artifacts"])
    with self.assertRaises(SystemExit):
      arg_parser.parse_args([f"--target_device_name"])
    with self.assertRaises(SystemExit):
      arg_parser.parse_args([f"--execution_benchmark_config"])

  def test_parser_default_execution_benchmark_config(self):
    arg_parser = common_arguments.Parser()

    with tempfile.TemporaryDirectory() as tempdir:
      args = arg_parser.parse_args(
          [f"--e2e_test_artifacts_dir={tempdir}", "--target_device_name=test"])

      self.assertEqual(
          args.execution_benchmark_config,
          pathlib.Path(tempdir) /
          benchmark_config.DEFAULT_EXECUTION_BENCHMARK_CONFIG)


if __name__ == "__main__":
  unittest.main()
