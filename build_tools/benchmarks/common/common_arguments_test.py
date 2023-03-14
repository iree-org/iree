#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import shutil
import tempfile

import common.common_arguments


class CommonArgumentsTest(unittest.TestCase):

  def test_parser(self):
    with tempfile.TemporaryDirectory() as tempdir:
      common.common_arguments.Parser().parse_args([
          "--normal_benchmark_tool_dir=" + tempdir,
          "--traced_benchmark_tool_dir=" + tempdir,
          "--trace_capture_tool=" + shutil.which("ls"), "."
      ])

  def test_parser_check_build_dir(self):
    arg_parser = common.common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["nonexistent"])

  def test_parser_check_normal_benchmark_tool(self):
    arg_parser = common.common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--normal_benchmark_tool_dir=nonexistent", "."])

  def test_parser_check_traced_benchmark_tool(self):
    arg_parser = common.common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--traced_benchmark_tool_dir=nonexistent", "."])

  def test_parser_check_trace_capture_tool(self):
    arg_parser = common.common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--trace_capture_tool=nonexistent", "."])

  def test_parser_e2e_test_artifacts_dir_needs_execution_benchmark_config(self):
    arg_parser = common.common_arguments.Parser()
    with tempfile.TemporaryDirectory() as tempdir:
      with self.assertRaises(SystemExit):
        arg_parser.parse_args([f"--e2e_test_artifacts_dir={tempdir}"])

  def test_parser_only_execution_benchmark_config_or_target_device_name(self):
    arg_parser = common.common_arguments.Parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args([f"--execution_benchmark_config"])
    with self.assertRaises(SystemExit):
      arg_parser.parse_args([f"--target_device_name"])


if __name__ == "__main__":
  unittest.main()
