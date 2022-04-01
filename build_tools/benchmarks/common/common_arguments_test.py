#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from common.common_arguments import build_common_argument_parser


class CommonArgumentsTest(unittest.TestCase):

  def test_build_common_argument_parser(self):
    arg_parser = build_common_argument_parser()
    arg_parser.parse_args([
        "--normal_benchmark_tool_dir=/tmp", "--traced_benchmark_tool_dir=/tmp",
        "--trace_capture_tool=/bin/ls", "."
    ])

  def test_build_common_argument_parser_check_build_dir(self):
    arg_parser = build_common_argument_parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["nonexistent"])

  def test_build_common_argument_parser_check_normal_benchmark_tool(self):
    arg_parser = build_common_argument_parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--normal_benchmark_tool_dir=nonexistent", "."])

  def test_build_common_argument_parser_check_traced_benchmark_tool(self):
    arg_parser = build_common_argument_parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--traced_benchmark_tool_dir=nonexistent", "."])

  def test_build_common_argument_parser_check_trace_capture_tool(self):
    arg_parser = build_common_argument_parser()
    with self.assertRaises(SystemExit):
      arg_parser.parse_args(["--trace_capture_tool=nonexistent", "."])


if __name__ == "__main__":
  unittest.main()
