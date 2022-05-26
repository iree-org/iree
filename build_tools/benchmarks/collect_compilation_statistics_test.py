#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tempfile
import unittest

from collect_compilation_statistics import match_module_cmake_target, parse_compilation_time_from_ninja_log


class CollectCompilationStatistics(unittest.TestCase):

  def test_match_module_cmake_target(self):
    target = match_module_cmake_target(
        "iree/iree-build/benchmark_suites/TFLite/vmfb/test.vmfb")

    self.assertEqual(target, "benchmark_suites/TFLite/vmfb/test.vmfb")

  def test_match_module_cmake_target_not_match(self):
    target = match_module_cmake_target("benchmark_suites/TFLite/vmfb/test.mlir")

    self.assertIsNone(target)

  def test_parse_compilation_time_from_ninja_log(self):
    target1 = "benchmark_suites/TFLite/vmfb/deeplabv3.vmfb"
    target2 = "benchmark_suites/TFLite/vmfb/mobilessd.vmfb"
    ninja_log = ("# ninja log v5\n"
                 f"0\t100\t100\tbuild/{target1}\t124\n"
                 f"130\t200\t200\tbuild/{target2}\t224\n")

    with tempfile.NamedTemporaryFile("w") as f:
      f.write(ninja_log)
      f.flush()
      target_map = parse_compilation_time_from_ninja_log(f.name)

    self.assertEqual(target_map, {target1: 100, target2: 70})


if __name__ == "__main__":
  unittest.main()
