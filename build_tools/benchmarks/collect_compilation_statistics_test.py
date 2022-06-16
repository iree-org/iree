#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import zipfile

from io import BytesIO, StringIO

from common.benchmark_definition import ModuleComponentSizes
from collect_compilation_statistics import CONST_COMPONENT_NAME, VM_COMPONENT_NAME, get_module_component_info, get_module_path, match_module_cmake_target, parse_compilation_time_from_ninja_log


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
    ninja_log = StringIO("# ninja log v5\n"
                         f"0\t100\taaa\tbuild/{target1}\taaa\n"
                         f"130\t200\tbbb\tbuild/{target2}\tbbb\n")

    target_map = parse_compilation_time_from_ninja_log(ninja_log)

    self.assertEqual(target_map, {target1: 100, target2: 70})

  def test_get_module_component_info(self):
    module_file = BytesIO()
    with zipfile.ZipFile(module_file, "w") as zip:
      zip.writestr(VM_COMPONENT_NAME, b"abcd")
      zip.writestr(CONST_COMPONENT_NAME, b"123")
      zip.writestr("main_dispatch_0_vulkan_spirv_fb.fb", b"bindata1")
      zip.writestr("main_dispatch_1_vulkan_spirv_fb.fb", b"bindata2")
    module_file_data = module_file.getvalue()

    component_sizes = get_module_component_info(BytesIO(module_file_data),
                                                len(module_file_data))

    self.assertEqual(
        component_sizes,
        ModuleComponentSizes(file_size=len(module_file_data),
                             vm_component_size=4,
                             const_component_size=3,
                             total_dispatch_component_size=16))

  def test_get_module_component_info_unknown_components(self):
    module_file = BytesIO()
    with zipfile.ZipFile(module_file, "w") as zip:
      zip.writestr(VM_COMPONENT_NAME, b"abcd")
      zip.writestr(CONST_COMPONENT_NAME, b"123")
      zip.writestr("main_dispatch_0_unknown.fb", b"bindata")
    module_file_data = module_file.getvalue()

    self.assertRaises(
        RuntimeError, lambda: get_module_component_info(
            BytesIO(module_file_data), len(module_file_data)))

  def test_get_module_path(self):
    flag_file = StringIO(
        f"--function_inputs=1x2x3xf32\n--module_file=/abcd.vmfb")

    moduel_path = get_module_path(flag_file)

    self.assertEqual(moduel_path, "/abcd-compile-stats.vmfb")


if __name__ == "__main__":
  unittest.main()
