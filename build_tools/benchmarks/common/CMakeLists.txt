# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
# Tests
################################################################################

benchmark_tool_py_test(
  NAME
    linux_device_utils_test
  SRC
    "linux_device_utils_test.py"
)

benchmark_tool_py_test(
  NAME
    common_arguments_test
  SRC
    "common_arguments_test.py"
)

benchmark_tool_py_test(
  NAME
    benchmark_config_test
  SRC
    "benchmark_config_test.py"
)

benchmark_tool_py_test(
  NAME
    benchmark_suite_test
  SRC
    "benchmark_suite_test.py"
)

benchmark_tool_py_test(
  NAME
    benchmark_driver_test
  SRC
    "benchmark_driver_test.py"
)
