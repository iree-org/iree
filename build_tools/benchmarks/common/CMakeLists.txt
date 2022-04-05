# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
# Tests
################################################################################

if(IREE_BUILD_TESTS AND NOT ANDROID)

iree_py_test(
  NAME
    linux_device_utils_test
  SRCS
    "linux_device_utils_test.py"
)

iree_py_test(
  NAME
    common_arguments_test
  SRCS
    "common_arguments_test.py"
)

iree_py_test(
  NAME
    benchmark_config_test
  SRCS
    "benchmark_config_test.py"
)

# TODO(#8708): Temporary solution to fix python path for tests.
set_property(TEST "build_tools/benchmarks/common/linux_device_utils_test"
    APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${BENCHMARKS_TOOL_PYTHON_DIR}:$ENV{PYTHONPATH}")
set_property(TEST "build_tools/benchmarks/common/common_arguments_test"
  APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${BENCHMARKS_TOOL_PYTHON_DIR}:$ENV{PYTHONPATH}")
set_property(TEST "build_tools/benchmarks/common/benchmark_config_test"
  APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${BENCHMARKS_TOOL_PYTHON_DIR}:$ENV{PYTHONPATH}")

endif()
