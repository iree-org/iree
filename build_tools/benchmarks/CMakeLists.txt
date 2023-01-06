# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(BENCHMARKS_TOOL_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# benchmark_tool_py_test()
#
# CMake function to test benchmark python tools.
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
function(benchmark_tool_py_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    "ARGS;LABELS"
    ${ARGN}
  )

  iree_build_tools_py_test(
    NAME
      "${_RULE_NAME}"
    SRC
      "${_RULE_SRC}"
    ARGS
      ${_RULE_ARGS}
    LABELS
      ${_RULE_LABELS}
    PACKAGE_DIRS
      ${BENCHMARKS_TOOL_PYTHON_DIR}
  )
endfunction()

add_subdirectory(common)

benchmark_tool_py_test(
  NAME
    collect_compilation_statistics_test
  SRC
    "collect_compilation_statistics_test.py"
)

benchmark_tool_py_test(
  NAME
    export_benchmark_config_test
  SRC
    "export_benchmark_config_test.py"
)

benchmark_tool_py_test(
  NAME
    post_benchmark_comment_test
  SRC
    "post_benchmark_comment_test.py"
)
