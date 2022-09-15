# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(BUILD_TOOLS_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# build_tools_py_module_test()
#
# CMake function to test build_tools python modules.
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
function(build_tools_py_module_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    "ARGS;LABELS"
    ${ARGN}
  )

  iree_local_py_test(
    NAME
      "${_RULE_NAME}"
    SRC
      "${_RULE_SRC}"
    ARGS
      ${_RULE_ARGS}
    LABELS
      ${_RULE_LABELS}
    PACKAGE_DIRS
      ${BUILD_TOOLS_PYTHON_DIR}
  )
endfunction()

add_subdirectory(e2e_test_framework)
