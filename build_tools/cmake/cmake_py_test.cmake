# Copied from https://github.com/google/iree/blob/main/build_tools/cmake/cmake_cc_test.cmake
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include(cmake_installed_test)

# cmake_py_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target.
# SRCS: List of source files
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# ARGS command line arguments for the test.
#
# Note:
# cmake_cc_test will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# cmake_base_foo_test.
#
function(cmake_py_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    "GENERATED_IN_BINARY_DIR"
    "NAME;SRCS"
    "ARGS;LABELS;DATA"
    ${ARGN}
  )

  # Switch between source and generated tests.
  set(_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(_RULE_GENERATED_IN_BINARY_DIR)
    set(_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  rename_bazel_targets(_NAME "${_RULE_NAME}")

  cmake_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  cmake_add_installed_test(
    TEST_NAME "${_NAME_PATH}"
    LABELS "${_RULE_LABELS}"
    ENVIRONMENT
      "PYTHONPATH=${IREE_BINARY_DIR}/compiler-api/python_package:${IREE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}"
    COMMAND
      "${IREE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
      "${Python3_EXECUTABLE}"
      "${_SRC_DIR}/${_RULE_SRCS}"
      ${_RULE_ARGS}
    INSTALLED_COMMAND
      python
      "${_PACKAGE_PATH}/${_RULE_SRCS}"
  )

  if(_RULE_DATA)
    rename_bazel_targets(_DATA "${_RULE_DATA}")
    add_dependencies(${_NAME} ${_DATA})
  endif()

  install(FILES ${_RULE_SRCS}
    DESTINATION "tests/${_PACKAGE_PATH}"
    COMPONENT Tests
  )

  # TODO(boian): Find out how to add deps to tests.
  # CMake seems to not allow build targets to be dependencies for tests.
  # One way to achieve this is to make the test execution a target.
endfunction()
