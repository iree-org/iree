# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_lit_test()
#
# Creates a lit test for the specified source file.
#
# Mirrors the bzl rule of the same name.
#
# Parameters:
# NAME: Name of the target
# TEST_FILE: Test file to run with the lit runner.
# DATA: Additional data dependencies invoked by the test (e.g. binaries
#   called in the RUN line)
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# TODO(gcmn): allow using alternative driver
# A driver other than the default iree/tools/run_lit.sh is not currently supported.
function(iree_lit_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Note: lit tests are not *required* to be "compiler" tests, but we only use
  # them for compiler tests in practice.
  if(NOT IREE_BUILD_COMPILER)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST_FILE"
    "DATA;LABELS"
    ${ARGN}
  )

  if(CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _RULE_LABELS)
    return()
  endif()

  iree_package_ns(_PACKAGE_NS)
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  get_filename_component(_TEST_FILE_PATH ${_RULE_TEST_FILE} ABSOLUTE)

  list(TRANSFORM _RULE_DATA REPLACE "^::" "${_PACKAGE_NS}::")
  set(_DATA_DEP_PATHS)
  foreach(_DATA_DEP ${_RULE_DATA})
    list(APPEND _DATA_DEP_PATHS $<TARGET_FILE:${_DATA_DEP}>)
  endforeach(_DATA_DEP)

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  add_test(
    NAME
      ${_NAME_PATH}
    COMMAND
      # We run all our tests through a custom test runner to allow setup
      # and teardown.
      "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
      "${CMAKE_SOURCE_DIR}/iree/tools/run_lit.${IREE_HOST_SCRIPT_EXT}"
      ${_TEST_FILE_PATH}
      ${_DATA_DEP_PATHS}
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
  )

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY REQUIRED_FILES "${_TEST_FILE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT "TEST_TMPDIR=${_NAME}_test_tmpdir")
  iree_add_test_environment_properties(${_NAME_PATH})

  # TODO(gcmn): Figure out how to indicate a dependency on _RULE_DATA being built
endfunction()


# iree_lit_test_suite()
#
# Creates a suite of lit tests for a list of source files.
#
# Mirrors the bzl rule of the same name.
#
# Parameters:
# NAME: Name of the target
# SRCS: List of test files to run with the lit runner. Creates one test per source.
# DATA: Additional data dependencies invoked by the test (e.g. binaries
#   called in the RUN line)
# LABELS: Additional labels to apply to the generated tests. The package path is
#     added automatically.
#
# TODO(gcmn): allow using alternative driver
# A driver other than the default iree/tools/run_lit.sh is not currently supported.
function(iree_lit_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Note: we could check IREE_BUILD_COMPILER here, but cross compilation makes
  # that a little tricky. Instead, we let iree_check_test handle the checks,
  # meaning this function may run some configuration but generate no targets.

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DATA;LABELS"
    ${ARGN}
  )

  foreach(_TEST_FILE ${_RULE_SRCS})
    get_filename_component(_TEST_BASENAME ${_TEST_FILE} NAME)
    iree_lit_test(
      NAME
        "${_TEST_BASENAME}.test"
      TEST_FILE
        "${_TEST_FILE}"
      DATA
        "${_RULE_DATA}"
      LABELS
        "${_RULE_LABELS}"
    )
  endforeach()
endfunction()
