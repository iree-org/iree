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
# TOOLS: Tools that should be included on the PATH
# DATA: Additional data dependencies invoked by the test (e.g. binaries
#   called in the RUN line)
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
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
    "DATA;TOOLS;LABELS;TIMEOUT"
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
  list(TRANSFORM _RULE_TOOLS REPLACE "^::" "${_PACKAGE_NS}::")
  set(_DATA_DEP_PATHS)
  foreach(_DATA_DEP IN LISTS _RULE_DATA _RULE_TOOLS)
    list(APPEND _DATA_DEP_PATHS $<TARGET_FILE:${_DATA_DEP}>)
  endforeach()

  set(_LIT_PATH_ARGS)
  foreach(_TOOL IN LISTS _RULE_TOOLS)
    list(APPEND _LIT_PATH_ARGS "--path" "$<TARGET_FILE_DIR:${_TOOL}>")
  endforeach()

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  add_test(
    NAME
      ${_NAME_PATH}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${LLVM_EXTERNAL_LIT}"
      ${_LIT_PATH_ARGS}
      ${_TEST_FILE_PATH}
  )

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY REQUIRED_FILES "${_TEST_FILE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT
    "LIT_OPTS=-v"
    "FILECHECK_OPTS=--enable-var-scope")
  set_property(TEST ${_NAME_PATH} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
  iree_configure_test(${_NAME_PATH})

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
# TOOLS: Tools that should be included on the PATH
# DATA: Additional data dependencies used by the test
# LABELS: Additional labels to apply to the generated tests. The package path is
#     added automatically.
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
    "SRCS;DATA;TOOLS;LABELS;TIMEOUT"
    ${ARGN}
  )

  if (NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  foreach(_TEST_FILE ${_RULE_SRCS})
    get_filename_component(_TEST_BASENAME ${_TEST_FILE} NAME)
    iree_lit_test(
      NAME
        "${_TEST_BASENAME}.test"
      TEST_FILE
        "${_TEST_FILE}"
      DATA
        "${_RULE_DATA}"
      TOOLS
        "${_RULE_TOOLS}"
      LABELS
        "${_RULE_LABELS}"
      TIMEOUT
        ${_RULE_TIMEOUT}
    )
  endforeach()
endfunction()
