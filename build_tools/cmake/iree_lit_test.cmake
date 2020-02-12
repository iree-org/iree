# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
#
# TODO(gcmn): allow using alternative driver
# A driver other than the default iree/tools/run_lit.sh is not currently supported.
function(iree_lit_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST_FILE"
    "DATA"
    ${ARGN}
  )
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  get_filename_component(_TEST_FILE_PATH ${_RULE_TEST_FILE} ABSOLUTE)

  add_test(
    NAME ${_NAME}
    # We run all our tests through a custom test runner to allow setup and teardown.
    COMMAND ${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.sh ${CMAKE_SOURCE_DIR}/iree/tools/run_lit.sh ${_TEST_FILE_PATH}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" # Make sure the lit runner can find all the binaries
  )
  set_property(TEST ${_NAME} PROPERTY ENVIRONMENT "TEST_TMPDIR=${_NAME}_test_tmpdir")
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
#
# TODO(gcmn): allow using alternative driver
# A driver other than the default iree/tools/run_lit.sh is not currently supported.
function(iree_lit_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DATA"
    ${ARGN}
  )
  IF(NOT IREE_BUILD_TESTS)
    return()
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
    )
  endforeach()
endfunction()
