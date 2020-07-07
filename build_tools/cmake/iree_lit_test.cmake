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
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# TODO(gcmn): allow using alternative driver
# A driver other than the default iree/tools/run_lit.sh is not currently supported.
function(iree_lit_test)
  if(NOT IREE_BUILD_TESTS)
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

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  get_filename_component(_TEST_FILE_PATH ${_RULE_TEST_FILE} ABSOLUTE)

  set(_DATA_DEP_PATHS)
  foreach(_DATA_DEP ${_RULE_DATA})
    string(REPLACE "::" "_" _DATA_DEP_NAME ${_DATA_DEP})
    # TODO(*): iree_sh_binary so we can avoid this.
    if("${_DATA_DEP_NAME}" STREQUAL "iree_tools_IreeFileCheck")
      list(APPEND _DATA_DEP_PATHS "${CMAKE_SOURCE_DIR}/iree/tools/IreeFileCheck.sh")
    else()
      iree_get_target_path(_DATA_DEP_PATH ${_DATA_DEP_NAME})
      list(APPEND _DATA_DEP_PATHS "${_DATA_DEP_PATH}")
    endif()
  endforeach(_DATA_DEP)

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}:${_RULE_NAME}")
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
      "${CMAKE_BINARY_DIR}"
  )

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT "TEST_TMPDIR=${_NAME}_test_tmpdir")
  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY REQUIRED_FILES "${_TEST_FILE_PATH}")

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
