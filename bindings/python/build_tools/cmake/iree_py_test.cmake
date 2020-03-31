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

# iree_py_test()
#
# CMake function to imitate Bazel's iree_py_test rule.
#
# Parameters:
# NAME: name of test
# SRCS: List of source file
# DEPS: List of deps the test requires
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.

function(iree_py_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;LABELS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  string(REPLACE "_" "/" _PACKAGE_PATH ${_PACKAGE_NAME})
  set(_NAME_PATH "${_PACKAGE_PATH}:${_RULE_NAME}")

  add_test(
    NAME ${_NAME_PATH}
    COMMAND ${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.sh ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRCS}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH};TEST_TMPDIR=${_NAME}_test_tmpdir")
  # TODO(marbre): Find out how to add deps to tests.
  #               Similar to _RULE_DATA in iree_lit_test().

endfunction()
