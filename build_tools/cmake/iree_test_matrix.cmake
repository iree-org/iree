# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# Generates a matrix of permuted test cases from a description in the
# test_matrix.yaml file in the current source directory.
# The resulting test files will be written to:
#   ${CMAKE_CURRENT_BINARY_DIR}/generated
# Typically this should be followed by:
#   iree_test_matrix_glob_py_tests
#
# (type of tests to add are based on the runners defined in the
# generation script)
# TODO: Add a `generate` step conditioned on the test_matrix.yaml file, so that
# reconfigure is automatic.
function(iree_test_matrix_gen)
  set(_test_matrix_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_matrix.yaml")
  set(_generator_script "${PROJECT_SOURCE_DIR}/build_tools/testing/gen_test_matrix.py")
  message(STATUS "Generating tests for ${CMAKE_CURRENT_SOURCE_DIR}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
      "${_generator_script}"
      --dir "${CMAKE_CURRENT_SOURCE_DIR}"
      --output_dir "${CMAKE_CURRENT_BINARY_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RESULTS_VARIABLE _gen_result)

  if(_gen_result)
    message(SEND_ERROR "Error generating tests for ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endfunction()

# Adds individual py_tests for each generated *.py file from an
# iree_test_matrix_gen invocation.
#
# Parameters:
#   GLOB: The glob expression to use (defaults to *.py)
#   LABELS: Optional labels to add
function(iree_test_matrix_glob_py_tests)
  cmake_parse_arguments(ARG
    ""
    "GLOB"
    "LABELS"
    ${ARGN})
  set(_glob "*.py")
  if(ARG_GLOB)
    set(_glob "${ARG_GLOB}")
  endif()

  file(GLOB
    _found_test_files
    LIST_DIRECTORIES NO
    RELATIVE "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/${_glob}")

  if(NOT _found_test_files)
    message(SEND_ERROR
      "No generated tests (${_glob}) found under ${CMAKE_CURRENT_BINARY_DIR}/generated")
  endif()

  foreach(test_file ${_found_test_files})
    iree_py_test(
      NAME "generated/${test_file}"
      GENERATED_IN_BINARY_DIR
      SRCS "generated/${test_file}"
      LABELS
        generated
        ${ARG_LABELS}
    )
  endforeach()
endfunction()
