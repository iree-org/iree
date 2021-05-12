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

# iree_run_binary_test()
#
# Creates a test that runs the specified binary with the specified arguments.
#
# Mirrors the bzl function of the same name.
#
# Parameters:
# NAME: name of target
# ARGS: arguments passed to the test binary.
# TEST_BINARY: binary target to run as the test.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# Note: the DATA argument is not supported because CMake doesn't have a good way
# to specify a data dependency for a test.
#
#
# Usage:
# iree_cc_binary(
#   NAME
#     requires_args_to_run
#   ...
# )
# iree_run_binary_test(
#   NAME
#     requires_args_to_run_test
#   ARGS
#    --do-the-right-thing
#   TEST_BINARY
#     ::requires_args_to_run
# )

function(iree_run_binary_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST_BINARY"
    "ARGS;LABELS"
    ${ARGN}
  )

  # Prefix the test with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  # Replace binary passed by relative ::name with iree::package::name
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _TEST_BINARY_TARGET ${_RULE_TEST_BINARY})

  if(ANDROID)
    set(_ANDROID_REL_DIR "${_PACKAGE_PATH}/${_RULE_NAME}")
    set(_ANDROID_ABS_DIR "/data/local/tmp/${_ANDROID_REL_DIR}")

    # Define a custom target for pushing and running the test on Android device.
    set(_TEST_NAME ${_TEST_NAME}_on_android_device)
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_android_test.${IREE_HOST_SCRIPT_EXT}"
        "${_ANDROID_REL_DIR}/$<TARGET_FILE_NAME:${_TEST_BINARY_TARGET}>"
        ${_RULE_ARGS}
    )
    # Use environment variables to instruct the script to push artifacts
    # onto the Android device before running the test. This needs to match
    # with the expectation of the run_android_test.{sh|bat|ps1} script.
    set(
      _ENVIRONMENT_VARS
        TEST_ANDROID_ABS_DIR=${_ANDROID_ABS_DIR}
        TEST_EXECUTABLE=$<TARGET_FILE:${_TEST_BINARY_TARGET}>
        TEST_TMPDIR=${_ANDROID_ABS_DIR}/test_tmpdir
    )
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
  else()
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
        "$<TARGET_FILE:${_TEST_BINARY_TARGET}>"
        ${_RULE_ARGS}
    )
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT "TEST_TMPDIR=${CMAKE_BINARY_DIR}/${_NAME}_test_tmpdir")
    iree_add_test_environment_properties(${_TEST_NAME})
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
endfunction()
