# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_native_test()
#
# Creates a test that runs the specified binary with the specified arguments.
#
# Mirrors the bzl function of the same name.
#
# Parameters:
# NAME: name of target
# DRIVER: If specified, will pass --driver=DRIVER to the test binary and adds
#     a driver label to the test.
# TEST_INPUT_FILE_ARG: If specified, the input file will be added to DATA and
#     its device path appended to ARGS. Note that the device path may be
#     different from the host path, so this parameter should be used to portably
#     pass file arguments to tests.
# DATA: Additional input files needed by the test binary. When running tests on
#     a separate device (e.g. Android), these files will be pushed to the
#     device. TEST_INPUT_FILE_ARG is automatically added if specified.
# ARGS: additional arguments passed to the test binary. TEST_INPUT_FILE_ARG and
#     --driver=DRIVER are automatically added if specified.
# SRC: binary target to run as the test.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# Note: the DATA argument is not actually adding dependencies because CMake
# doesn't have a good way to specify a data dependency for a test.
#
# Usage:
# iree_cc_binary(
#   NAME
#     requires_args_to_run
#   ...
# )
# iree_native_test(
#   NAME
#     requires_args_to_run_test
#   ARGS
#    --do-the-right-thing
#   SRC
#     ::requires_args_to_run
# )

function(iree_native_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;DRIVER;TEST_INPUT_FILE_ARG"
    "ARGS;LABELS;DATA"
    ${ARGN}
  )

  # Prefix the test with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  # If driver was specified, add the corresponding test arg and label.
  if (DEFINED _RULE_DRIVER)
    list(APPEND _RULE_ARGS "--driver=${_RULE_DRIVER}")
    list(APPEND _RULE_LABELS "driver=${_RULE_DRIVER}")
  endif()

  if(ANDROID)
    set(_ANDROID_ABS_DIR "/data/local/tmp/${_PACKAGE_PATH}/${_RULE_NAME}")
  endif()

  if (DEFINED _RULE_TEST_INPUT_FILE_ARG)
    if (ANDROID)
      get_filename_component(_TEST_INPUT_FILE_BASENAME "${_RULE_TEST_INPUT_FILE_ARG}" NAME)
      list(APPEND _RULE_ARGS "${_ANDROID_ABS_DIR}/${_TEST_INPUT_FILE_BASENAME}")
    else()
      list(APPEND _RULE_ARGS "${_RULE_TEST_INPUT_FILE_ARG}")
    endif()
    list(APPEND _RULE_DATA "${_RULE_TEST_INPUT_FILE_ARG}")
  endif()

  # Replace binary passed by relative ::name with iree::package::name
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _SRC_TARGET ${_RULE_SRC})

  if(ANDROID)
    # Define a custom target for pushing and running the test on Android device.
    set(_TEST_NAME ${_TEST_NAME}_on_android_device)
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_android_test.${IREE_HOST_SCRIPT_EXT}"
        "${_ANDROID_ABS_DIR}/$<TARGET_FILE_NAME:${_SRC_TARGET}>"
        ${_RULE_ARGS}
    )
    # Use environment variables to instruct the script to push artifacts
    # onto the Android device before running the test. This needs to match
    # with the expectation of the run_android_test.{sh|bat|ps1} script.
    string (REPLACE ";" " " _DATA_SPACE_SEPARATED "${_RULE_DATA}")
    set(
      _ENVIRONMENT_VARS
        "TEST_ANDROID_ABS_DIR=${_ANDROID_ABS_DIR}"
        "TEST_EXECUTABLE=$<TARGET_FILE:${_SRC_TARGET}>"
        "TEST_DATA=${_DATA_SPACE_SEPARATED}"
        "TEST_TMPDIR=${_ANDROID_ABS_DIR}/test_tmpdir"
    )
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
  else()
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
        "$<TARGET_FILE:${_SRC_TARGET}>"
        ${_RULE_ARGS}
    )
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT "TEST_TMPDIR=${CMAKE_BINARY_DIR}/${_NAME}_test_tmpdir")
    iree_add_test_environment_properties(${_TEST_NAME})
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST "${_TEST_NAME}" PROPERTY REQUIRED_FILES "${_RULE_DATA}")
endfunction()
