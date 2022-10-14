# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_parse_input_file_arg
#
# A helper function to split the flag and the file path from
# `TEST_INPUT_FILE_ARGS`. it is for `iree_run_module_test` as `iree-run-module`
# can have multiple input files while are passed through different flags.
# The file path is then handled in `iree_native_test` for portability.
function(iree_parse_input_file_arg OUT_FLAG OUT_FILE INPUT_ARG)
  if(${INPUT_ARG} MATCHES "^--")
    string(FIND "${INPUT_ARG}" "=" _FLAG_IDX)
    if(${_FLAG_IDX} GREATER 0)  # Add `=` back to _OUT_FLAG
      math(EXPR _FLAG_IDX "${_FLAG_IDX} + 1")
      cmake_path(GET INPUT_ARG EXTENSION LAST_ONLY _FILE_TYPE)
      if(_FILE_TYPE STREQUAL ".npy")  # Add `@` back to _OUT_FLAG
        math(EXPR _FLAG_IDX "${_FLAG_IDX} + 1")
      endif()
    endif()
    string(SUBSTRING "${INPUT_ARG}" 0 ${_FLAG_IDX} _OUT_FLAG)
    string(SUBSTRING "${INPUT_ARG}" ${_FLAG_IDX} -1 _OUT_FILE)
  else()
    set(_OUT_FLAG "")
    set(_OUT_FILE "${INPUT_ARG}")
  endif()
  set(${OUT_FLAG} "${_OUT_FLAG}" PARENT_SCOPE)
  set(${OUT_FILE} "${_OUT_FILE}" PARENT_SCOPE)
endfunction()

# iree_native_test()
#
# Creates a test that runs the specified binary with the specified arguments.
#
# Mirrors the bzl function of the same name.
#
# Parameters:
# NAME: name of target
# DRIVER: If specified, will pass --device=DRIVER to the test binary and adds
#     a driver label to the test.
# TEST_INPUT_FILE_ARGS: If specified, the input files will be added to DATA and
#     their device paths appended to ARGS. Note that the device path may be
#     different from the host path, so this parameter should be used to portably
#     pass file arguments to tests.
# DATA: Additional input files needed by the test binary. When running tests on
#     a separate device (e.g. Android), these files will be pushed to the
#     device. TEST_INPUT_FILE_ARGS is automatically added if specified.
# ARGS: additional arguments passed to the test binary. TEST_INPUT_FILE_ARGS and
#     --device=DRIVER are automatically added if specified.
# SRC: binary target to run as the test.
# WILL_FAIL: The target will run, but its pass/fail status will be inverted.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# TIMEOUT: Test target timeout in seconds.
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
    "NAME;SRC;DRIVER;WILL_FAIL"
    "ARGS;TEST_INPUT_FILE_ARGS;LABELS;DATA;TIMEOUT"
    ${ARGN}
  )

  # Prefix the test with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  # If driver was specified, add the corresponding test arg and label.
  if(DEFINED _RULE_DRIVER)
    list(APPEND _RULE_ARGS "--device=${_RULE_DRIVER}")
    list(APPEND _RULE_LABELS "driver=${_RULE_DRIVER}")
  endif()

  if(ANDROID)
    set(_ANDROID_ABS_DIR "/data/local/tmp/${_PACKAGE_PATH}/${_RULE_NAME}")
  endif()

  if(DEFINED _RULE_TEST_INPUT_FILE_ARGS)
    foreach(_INPUT_FILE_ARG ${_RULE_TEST_INPUT_FILE_ARGS})
      iree_parse_input_file_arg(_INPUT_FLAG _INPUT_FILE ${_INPUT_FILE_ARG})
      if(ANDROID)
        get_filename_component(_TEST_INPUT_FILE_BASENAME "${_INPUT_FILE}" NAME)
        list(APPEND _RULE_ARGS "${_INPUT_FLAG}${_ANDROID_ABS_DIR}/${_TEST_INPUT_FILE_BASENAME}")
      else()
        list(APPEND _RULE_ARGS "${_INPUT_FLAG}${_INPUT_FILE}")
      endif()
      list(APPEND _RULE_DATA "${_INPUT_FILE}")
    endforeach()
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
    string(REPLACE ";" " " _DATA_SPACE_SEPARATED "${_RULE_DATA}")
    set(
      _ENVIRONMENT_VARS
        "TEST_ANDROID_ABS_DIR=${_ANDROID_ABS_DIR}"
        "TEST_EXECUTABLE=$<TARGET_FILE:${_SRC_TARGET}>"
        "TEST_DATA=${_DATA_SPACE_SEPARATED}"
        "TEST_TMPDIR=${_ANDROID_ABS_DIR}/test_tmpdir"
    )
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
  elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" OR
          CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv32") AND
         CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The test target needs to run within the QEMU emulator for RV64 Linux
    # crosscompile build or on-device.
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${IREE_ROOT_DIR}/build_tools/cmake/run_riscv_test.sh"
        "$<TARGET_FILE:${_SRC_TARGET}>"
        ${_RULE_ARGS}
    )
    iree_configure_test(${_TEST_NAME})
  else()
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "$<TARGET_FILE:${_SRC_TARGET}>"
        ${_RULE_ARGS}
    )
    iree_configure_test(${_TEST_NAME})
  endif()

  if (NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST "${_TEST_NAME}" PROPERTY REQUIRED_FILES "${_RULE_DATA}")
  set_property(TEST ${_TEST_NAME} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
  if(_RULE_WILL_FAIL)
    set_property(TEST ${_TEST_NAME} PROPERTY WILL_FAIL ${_RULE_WILL_FAIL})
  endif()
endfunction()
