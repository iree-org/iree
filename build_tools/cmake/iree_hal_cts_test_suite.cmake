# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_hal_cts_test_suite()
#
# Creates a set of tests for a provided Hardware Abstraction Layer (HAL) driver,
# with one generated test for each test in the Conformance Test Suite (CTS).
#
# Parameters:
#   DRIVER_NAME: The name of the driver to test. Used for both target names and
#       for `iree_hal_driver_registry_try_create_by_name()` within test code.
#   DRIVER_REGISTRATION_HDR: The C #include path for `DRIVER_REGISTRATION_FN`.
#   DRIVER_REGISTRATION_FN: The C function which registers `DRIVER_NAME`.
#   DEPS: List of other libraries to link in to the binary targets (typically
#       the dependency for `DRIVER_REGISTRATION_HDR`).
#   EXCLUDED_TESTS: List of test names from `IREE_ALL_CTS_TESTS` to
#       exclude from the test suite for this driver.
#   LABELS: Additional labels to forward to `iree_cc_test`. The package path
#     and "driver=${DRIVER}" are added automatically.
function(iree_hal_cts_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "DRIVER_NAME;DRIVER_REGISTRATION_HDR;DRIVER_REGISTRATION_FN"
    "DEPS;EXCLUDED_TESTS;LABELS"
    ${ARGN}
  )

  list(APPEND _RULE_LABELS "driver=${_RULE_DRIVER_NAME}")

  foreach(_TEST_NAME ${IREE_ALL_CTS_TESTS})
    if("${_TEST_NAME}" IN_LIST _RULE_EXCLUDED_TESTS)
      continue()
    endif()

    # Note: driver names may contain dashes and other special characters. We
    # could sanitize for file and target names, but passing through directly
    # may be more intuitive.
    set(_TEST_SOURCE_NAME "${_TEST_NAME}_${_RULE_DRIVER_NAME}_test.cc")
    set(_TEST_LIBRARY_DEP "iree::hal::cts::${_TEST_NAME}_test_library")

    # Generate the source file for this [test x driver] pair.
    # TODO(scotttodd): Move to build time instead of configure time?
    set(IREE_CTS_TEST_FILE_PATH "iree/hal/cts/${_TEST_NAME}_test.h")
    set(IREE_CTS_DRIVER_REGISTRATION_HDR "${_RULE_DRIVER_REGISTRATION_HDR}")
    set(IREE_CTS_DRIVER_REGISTRATION_FN "${_RULE_DRIVER_REGISTRATION_FN}")
    set(IREE_CTS_TEST_CLASS_NAME "${_TEST_NAME}_test")
    set(IREE_CTS_DRIVER_NAME "${_RULE_DRIVER_NAME}")
    configure_file(
      "${IREE_ROOT_DIR}/iree/hal/cts/cts_test_template.cc.in"
      ${_TEST_SOURCE_NAME}
    )

    iree_cc_test(
      NAME
        ${_RULE_DRIVER_NAME}_${_TEST_NAME}_test
      SRCS
        "${CMAKE_CURRENT_BINARY_DIR}/${_TEST_SOURCE_NAME}"
      DEPS
        ${_RULE_DEPS}
        ${_TEST_LIBRARY_DEP}
        iree::base
        iree::hal
        iree::hal::cts::cts_test_base
        iree::testing::gtest_main
      LABELS
        ${_RULE_LABELS}
    )
  endforeach()
endfunction()
