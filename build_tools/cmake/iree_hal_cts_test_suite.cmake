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
#       for `iree_hal_driver_registry_try_create()` within test code.
#   VARIANT_SUFFIX: Suffix to add to the test names, separate from the driver
#       name. Useful when specifying multiple configurations using `ARGS` or
#       other parameters.
#   DRIVER_REGISTRATION_HDR: The C #include path for `DRIVER_REGISTRATION_FN`.
#   DRIVER_REGISTRATION_FN: The C function which registers `DRIVER_NAME`.
#   COMPILER_TARGET_BACKEND: Optional target backend name to pass to the
#       `-iree-hal-target-backends` option of `iree-compile` to use for
#       executable generation. If this is omitted, or the associated compiler
#       target is not enabled, tests which use executables will be disabled.
#   EXECUTABLE_FORMAT: Executable format identifier. Will be interpreted
#       literally in C++ and may include macros like IREE_ARCH as needed.
#       Examples:
#           "\"vmvx-bytecode-fb\"" -> "vmvx-bytecode-fb"
#           "\"embedded-elf-\" IREE_ARCH" -> "embedded-elf-x86_64"
#   ARGS: List of command line arguments to pass to test binaries.
#   DEPS: List of other libraries to link in to the binary targets (typically
#       the dependency for `DRIVER_REGISTRATION_HDR`).
#   INCLUDED_TESTS: List of test names to include from the test suite.
#       Defaults to `IREE_ALL_CTS_TESTS`.
#   EXCLUDED_TESTS: List of test names to exclude from the test suite for this
#       driver. Generally set either this _or_ `INCLUDED_TESTS`.
#   LABELS: Additional labels to forward to `iree_cc_test`. The package path
#     and "driver=${DRIVER}" are added automatically.
function(iree_hal_cts_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "DRIVER_NAME;VARIANT_SUFFIX;DRIVER_REGISTRATION_HDR;DRIVER_REGISTRATION_FN;COMPILER_TARGET_BACKEND;EXECUTABLE_FORMAT"
    "DEPS;ARGS;INCLUDED_TESTS;EXCLUDED_TESTS;LABELS"
    ${ARGN}
  )

  list(APPEND _RULE_LABELS "driver=${_RULE_DRIVER_NAME}")

  # Enable executable tests if a compiler target backend capable of producing
  # executables compatible with the driver is provided and enabled.
  set(_ENABLE_EXECUTABLE_TESTS OFF)
  if(DEFINED _RULE_COMPILER_TARGET_BACKEND)
    string(TOUPPER ${_RULE_COMPILER_TARGET_BACKEND} _UPPERCASE_TARGET_BACKEND)
    string(REPLACE "-" "_" _NORMALIZED_TARGET_BACKEND ${_UPPERCASE_TARGET_BACKEND})
    if(NOT DEFINED IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
      message(SEND_ERROR "Unknown backend '${_RULE_COMPILER_TARGET_BACKEND}'. Check IREE_TARGET_BACKEND_* options.")
    endif()
    if(IREE_HOST_BIN_DIR)
      # If we're not building the host tools from source under this configuration,
      # such as when cross compiling, then we can't easily check for which
      # compiler target backends are enabled. Just assume all are enabled and only
      # rely on the runtime HAL driver check above for filtering.
      set(_ENABLE_EXECUTABLE_TESTS ON)
    else()
      # We are building the host tools, so check enabled compiler target backends.
      if(IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
        set(_ENABLE_EXECUTABLE_TESTS ON)
      endif()
    endif()
  endif()

  # Generate testdata if executable tests are enabled.
  set(_EXECUTABLE_DEPS "")
  if(_ENABLE_EXECUTABLE_TESTS)
    set(_EXECUTABLES_TESTDATA_NAME "${_RULE_COMPILER_TARGET_BACKEND}_executables")

    set(_TRANSLATE_FLAGS
      "--compile-mode=hal-executable"
      "--iree-hal-target-backends=${_RULE_COMPILER_TARGET_BACKEND}"
    )

    # Skip if already created (multiple suites using the same compiler setting).
    iree_package_name(_PACKAGE_NAME)
    if(NOT TARGET ${_PACKAGE_NAME}_${_EXECUTABLES_TESTDATA_NAME}_c)
      set(_EMBED_DATA_SOURCES "")
      foreach(_FILE_NAME ${IREE_ALL_CTS_EXECUTABLE_SOURCES})
        # Note: this is an abuse of naming. We are not building a bytecode
        # module, but this CMake rule already wraps iree-compile
        # We should add a new function like `iree_hal_executable()`.
        iree_bytecode_module(
          NAME
            ${_RULE_COMPILER_TARGET_BACKEND}_${_FILE_NAME}_module
          MODULE_FILE_NAME
            "${_RULE_COMPILER_TARGET_BACKEND}_${_FILE_NAME}.bin"
          SRC
            "${IREE_ROOT_DIR}/runtime/src/iree/hal/cts/testdata/${_FILE_NAME}.mlir"
          FLAGS
            ${_TRANSLATE_FLAGS}
          PUBLIC
          TESTONLY
        )
        list(APPEND _EMBED_DATA_SOURCES "${_RULE_COMPILER_TARGET_BACKEND}_${_FILE_NAME}.bin")
      endforeach()

      iree_c_embed_data(
        NAME
          ${_EXECUTABLES_TESTDATA_NAME}_c
        SRCS
          ${_EMBED_DATA_SOURCES}
        C_FILE_OUTPUT
          "${_EXECUTABLES_TESTDATA_NAME}_c.c"
        H_FILE_OUTPUT
          "${_EXECUTABLES_TESTDATA_NAME}_c.h"
        IDENTIFIER
          "iree_cts_testdata_executables"
        STRIP_PREFIX
          "${_RULE_COMPILER_TARGET_BACKEND}_"
        FLATTEN
        PUBLIC
        TESTONLY
      )

    endif()

    list(APPEND _EXECUTABLE_DEPS
      ::${_EXECUTABLES_TESTDATA_NAME}_c
    )
  endif()

  if(_RULE_INCLUDED_TESTS)
    set(_INCLUDED_TESTS ${_RULE_INCLUDED_TESTS})
  else()
    set(_INCLUDED_TESTS ${IREE_ALL_CTS_TESTS})
  endif()

  foreach(_TEST_NAME ${_INCLUDED_TESTS})
    if("${_TEST_NAME}" IN_LIST _RULE_EXCLUDED_TESTS)
      continue()
    endif()

    # Set vars (and possibly skip) based on if this test uses executables.
    if("${_TEST_NAME}" IN_LIST IREE_EXECUTABLE_CTS_TESTS)
      if(NOT _ENABLE_EXECUTABLE_TESTS)
        continue()
      endif()

      set(_TEST_DEPS
        "${_RULE_DEPS}"
        "${_EXECUTABLE_DEPS}"
      )
      set(IREE_CTS_EXECUTABLE_FORMAT "${_RULE_EXECUTABLE_FORMAT}")
      set(IREE_CTS_EXECUTABLES_TESTDATA_HDR "${_EXECUTABLES_TESTDATA_NAME}_c.h")
    else()
      set(_TEST_DEPS "${_RULE_DEPS}" )
      set(IREE_CTS_EXECUTABLE_FORMAT "")
      set(IREE_CTS_EXECUTABLES_TESTDATA_HDR "")
    endif()

    # Note: driver names may contain dashes and other special characters. We
    # could sanitize for file and target names, but passing through directly
    # may be more intuitive.
    if(DEFINED _RULE_VARIANT_SUFFIX)
      set(_TEST_TARGET_NAME "${_RULE_DRIVER_NAME}_${_RULE_VARIANT_SUFFIX}_${_TEST_NAME}_test")
    else()
      set(_TEST_TARGET_NAME "${_RULE_DRIVER_NAME}_${_TEST_NAME}_test")
    endif()
    set(_TEST_SOURCE_NAME "${_TEST_TARGET_NAME}.cc")
    set(_TEST_LIBRARY_DEP "iree::hal::cts::${_TEST_NAME}_test_library")

    # Generate the source file for this [test x driver] pair.
    # TODO(scotttodd): Move to build time instead of configure time?
    set(IREE_CTS_TEST_FILE_PATH "runtime/src/iree/hal/cts/${_TEST_NAME}_test.h")
    set(IREE_CTS_DRIVER_REGISTRATION_HDR "${_RULE_DRIVER_REGISTRATION_HDR}")
    set(IREE_CTS_DRIVER_REGISTRATION_FN "${_RULE_DRIVER_REGISTRATION_FN}")
    set(IREE_CTS_TEST_CLASS_NAME "${_TEST_NAME}_test")
    set(IREE_CTS_DRIVER_NAME "${_RULE_DRIVER_NAME}")
    set(IREE_CTS_TARGET_BACKEND "${_RULE_COMPILER_TARGET_BACKEND}")

    configure_file(
      "${IREE_ROOT_DIR}/runtime/src/iree/hal/cts/cts_test_template.cc.in"
      ${_TEST_SOURCE_NAME}
    )

    iree_cc_test(
      NAME
        ${_TEST_TARGET_NAME}
      ARGS
        ${_RULE_ARGS}
      SRCS
        "${CMAKE_CURRENT_BINARY_DIR}/${_TEST_SOURCE_NAME}"
      DEPS
        ${_TEST_DEPS}
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
