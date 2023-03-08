# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

set(IREE_TARGET_BACKENDS_SUPPORTING_TARGET_CPU_FEATURES
  llvm-cpu
  vmvx
)

# Helper for iree_check_test and iree_trace_runner_test.
# Just a thin wrapper around iree_bytecode_module, passing it some
# common flags, including the appropriate --iree-llvm-target-triple in the
# Android case.
function(iree_bytecode_module_for_iree_check_test_and_friends)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "MODULE_NAME;SRC;TARGET_BACKEND;MODULE_FILE_NAME"
    "FLAGS;TARGET_CPU_FEATURES"
    ${ARGN}
  )

  if(_RULE_TARGET_CPU_FEATURES)
    if(NOT _RULE_TARGET_BACKEND IN_LIST IREE_TARGET_BACKENDS_SUPPORTING_TARGET_CPU_FEATURES)
      message(SEND_ERROR "TARGET_CPU_FEATURES should be empty when \
TARGET_BACKEND is not in the list (${IREE_TARGET_BACKENDS_SUPPORTING_TARGET_CPU_FEATURES}). Actual values: \
TARGET_CPU_FEATURES=${_RULE_TARGET_CPU_FEATURES}, TARGET_BACKEND=${_RULE_TARGET_BACKEND}.")
    endif()
    list(APPEND _RULE_FLAGS "--iree-llvm-target-cpu-features=${_RULE_TARGET_CPU_FEATURES}")
  endif()

  iree_bytecode_module(
    NAME
      "${_RULE_MODULE_NAME}"
    MODULE_FILE_NAME
      "${_RULE_MODULE_FILE_NAME}"
    SRC
      "${_RULE_SRC}"
    FLAGS
      "--mlir-print-op-on-diagnostic=false"
      "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}"
      ${_RULE_FLAGS}
    TESTONLY
  )
endfunction()

# iree_check_test()
#
# Creates a test using iree-check-module for the specified source file.
#
# Mirrors the bzl rule of the same name.
#
# Parameters:
#   NAME: Name of the target
#   SRC: mlir source file to be compiled to an IREE module.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with. This can be omitted to test only
#       compilation, but consider omiting the driver as a hacky abuse of the
#       rule since compilation on its own not use iree-check-module.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode output
#       format and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to iree-check-module. The driver
#       and input file are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   MODULE_FILE_NAME: Optional, specifies the absolute path to the filename
#       to use for the generated IREE module (.vmfb).
#   TARGET_CPU_FEATURES: If specified, a string passed as argument to
#       --iree-llvm-target-cpu-features.
function(iree_check_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Check tests require (by way of iree_bytecode_module) some tools.
  #
  # These can either be built from source, if IREE_BUILD_COMPILER is set, or
  # be located under IREE_HOST_BIN_DIR (required if cross-compiling).
  if(NOT IREE_BUILD_COMPILER AND NOT IREE_HOST_BIN_DIR)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;TARGET_BACKEND;DRIVER;MODULE_FILE_NAME"
    "COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES;TIMEOUT"
    ${ARGN}
  )

  if(CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _RULE_LABELS)
    return()
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_MODULE_NAME "${_RULE_NAME}_module")

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_MODULE_NAME}.vmfb")
  endif(DEFINED _RULE_MODULE_FILE_NAME)

  iree_bytecode_module_for_iree_check_test_and_friends(
    MODULE_NAME
      "${_MODULE_NAME}"
    MODULE_FILE_NAME
      "${_MODULE_FILE_NAME}"
    SRC
      "${_RULE_SRC}"
    TARGET_BACKEND
      "${_RULE_TARGET_BACKEND}"
    FLAGS
      ${_RULE_COMPILER_FLAGS}
    TARGET_CPU_FEATURES
      ${_RULE_TARGET_CPU_FEATURES}
  )

  set(_RUNNER_TARGET "iree-check-module")

  # A target specifically for the test. We could combine this with the above,
  # but we want that one to get pulled into iree_bytecode_module.
  add_custom_target("${_NAME}" ALL)
  add_dependencies(
    "${_NAME}"
    "${_NAME}_module"
    "${_RUNNER_TARGET}"
  )

  add_dependencies(iree-test-deps "${_NAME}")

  if(NOT DEFINED _RULE_DRIVER)
    return()
  endif()

  iree_native_test(
    NAME
      "${_RULE_NAME}"
    DRIVER
      "${_RULE_DRIVER}"
    SRC
      "${_RUNNER_TARGET}"
    ARGS
      "{{${_MODULE_FILE_NAME}}}"
      ${_RULE_RUNNER_ARGS}
    LABELS
      ${_RULE_LABELS}
      ${_RULE_TARGET_CPU_FEATURES}
    TIMEOUT
      ${_RULE_TIMEOUT}
  )
endfunction()

# iree_check_single_backend_test_suite()
#
# Creates a test suite of iree-check-module tests for a single backend/driver pair.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source file.
# Parameters:
#   NAME: name of the generated test suite.
#   SRCS: source mlir files containing the module.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with. This can be omitted to test only
#       compilation, but consider omiting the driver as a hacky abuse of the
#       rule since compilation on its own not use iree-check-module.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode output
#       format and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to the underlying iree-check-module
#       tests. The driver and input file are passed automatically. To use
#       different args per test, create a separate suite or iree_check_test.
#   LABELS: Additional labels to apply to the generated tests. The package path
#       is added automatically.
#   TARGET_CPU_FEATURES: If specified, a string passed as argument to
#       --iree-llvm-target-cpu-features.
function(iree_check_single_backend_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Note: we could check IREE_BUILD_COMPILER here, but cross compilation makes
  # that a little tricky. Instead, we let iree_check_test handle the checks,
  # meaning this function may run some configuration but generate no targets.

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET_BACKEND;DRIVER"
    "SRCS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES;TIMEOUT"
    ${ARGN}
  )

  string(TOUPPER "${IREE_EXTERNAL_HAL_DRIVERS}" _UPPERCASE_EXTERNAL_DRIVERS)
  string(REPLACE "-" "_" _NORMALIZED_EXTERNAL_DRIVERS "${_UPPERCASE_EXTERNAL_DRIVERS}")

  # Omit tests for which the specified driver or target backend is not enabled.
  # This overlaps with directory exclusions and other filtering mechanisms.
  #
  # Note: omitting the DRIVER arg is allowed (though it is a hack). If it is
  # omitted, we don't need to test for a driver being enabled.
  if(DEFINED _RULE_DRIVER)
    string(TOUPPER ${_RULE_DRIVER} _UPPERCASE_DRIVER)
    string(REPLACE "-" "_" _NORMALIZED_DRIVER ${_UPPERCASE_DRIVER})
    if((NOT DEFINED IREE_HAL_DRIVER_${_NORMALIZED_DRIVER}) AND (NOT ${_NORMALIZED_DRIVER} IN_LIST _NORMALIZED_EXTERNAL_DRIVERS))
      message(SEND_ERROR "Unknown driver '${_RULE_DRIVER}'. Check IREE_HAL_DRIVER_*/IREE_EXTERNAL_HAL_DRIVERS options.")
    endif()
    if((NOT IREE_HAL_DRIVER_${_NORMALIZED_DRIVER}) AND (NOT IREE_EXTERNAL_${_NORMALIZED_DRIVER}_HAL_DRIVER_FOUND))
      return()
    endif()
  endif()
  string(TOUPPER ${_RULE_TARGET_BACKEND} _UPPERCASE_TARGET_BACKEND)
  string(REPLACE "-" "_" _NORMALIZED_TARGET_BACKEND ${_UPPERCASE_TARGET_BACKEND})
  if(NOT DEFINED IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
    message(SEND_ERROR "Unknown backend '${_RULE_TARGET_BACKEND}'. Check IREE_TARGET_BACKEND_* options.")
  endif()
  if(IREE_HOST_BIN_DIR)
    # If we're not building the host tools from source under this configuration,
    # such as when cross compiling, then we can't easily check for which
    # compiler target backends are enabled. Just assume all are enabled and only
    # rely on the runtime HAL driver check above for filtering.

    # No driver, so this is a special configuration. The assumption above
    # might not be true, so skip (these tests are _probably_ already being
    # built on the host anyway, so no need to build when cross compiling).
    # TODO(#11354): Use a different test function / move compile to test-time
    if(NOT DEFINED _RULE_DRIVER)
      return()
    endif()
  else()
    # We are building the host tools, so check enabled compiler target backends.
    if(NOT IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
      return()
    endif()
  endif()

  foreach(_SRC IN LISTS _RULE_SRCS)
    get_filename_component(_BASE_NAME ${_SRC} NAME)
    set(_TEST_NAME "${_RULE_NAME}_${_BASE_NAME}")
    iree_check_test(
      NAME
        ${_TEST_NAME}
      SRC
        ${_SRC}
      TARGET_BACKEND
        ${_RULE_TARGET_BACKEND}
      DRIVER
        ${_RULE_DRIVER}
      COMPILER_FLAGS
        ${_RULE_COMPILER_FLAGS}
      RUNNER_ARGS
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      TARGET_CPU_FEATURES
        ${_RULE_TARGET_CPU_FEATURES}
      TIMEOUT
        ${_RULE_TIMEOUT}
    )
  endforeach()
endfunction()

# Helper function parsing a string occurring as an entry in TARGET_CPU_FEATURES_VARIANTS.
#
# This function has 3 output-params: variables that it sets with PARENT_SCOPE:
# _ENABLED, _TARGET_CPU_FEATURES, _TARGET_CPU_FEATURES_SUFFIX.
#
# "default" is handled specially. _ENABLED is always set to "TRUE" and
# _TARGET_CPU_FEATURES, and _TARGET_CPU_FEATURES_SUFFIX are set to
# the empty string.
#
# Other values are parsed as "arch:features", the parsed arch is matched with
# `CMAKE_SYSTEM_PROCESSOR`, `_ENABLED` is set to "TRUE" if and only if they
# match, `_TARGET_CPU_FEATURES_SUFFIX` is set to a string based on the
# features that is appropriate to include in a CMake target or test name, and
# More than one target cpu feature is currently unsupported.
#
# aarch64:+dotprod ->_ENABLED="TRUE" if the target architecture is aarch64,
#                    _TARGET_CPU_FEATURES="+dotprod",
#                    _TARGET_CPU_FEATURES_SUFFIX="_dotprod",
# default -> _ENABLED="TRUE" unconditionally,
#            other output strings are "".
function(process_target_cpu_features _INPUT_TARGET_CPU_FEATURES _ENABLED
         _TARGET_CPU_FEATURES _TARGET_CPU_FEATURES_SUFFIX)
  set(_TARGET_CPU_FEATURES "" PARENT_SCOPE)
  set(_TARGET_CPU_FEATURES_SUFFIX "" PARENT_SCOPE)
  if("${_INPUT_TARGET_CPU_FEATURES}" STREQUAL "default")
    set(_ENABLED "TRUE" PARENT_SCOPE)
    return()
  endif()
  string(REGEX MATCHALL "[^:]+" _COMPONENTS "${_INPUT_TARGET_CPU_FEATURES}")
  list(LENGTH _COMPONENTS _NUM_COMPONENTS)
  if(NOT _NUM_COMPONENTS EQUAL 2)
    message(SEND_ERROR "TARGET_CPU_FEATURES should be of the form \
_FILTER_ARCH:_TARGET_CPU_FEATURES. Got: ${_INPUT_TARGET_CPU_FEATURES}")
    return()
  endif()
  # TARGET_CPU_FEATURES_VARIANT is of the form _FILTER_ARCH:_TARGET_CPU_FEATURE.
  list(GET _COMPONENTS 0 _FILTER_ARCH)
  list(GET _COMPONENTS 1 _TARGET_CPU_FEATURES)
  if(_FILTER_ARCH STREQUAL CMAKE_SYSTEM_PROCESSOR)
    set(_ENABLED "TRUE" PARENT_SCOPE)
    set(_TARGET_CPU_FEATURES "${_TARGET_CPU_FEATURES}" PARENT_SCOPE)
    # TODO: the logic to generate the suffix from the list of target CPU features
    # will need to be generalized when the lists have more than 1 element, when
    # some features are being disabled by a "-" sign, and if some features involve
    # any character that's not wanted in a cmake rule name.
    # For now, let's just generate errors in those cases:
    list(LENGTH _TARGET_CPU_FEATURES _NUM_TARGET_CPU_FEATURES)
    if(NOT _NUM_TARGET_CPU_FEATURES EQUAL 1)
      message(SEND_ERROR "Current limitation: \
TARGET_CPU_FEATURES should have length 1")
    endif()
    string(SUBSTRING "${_TARGET_CPU_FEATURES}" 0 1 _TARGET_CPU_FEATURES_FIRST_CHAR)
    string(SUBSTRING "${_TARGET_CPU_FEATURES}" 1 -1 _TARGET_CPU_FEATURES_AFTER_FIRST_CHAR)
    if(NOT _TARGET_CPU_FEATURES_FIRST_CHAR STREQUAL "+")
      message(SEND_ERROR "Current limitation: \
TARGET_CPU_FEATURES should start with a +. Got: ${_TARGET_CPU_FEATURES}.")
    endif()
    if(NOT _TARGET_CPU_FEATURES_AFTER_FIRST_CHAR MATCHES "[a-zA-Z0-9_]+")
      message(SEND_ERROR "Current limitation: \
TARGET_CPU_FEATURES should match [a-zA-Z0-9]+ after the initial +. \
Got: ${_TARGET_CPU_FEATURES}.")
    endif()
    # Generate the target cpu features suffix string with underscores ('_')
    # separating the features.
    string(REPLACE "+" "_" _TARGET_CPU_FEATURES_SUFFIX_LOCAL "${_TARGET_CPU_FEATURES}")
    set(_TARGET_CPU_FEATURES_SUFFIX "${_TARGET_CPU_FEATURES_SUFFIX_LOCAL}" PARENT_SCOPE)
  else()
    set(_ENABLED "FALSE" PARENT_SCOPE)
  endif()
endfunction()

# iree_check_test_suite()
#
# Creates a test suite of iree-check-module tests.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source and backend/driver pair.
# Parameters:
#   NAME: name of the generated test suite.
#   SRCS: source mlir files containing the module.
#   TARGET_BACKENDS: backends to compile the module for. These form pairs with
#       the DRIVERS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   DRIVERS: drivers to run the module with. These form pairs with the
#       TARGET_BACKENDS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   RUNNER_ARGS: additional args to pass to the underlying iree-check-module tests. The
#       driver and input file are passed automatically. To use different args per
#       test, create a separate suite or iree_check_test.
#   LABELS: Additional labels to apply to the generated tests. The package path is
#       added automatically.
#   TARGET_CPU_FEATURES_VARIANTS: list of target cpu features variants. Only used
#       for drivers that vary based on the target CPU features. For each list
#       element, a separate test is created, with the list element passed as
#       argument to --iree-llvm-target-cpu-features. The special value "default"
#       is interpreted as no --iree-llvm-target-cpu-features flag to work around
#       corner cases with empty entries in CMake lists.
function(iree_check_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;TARGET_BACKENDS;DRIVERS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES_VARIANTS;TIMEOUT"
    ${ARGN}
  )

  if(NOT DEFINED _RULE_TARGET_BACKENDS AND NOT DEFINED _RULE_DRIVERS)
    set(_RULE_TARGET_BACKENDS "vmvx" "vulkan-spirv" "llvm-cpu")
    set(_RULE_DRIVERS "local-task" "vulkan" "local-task")
  endif()

  list(LENGTH _RULE_TARGET_BACKENDS _TARGET_BACKEND_COUNT)
  list(LENGTH _RULE_DRIVERS _DRIVER_COUNT)

  if(NOT _TARGET_BACKEND_COUNT EQUAL _DRIVER_COUNT)
    message(SEND_ERROR
        "TARGET_BACKENDS count ${_TARGET_BACKEND_COUNT} does not match DRIVERS count ${_DRIVER_COUNT}")
  endif()

  math(EXPR _MAX_INDEX "${_TARGET_BACKEND_COUNT} - 1")
  foreach(_INDEX RANGE "${_MAX_INDEX}")
    list(GET _RULE_TARGET_BACKENDS ${_INDEX} _TARGET_BACKEND)
    list(GET _RULE_DRIVERS ${_INDEX} _DRIVER)
    if(_TARGET_BACKEND STREQUAL "llvm-cpu" AND _RULE_TARGET_CPU_FEATURES_VARIANTS)
      set(_TARGET_CPU_FEATURES_VARIANTS "${_RULE_TARGET_CPU_FEATURES_VARIANTS}")
    else()
      set(_TARGET_CPU_FEATURES_VARIANTS "default")
    endif()
    foreach(_TARGET_CPU_FEATURES_LIST_ELEM IN LISTS _TARGET_CPU_FEATURES_VARIANTS)
      process_target_cpu_features("${_TARGET_CPU_FEATURES_LIST_ELEM}" _ENABLED _TARGET_CPU_FEATURES _TARGET_CPU_FEATURES_SUFFIX)
      if(NOT _ENABLED)
        # The current entry is disabled on the target CPU architecture.
        continue()
      endif()
      iree_check_single_backend_test_suite(
        NAME
          "${_RULE_NAME}_${_TARGET_BACKEND}_${_DRIVER}${_TARGET_CPU_FEATURES_SUFFIX}"
        SRCS
          ${_RULE_SRCS}
        TARGET_BACKEND
          ${_TARGET_BACKEND}
        DRIVER
          ${_DRIVER}
        COMPILER_FLAGS
          ${_RULE_COMPILER_FLAGS}
        RUNNER_ARGS
          ${_RULE_RUNNER_ARGS}
        LABELS
          ${_RULE_LABELS}
        TARGET_CPU_FEATURES
          ${_TARGET_CPU_FEATURES}
        TIMEOUT
          ${_RULE_TIMEOUT}
      )
    endforeach()
  endforeach()
endfunction()
