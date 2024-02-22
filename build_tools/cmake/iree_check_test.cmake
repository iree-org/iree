# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

function(iree_is_bytecode_module_test_excluded_by_labels _DST_IS_EXCLUDED_VAR _SRC_LABELS)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" _LOWERCASE_BUILD_TYPE)
  if(((IREE_ARCH MATCHES "^riscv_") AND ("noriscv" IN_LIST _SRC_LABELS)) OR
     (EMSCRIPTEN AND ("nowasm" IN_LIST _SRC_LABELS)) OR
     (IREE_ENABLE_ASAN AND ("noasan" IN_LIST _SRC_LABELS)) OR
     (IREE_ENABLE_TSAN AND ("notsan" IN_LIST _SRC_LABELS)) OR
     (CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _RULE_LABELS) OR
     ((_LOWERCASE_BUILD_TYPE STREQUAL "debug") AND ( "optonly" IN_LIST _RULE_LABELS)))
    set("${_DST_IS_EXCLUDED_VAR}" TRUE PARENT_SCOPE)
  endif()
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
#       --iree-llvmcpu-target-cpu-features.
#   DEPENDS: Optional. Additional dependencies beyond SRC and the tools.
#   INPUT_TYPE: The value for the --iree-input-type= flag. Also disables tests
#       if no compiled support for that configuration.
function(iree_check_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;TARGET_BACKEND;DRIVER;MODULE_FILE_NAME;INPUT_TYPE"
    "COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES;DEPENDS;TIMEOUT"
    ${ARGN}
  )

  # ---------------------------------------------------------------------------
  # Bytecode module builds require
  #   1. the compiler, either in the same build or provided in IREE_HOST_BIN_DIR
  #   2. compiler support for _RULE_INPUT_TYPE
  #   3. compiler support for _RULE_TARGET_BACKEND
  set(_BYTECODE_MODULE_BUILD_ENABLED TRUE)

  # 1. Check for the compiler.
  if(NOT IREE_BUILD_COMPILER AND NOT IREE_HOST_BIN_DIR)
    set(_BYTECODE_MODULE_BUILD_ENABLED FALSE)
  endif()

  # 2. Check input type availability.
  # Note: we can only reliably check for this when building the compiler host
  # tools from source. If the tools are already built, we assume that all input
  # dialects are enabled. We could query the tools in the binary directory for
  # support dynamically if optionality would be useful.
  if(DEFINED _RULE_INPUT_TYPE AND NOT IREE_HOST_BIN_DIR)
    if("${_RULE_INPUT_TYPE}" STREQUAL "stablehlo" AND NOT IREE_INPUT_STABLEHLO)
      set(_BYTECODE_MODULE_BUILD_ENABLED FALSE)
    endif()
    if("${_RULE_INPUT_TYPE}" STREQUAL "tosa" AND NOT IREE_INPUT_TOSA)
      set(_BYTECODE_MODULE_BUILD_ENABLED FALSE)
    endif()
    if("${_RULE_INPUT_TYPE}" STREQUAL "torch" AND NOT IREE_INPUT_TORCH)
      set(_BYTECODE_MODULE_BUILD_ENABLED FALSE)
    endif()
  endif()

  # 3. Check target backend availability.
  # Note: we can only reliably check for this when building the compiler host
  # tools from source. If the tools are already built, we assume that all target
  # backends are enabled. We could query the tools in the binary directory for
  # support dynamically if optionality would be useful.
  if(NOT IREE_HOST_BIN_DIR)
    string(TOUPPER ${_RULE_TARGET_BACKEND} _UPPERCASE_TARGET_BACKEND)
    string(REPLACE "-" "_" _NORMALIZED_TARGET_BACKEND ${_UPPERCASE_TARGET_BACKEND})
    # TODO(scotttodd): allow plugins to provide external backends here
    if(NOT DEFINED IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
      message(SEND_ERROR "Unknown backend '${_RULE_TARGET_BACKEND}'. Check IREE_TARGET_BACKEND_* options.")
    endif()
    if(NOT IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
      set(_BYTECODE_MODULE_BUILD_ENABLED FALSE)
    endif()
  endif()
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # Tests are defined if _RULE_DRIVER is defined.
  set(_TEST_DEFINED TRUE)
  if(NOT DEFINED _RULE_DRIVER)
    set(_TEST_DEFINED FALSE)
  endif()

  # Test execution requires
  #   1. the bytecode module build to be enabled
  #   2. _RULE_DRIVER is defined and runtime support is enabled
  #   3. no other label exclusions (e.g. 'optonly' test with 'debug' config)
  set(_TEST_DISABLED FALSE)

  # 1. Check bytecode module build.
  if(NOT _BYTECODE_MODULE_BUILD_ENABLED)
    set(_TEST_DISABLED TRUE)
  endif()

  # 2. Check driver availability.
  if(DEFINED _RULE_DRIVER)
    string(TOUPPER ${_RULE_DRIVER} _UPPERCASE_DRIVER)
    string(REPLACE "-" "_" _NORMALIZED_DRIVER ${_UPPERCASE_DRIVER})
    string(TOUPPER "${IREE_EXTERNAL_HAL_DRIVERS}" _UPPERCASE_EXTERNAL_DRIVERS)
    string(REPLACE "-" "_" _NORMALIZED_EXTERNAL_DRIVERS "${_UPPERCASE_EXTERNAL_DRIVERS}")
    if((NOT DEFINED IREE_HAL_DRIVER_${_NORMALIZED_DRIVER}) AND
       (NOT ${_NORMALIZED_DRIVER} IN_LIST _NORMALIZED_EXTERNAL_DRIVERS))
      message(SEND_ERROR "Unknown driver '${_RULE_DRIVER}'. Check IREE_HAL_DRIVER_*/IREE_EXTERNAL_HAL_DRIVERS options.")
    endif()
    if((NOT IREE_HAL_DRIVER_${_NORMALIZED_DRIVER}) AND
       (NOT IREE_EXTERNAL_${_NORMALIZED_DRIVER}_HAL_DRIVER_FOUND))
      set(_TEST_DISABLED TRUE)
    endif()
  endif()

  # 3. Check label exclusions.
  iree_is_bytecode_module_test_excluded_by_labels(_EXCLUDED_BY_LABELS "${_RULE_LABELS}")
  if(_EXCLUDED_BY_LABELS)
    set(_TEST_DISABLED TRUE)
  endif()

  if((_TEST_DISABLED OR NOT _TEST_DEFINED) AND NOT IREE_BUILD_ALL_CHECK_TEST_MODULES)
    set(_BYTECODE_MODULE_BUILD_ENABLED FALSE)
  endif()
  # ---------------------------------------------------------------------------

  iree_package_name(_PACKAGE_NAME)

  set(_MODULE_NAME "${_RULE_NAME}_module")
  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else()
    set(_MODULE_FILE_NAME "${_MODULE_NAME}.vmfb")
  endif(DEFINED _RULE_MODULE_FILE_NAME)

  set(_BASE_COMPILER_FLAGS "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}")
  if(_RULE_INPUT_TYPE)
    list(APPEND _BASE_COMPILER_FLAGS "--iree-input-type=${_RULE_INPUT_TYPE}")
  endif()
  if(_RULE_TARGET_CPU_FEATURES)
    list(APPEND _BASE_COMPILER_FLAGS "--iree-llvmcpu-target-cpu-features=${_RULE_TARGET_CPU_FEATURES}")
  endif()

  if(_BYTECODE_MODULE_BUILD_ENABLED)
    iree_bytecode_module(
      NAME
        "${_MODULE_NAME}"
      MODULE_FILE_NAME
        "${_MODULE_FILE_NAME}"
      SRC
        "${_RULE_SRC}"
      FLAGS
        "${_BASE_COMPILER_FLAGS}"
        "${_RULE_COMPILER_FLAGS}"
      DEPENDS
        "${_RULE_DEPENDS}"
    )
  endif()

  set(_RUNNER_TARGET "iree-check-module")

  # Add a custom build target specifically for the test and its deps.
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_NAME}" ALL)
  if(_BYTECODE_MODULE_BUILD_ENABLED)
    add_dependencies(
      "${_NAME}"
      "${_NAME}_module"
      "${_RUNNER_TARGET}"
    )
  endif()
  add_dependencies(iree-test-deps "${_NAME}")

  if(_TEST_DEFINED)
    iree_native_test(
      NAME
        "${_RULE_NAME}"
      DRIVER
        "${_RULE_DRIVER}"
      SRC
        "${_RUNNER_TARGET}"
      ARGS
        "--module={{${_MODULE_FILE_NAME}}}"
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      TIMEOUT
        ${_RULE_TIMEOUT}
      DISABLED
        ${_TEST_DISABLED}
    )
  endif()
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
#       --iree-llvmcpu-target-cpu-features.
#   DEPENDS: Optional. Additional dependencies beyond SRC and the tools.
#   INPUT_TYPE: The value for the --iree-input-type= flag. Also disables tests
#       if no compiled support for that configuration.
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
    "NAME;TARGET_BACKEND;DRIVER;INPUT_TYPE"
    "SRCS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES;DEPENDS;TIMEOUT"
    ${ARGN}
  )

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
      INPUT_TYPE
        ${_RULE_INPUT_TYPE}
      RUNNER_ARGS
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      TARGET_CPU_FEATURES
        ${_RULE_TARGET_CPU_FEATURES}
      DEPENDS
        ${_RULE_DEPENDS}
      TIMEOUT
        ${_RULE_TIMEOUT}
    )
  endforeach()
endfunction()

function(iree_check_cpu_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET_BACKEND;DRIVER;INPUT_TYPE"
    "SRCS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES;DEPENDS;TIMEOUT"
    ${ARGN}
  )

  # TODO(scotttodd): could add `CONDITION` options to
  #   iree_check_single_backend_test_suite that would allow for
  #   tests being defined but marked DISABLED, rather than wrap these with
  #   all-or-nothing CMake `if(...)` code...?

  set(_BASE_LABELS ${_RULE_LABELS})
  list(APPEND _BASE_LABELS "notsan")
  iree_check_single_backend_test_suite(
    NAME ${_RULE_NAME}
    TARGET_BACKEND ${_RULE_TARGET_BACKEND}
    DRIVER ${_RULE_DRIVER}
    INPUT_TYPE ${_RULE_INPUT_TYPE}
    SRCS ${_RULE_SRCS}
    COMPILER_FLAGS ${_RULE_COMPILER_FLAGS}
    RUNNER_ARGS ${_RULE_RUNNER_ARGS}
    LABELS ${_BASE_LABELS}
    TARGET_CPU_FEATURES ${_RULE_TARGET_CPU_FEATURES}
    DEPENDS ${_RULE_DEPENDS}
    TIMEOUT ${_RULE_TIMEOUT}
  )

  if(IREE_ENABLE_ASAN)
    set(_ASAN_COMPILER_FLAGS ${_RULE_COMPILER_FLAGS})
    list(APPEND _ASAN_COMPILER_FLAGS "--iree-llvmcpu-link-embedded=false")
    list(APPEND _ASAN_COMPILER_FLAGS "--iree-llvmcpu-sanitize=address")
    set(_ASAN_LABELS ${_RULE_LABELS})
    list(APPEND _ASAN_LABELS "notsan")
    iree_check_single_backend_test_suite(
      NAME "${_RULE_NAME}_asan"
      TARGET_BACKEND ${_RULE_TARGET_BACKEND}
      DRIVER ${_RULE_DRIVER}
      INPUT_TYPE ${_RULE_INPUT_TYPE}
      SRCS ${_RULE_SRCS}
      COMPILER_FLAGS ${_ASAN_COMPILER_FLAGS}
      RUNNER_ARGS ${_RULE_RUNNER_ARGS}
      LABELS ${_ASAN_LABELS}
      TARGET_CPU_FEATURES ${_RULE_TARGET_CPU_FEATURES}
      DEPENDS ${_RULE_DEPENDS}
      TIMEOUT ${_RULE_TIMEOUT}
    )
  endif()

  if(IREE_ENABLE_TSAN)
    set(_TSAN_COMPILER_FLAGS ${_RULE_COMPILER_FLAGS})
    list(APPEND _TSAN_COMPILER_FLAGS "--iree-llvmcpu-link-embedded=false")
    list(APPEND _TSAN_COMPILER_FLAGS "--iree-llvmcpu-sanitize=thread")
    iree_check_single_backend_test_suite(
      NAME "${_RULE_NAME}_tsan"
      TARGET_BACKEND ${_RULE_TARGET_BACKEND}
      DRIVER ${_RULE_DRIVER}
      INPUT_TYPE ${_RULE_INPUT_TYPE}
      SRCS ${_RULE_SRCS}
      COMPILER_FLAGS ${_TSAN_COMPILER_FLAGS}
      RUNNER_ARGS ${_RULE_RUNNER_ARGS}
      LABELS ${_RULE_LABELS}
      TARGET_CPU_FEATURES ${_RULE_TARGET_CPU_FEATURES}
      DEPENDS ${_RULE_DEPENDS}
      TIMEOUT ${_RULE_TIMEOUT}
    )
  endif()
endfunction()

# Helper function parsing a string occurring as an entry in TARGET_CPU_FEATURES_VARIANTS.
#
# This function has 3 output-params: variables that it sets with PARENT_SCOPE:
# _ENABLED, _FEATURES_NAME, _FEATURES.
#
# "default" is handled specially. _ENABLED is always set to "TRUE" and
# _FEATURES_NAME and _FEATURES are set to
# the empty string.
#
# Other values are parsed as "arch:features_name:features". The `arch`
# component is  matched with `IREE_ARCH`, `_ENABLED` is set to "TRUE" if and
# only if they match. In that case:
#   `_FEATURES_NAME` is set to `features_name`.
#   `_FEATURES` is set to `features`.
#
# Examples:
#
# default:
#    _ENABLED="TRUE" unconditionally,
#        other output strings are "".
#
# aarch64:dotprod:+dotprod:
#    _ENABLED="TRUE" if the target architecture is aarch64, and in that case:
#        _FEATURES_NAME="dotprod".
#        _FEATURES="+dotprod".
function(parse_target_cpu_features_variant _VARIANT_STRING _ENABLED_VAR
             _FEATURES_NAME_VAR _FEATURES_VAR)
  set("${_ENABLED_VAR}" FALSE PARENT_SCOPE)
  set("${_FEATURES_NAME_VAR}" "" PARENT_SCOPE)
  set("${_FEATURES_VAR}" "" PARENT_SCOPE)
  if("${_VARIANT_STRING}" STREQUAL "default")
    set("${_ENABLED_VAR}" TRUE PARENT_SCOPE)
    return()
  endif()
  # Interpret _VARIANT_STRING as a CMake list (;-separated).
  string(REPLACE ":" ";" _COMPONENTS "${_VARIANT_STRING}")
  list(LENGTH _COMPONENTS _NUM_COMPONENTS)
  if(NOT _NUM_COMPONENTS EQUAL 3)
    message(SEND_ERROR "TARGET_CPU_FEATURES_VARIANTS should be of the form \
    \"arch:features_name:features\". Got: \"${_VARIANT_STRING}\"")
    return()
  endif()
  list(GET _COMPONENTS 0 _FILTER_ARCH)
  list(GET _COMPONENTS 1 _FEATURES_NAME)
  list(GET _COMPONENTS 2 _FEATURES)
  if(_FILTER_ARCH STREQUAL IREE_ARCH)
    set("${_ENABLED_VAR}" TRUE PARENT_SCOPE)
    set("${_FEATURES_NAME_VAR}" "${_FEATURES_NAME}" PARENT_SCOPE)
    set("${_FEATURES_VAR}" "${_FEATURES}" PARENT_SCOPE)
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
#   TARGET_CPU_FEATURES_VARIANTS: list of target cpu features variants. Each
#       entry is either "default" for the architecture defaults, or a colon-
#       separated triple "arch:name:cpu_features" where "arch" filters
#       for a target CPU architecture (in IREE_ARCH format), "name" is a
#       short name for the CPU features set (used to generate target names)
#       and cpu_features is a comma-separated list of LLVM target attributes
#       to enable. Example:
#         x86_64:avx2_fma:+avx,+avx2,+fma
#   INPUT_TYPE: The value for the --iree-input-type= flag. Also disables tests
#       if no compiled support for that configuration.
function(iree_check_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;INPUT_TYPE"
    "SRCS;TARGET_BACKENDS;DRIVERS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES_VARIANTS;TIMEOUT"
    ${ARGN}
  )

  iree_is_bytecode_module_test_excluded_by_labels(_EXCLUDED_BY_LABELS "${_RULE_LABELS}")
  if(_EXCLUDED_BY_LABELS)
    return()
  endif()

  if(_RULE_TARGET_CPU_FEATURES_VARIANTS)
    set(_TARGET_CPU_FEATURES_VARIANTS "${_RULE_TARGET_CPU_FEATURES_VARIANTS}")
  else()
    set(_TARGET_CPU_FEATURES_VARIANTS "default")
  endif()

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
    foreach(_VARIANT_STRING IN LISTS _TARGET_CPU_FEATURES_VARIANTS)
      parse_target_cpu_features_variant("${_VARIANT_STRING}"
        _ENABLED _TARGET_CPU_FEATURES_NAME _TARGET_CPU_FEATURES)
      if(NOT _ENABLED)
        # The current entry is disabled on the target CPU architecture.
        continue()
      endif()
      set(_TARGET_CPU_FEATURES_SUFFIX "")
      set(_LABELS "${_RULE_LABELS}")
      if (_TARGET_CPU_FEATURES_NAME)
        set(_TARGET_CPU_FEATURES_SUFFIX "_${_TARGET_CPU_FEATURES_NAME}")
        list(APPEND _LABELS "cpu_features=${_TARGET_CPU_FEATURES_NAME}")
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
          ${_LABELS}
        TARGET_CPU_FEATURES
          ${_TARGET_CPU_FEATURES}
        TIMEOUT
          ${_RULE_TIMEOUT}
        INPUT_TYPE
          ${_RULE_INPUT_TYPE}
      )
    endforeach()
  endforeach()
endfunction()
