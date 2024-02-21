# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_e2e_matmul_test()
#
# Creates a test using a specified test runner program for the specified
# matmul test files.
#
# Parameters:
#   NAME: Name of the target
#   MATMULS_SRC: mlir source file with matmuls to be compiled to an IREE module.
#   MATMULS_VMFB: specifies the path to use for the generated IREE module.
#   CALLS_SRC: mlir source file with calls to be compiled to an IREE module.
#   CALLS_VMFB: specifies the path to use for the generated IREE module.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode output
#       format and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to the trace-runner program. The driver
#       and input file flags are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   TEST_RUNNER: trace-runner program to run.
#   TARGET_CPU_FEATURES: If specified, a string passed as argument to
#       --iree-llvmcpu-target-cpu-features.
#   TEST_DEFINED: Whether to define a test target.
#   TEST_DISABLED: The test target will be skipped and its status will be
#       'Not Run'.
function(iree_e2e_matmul_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # See comment in iree_check_test about this condition.
  if(NOT IREE_BUILD_COMPILER AND NOT IREE_HOST_BIN_DIR)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;MATMULS_SRC;MATMULS_VMFB;CALLS_SRC;CALLS_VMFB;TRACE;TARGET_BACKEND;DRIVER;TEST_RUNNER;TEST_DEFINED;TEST_DISABLED"
    "COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES"
    ${ARGN}
  )

  iree_is_bytecode_module_test_excluded_by_labels(_EXCLUDED_BY_LABELS "${_RULE_LABELS}")
  if(_EXCLUDED_BY_LABELS)
    return()
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_BASE_COMPILER_FLAGS
    "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}"
  )
  if (_RULE_TARGET_CPU_FEATURES)
    list(APPEND _BASE_COMPILER_FLAGS "--iree-llvmcpu-target-cpu-features=${_RULE_TARGET_CPU_FEATURES}")
  endif()

  iree_bytecode_module(
    NAME
      "${_RULE_NAME}_matmuls_module"
    MODULE_FILE_NAME
      "${_RULE_MATMULS_VMFB}"
    SRC
      "${_RULE_MATMULS_SRC}"
    FLAGS
      "${_BASE_COMPILER_FLAGS}"
      "${_RULE_COMPILER_FLAGS}"
  )

  iree_bytecode_module(
    NAME
      "${_RULE_NAME}_calls_module"
    MODULE_FILE_NAME
      "${_RULE_CALLS_VMFB}"
    SRC
      "${_RULE_CALLS_SRC}"
    FLAGS
      "${_BASE_COMPILER_FLAGS}"
      "${_RULE_COMPILER_FLAGS}"
  )

  # A target specifically for the test. We could combine this with the above,
  # but we want that one to get pulled into iree_bytecode_module.
  add_custom_target("${_NAME}" ALL)
  add_dependencies(
    "${_NAME}"
    "${_NAME}_matmuls_module"
    "${_NAME}_calls_module"
    "${_RULE_TEST_RUNNER}"
  )

  add_dependencies(iree-test-deps "${_NAME}")

  if(_RULE_TEST_DEFINED)
    iree_native_test(
      NAME
        "${_RULE_NAME}"
      DRIVER
        "${_RULE_DRIVER}"
      SRC
        "${_RULE_TEST_RUNNER}"
      DATA
        ${_MATMULS_VMFB}
        ${_CALLS_VMFB}
      ARGS
        "--module={{${_MATMULS_VMFB}}}"
        "--module={{${_CALLS_VMFB}}}"
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      DISABLED
        ${_RULE_TEST_DISABLED}
    )
  endif()
endfunction()

# iree_single_backend_e2e_matmul_test()
#
# Parameters:
#   NAME: Name of the target
#   GENERATOR: Program (at the moment, must be Python3) to run to generate the
#       source file (and possibly a trace file and module path). It will be
#       invoked with the following standard flags, in addition to GENERATOR_ARGS:
#         --output_matmuls_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_matmuls.mlir
#         --output_calls_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_calls.mlir
#       and if TARGET_CPU_FEATURES is not empty:
#         --requirements=${TARGET_CPU_FEATURES}
#   GENERATOR_ARGS: additional args to pass to the generator program.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode output
#       format and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to the trace-runner program. The driver
#       and input file flags are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   TEST_RUNNER: trace-runner program to run.
#   TARGET_CPU_FEATURES: If specified, a string passed as argument to
#       --iree-llvmcpu-target-cpu-features.
function(iree_single_backend_e2e_matmul_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Copied from iree_check_test. Refer to the comment there.
  if(NOT IREE_BUILD_COMPILER AND NOT IREE_HOST_BIN_DIR)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;TARGET_BACKEND;DRIVER;TEST_RUNNER"
    "GENERATOR_ARGS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES"
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

  # 2. Check target backend availability.
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
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_MATMULS_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_matmuls.mlir")
  set(_CALLS_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_calls.mlir")
  set(_MATMULS_VMFB "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_matmuls.vmfb")
  set(_CALLS_VMFB "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_calls.vmfb")

  list(APPEND _GENERATOR_STANDARD_FLAGS "--output_matmuls_mlir=${_MATMULS_SRC}")
  list(APPEND _GENERATOR_STANDARD_FLAGS "--output_calls_mlir=${_CALLS_SRC}")
  if(_RULE_TARGET_CPU_FEATURES)
    list(APPEND _GENERATOR_STANDARD_FLAGS "--requirements=${_RULE_TARGET_CPU_FEATURES}")
  endif()

  if(NOT _BYTECODE_MODULE_BUILD_ENABLED)
    return()
  endif()

  add_custom_command(
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_GENERATOR}"
      ${_GENERATOR_STANDARD_FLAGS}
      ${_RULE_GENERATOR_ARGS}
    OUTPUT
      ${_MATMULS_SRC}
      ${_CALLS_SRC}
    DEPENDS
      ${_RULE_GENERATOR}
  )

  add_custom_target(
    "${_NAME}_generated_files"
    DEPENDS
      ${_MATMULS_SRC}
      ${_CALLS_SRC}
  )

  iree_e2e_matmul_test(
    NAME
      "${_RULE_NAME}"
    MATMULS_SRC
      "${_MATMULS_SRC}"
    MATMULS_VMFB
      "${_MATMULS_VMFB}"
    CALLS_SRC
      "${_CALLS_SRC}"
    CALLS_VMFB
      "${_CALLS_VMFB}"
    TEST_RUNNER
      "${_RULE_TEST_RUNNER}"
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
    TEST_DEFINED
      ${_TEST_DEFINED}
    TEST_DISABLED
      ${_TEST_DISABLED}
  )

  # Note we are relying on the fact that the target created by
  # iree_e2e_matmul_test is _NAME, even though we passed _RULE_NAME to it,
  # i.e. we are relying on the prefixing to be identical.
  add_dependencies("${_NAME}" "${_NAME}_generated_files")
endfunction()


# iree_generated_e2e_matmul_test()
#
# Creates a set of iree_single_backend_e2e_matmul_test's differing
# by target backend and driver.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source and backend/driver pair.
# Parameters:
#   NAME: Name of the target
#   GENERATOR: Program (at the moment, must be Python3) to run to generate the
#       source file (and possibly a trace file and module path). It will be
#       invoked with the following standard flags, in addition to GENERATOR_ARGS:
#         --output_matmuls_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_matmuls.mlir
#         --output_calls_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_calls.mlir
#   GENERATOR_ARGS: additional args to pass to the generator program.
#   TARGET_BACKENDS: backends to compile the module for. These form pairs with
#       the DRIVERS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   DRIVERS: drivers to run the module with. These form pairs with the
#       TARGET_BACKENDS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode output
#       format and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to the trace-runner program. The driver
#       and input file flags are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   TEST_RUNNER: trace-runner program to run.
#   TARGET_CPU_FEATURES_VARIANTS:list of target cpu features variants. Each
#       entry is either "default" for the architecture defaults, or a colon-
#       separated triple "arch:name:cpu_features" where "arch" filters
#       for a target CPU architecture (in IREE_ARCH format), "name" is a
#       short name for the CPU features set (used to generate target names)
#       and cpu_features is a comma-separated list of LLVM target attributes
#       to enable. Example:
#         x86_64:avx2_fma:+avx,+avx2,+fma
function(iree_generated_e2e_matmul_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;TEST_RUNNER"
    "TARGET_BACKENDS;DRIVERS;GENERATOR_ARGS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;TARGET_CPU_FEATURES_VARIANTS"
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
      iree_single_backend_e2e_matmul_test(
        NAME
          "${_RULE_NAME}_${_TARGET_BACKEND}_${_DRIVER}${_TARGET_CPU_FEATURES_SUFFIX}"
        GENERATOR
          ${_RULE_GENERATOR}
        GENERATOR_ARGS
          ${_RULE_GENERATOR_ARGS}
        TEST_RUNNER
          ${_RULE_TEST_RUNNER}
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
      )
    endforeach()
  endforeach()
endfunction()
