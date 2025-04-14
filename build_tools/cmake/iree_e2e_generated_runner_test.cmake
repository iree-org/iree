# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_e2e_runner_test()
#
# Creates a test using a specified test runner program for the specified
# test files.
#
# Parameters:
#   NAME: Name of the target
#   TEST_TYPE: Type of test (Currently, matmul and conv2d are supported).
#   VARIANT_NAME: Variant name to suffix NAME with.
#       Will reuse the same TEST_TYPE/calls vmfb files.
#   TESTS_SRC: mlir source file with TEST_TYPE to be compiled to an IREE module.
#   TESTS_VMFB: specifies the path to use for the generated IREE module.
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
#   TEST_DEFINED: Whether to define a test target.
#   TEST_DISABLED: The test target will be skipped and its status will be
#       'Not Run'.
function(iree_e2e_runner_test)
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
    "NAME;TEST_TYPE;VARIANT_NAME;TESTS_SRC;TESTS_VMFB;CALLS_SRC;CALLS_VMFB;TRACE;TARGET_BACKEND;DRIVER;TEST_RUNNER;TEST_DEFINED;TEST_DISABLED"
    "COMPILER_FLAGS;RUNNER_ARGS;LABELS"
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

  if(NOT TARGET "${_NAME}_${_RULE_TEST_TYPE}_module")
    iree_bytecode_module(
      NAME
        "${_RULE_NAME}_${_RULE_TEST_TYPE}_module"
      MODULE_FILE_NAME
        "${_RULE_TESTS_VMFB}"
      SRC
        "${_RULE_TESTS_SRC}"
      FLAGS
        "${_BASE_COMPILER_FLAGS}"
        "${_RULE_COMPILER_FLAGS}"
    )
  endif()

  if(NOT TARGET "${_NAME}_calls_module")
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
  endif()

  # A target specifically for the test. We could combine this with the above,
  # but we want that one to get pulled into iree_bytecode_module.
  add_custom_target("${_NAME}${_RULE_VARIANT_NAME}" ALL)
  add_dependencies(
    "${_NAME}${_RULE_VARIANT_NAME}"
    "${_NAME}_${_RULE_TEST_TYPE}_module"
    "${_NAME}_calls_module"
    "${_RULE_TEST_RUNNER}"
  )

  add_dependencies(iree-test-deps "${_NAME}${_RULE_VARIANT_NAME}")

  if(_RULE_TEST_DEFINED)
    iree_native_test(
      NAME
        "${_RULE_NAME}${_RULE_VARIANT_NAME}"
      DRIVER
        "${_RULE_DRIVER}"
      SRC
        "${_RULE_TEST_RUNNER}"
      DATA
        ${_TESTS_VMFB}
        ${_CALLS_VMFB}
      ARGS
        "--module={{${_TESTS_VMFB}}}"
        "--module={{${_CALLS_VMFB}}}"
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      DISABLED
        ${_RULE_TEST_DISABLED}
    )
  endif()
endfunction()

# iree_single_backend_e2e_runner_test()
#
# Parameters:
#   NAME: Name of the target
#   TEST_TYPE: Type of test (Currently, matmul and conv are supported).
#   GENERATOR: Program (at the moment, must be Python3) to run to generate the
#       source file (and possibly a trace file and module path). It will be
#       invoked with the following standard flags, in addition to GENERATOR_ARGS:
#         --output_${TEST_TYPE}_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_${TEST_TYPE}.mlir
#         --output_calls_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_calls.mlir
#       and if COMPILER_FLAGS contains "--iree-llvmcpu-target-cpu_features=${TARGET_CPU_FEATURES}":
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
function(iree_single_backend_e2e_runner_test)
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
    "NAME;TEST_TYPE;GENERATOR;TARGET_BACKEND;DRIVER;TEST_RUNNER"
    "GENERATOR_ARGS;COMPILER_FLAGS;RUNNER_ARGS;LABELS"
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

  set(_TESTS_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_${_RULE_TEST_TYPE}.mlir")
  set(_CALLS_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_calls.mlir")
  set(_TESTS_VMFB "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_${_RULE_TEST_TYPE}.vmfb")
  set(_CALLS_VMFB "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_calls.vmfb")

  list(APPEND _GENERATOR_STANDARD_FLAGS "--output_${_RULE_TEST_TYPE}_mlir=${_TESTS_SRC}")
  list(APPEND _GENERATOR_STANDARD_FLAGS "--output_calls_mlir=${_CALLS_SRC}")
  foreach(_COMPILER_FLAG IN LISTS _RULE_COMPILER_FLAGS)
    set(_CPU_FEATURES_REGEX "^--iree-llvmcpu-target-cpu-features=")
    if (_COMPILER_FLAG MATCHES "${_CPU_FEATURES_REGEX}")
      string(REGEX REPLACE "${_CPU_FEATURES_REGEX}" "" _CPU_FEATURES "${_COMPILER_FLAG}")
      list(APPEND _GENERATOR_STANDARD_FLAGS "--requirements=${_CPU_FEATURES}")
    endif()
  endforeach()

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
      ${_TESTS_SRC}
      ${_CALLS_SRC}
    DEPENDS
      ${_RULE_GENERATOR}
  )

  add_custom_target(
    "${_NAME}_generated_files"
    DEPENDS
      ${_TESTS_SRC}
      ${_CALLS_SRC}
  )

  # When using the llvm-cpu backend, the runtime build config may need to
  # match the compiled executable config using (`--iree-llvmcpu-sanitize=`):
  #
  # | Runtime type         | Compatible with these executable types |
  # | -------------------- | -------------------------------------- |
  # | Base (no sanitizers) | Base, ASan                             |
  # | ASan                 | Base, ASan                             |
  # | TSan                 | TSan (ABI break)                       |

  # Define the regular test suite, unless the config is llvm-cpu + TSan.
  if(NOT _RULE_TARGET_BACKEND STREQUAL "llvm-cpu" OR NOT IREE_ENABLE_TSAN)
    iree_e2e_runner_test(
      NAME ${_RULE_NAME}
      TEST_TYPE ${_RULE_TEST_TYPE}
      VARIANT_NAME ""
      TESTS_SRC ${_TESTS_SRC}
      TESTS_VMFB ${_TESTS_VMFB}
      CALLS_SRC ${_CALLS_SRC}
      CALLS_VMFB ${_CALLS_VMFB}
      TEST_RUNNER ${_RULE_TEST_RUNNER}
      TARGET_BACKEND ${_RULE_TARGET_BACKEND}
      DRIVER ${_RULE_DRIVER}
      COMPILER_FLAGS ${_RULE_COMPILER_FLAGS}
      RUNNER_ARGS ${_RULE_RUNNER_ARGS}
      LABELS ${_RULE_LABELS}
      TEST_DEFINED ${_TEST_DEFINED}
      TEST_DISABLED ${_TEST_DISABLED}
    )
    # Note we are relying on the fact that the target created by
    # iree_e2e_runner_test is _NAME, even though we passed _RULE_NAME to it,
    # i.e. we are relying on the prefixing to be identical.
    add_dependencies("${_NAME}" "${_NAME}_generated_files")
  endif()

  # Define tests for AddressSanitizer (ASan) and ThreadSanitizer (TSan).
  # Normally test suites should do this sort of branching at the leaves rather
  # than modify the base CMake function directly, but sanitizers are applied
  # at the build system uniformly, so until we decouple the test suites from
  # source builds further this felt like a reasonable compromise.
  if(_RULE_TARGET_BACKEND STREQUAL "llvm-cpu")
    if(IREE_ENABLE_ASAN)
      set(_ASAN_COMPILER_FLAGS ${_RULE_COMPILER_FLAGS})
      list(APPEND _ASAN_COMPILER_FLAGS "--iree-llvmcpu-link-embedded=false")
      list(APPEND _ASAN_COMPILER_FLAGS "--iree-llvmcpu-sanitize=address")
      iree_e2e_runner_test(
        NAME ${_RULE_NAME}
        TEST_TYPE ${_RULE_TEST_TYPE}
        VARIANT_NAME "_asan"
        TESTS_SRC ${_TESTS_SRC}
        TESTS_VMFB ${_TESTS_VMFB}
        CALLS_SRC ${_CALLS_SRC}
        CALLS_VMFB ${_CALLS_VMFB}
        TEST_RUNNER ${_RULE_TEST_RUNNER}
        TARGET_BACKEND ${_RULE_TARGET_BACKEND}
        DRIVER ${_RULE_DRIVER}
        COMPILER_FLAGS ${_ASAN_COMPILER_FLAGS}
        RUNNER_ARGS ${_RULE_RUNNER_ARGS}
        LABELS ${_RULE_LABELS}
        TEST_DEFINED ${_TEST_DEFINED}
        TEST_DISABLED ${_TEST_DISABLED}
      )
      # Note we are relying on the fact that the target created by
      # iree_e2e_runner_test is _NAME, even though we passed _RULE_NAME to it,
      # i.e. we are relying on the prefixing to be identical.
      add_dependencies("${_NAME}_asan" "${_NAME}_generated_files")
    endif()

    if(IREE_ENABLE_TSAN)
      set(_TSAN_COMPILER_FLAGS ${_RULE_COMPILER_FLAGS})
      list(APPEND _TSAN_COMPILER_FLAGS "--iree-llvmcpu-link-embedded=false")
      list(APPEND _TSAN_COMPILER_FLAGS "--iree-llvmcpu-sanitize=thread")
      iree_e2e_runner_test(
        NAME ${_RULE_NAME}
        VARIANT_NAME "_tsan"
        TESTS_SRC ${_TESTS_SRC}
        TESTS_VMFB ${_TESTS_VMFB}
        CALLS_SRC ${_CALLS_SRC}
        CALLS_VMFB ${_CALLS_VMFB}
        TEST_RUNNER ${_RULE_TEST_RUNNER}
        TARGET_BACKEND ${_RULE_TARGET_BACKEND}
        DRIVER ${_RULE_DRIVER}
        COMPILER_FLAGS ${_TSAN_COMPILER_FLAGS}
        RUNNER_ARGS ${_RULE_RUNNER_ARGS}
        LABELS ${_RULE_LABELS}
        TEST_DEFINED ${_TEST_DEFINED}
        TEST_DISABLED ${_TEST_DISABLED}
      )
      # Note we are relying on the fact that the target created by
      # iree_e2e_runner_test is _NAME, even though we passed _RULE_NAME to it,
      # i.e. we are relying on the prefixing to be identical.
      add_dependencies("${_NAME}_tsan" "${_NAME}_generated_files")
    endif()
  endif()
endfunction()


# iree_generated_e2e_runner_test()
#
# Creates a set of iree_single_backend_e2e_runner_test's differing
# by target backend and driver.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source and backend/driver pair.
# Parameters:
#   NAME: Name of the target
#   TEST_TYPE: Type of test (Currently, matmul and conv are supported).
#   GENERATOR: Program (at the moment, must be Python3) to run to generate the
#       source file (and possibly a trace file and module path). It will be
#       invoked with the following standard flags, in addition to GENERATOR_ARGS:
#         --output_${TEST_TYPE}_mlir=${CMAKE_CURRENT_BINARY_DIR}/name_${TEST_TYPE}.mlir
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
#   TARGET_CPU_FEATURES_VARIANTS: list of target cpu features variants. Each
#       entry is either "generic" for the architecture defaults, or "host" for
#       the host CPU, or a colon-separated triple "arch:name:cpu_features" where "arch" filters
#       for a target CPU architecture (in IREE_ARCH format), "name" is a
#       short name for the CPU features set (used to generate target names)
#       and cpu_features is a comma-separated list of LLVM target attributes
#       to enable. Example:
#         x86_64:avx2_fma:+avx,+avx2,+fma
function(iree_generated_e2e_runner_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TEST_TYPE;GENERATOR;TEST_RUNNER"
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
    set(_TARGET_CPU_FEATURES_VARIANTS "generic")
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
      if(_TARGET_BACKEND STREQUAL "llvm-cpu")
        parse_target_cpu_features_variant("${_VARIANT_STRING}"
          _ENABLED _TARGET_CPU_FEATURES_NAME _VARIANT_COMPILER_FLAGS)
        if(NOT _ENABLED)
          # The current entry is disabled on the target CPU architecture.
          continue()
        endif()
      endif()
      set(_TARGET_CPU_FEATURES_SUFFIX "")
      set(_LABELS "${_RULE_LABELS}")
      if (_TARGET_CPU_FEATURES_NAME)
        set(_TARGET_CPU_FEATURES_SUFFIX "_${_TARGET_CPU_FEATURES_NAME}")
        list(APPEND _LABELS "cpu_features=${_TARGET_CPU_FEATURES_NAME}")
      endif()
      iree_single_backend_e2e_runner_test(
        NAME
          "${_RULE_NAME}_${_TARGET_BACKEND}_${_DRIVER}${_TARGET_CPU_FEATURES_SUFFIX}"
        TEST_TYPE
          ${_RULE_TEST_TYPE}
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
          ${_VARIANT_COMPILER_FLAGS}
        RUNNER_ARGS
          ${_RULE_RUNNER_ARGS}
        LABELS
          ${_LABELS}
      )
    endforeach()
  endforeach()
endfunction()
