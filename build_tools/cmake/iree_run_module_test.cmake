# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Utility function to return the platform name in crosscompile.
# List of CMAKE_SYSTEM_NAME values:
#   https://gitlab.kitware.com/cmake/cmake/-/issues/21489#note_1077167
# Examples: arm_64-Android arm_32-Linux x86_64-Windows arm_64-iOS riscv_64-Linux
function(iree_get_platform PLATFORM)
  set(${PLATFORM} "${IREE_ARCH}-${CMAKE_SYSTEM_NAME}" PARENT_SCOPE)
endfunction()

# iree_run_module_test()
#
# Creates a test using iree-run-module to run an IREE module (vmfb).
#
# The function is unimplemented in Bazel because it is not used there.
#
# Parameters:
#   NAME: Name of the target
#   MODULE_SRC: IREE module (vmfb) file.
#   DRIVER: Driver to run the module with.
#   RUNNER_ARGS: additional args to pass to iree-run-module. The driver
#       and input file are passed automatically.
#   EXPECTED_OUTPUT: A string representing the expected output from executing
#       the module in the format accepted by `iree-run-module` or a file
#       containing the same.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   XFAIL_PLATFORMS: List of platforms (see iree_get_platform) for which the
#       test is expected to fail. The test pass/fail status is inverted.
#   UNSUPPORTED_PLATFORMS: List of platforms (see iree_get_platform) for which
#       the test is skipped entirely.
#   DEPS: (Optional) List of targets to build the test artifacts.
#   TIMEOUT: (optional) Test timeout.
#
# Examples:
#
# iree_run_module_test(
#   NAME
#     iree_run_module_correctness_test
#   MODULE_SRC
#     "iree_run_module_bytecode_module_llvm_cpu.vmfb"
#   DRIVER
#     "local-sync"
#   RUNNER_ARGS
#     "--function=abs"
#     "--input=f32=-10"
#   EXPECTED_OUTPUT
#     "f32=10"
# )
#
# iree_run_module_test(
#   NAME
#     mobilenet_v1_fp32_correctness_test
#   MODULE_SRC
#     "mobilenet_v1_fp32.vmfb"
#   DRIVER
#     "local-sync"
#   RUNNER_ARGS
#     "--function=main"
#     "--input=1x224x224x3xf32=0"
#   EXPECTED_OUTPUT
#     "mobilenet_v1_fp32_expected_output.txt"
#   UNSUPPORTED_PLATFORMS
#     "arm_64-Android"
#     "riscv_32-Linux"
# )

function(iree_run_module_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;MODULE_SRC;DRIVER;EXPECTED_OUTPUT;TIMEOUT"
    "RUNNER_ARGS;LABELS;XFAIL_PLATFORMS;UNSUPPORTED_PLATFORMS;DEPS"
    ${ARGN}
  )

  iree_get_platform(_PLATFORM)
  if(_PLATFORM IN_LIST _RULE_UNSUPPORTED_PLATFORMS)
    return()
  endif()

  if(NOT DEFINED _RULE_DRIVER)
    message(SEND_ERROR "The DRIVER argument is required.")
  endif()

  iree_package_path(_PACKAGE_PATH)

  # All the file paths referred in the _RUNNER_FILE_ARGS are absolute paths and
  # the portability is handled by `iree_native_test`.
  list(APPEND _RUNNER_FILE_ARGS "--module={{${_RULE_MODULE_SRC}}}")

  # A target specifically for the test.
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET_NAME}" ALL)

  if(_RULE_EXPECTED_OUTPUT)
    # this may be a file or a literal output. In the latter case, the
    # extension variable will be empty.
    cmake_path(GET _RULE_EXPECTED_OUTPUT EXTENSION LAST_ONLY _OUTPUT_FILE_TYPE)
    if(NOT _OUTPUT_FILE_TYPE)  # The expected output is listed in the field.
      list(APPEND _RULE_RUNNER_ARGS "--expected_output=\"${_RULE_EXPECTED_OUTPUT}\"")
    elseif(_OUTPUT_FILE_TYPE STREQUAL ".txt")
      file(REAL_PATH "${_RULE_EXPECTED_OUTPUT}" _OUTPUT_FILE_ABS_PATH)
      # Process the text input to remove the line breaks.
      file(READ "${_OUTPUT_FILE_ABS_PATH}" _EXPECTED_OUTPUT)
      string(REPLACE "\n" " " _EXPECTED_OUTPUT_STR "${_EXPECTED_OUTPUT}")
      set(_EXPECTED_OUTPUT_STR "--expected_output=\"${_EXPECTED_OUTPUT_STR}\"")
      list(APPEND _RULE_RUNNER_ARGS ${_EXPECTED_OUTPUT_STR})
    elseif(_OUTPUT_FILE_TYPE STREQUAL ".npy")
      # Large npy files are not stored in the codebase. Need to download them
      # from GCS iree-model-artifacts first and store them in the following possible
      # paths.
      find_file(_OUTPUT_FILE_ABS_PATH
        NAME
          "${_RULE_EXPECTED_OUTPUT}"
        PATHS
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${IREE_E2E_TEST_ARTIFACTS_DIR}"
        NO_CACHE
        NO_DEFAULT_PATH
      )
      # If the expected output npy file is not found, try to fetch it.
      if(NOT _OUTPUT_FILE_ABS_PATH)
        set(_FETCH_NAME "model-expected-output-${_RULE_NAME}")
        iree_fetch_artifact(
          NAME "${_FETCH_NAME}"
          SOURCE_URL "https://storage.googleapis.com/iree-model-artifacts/${_RULE_EXPECTED_OUTPUT}"
          OUTPUT "${IREE_E2E_TEST_ARTIFACTS_DIR}/${_RULE_EXPECTED_OUTPUT}"
        )
        add_dependencies(${_TARGET_NAME}
          "${_PACKAGE_NAME}_${_FETCH_NAME}"
        )
      else()
        list(APPEND _RUNNER_FILE_ARGS
          "--expected_output=@{{${_OUTPUT_FILE_ABS_PATH}}}")
      endif()
    else()
      message(SEND_ERROR "Unsupported expected output file type: ${_RULE_EXPECTED_OUTPUT}")
    endif(NOT _OUTPUT_FILE_TYPE)
  endif(_RULE_EXPECTED_OUTPUT)

  # Dump the flags into a flag file to avoid CMake's naive handling of spaces
  # in expected output. `--module` is coded separatedly to make it portable.
  if(_RULE_RUNNER_ARGS)
    # Write each argument in a new line.
    string(REPLACE ";" "\n" _OUTPUT_FLAGS "${_RULE_RUNNER_ARGS}")
    file(CONFIGURE
      OUTPUT
        "${_RULE_NAME}_flagfile"
      CONTENT
        "${_OUTPUT_FLAGS}"
    )
    list(APPEND _RUNNER_FILE_ARGS
      "--flagfile={{${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_flagfile}}")
  endif()

  # Set expect failure cases.
  set(_TEST_XFAIL FALSE)
  if(_PLATFORM IN_LIST _RULE_XFAIL_PLATFORMS OR
     _RULE_XFAIL_PLATFORMS STREQUAL "all")
    set(_TEST_XFAIL TRUE)
  endif()

  set(_RUNNER_TARGET "iree-run-module")

  iree_native_test(
    NAME
      "${_RULE_NAME}"
    DRIVER
      "${_RULE_DRIVER}"
    SRC
      "${_RUNNER_TARGET}"
    ARGS
      ${_RUNNER_FILE_ARGS}
    WILL_FAIL
      ${_TEST_XFAIL}
    LABELS
      "test-type=run-module-test"
      ${_RULE_LABELS}
    TIMEOUT
      ${_RULE_TIMEOUT}
  )

  if(_RULE_DEPS)
    add_dependencies(${_TARGET_NAME}
      ${_RULE_DEPS}
    )
  endif()

  add_dependencies(iree-test-deps "${_TARGET_NAME}")
  add_dependencies(iree-run-module-test-deps "${_TARGET_NAME}")
endfunction()

# iree_benchmark_suite_module_test()
#
# The function is unimplemented in Bazel because it is not used there.
#
# Creates a test using iree-run-module to run a benchmark suite module.
#
# Parameters:
#   NAME: Name of the target
#   DRIVER: Driver to run the module with.
#   MODULES: Platform-module path list (relative to IREE_E2E_TEST_ARTIFACTS_DIR)
#       for the supported platforms. Each item is in the format:
#       "${PLATFORM}=${MODULE_PATH}"
#   RUNNER_ARGS: additional args to pass to iree-run-module. The driver
#       and input file are passed automatically.
#   EXPECTED_OUTPUT: A file of expected output to compare with the output from
#       iree-run-module
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   XFAIL_PLATFORMS: List of platforms (see iree_get_platform) for which the
#       test is expected to fail. The test pass/fail status is inverted.
#   UNSUPPORTED_PLATFORMS: List of platforms (see iree_get_platform) for which
#       the test is skipped entirely.
#   TIMEOUT: (optional) Test timeout.
#
# Example:
#
# iree_benchmark_suite_module_test(
#   NAME
#     mobilenet_v1_fp32_correctness_test
#   DRIVER
#     "local-sync"
#   MODULES
#     "riscv32-Linux=iree_module_EfficientNet_int8_riscv32/module.vmfb"
#     "x86_64=iree_module_EfficientNet_int8_x86_64/module.vmfb"
#   RUNNER_ARGS
#     "--function=main"
#     "--input=1x224x224x3xf32=0"
#   EXPECTED_OUTPUT
#     "mobilenet_v1_fp32_expected_output.txt"
# )
function(iree_benchmark_suite_module_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;DRIVER;EXPECTED_OUTPUT;TIMEOUT"
    "MODULES;RUNNER_ARGS;LABELS;XFAIL_PLATFORMS;UNSUPPORTED_PLATFORMS"
    ${ARGN}
  )

  # Benchmark artifacts needs to be stored at the location of
  # `IREE_E2E_TEST_ARTIFACTS_DIR` or the test target is bypassed.
  if("${IREE_E2E_TEST_ARTIFACTS_DIR}" STREQUAL "")
    return()
  endif()

  iree_get_platform(_PLATFORM)

  foreach(_PLATFORM_MODULE_PAIR IN LISTS _RULE_MODULES)
    string(REGEX MATCH "^${_PLATFORM}=(.+)$" _MATCHED_PAIR "${_PLATFORM_MODULE_PAIR}")
    if (_MATCHED_PAIR)
      set(_MODULE_PATH "${CMAKE_MATCH_1}")
      break()
    endif()
  endforeach()

  # Platform is not in the supported module list, skip the test.
  if (NOT _MODULE_PATH)
    return()
  endif()

  # Build module locally if IREE_BUILD_E2E_TEST_ARTIFACTS is set.
  if (IREE_BUILD_E2E_TEST_ARTIFACTS)
    # The module path follows the format:
    # iree_module_abc/module.vmfb
    # Ane the corresponding build target from e2e test artifacts is:
    # iree-module-abc
    cmake_path(GET _MODULE_PATH PARENT_PATH _MODULE_DIR)
    string(REGEX REPLACE "^iree_module_" "iree-module-" _DEP_NAME "${_MODULE_DIR}")
    # Append the prefix of package name.
    set(_MODULE_DEP_TARGET "iree_tests_e2e_test_artifacts_${_DEP_NAME}")
  else()
    set(_MODULE_DEP_TARGET "")
  endif()

  iree_run_module_test(
    NAME
      "${_RULE_NAME}"
    MODULE_SRC
      "${IREE_E2E_TEST_ARTIFACTS_DIR}/${_MODULE_PATH}"
    DRIVER
      "${_RULE_DRIVER}"
    EXPECTED_OUTPUT
      "${_RULE_EXPECTED_OUTPUT}"
    RUNNER_ARGS
      ${_RULE_RUNNER_ARGS}
    XFAIL_PLATFORMS
      ${_RULE_XFAIL_PLATFORMS}
    UNSUPPORTED_PLATFORMS
      ${_RULE_UNSUPPORTED_PLATFORMS}
    LABELS
      ${_RULE_LABELS}
    TIMEOUT
      ${_RULE_TIMEOUT}
    DEPS
      "${_MODULE_DEP_TARGET}"
  )
endfunction()
