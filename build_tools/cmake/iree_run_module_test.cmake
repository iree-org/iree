# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Utility function to return the platform name in crosscompile.
function(iree_get_platform PLATFORM)
  if(ANDROID AND CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(_PLATFORM "android-arm64-v8a")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_PLATFORM "x86_64")
  else()
    set(_PLATFORM "${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}")
  endif()
  set(${PLATFORM} "${_PLATFORM}" PARENT_SCOPE)
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
#   XFAIL_PLATFORMS: List of platforms (all, x86_64, android-arm64-v8a,
#       riscv64-Linux, riscv32-Linux) for which the test is expected to fail
#       e.g. due to issues with the upstream llvm backend. The target will be
#       run, but its pass/fail status will be inverted.
#   UNSUPPORTED_PLATFORMS: List of platforms (x86_64, android-arm64-v8a,
#       riscv64-Linux, riscv32-Linux) not supported by the test target. The
#       target will be skipped entirely.
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
#     "--entry_function=abs"
#     "--function_input=f32=-10"
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
#     "--entry_function=main"
#     "--function_input=1x224x224x3xf32=0"
#   EXPECTED_OUTPUT
#     "mobilenet_v1_fp32_expected_output.txt"
#   UNSUPPORTED_PLATFORMS
#     "android-arm64-v8a"
#     "riscv32-Linux"
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

  file(RELATIVE_PATH _SRC "${CMAKE_CURRENT_BINARY_DIR}" "${_RULE_MODULE_SRC}")

  if(_RULE_EXPECTED_OUTPUT)
    # this may be a file or a literal output. In the latter case, the
    # extension variable will be empty.
    cmake_path(GET _RULE_EXPECTED_OUTPUT EXTENSION LAST_ONLY _OUTPUT_FILE_TYPE)
    if(NOT _OUTPUT_FILE_TYPE)  # The expected output is listed in the field.
      list(APPEND _RULE_RUNNER_ARGS "--expected_output=\"${_RULE_EXPECTED_OUTPUT}\"")
    elseif(_OUTPUT_FILE_TYPE STREQUAL ".txt")
      file(REAL_PATH "${_RULE_EXPECTED_OUTPUT}" _OUTPUT_FILE_SRC)
      # Process the text input to remove the line breaks.
      file(READ "${_OUTPUT_FILE_SRC}" _EXPECTED_OUTPUT)
      string(REPLACE "\n" " " _EXPECTED_OUTPUT_STR "${_EXPECTED_OUTPUT}")
      set(_EXPECTED_OUTPUT_STR "--expected_output=\"${_EXPECTED_OUTPUT_STR}\"")
      list(APPEND _RULE_RUNNER_ARGS ${_EXPECTED_OUTPUT_STR})
    elseif(_OUTPUT_FILE_TYPE STREQUAL ".npy")
      # Large npy files are not stored in the codebase. Need to download them
      # from GCS iree-model-artifacts first and store them in the following possible
      # paths.
      find_file(_OUTPUT_FILE_SRC_PATH
        NAME
          "${_RULE_EXPECTED_OUTPUT}"
        PATHS
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${IREE_BENCHMARK_SUITE_DIR}"
        NO_CACHE
        NO_DEFAULT_PATH
      )
      # If the expected output npy file is not found (the large file is not
      # loaded from GCS to `IREE_BENCHMARK_SUITE_DIR` benchmark suite test),
      # report error.
      if(NOT _OUTPUT_FILE_SRC_PATH)
        message(SEND_ERROR "${_RULE_EXPECTED_OUTPUT} is not found in\n\
          ${CMAKE_CURRENT_SOURCE_DIR}\n\
          ${CMAKE_CURRENT_BINARY_DIR}\n\
          ${IREE_BENCHMARK_SUITE_DIR}\n\
          Please check if you need to download it first.")
      else()
        file(RELATIVE_PATH _OUTPUT_FILE_SRC
          "${CMAKE_CURRENT_BINARY_DIR}" "${_OUTPUT_FILE_SRC_PATH}")
        list(APPEND _RULE_RUNNER_ARGS "--expected_output=@${_OUTPUT_FILE_SRC}")
      endif()
    else()
      message(SEND_ERROR "Unsupported expected output file type: ${_RULE_EXPECTED_OUTPUT}")
    endif(NOT _OUTPUT_FILE_TYPE)
  endif(_RULE_EXPECTED_OUTPUT)

  # Dump the flags into a flag file to avoid CMake's naive handling of spaces
  # in expected output. `--module_file` is coded separatedly to make it portable.
  if(_RULE_RUNNER_ARGS)
    set(_OUTPUT_FLAGFILE "${_RULE_NAME}_flagfile")
    # Write each argument in a new line.
    string(REPLACE ";" "\n" _OUTPUT_FLAGS "${_RULE_RUNNER_ARGS}")
    file(CONFIGURE
      OUTPUT
        "${_OUTPUT_FLAGFILE}"
      CONTENT
        "${_OUTPUT_FLAGS}"
    )
  endif()

  # A target specifically for the test.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_custom_target("${_NAME}" ALL)

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
      "--module_file=${_SRC}"
      "--flagfile=${_OUTPUT_FLAGFILE}"
    WILL_FAIL
      ${_TEST_XFAIL}
    LABELS
      ${_RULE_LABELS}
    TIMEOUT
      ${_RULE_TIMEOUT}
  )

  if(_RULE_DEPS)
    add_dependencies(${_NAME}
      ${_RULE_DEPS}
    )
  endif()

  add_dependencies(iree-test-deps "${_NAME}")
endfunction()

# iree_benchmark_suite_module_test()
#
# Creates a test using iree-run-module to run a benchmark suite module.
#
# The function is unimplemented in Bazel because it is not used there.
#
# Parameters:
#   NAME: Name of the target
#   BENCHMARK_MODULE_SRC: IREE module flagfile path built from benchmark_suite.
#       The flagfile for different compile configurations are stored in the
#       subdirectories.
#   DRIVER: Driver to run the module with.
#   RUNNER_ARGS: additional args to pass to iree-run-module. The driver
#       and input file are passed automatically.
#   EXPECTED_OUTPUT: A file of expected output to compare with the output from
#       iree-run-module
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   XFAIL_PLATFORMS: List of platforms (all, x86_64, android-arm64-v8a,
#       riscv64-Linux, riscv32-Linux) for which the test is expected to fail
#       e.g. due to issues with the upstream llvm backend. The target will be
#       run, but its pass/fail status will be inverted.
#   UNSUPPORTED_PLATFORMS: List of platforms (x86_64, android-arm64-v8a,
#       riscv64-Linux, riscv32-Linux) not supported by the test target. The
#       target will be skipped entirely.
#   TIMEOUT: (optional) Test timeout.
#
# Example:
#
# iree_benchmark_suite_module_test(
#   NAME
#     mobilenet_v1_fp32_correctness_test
#   BENCHMARK_MODULE_SRC
#     "TFLite/MobileNetV1-fp32,imagenet"
#   DRIVER
#     "local-sync"
#   RUNNER_ARGS
#     "--entry_function=main"
#     "--function_input=1x224x224x3xf32=0"
#   EXPECTED_OUTPUT
#     "mobilenet_v1_fp32_expected_output.txt"
#   UNSUPPORTED_PLATFORMS
#     "android-arm64-v8a"
#     "riscv32-Linux"
# )
function(iree_benchmark_suite_module_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;BENCHMARK_MODULE_SRC;DRIVER;EXPECTED_OUTPUT;TIMEOUT"
    "RUNNER_ARGS;LABELS;XFAIL_PLATFORMS;UNSUPPORTED_PLATFORMS"
    ${ARGN}
  )

  # Benchmark suite needs to be stored at the location of
  # `IREE_BENCHMARK_SUITE_DIR` or the test target is bypassed.
  if(NOT DEFINED IREE_BENCHMARK_SUITE_DIR)
    return()
  endif()

  iree_get_platform(_PLATFORM)
  if(_PLATFORM IN_LIST _RULE_UNSUPPORTED_PLATFORMS)
    return()
  endif()

  set(_MODULE_FLAG_DIR "${IREE_BENCHMARK_SUITE_DIR}/${_RULE_BENCHMARK_MODULE_SRC}/")
  # Find the platform specific module flag file with matching path name.
  # TODO(#10391): Update this logic with the new benchmark framework.
  if(_PLATFORM STREQUAL "riscv64-Linux")
    set(_FLAGFILE_HINT_PATH "${_MODULE_FLAG_DIR}/iree-llvm-cpu*RV64*__full-inference,default-flags/flagfile")
  elseif(_PLATFORM STREQUAL "riscv32-Linux")
    set(_FLAGFILE_HINT_PATH "${_MODULE_FLAG_DIR}/iree-llvm-cpu*RV32*__full-inference,default-flags/flagfile")
  elseif(_PLATFORM STREQUAL "android-arm64-v8a")
    set(_FLAGFILE_HINT_PATH "${_MODULE_FLAG_DIR}/iree-llvm-cpu*ARM64-v8A*__big-core,full-inference,default-flags/flagfile")
  else()  # X86_64
    set(_FLAGFILE_HINT_PATH "${_MODULE_FLAG_DIR}/iree-llvm-cpu*x86_64*__full-inference,default-flags/flagfile")
  endif()
  file(GLOB _FLAGFILE_PATH
      LIST_DIRECTORIES FALSE
      "${_FLAGFILE_HINT_PATH}"
    )
  if(NOT _FLAGFILE_PATH)
    message(SEND_ERROR "Could not locate flagfile matching '${_FLAGFILE_HINT_PATH}' for ${_RULE_BENCHMARK_MODULE_SRC}")
    return()
  endif()

  list(LENGTH _FLAGFILE_PATH _FLAGFILE_COUNT)
  if(_FLAGFILE_COUNT GREATER 1)
    message(SEND_ERROR "Found multiple files matching '${_FLAGFILE_HINT_PATH}' for ${_RULE_BENCHMARK_MODULE_SRC}: ${_FLAGFILE_PATH}")
  endif()

  cmake_path(GET _FLAGFILE_PATH PARENT_PATH _FLAG_FILE_DIR)
  file(STRINGS "${_FLAGFILE_PATH}" _FLAGS ENCODING UTF-8)
  # Parse the flagfile to find the vmfb location.
  # TODO(#10391): Update this logic with the new benchmark framework.
  foreach(_FLAG ${_FLAGS})
    if(_FLAG MATCHES "--module_file=")
      string(REPLACE "--module_file=" "" _SRC "${_FLAG}")
      set(_SRC "${_FLAG_FILE_DIR}/${_SRC}")
    endif()
  endforeach(_FLAG)

  iree_run_module_test(
    NAME
      "${_RULE_NAME}"
    MODULE_SRC
      "${_SRC}"
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
  )
endfunction()
