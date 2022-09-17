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
# The function supports cross compile and checks the benchmark suite prefixes
# based on the cross compile platform settings, so there is no matching bzl rule
# until there is a bazel --build_config for cross compile targets in place.
#
# Parameters:
#   NAME: Name of the target
#   MODULE_SRC: IREE module (vmfb) file.
#   DRIVER: Driver to run the module with. If specified, it has to be consistent
#       with the one provided by the benchmark_suite's flagfile.
#   RUNNER_ARGS: additional args to pass to iree-run-module. The driver
#       and input file are passed automatically.
#   EXPECTED_OUTPUT: A file of expected output to compare with the output from
#       iree-run-module
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   XFAIL: When set for x86_64|android-arm64-v8a|riscv64-Linux|riscv32-Linux,
#       the test is expected to fail for the particular platforms/architectures
#       for various reasons, e.g., upstream llvm backend.
#   UNSUPPORTED_PLATFORMS: Platforms
#       (android-arm64-v8a|riscv64-Linux|riscv32-Linux) not supported by the
#       test target. The target will be skipped during crosscompile.
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
#     person_detect_int8_correctness_test
#   MODULE_SRC
#     "person_detect_int8.vmfb"
#   DRIVER
#     "local-sync"
#   RUNNER_ARGS
#     "--entry_function=main"
#     "--function_input=1x96x96x1xi8=0"
#   EXPECTED_OUTPUT
#     "1x2xi8=[72 -72]"
#   UNSUPPORTED_PLATFORMS
#     "android-arm64-v8a"
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
    "RUNNER_ARGS;LABELS;XFAIL;UNSUPPORTED_PLATFORMS;DEPS"
    ${ARGN}
  )

  if(CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _RULE_LABELS)
    return()
  endif()

  iree_get_platform(_PLATFORM)
  if(_PLATFORM IN_LIST _RULE_UNSUPPORTED_PLATFORMS)
    return()
  endif()

  set(_RUNNER_TARGET "iree-run-module")
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  get_filename_component(_SRC "${_RULE_MODULE_SRC}" REALPATH)
  list(APPEND _RULE_RUNNER_ARGS "--module_file=${_SRC}")

  if(_RULE_EXPECTED_OUTPUT)
    get_filename_component(_OUTPUT_FILE_TYPE "${_RULE_EXPECTED_OUTPUT}" EXT)
    if(_OUTPUT_FILE_TYPE STREQUAL ".txt")
      get_filename_component(_OUTPUT_FILE_SRC "${_RULE_EXPECTED_OUTPUT}" REALPATH)
      # Process the text input to remove the line breaks.
      file(STRINGS "${_OUTPUT_FILE_SRC}" _EXPECTED_OUTPUT ENCODING UTF-8)
      string(REPLACE ";" " " _EXPECTED_OUTPUT_STR "${_EXPECTED_OUTPUT}")
      set(_EXPECTED_OUTPUT_STR "--expected_output=\"${_EXPECTED_OUTPUT_STR}\"")
      list(APPEND _RULE_RUNNER_ARGS ${_EXPECTED_OUTPUT_STR})
      list(APPEND _SRC ${_OUTPUT_FILE_SRC})
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
        get_filename_component(_OUTPUT_FILE_SRC "${_OUTPUT_FILE_SRC_PATH}" REALPATH)
        list(APPEND _RULE_RUNNER_ARGS "--expected_output=@${_OUTPUT_FILE_SRC}")
        list(APPEND _SRC ${_OUTPUT_FILE_SRC})
      endif()
    else()  # The expected output is listed in the field.
      list(APPEND _RULE_RUNNER_ARGS "--expected_output=\"${_RULE_EXPECTED_OUTPUT}\"")
    endif()
  endif(_RULE_EXPECTED_OUTPUT)

  if(NOT DEFINED _RULE_DRIVER)
    return()
  endif()

  # Dump the flags into a flag file to avoid CMake's naive handling of spaces
  # in expected output.
  if(_RULE_RUNNER_ARGS)
    set(_OUTPUT_FLAGFILE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_flagfile")
    # Write each argument in a new line.
    string(REPLACE ";" "\n" _OUTPUT_FLAGS "${_RULE_RUNNER_ARGS}")
    file(WRITE "${_OUTPUT_FLAGFILE}" "${_OUTPUT_FLAGS}")
    list(APPEND _SRC "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_flagfile")
  endif()

  # A target specifically for the test.
  add_custom_target("${_NAME}" ALL)

  iree_native_test(
    NAME
      "${_RULE_NAME}"
    DRIVER
      "${_RULE_DRIVER}"
    SRC
      "${_RUNNER_TARGET}"
    ARGS
      "--flagfile=${_OUTPUT_FLAGFILE}"
    DATA
      "${_SRC}"
    LABELS
      ${_RULE_LABELS}
    TIMEOUT
      ${_RULE_TIMEOUT}
  )

  # Replace dependencies passed by ::name with iree::package::name
  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  if(_RULE_DEPS)
    add_dependencies(${_NAME}
      ${_RULE_DEPS}
    )
  endif()

  # Set expect failure cases.
  if(_PLATFORM IN_LIST _RULE_XFAIL)
    set(_XFAIL TRUE)
  else()
    set(_XFAIL FALSE)
  endif()
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")
  if(_XFAIL)
    set_property(TEST ${_TEST_NAME} PROPERTY WILL_FAIL TRUE)
  endif()
  add_dependencies(iree-test-deps "${_NAME}")
endfunction()

# iree_benchmark_suite_module_test()
#
# Creates a test using iree-run-module to run a benchmark suite module.
#
# The function supports cross compile and checks the benchmark suite prefixes
# based on the cross compile platform settings, so there is no matching bzl rule
# until there is a bazel --build_config for cross compile targets in place.
#
# Parameters:
#   NAME: Name of the target
#   BENCHMARK_MODULE_SRC: IREE module from benchmark_suite.
#   DRIVER: Driver to run the module with. If specified, it has to be consistent
#       with the one provided by the benchmark_suite's flagfile.
#   RUNNER_ARGS: additional args to pass to iree-run-module. The driver
#       and input file are passed automatically.
#   EXPECTED_OUTPUT: A file of expected output to compare with the output from
#       iree-run-module
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   XFAIL: When set for x86_64|android-arm64-v8a|riscv64-Linux|riscv32-Linux,
#       the test is expected to fail for the particular platforms/architectures
#       for various reasons, e.g., upstream llvm backend.
#   UNSUPPORTED_PLATFORMS: Platforms
#       (android-arm64-v8a|riscv64-Linux|riscv32-Linux) not supported by the
#       test target. The target will be skipped during crosscompile.
#   TIMEOUT: (optional) Test timeout.
#
# Examples:
#
# iree_benchmark_suite_module_test(
#   NAME
#     person_detect_int8_correctness_test
#   BENCHMARK_MODULE_SRC
#     "TFLite/PersonDetect-int8"
#   DRIVER
#     "local-sync"
#   RUNNER_ARGS
#     "--entry_function=main"
#     "--function_input=1x96x96x1xi8=0"
#   EXPECTED_OUTPUT
#     "1x2xi8=[72 -72]"
#   UNSUPPORTED_PLATFORMS
#     "android-arm64-v8a"
# )
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
    "RUNNER_ARGS;LABELS;XFAIL;UNSUPPORTED_PLATFORMS"
    ${ARGN}
  )

  if(CMAKE_CROSSCOMPILING AND "hostonly" IN_LIST _RULE_LABELS)
    return()
  endif()

  iree_get_platform(_PLATFORM)
  if(_PLATFORM IN_LIST _RULE_UNSUPPORTED_PLATFORMS)
    return()
  endif()

  # Benchmark suite needs to be stored at the location of
  # `IREE_BENCHMARK_SUITE_DIR` or the test target is bypassed.
  if(_RULE_BENCHMARK_MODULE_SRC AND NOT DEFINED IREE_BENCHMARK_SUITE_DIR)
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
  if(_FLAGFILE_PATH)
    list(LENGTH _FLAGFILE_PATH _FLAGFILE_NUM)
    if(_FLAGFILE GREATER 1)
      message(SEND_ERROR "Found multiple flagfile with '${_FLAGFILE_HINT_PATH}' for ${_RULE_BENCHMARK_MODULE_SRC}")
    endif()
    get_filename_component(_FLAG_FILE_DIR "${_FLAGFILE_PATH}" DIRECTORY)
    file(STRINGS "${_FLAGFILE_PATH}" _FLAGS ENCODING UTF-8)
    # Parse the flagfile to find the vmfb location.
    # TODO(#10391): Update this logic with the new benchmark framework.
    foreach(_FLAG ${_FLAGS})
      if(_FLAG MATCHES "--module_file=")
        string(REPLACE "--module_file=" "" _SRC "${_FLAG}")
        set(_SRC "${_FLAG_FILE_DIR}/${_SRC}")
      endif()
    endforeach(_FLAG)
  else()
    message(SEND_ERROR "Could not locate flagfile matching '${_FLAGFILE_HINT_PATH}' for ${_RULE_BENCHMARK_MODULE_SRC}")
  endif(_FLAGFILE_PATH)

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
    XFAIL
      ${_RULE_XFAIL}
    UNSUPPORTED_PLATFORMS
      ${_RULE_UNSUPPORTED_PLATFORMS}
    LABELS
      ${_RULE_LABELS}
    TIMEOUT
      ${_RULE_TIMEOUT}
  )
endfunction()
