# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(IREE_PJRT)
cmake_policy(SET CMP0069 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

enable_testing()

# This project uses explicit symbol visibility.
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

# All python binaries go into one tree.
set(IREE_PJRT_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
set(IREE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Enable LTO if supported.
option(IREE_PJRT_ENABLE_LTO "Enable LTO (link time optimization) if supported" OFF)
include(CheckIPOSupported)
check_ipo_supported(RESULT _ireert_lto_supported OUTPUT error)
if(IREE_PJRT_ENABLE_LTO)
  if(_ireert_lto_supported)
    message(STATUS "Enabling LTO")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO not supported by toolchain bit requested (ignored)")
  endif()
endif()

# Customize defaults.
option(IREE_BUILD_COMPILER "Disable compiler for runtime-library build" OFF)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)

# Include IREE.
message(STATUS "Including IREE from ${IREE_ROOT_DIR}")
add_subdirectory("${IREE_ROOT_DIR}" "iree_core" EXCLUDE_FROM_ALL)

# Include local sources.
# Handle various global definitions that need to be set at the global
# toolchain level.
iree_setup_toolchain()

add_subdirectory(src)
add_subdirectory(third_party/pjrt_c_api)

#-------------------------------------------------------------------------------
# Convenience test target.
#-------------------------------------------------------------------------------

add_custom_target(iree-pjrt-test-deps)

set(IREE_PJRT_CTEST_ARGS "-L;^iree_pjrt.+")
add_custom_target(iree-pjrt-run-tests
  COMMENT
    "Run IREE PJRT Tests"
    WORKING_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL
  VERBATIM
  COMMAND_EXPAND_LISTS
  COMMAND
    "${CMAKE_COMMAND}" -E echo
    "The 'iree-pjrt-run-tests' target is a helper for running ctest. For advanced"
    "options, build dependencies and invoke ctest independently as in:"
  COMMAND
    "${CMAKE_COMMAND}" -E echo
    "  (cd ${CMAKE_CURRENT_BINARY_DIR} && cmake --build . --target iree-pjrt-run-tests && ctest ${IREE_PJRT_CTEST_ARGS} --output-on-failure)"
  COMMAND
    "${CMAKE_COMMAND}" -E echo
    "Run tests in parallel by setting a variable like CTEST_PARALLEL_LEVEL=25."
  COMMAND
    "${CMAKE_CTEST_COMMAND}" ${IREE_PJRT_CTEST_ARGS} --output-on-failure
)
add_dependencies(iree-pjrt-run-tests iree-pjrt-test-deps)
