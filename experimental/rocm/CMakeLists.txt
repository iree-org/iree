# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
# Canonicalize path.
cmake_path(ABSOLUTE_PATH IREE_PACKAGE_ROOT_DIR
  BASE_DIRECTORY ${IREE_PACKAGE_ROOT_DIR}
  NORMALIZE
  OUTPUT_VARIABLE IREE_PACKAGE_ROOT_DIR)
set(IREE_PACKAGE_ROOT_PREFIX iree)

iree_add_all_subdirs()

if(NOT DEFINED ROCM_HEADERS_API_ROOT)
  set(ROCM_HEADERS_API_ROOT "${IREE_SOURCE_DIR}/third_party/hip-build-deps/include")
endif()

if(NOT EXISTS "${ROCM_HEADERS_API_ROOT}/hip/hip_version.h")
  message(SEND_ERROR "Could not find HIP headers at: ${ROCM_HEADERS_API_ROOT}")
endif()

iree_cc_library(
  NAME
    rocm
  HDRS
    "api.h"
  SRCS
    "api.h"
    "context_wrapper.h"
    "rocm_allocator.c"
    "rocm_allocator.h"
    "rocm_buffer.c"
    "rocm_buffer.h"
    "rocm_device.c"
    "rocm_device.h"
    "rocm_driver.c"
    "rocm_event.c"
    "rocm_event.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "direct_command_buffer.c"
    "direct_command_buffer.h"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "status_util.c"
    "status_util.h"
    "tracing.c"
    "tracing.h"
  INCLUDES
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${PROJECT_BINARY_DIR}"
    "${ROCM_HEADERS_API_ROOT}"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::flatcc::parsing
    iree::base::internal::synchronization
    iree::hal
    iree::hal::utils::file_transfer
    iree::hal::utils::memory_file
    iree::hal::utils::semaphore_base
    iree::schemas::rocm_executable_def_c_fbs
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
  TEXTUAL_HDRS
    "dynamic_symbol_tables.h"
  SRCS
    "rocm_headers.h"
    "dynamic_symbols.c"
  INCLUDES
    "${ROCM_HEADERS_API_ROOT}"
    "${CMAKE_CURRENT_LIST_DIR}/../.."
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
  DEPS
    iree::base
    iree::base::internal::dynamic_library
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    "dynamic_symbols_test.cc"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=rocm"
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
)
