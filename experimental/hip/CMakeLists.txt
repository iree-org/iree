# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Set the root for package namespacing to the current directory.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/experimental/hip")

iree_add_all_subdirs()

if(NOT DEFINED HIP_API_HEADERS_ROOT)
  set(HIP_API_HEADERS_ROOT "${IREE_SOURCE_DIR}/third_party/hip-build-deps/include")
endif()

if(NOT EXISTS "${HIP_API_HEADERS_ROOT}/hip/hip_version.h")
  message(SEND_ERROR "Could not find HIP headers at: ${HIP_API_HEADERS_ROOT}")
endif()

iree_cc_library(
  NAME
    hip
  HDRS
    "api.h"
  SRCS
    "api.h"
    "event_pool.c"
    "event_pool.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "graph_command_buffer.c"
    "graph_command_buffer.h"
    "hip_allocator.c"
    "hip_allocator.h"
    "hip_buffer.c"
    "hip_buffer.h"
    "hip_device.c"
    "hip_device.h"
    "hip_driver.c"
    "memory_pools.c"
    "memory_pools.h"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "pending_queue_actions.c"
    "pending_queue_actions.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "stream_command_buffer.c"
    "stream_command_buffer.h"
    "timepoint_pool.c"
    "timepoint_pool.h"
  INCLUDES
    "${HIP_API_HEADERS_ROOT}"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::atomic_slist
    iree::base::internal::event_pool
    iree::base::internal::synchronization
    iree::base::internal::threading
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::utils::collective_batch
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::resource_set
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
    "status_util.h"
  TEXTUAL_HDRS
    "dynamic_symbol_tables.h"
  SRCS
    "dynamic_symbols.c"
    "hip_headers.h"
    "status_util.c"
  INCLUDES
    "${HIP_API_HEADERS_ROOT}"
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
  DEPS
    iree::base
    iree::base::core_headers
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
    "driver=hip"
  COPTS
    "-D__HIP_PLATFORM_HCC__=1"
)
