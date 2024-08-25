# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Set the root for package namespacing to the current directory.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/experimental/hsa")

iree_add_all_subdirs()

set(IREE_ROCM_TARGET_CHIP_DEFAULT "gfx1103")
set(IREE_ROCM_TARGET_CHIP "${IREE_ROCM_TARGET_CHIP_DEFAULT}" CACHE STRING
  "Target chip for ROCm. This influences conformance tests that need to compile device code. Defaults to \"${IREE_ROCM_TARGET_CHIP_DEFAULT}\".")

if(NOT DEFINED HSA_API_HEADERS_ROOT)
  set(HSA_API_HEADERS_ROOT "${IREE_SOURCE_DIR}/third_party/hsa-build-deps/runtime/hsa-runtime/inc")
endif()


iree_cc_library(
  NAME
    hsa
  HDRS
    "api.h"
  SRCS
    "api.h"
    "event_pool.c"
    "event_pool.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "hsa_allocator.c"
    "hsa_allocator.h"
    "hsa_buffer.c"
    "hsa_buffer.h"
    "hsa_device.c"
    "hsa_device.h"
    "hsa_driver.c"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "pending_queue_actions.c"
    "pending_queue_actions.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "queue_command_buffer.c"
    "queue_command_buffer.h"
    "timepoint_pool.c"
    "timepoint_pool.h"
  INCLUDES
    "${HSA_API_HEADERS_ROOT}"
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
    iree::base::internal::wait_handle
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::utils::collective_batch
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::memory_file
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
    iree::schemas::rocm_executable_def_c_fbs
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
    "hsa_headers.h"
    "status_util.c"
  INCLUDES
    "${HSA_API_HEADERS_ROOT}"
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
    "driver=hsa"
)
