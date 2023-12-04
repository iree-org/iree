# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Set the root for package namespacing to the current directory.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/experimental/cuda2")

iree_add_all_subdirs()

iree_cc_library(
  NAME
    cuda2
  HDRS
    "api.h"
  SRCS
    "api.h"
    "cuda_allocator.c"
    "cuda_allocator.h"
    "cuda_buffer.c"
    "cuda_buffer.h"
    "cuda_device.c"
    "cuda_device.h"
    "cuda_driver.c"
    "event_pool.c"
    "event_pool.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "graph_command_buffer.c"
    "graph_command_buffer.h"
    "memory_pools.c"
    "memory_pools.h"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "nccl_channel.c"
    "nccl_channel.h"
    "pending_queue_actions.c"
    "pending_queue_actions.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "stream_command_buffer.c"
    "stream_command_buffer.h"
    "timepoint_pool.c"
    "timepoint_pool.h"
    "tracing.c"
    "tracing.h"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::event_pool
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::utils::buffer_transfer
    iree::hal::utils::collective_batch
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::memory_file
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
    iree::schemas::cuda_executable_def_c_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "cuda_dynamic_symbols.h"
    "cuda_status_util.h"
    "nccl_dynamic_symbols.h"
    "nccl_status_util.h"
  TEXTUAL_HDRS
    "cuda_dynamic_symbol_table.h"
    "nccl_dynamic_symbol_table.h"
  SRCS
    "cuda_dynamic_symbols.c"
    "cuda_headers.h"
    "cuda_status_util.c"
    "nccl_dynamic_symbols.c"
    "nccl_headers.h"
    "nccl_status_util.c"
  DEPS
    iree::base
    iree::base::internal::dynamic_library
    iree_cuda::headers
    nccl::headers
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
    "driver=cuda2"
)
