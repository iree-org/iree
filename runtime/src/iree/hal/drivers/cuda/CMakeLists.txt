# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT CUDAToolkit_INCLUDE_DIRS)
  message(FATAL_ERROR "No CUDA SDK includes found: should have been set globally")
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    cuda
  HDRS
    "api.h"
  SRCS
    "api.h"
    "context_wrapper.h"
    "cuda_allocator.c"
    "cuda_allocator.h"
    "cuda_buffer.c"
    "cuda_buffer.h"
    "cuda_device.c"
    "cuda_device.h"
    "cuda_driver.c"
    "cuda_event.c"
    "cuda_event.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "graph_command_buffer.c"
    "graph_command_buffer.h"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "status_util.c"
    "status_util.h"
    "stream_command_buffer.c"
    "stream_command_buffer.h"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::flatcc::parsing
    iree::base::internal::synchronization
    iree::base::tracing
    iree::hal
    iree::hal::utils::buffer_transfer
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
    iree::schemas::cuda_executable_def_c_fbs
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
    "cuda_headers.h"
    "dynamic_symbols.c"
  INCLUDES
    ${CUDAToolkit_INCLUDE_DIRS}
  DEPS
    iree::base::core_headers
    iree::base::internal::dynamic_library
    iree::base::tracing
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
    "driver=cuda"
)
