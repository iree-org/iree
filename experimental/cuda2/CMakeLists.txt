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
    "cuda_driver.c"
    "memory_pools.c"
    "memory_pools.h"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::base::tracing
    iree::hal
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
    iree::base::core_headers
    iree::base::internal::dynamic_library
    iree::base::tracing
    iree_cuda::headers
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
