# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(NOT DEFINED HSA_API_HEADERS_ROOT)
  if(IREE_USE_SYSTEM_DEPS)
    find_path(HSA_API_HEADERS_ROOT
      NAMES hsa/hsa.h
      PATHS /opt/rocm/include
            /usr/include
      DOC "Path to HSA headers")
  else()
    set(HSA_API_HEADERS_ROOT "/opt/rocm/include")
  endif()
endif()

if(NOT EXISTS "${HSA_API_HEADERS_ROOT}/hsa/hsa.h")
  message(SEND_ERROR "Could not find HSA headers at: ${HSA_API_HEADERS_ROOT}")
endif()

iree_cc_library(
  NAME
    hsa
  HDRS
    "api.h"
  SRCS
    "api.h"
    "hsa_allocator.c"
    "hsa_allocator.h"
    "hsa_buffer.c"
    "hsa_buffer.h"
    "hsa_driver.c"
    "hsa_device.h"
    "hsa_device.c"
    "hsa_semaphore.c"
    "hsa_semaphore.h"
    "native_executable.c"
    "native_executable.h"
    "native_executable_hsaf.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "per_device_information.h"
    "stream_command_buffer.c"
    "stream_command_buffer.h"
  INCLUDES
    "${HSA_API_HEADERS_ROOT}"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::atomic_slist
    iree::base::internal::synchronization
    iree::base::internal::threading
    iree::base::internal::wait_handle
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::utils::collective_batch
    iree::hal::utils::executable_debug_info
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::executable_header
    iree::hal::utils::file_transfer
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
    iree::hal::utils::queue_host_call_emulation
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
    iree::schemas::executable_debug_info_c_fbs
    iree::schemas::hip_executable_def_c_fbs
    iree::hal::drivers::hip::hip
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

