# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    metal
  HDRS
    "api.h"
  SRCS
    "api.h"
    "builtin_executables.h"
    "builtin_executables.m"
    "direct_allocator.h"
    "direct_allocator.m"
    "direct_command_buffer.h"
    "direct_command_buffer.m"
    "kernel_library.h"
    "kernel_library.m"
    "metal_buffer.h"
    "metal_buffer.m"
    "metal_device.m"
    "metal_driver.m"
    "nop_executable_cache.h"
    "nop_executable_cache.m"
    "pipeline_layout.h"
    "pipeline_layout.m"
    "shared_event.h"
    "shared_event.m"
    "staging_buffer.h"
    "staging_buffer.m"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::drivers::metal::builtin
    iree::hal::utils::file_transfer
    iree::hal::utils::memory_file
    iree::hal::utils::resource_set
    iree::schemas::metal_executable_def_c_fbs
    "-framework Foundation"
    "-framework Metal"
  PUBLIC
)
