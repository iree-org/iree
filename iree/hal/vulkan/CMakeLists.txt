# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT ${IREE_HAL_DRIVER_VULKAN})
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    vulkan
  HDRS
    "api.h"
    "vulkan_device.h"
    "vulkan_driver.h"
  SRCS
    "api.cc"
    "command_queue.h"
    "debug_reporter.cc"
    "debug_reporter.h"
    "descriptor_pool_cache.cc"
    "descriptor_pool_cache.h"
    "descriptor_set_arena.cc"
    "descriptor_set_arena.h"
    "direct_command_buffer.cc"
    "direct_command_buffer.h"
    "direct_command_queue.cc"
    "direct_command_queue.h"
    "emulated_semaphore.cc"
    "emulated_semaphore.h"
    "extensibility_util.cc"
    "extensibility_util.h"
    "handle_util.h"
    "internal_vk_mem_alloc.cc"
    "internal_vk_mem_alloc.h"
    "native_descriptor_set.cc"
    "native_descriptor_set.h"
    "native_descriptor_set_layout.cc"
    "native_descriptor_set_layout.h"
    "native_event.cc"
    "native_event.h"
    "native_executable.cc"
    "native_executable.h"
    "native_executable_layout.cc"
    "native_executable_layout.h"
    "native_semaphore.cc"
    "native_semaphore.h"
    "nop_executable_cache.cc"
    "nop_executable_cache.h"
    "serializing_command_queue.cc"
    "serializing_command_queue.h"
    "status_util.c"
    "status_util.h"
    "timepoint_util.cc"
    "timepoint_util.h"
    "vma_allocator.cc"
    "vma_allocator.h"
    "vma_buffer.cc"
    "vma_buffer.h"
    "vulkan_device.cc"
    "vulkan_driver.cc"
    "vulkan_headers.h"
  DEPS
    ::dynamic_symbols
    Vulkan::Headers
    absl::core_headers
    absl::flat_hash_map
    absl::inlined_vector
    absl::memory
    absl::span
    absl::strings
    absl::synchronization
    iree::base::api
    iree::base::core_headers
    iree::base::flatcc
    iree::base::internal
    iree::base::logging
    iree::base::status
    iree::base::synchronization
    iree::base::tracing
    iree::hal::api
    iree::hal::vulkan::util::arena
    iree::hal::vulkan::util::intrusive_list
    iree::hal::vulkan::util::ref_ptr
    iree::schemas::spirv_executable_def_c_fbs
    vulkan_memory_allocator
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbol_tables.h"
    "dynamic_symbols.h"
  SRCS
    "dynamic_symbols.cc"
    "vulkan_headers.h"
  DEPS
    Vulkan::Headers
    absl::core_headers
    absl::memory
    absl::span
    absl::strings
    iree::base::core_headers
    iree::base::dynamic_library
    iree::base::status
    iree::base::tracing
    iree::hal::vulkan::util::ref_ptr
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    "dynamic_symbols_test.cc"
  DEPS
    ::dynamic_symbols
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=vulkan"
)
