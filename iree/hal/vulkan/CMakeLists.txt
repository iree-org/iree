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
    api
  HDRS
    api.h
  SRCS
    api.cc
  DEPS
    ::utils
    ::vulkan
    iree::base::api
    iree::base::tracing
    iree::hal::api
  PUBLIC
)

iree_cc_library(
  NAME
    utils
  HDRS
    debug_reporter.h
    dynamic_symbol_tables.h
    dynamic_symbols.h
    extensibility_util.h
    handle_util.h
    renderdoc_capture_manager.h
    status_util.h
    timepoint_util.h
    vulkan_headers.h
  SRCS
    debug_reporter.cc
    dynamic_symbols.cc
    extensibility_util.cc
    renderdoc_capture_manager.cc
    status_util.cc
    timepoint_util.cc
  DEPS
    Vulkan::Headers
    absl::core_headers
    absl::memory
    absl::span
    absl::strings
    absl::synchronization
    iree::base::core_headers
    iree::base::dynamic_library
    iree::base::intrusive_list
    iree::base::logging
    iree::base::ref_ptr
    iree::base::status
    iree::base::time
    iree::base::tracing
    iree::hal
    renderdoc_api::renderdoc_app
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    dynamic_symbols_test.cc
  DEPS
    ::utils
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    driver=vulkan
)

iree_cc_library(
  NAME
    vma_allocator
  HDRS
    vma_allocator.h
    vma_buffer.h
  SRCS
    internal_vk_mem_alloc.cc
    internal_vk_mem_alloc.h
    vma_allocator.cc
    vma_buffer.cc
  DEPS
    ::utils
    absl::flat_hash_map
    absl::memory
    absl::synchronization
    iree::base::core_headers
    iree::base::logging
    iree::base::status
    iree::base::tracing
    iree::hal
    vulkan_memory_allocator
  PUBLIC
)

iree_cc_library(
  NAME
    vulkan
  HDRS
    descriptor_pool_cache.h
    descriptor_set_arena.h
    direct_command_buffer.h
    direct_command_queue.h
    emulated_timeline_semaphore.h
    native_descriptor_set.h
    native_event.h
    native_timeline_semaphore.h
    pipeline_cache.h
    pipeline_executable.h
    pipeline_executable_layout.h
    serializing_command_queue.h
    vulkan_device.h
    vulkan_driver.h
  SRCS
    descriptor_pool_cache.cc
    descriptor_set_arena.cc
    direct_command_buffer.cc
    direct_command_queue.cc
    emulated_timeline_semaphore.cc
    native_descriptor_set.cc
    native_event.cc
    native_timeline_semaphore.cc
    pipeline_cache.cc
    pipeline_executable.cc
    pipeline_executable_layout.cc
    serializing_command_queue.cc
    vulkan_device.cc
    vulkan_driver.cc
  DEPS
    ::utils
    ::vma_allocator
    absl::core_headers
    absl::inlined_vector
    absl::memory
    absl::span
    absl::strings
    absl::synchronization
    iree::base::api
    iree::base::arena
    iree::base::core_headers
    iree::base::flatcc
    iree::base::intrusive_list
    iree::base::ref_ptr
    iree::base::status
    iree::base::time
    iree::base::tracing
    iree::hal
    iree::hal::command_buffer_validation
    iree::schemas::spirv_executable_def_c_fbs
  PUBLIC
)
