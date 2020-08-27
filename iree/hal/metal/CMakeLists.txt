# Copyright 2020 Google LLC
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

if(NOT ${IREE_HAL_DRIVER_METAL})
  return()
endif()

iree_cc_library(
  NAME
    metal_command_buffer
  HDRS
    "metal_command_buffer.h"
  SRCS
    "metal_command_buffer.mm"
  DEPS
    ::metal_kernel_library
    ::metal_pipeline_argument_buffer
    absl::flat_hash_map
    absl::inlined_vector
    iree::base::status
    iree::base::tracing
    iree::hal::command_buffer
  LINKOPTS
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_command_queue
  HDRS
    "metal_command_queue.h"
  SRCS
    "metal_command_queue.mm"
  DEPS
    ::metal_command_buffer
    ::metal_shared_event
    iree::base::status
    iree::base::time
    iree::base::tracing
    iree::hal::command_queue
  LINKOPTS
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_device
  HDRS
    "metal_device.h"
  SRCS
    "metal_device.mm"
  DEPS
    ::metal_command_buffer
    ::metal_command_queue
    ::metal_direct_allocator
    ::metal_pipeline_argument_buffer
    ::metal_pipeline_cache
    ::metal_shared_event
    absl::strings
    absl::span
    iree::base::status
    iree::base::time
    iree::base::tracing
    iree::hal::allocator
    iree::hal::command_queue
    iree::hal::device
    iree::hal::driver
    iree::hal::semaphore
  LINKOPTS
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_direct_allocator
  HDRS
    "metal_buffer.h"
    "metal_direct_allocator.h"
  SRCS
    "metal_buffer.mm"
    "metal_direct_allocator.mm"
  DEPS
    absl::memory
    iree::base::logging
    iree::base::status
    iree::base::tracing
    iree::hal::allocator
  LINKOPTS
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_driver
  HDRS
    "metal_driver.h"
  SRCS
    "metal_driver.mm"
  DEPS
    ::metal_device
    iree::base::status
    iree::base::tracing
    iree::hal::device_info
    iree::hal::driver
  LINKOPTS
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_driver_module
  SRCS
    "metal_driver_module.cc"
  DEPS
    ::metal_driver
    iree::base::init
    iree::base::status
    iree::hal::driver_registry
  ALWAYSLINK
  PUBLIC
)

iree_cc_library(
  NAME
    metal_kernel_library
  HDRS
    "metal_kernel_library.h"
  SRCS
    "metal_kernel_library.mm"
  DEPS
    absl::inlined_vector
    iree::base::memory
    iree::base::status
    iree::base::tracing
    iree::hal::executable
    iree::hal::executable_spec
    iree::schemas::metal_executable_def_cc_fbs
  LINKOPTS
    "-framework Foundation"
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_pipeline_argument_buffer
  HDRS
    "metal_pipeline_argument_buffer.h"
  SRCS
    "metal_pipeline_argument_buffer.cc"
  DEPS
    absl::inlined_vector
    absl::span
    iree::hal::descriptor_set_layout
    iree::hal::executable_layout
  PUBLIC
)

iree_cc_library(
  NAME
    metal_pipeline_cache
  HDRS
    "metal_pipeline_cache.h"
  SRCS
    "metal_pipeline_cache.mm"
  DEPS
    ::metal_kernel_library
    flatbuffers
    iree::base::status
    iree::base::tracing
    iree::hal::executable
    iree::hal::executable_cache
    iree::hal::executable_format
    iree::schemas::metal_executable_def_cc_fbs
  LINKOPTS
    "-framework Metal"
  PUBLIC
)

iree_cc_library(
  NAME
    metal_shared_event
  HDRS
    "metal_shared_event.h"
  SRCS
    "metal_shared_event.mm"
  DEPS
    iree::base::tracing
    iree::hal::semaphore
  LINKOPTS
    "-framework Metal"
  PUBLIC
)
