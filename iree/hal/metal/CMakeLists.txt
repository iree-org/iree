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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    metal
  HDRS
    "metal_buffer.h"
    "metal_capture_manager.h"
    "metal_command_buffer.h"
    "metal_command_queue.h"
    "metal_device.h"
    "metal_direct_allocator.h"
    "metal_driver.h"
    "metal_kernel_library.h"
    "metal_pipeline_argument_buffer.h"
    "metal_pipeline_cache.h"
    "metal_shared_event.h"
  SRCS
    "metal_buffer.mm"
    "metal_capture_manager.mm"
    "metal_command_buffer.mm"
    "metal_command_queue.mm"
    "metal_device.mm"
    "metal_direct_allocator.mm"
    "metal_driver.mm"
    "metal_kernel_library.mm"
    "metal_pipeline_argument_buffer.cc"
    "metal_pipeline_cache.mm"
    "metal_shared_event.mm"
  DEPS
    absl::flat_hash_map
    absl::inlined_vector
    absl::memory
    absl::span
    absl::strings
    iree::base::flatcc
    iree::base::file_io
    iree::base::logging
    iree::base::status
    iree::base::time
    iree::base::tracing
    iree::hal
    iree::schemas::metal_executable_def_c_fbs
  LINKOPTS
    "-framework Foundation"
    "-framework Metal"
  PUBLIC
)
