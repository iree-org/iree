// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_VULKAN_NATIVE_EXECUTABLE_H_
#define IREE_HAL_VULKAN_NATIVE_EXECUTABLE_H_

// clang-format off: Must be included before all other headers:
#include "iree/hal/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/hal/api.h"
#include "iree/hal/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  iree_string_view_t file_name;
  int line;
  iree_string_view_t func_name;
} iree_hal_vulkan_source_location_t;

// Creates a wrapper for one or more VkPipelines that are sourced from the same
// IREE executable. Each of the pipelines will share the same shader module
// and just differs by the entry point into the shader module they reference.
iree_status_t iree_hal_vulkan_native_executable_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkPipelineCache pipeline_cache,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable);

// Returns the source location for the given entry point. May be empty if not
// available.
void iree_hal_vulkan_native_executable_entry_point_source_location(
    iree_hal_executable_t* executable, iree_host_size_t entry_ordinal,
    iree_hal_vulkan_source_location_t* out_source_location);

// Returns the cached VkPipeline for the given executable |entry_ordinal|.
iree_status_t iree_hal_vulkan_native_executable_pipeline_for_entry_point(
    iree_hal_executable_t* executable, iree_host_size_t entry_ordinal,
    VkPipeline* out_pipeline_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_NATIVE_EXECUTABLE_H_
