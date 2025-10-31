// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_EXECUTABLE_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/pipeline_layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_source_location_t {
  iree_string_view_t file_name;
  int line;
  iree_string_view_t func_name;
} iree_hal_vulkan_source_location_t;

typedef struct iree_hal_vulkan_pipeline_t {
  VkPipeline handle;
  iree_hal_vulkan_pipeline_layout_t* layout;
  IREE_TRACE(iree_hal_vulkan_source_location_t source_location;)
} iree_hal_vulkan_pipeline_t;

// Infers the format of the executable and calculates its total size.
// If executable_data.data_length is 0 attempts to infer size from the data.
// Returns the canonical format string and total size of the executable data.
iree_status_t iree_hal_vulkan_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Creates a wrapper for one or more VkPipelines that are sourced from the same
// IREE executable. Each of the pipelines will share the same shader module
// and just differs by the entry point into the shader module they reference.
iree_status_t iree_hal_vulkan_native_executable_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkPipelineCache pipeline_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable);

// Returns the pipeline for the given |entry_point| in the |executable|.
iree_status_t iree_hal_vulkan_native_executable_lookup_pipeline(
    iree_hal_executable_t* executable, uint32_t entry_ordinal,
    const iree_hal_vulkan_pipeline_t** out_pipeline);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_EXECUTABLE_H_
