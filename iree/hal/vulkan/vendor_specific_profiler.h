// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VULKAN_VENDOR_SPECIFIC_PROFILER_H_
#define IREE_HAL_VULKAN_VENDOR_SPECIFIC_PROFILER_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_vendor_specific_profiler_context_t
    iree_hal_vulkan_vendor_specific_profiler_context_t;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Allocates a context for the vendor specific profiler for the given
// |physical_device|. Returns okay status with |out_context| set to null
// if the vendor is not supported.
//
// This function inspects the device name of |physical_device| to determine
// which vendor it is and create the corresponding profiler.
iree_status_t iree_hal_vulkan_vendor_specific_profiler_context_allocate(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_vendor_specific_profiler_context_t** out_context);

void iree_hal_vulkan_vendor_specific_profiler_context_free(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context);

// Starts the vendor specific profiler.
//
// This registers available GPU counters from the vendor specific profiler and
// spawns a thread to continously sampling those counters and upload to the IREE
// tracing framework.
iree_status_t iree_hal_vulkan_vendor_specific_profiler_start(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context);

void iree_hal_vulkan_vendor_specific_profiler_stop(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context);

#else

inline iree_status_t iree_hal_vulkan_vendor_specific_profiler_context_allocate(
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_vendor_specific_profiler_context_t** out_context) {
  *out_context = NULL;
  return iree_ok_status();
}

inline void iree_hal_vulkan_vendor_specific_profiler_context_free(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {}

inline iree_status_t iree_hal_vulkan_vendor_specific_profiler_start(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {
  return iree_ok_status();
}

inline void iree_hal_vulkan_vendor_specific_profiler_stop(
    iree_hal_vulkan_vendor_specific_profiler_context_t* context) {}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_VENDOR_SPECIFIC_PROFILER_H_
