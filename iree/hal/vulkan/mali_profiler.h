// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VULKAN_MALI_PROFILER_H_
#define IREE_HAL_VULKAN_MALI_PROFILER_H_

#include "iree/base/api.h"
#include "iree/base/tracing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

typedef struct iree_hal_vulkan_mali_profiler_context_t
    iree_hal_vulkan_mali_profiler_context_t;

iree_status_t iree_hal_vulkan_mali_profiler_context_allocate(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_mali_profiler_context_t** out_context);

void iree_hal_vulkan_mali_profiler_context_free(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_mali_profiler_context_t* context);

void iree_hal_vulkan_mali_profiler_start(
    iree_hal_vulkan_mali_profiler_context_t* context);

void iree_hal_vulkan_mali_profiler_stop(
    iree_hal_vulkan_mali_profiler_context_t* context);

void iree_hal_vulkan_mali_profiler_configure_counter(
    iree_hal_vulkan_mali_profiler_context_t* context,
    void (*counter_type_consumer)(const char* counter_name,
                                  uint8_t counter_type));

void iree_hal_vulkan_mali_profiler_sample(
    iree_hal_vulkan_mali_profiler_context_t* context,
    void (*counter_sample_consumer)(const char* counter_name,
                                    int64_t sample_value));

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_MALI_PROFILER_H_
