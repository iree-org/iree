// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_BIND_GROUP_CACHE_H_
#define IREE_HAL_DRIVERS_WEBGPU_BIND_GROUP_CACHE_H_

#include "experimental/webgpu/pipeline_layout.h"
#include "experimental/webgpu/platform/webgpu.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// NOTE: this is probably too small, but this is all a hack anyway.
// TODO(benvanik): build a real cache - today this is assuming the compiler does
// a much better job than it currently does as reducing the number of push
// descriptor sets.
#define IREE_HAL_WEBGPU_BIND_GROUP_CACHE_CAPACITY 32

// A subset of WGPUBindGroupEntry containing only what we need.
// WGPUBindGroupEntry is quite large (has sampler and texture information).
typedef struct iree_hal_webgpu_bind_group_binding_t {
  // TODO(benvanik): also track whether dynamic.
  WGPUBufferBindingType type;
  WGPUBuffer buffer;
  iree_device_size_t offset;
  iree_device_size_t length;
} iree_hal_webgpu_bind_group_binding_t;

typedef struct iree_hal_webgpu_bind_group_cache_entry_t {
  // Group layout this bind group conforms to.
  // It's possible to share bind groups with different compatible layouts but
  // we don't do that yet and require an exact match.
  WGPUBindGroupLayout group_layout;
  // Cached WebGPU bind group containing the bindings.
  WGPUBindGroup handle;
  // Each bit indicates a populated binding at the respective ordinal.
  iree_hal_webgpu_binding_mask_t binding_mask;
  // Each source binding to use for cache equality comparison.
  iree_hal_webgpu_bind_group_binding_t
      bindings[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT];
} iree_hal_webgpu_bind_group_cache_entry_t;

// Simple cache of WGPUBindGroups.
// Bind groups in WebGPU are immutable and we need to create new ones for each
// unique set of bindings.
typedef struct iree_hal_webgpu_bind_group_cache_t {
  WGPUDevice device;
  iree_host_size_t entry_count;
  iree_hal_webgpu_bind_group_cache_entry_t
      entries[IREE_HAL_WEBGPU_BIND_GROUP_CACHE_CAPACITY];
} iree_hal_webgpu_bind_group_cache_t;

// Initializes an empty bind group cache.
void iree_hal_webgpu_bind_group_cache_initialize(
    WGPUDevice device, iree_hal_webgpu_bind_group_cache_t* out_cache);

// Deinitializes the cache and drops all bind group handles.
void iree_hal_webgpu_bind_group_cache_deinitialize(
    iree_hal_webgpu_bind_group_cache_t* cache);

// Trims the cache down to its minimum size by dropping all bind groups.
// All WGPUBindGroup handles will be dropped.
void iree_hal_webgpu_bind_group_cache_trim(
    iree_hal_webgpu_bind_group_cache_t* cache);

// Acquires a bind group from the cache with the given |bindings|.
// Each bit of |binding_mask| indicates a binding that is used by the caller;
// this allows for matching of cached bind groups to match any with only the
// used bindings needing to match.
// Callers may use the returned bind group handle until the cache is trimmed.
WGPUBindGroup iree_hal_webgpu_bind_group_cache_acquire(
    iree_hal_webgpu_bind_group_cache_t* cache, WGPUBindGroupLayout group_layout,
    const iree_hal_webgpu_bind_group_binding_t* bindings,
    iree_hal_webgpu_binding_mask_t binding_mask);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_BIND_GROUP_CACHE_H_
