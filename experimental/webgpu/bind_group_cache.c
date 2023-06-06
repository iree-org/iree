// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/bind_group_cache.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "experimental/webgpu/buffer.h"
#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"

void iree_hal_webgpu_bind_group_cache_initialize(
    WGPUDevice device, iree_hal_webgpu_bind_group_cache_t* out_cache) {
  IREE_ASSERT_ARGUMENT(out_cache);
  IREE_TRACE_ZONE_BEGIN(z0);

  out_cache->device = device;
  out_cache->entry_count = IREE_HAL_WEBGPU_BIND_GROUP_CACHE_CAPACITY;

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_webgpu_bind_group_cache_deinitialize(
    iree_hal_webgpu_bind_group_cache_t* cache) {
  IREE_ASSERT_ARGUMENT(cache);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Trim is the same as deinit today.
  iree_hal_webgpu_bind_group_cache_trim(cache);
  cache->entry_count = 0;

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_webgpu_bind_group_cache_trim(
    iree_hal_webgpu_bind_group_cache_t* cache) {
  IREE_ASSERT_ARGUMENT(cache);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_webgpu_bind_group_cache_entry_t* entry = &cache->entries[i];
    if (entry->handle) iree_wgpuBindGroupDrop(entry->handle);
  }
  memset(cache->entries, 0, sizeof(cache->entries));

  IREE_TRACE_ZONE_END(z0);
}

WGPUBindGroup iree_hal_webgpu_bind_group_cache_acquire(
    iree_hal_webgpu_bind_group_cache_t* cache, WGPUBindGroupLayout group_layout,
    const iree_hal_webgpu_bind_group_binding_t* bindings,
    iree_hal_webgpu_binding_mask_t binding_mask) {
  IREE_ASSERT_ARGUMENT(cache);
  IREE_ASSERT_ARGUMENT(bindings);
  IREE_TRACE_ZONE_BEGIN(z0);

  // This is not a good algorithm :)
  // We should probably have a split index and a mechanism to partition it such
  // that lookups don't need to perform a full scan. This is cheaper than the
  // cost of creating a new bind group per dispatch (no need to call out to
  // WebGPU, allocate new objects, track those new objects lifetimes, etc) but
  // not cheap. Ideally we'd be relying on this for 4-5 bind groups per command
  // buffer at which point it doesn't matter but the compiler still needs to
  // improve a bit there.

  // Scan the cache for entries with a matching group layout and binding mask.
  // These should be the same today but in the future we may want to allow for
  // subsetting as defined by bind group compatibility.
  iree_host_size_t insertion_slot = cache->entry_count - 1;
  for (iree_host_size_t i = 0; i < cache->entry_count; ++i) {
    iree_hal_webgpu_bind_group_cache_entry_t* entry = &cache->entries[i];
    if (!entry->handle) {
      insertion_slot = iree_min(insertion_slot, i);
      continue;
    }
    if (entry->group_layout != group_layout) continue;
    if (entry->binding_mask != binding_mask) continue;

    // Found a potential match. Do a full comparison of the bindings.
    // TODO(benvanik): we only really need to compare the bindings that are
    // set in the mask, however memcmp over a few hundred bytes is usually
    // faster than what we'd have to do for that comparison.
    if (memcmp(bindings, entry->bindings, sizeof(entry->bindings)) == 0) {
      // Same exact bindings - cache hit!
      // TODO(benvanik): a real LRU that rearranges this to the front/back.
      IREE_TRACE_ZONE_END(z0);
      return entry->handle;
    }
  }

  // Evict an existing entry to store this new one or use the last unused slot.
  iree_hal_webgpu_bind_group_cache_entry_t* entry =
      &cache->entries[insertion_slot];
  if (entry->handle) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "evict");
    iree_wgpuBindGroupDrop(entry->handle);
  } else {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "miss");
  }
  entry->group_layout = group_layout;
  entry->binding_mask = binding_mask;
  memcpy(entry->bindings, bindings, sizeof(entry->bindings));

  // NOTE: we could change this to do bit scans over the binding_mask but I
  // haven't checked to see how expensive those are in WebAssembly. For now we
  // do a few more loop iterations with the assumption that doing a bit scan
  // may require hundreds of more instructions.
  uint32_t binding_count = 0;
  WGPUBindGroupEntry entries[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(entries); ++i) {
    if (!(binding_mask & (1u << i))) continue;
    entries[binding_count] = (WGPUBindGroupEntry){
        .nextInChain = NULL,
        .binding = binding_count,
        .buffer = bindings[i].buffer,
        .offset = (uint64_t)bindings[i].offset,
        .size = bindings[i].length,
    };
    ++binding_count;
  }

  const WGPUBindGroupDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
      .layout = group_layout,
      .entryCount = binding_count,
      .entries = entries,
  };
  entry->handle = wgpuDeviceCreateBindGroup(cache->device, &descriptor);

  IREE_TRACE_ZONE_END(z0);
  return entry->handle;
}
