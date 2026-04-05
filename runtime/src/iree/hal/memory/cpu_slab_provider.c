// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/cpu_slab_provider.h"

#if defined(IREE_PLATFORM_LINUX)
#include <sys/mman.h>
#endif  // IREE_PLATFORM_LINUX

typedef struct iree_hal_cpu_slab_provider_t {
  iree_hal_slab_provider_t base;
  iree_allocator_t host_allocator;
} iree_hal_cpu_slab_provider_t;

static const iree_hal_slab_provider_vtable_t iree_hal_cpu_slab_provider_vtable;

iree_status_t iree_hal_cpu_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;

  iree_hal_cpu_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  iree_hal_slab_provider_initialize(&iree_hal_cpu_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  *out_provider = &provider->base;
  return iree_ok_status();
}

static void iree_hal_cpu_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_allocator_t allocator = provider->host_allocator;
  iree_allocator_free(allocator, provider);
}

static iree_status_t iree_hal_cpu_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));
  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      provider->host_allocator, min_length, IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      /*offset=*/0, &ptr));
  out_slab->base_ptr = (uint8_t*)ptr;
  out_slab->length = min_length;
  out_slab->provider_handle = 0;
  return iree_ok_status();
}

static void iree_hal_cpu_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_allocator_free_aligned(provider->host_allocator, slab->base_ptr);
}

static iree_status_t iree_hal_cpu_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_byte_span_t data = {
      .data = slab->base_ptr + slab_offset,
      .data_length = (iree_host_size_t)allocation_size,
  };
  return iree_hal_heap_buffer_wrap(iree_hal_buffer_placement_undefined(),
                                   params.type, params.access, params.usage,
                                   allocation_size, data, release_callback,
                                   provider->host_allocator, out_buffer);
}

static void iree_hal_cpu_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  *out_memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
      IREE_HAL_MEMORY_TYPE_HOST_COHERENT | IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  *out_supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                         IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                         IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
}

// Forces the OS to back all virtual pages in the slab with physical memory
// and zero them. Without this, each page faults on first write — 65,536
// faults for a 256MB slab, scattered across whichever thread touches the
// memory first. Running this on the slab cache's background thread (which is
// NUMA-pinned) ensures pages are allocated on the correct NUMA node via
// first-touch policy.
static void iree_hal_cpu_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)slab->length);

  bool populated = false;

#if defined(IREE_PLATFORM_LINUX)
#ifdef MADV_HUGEPAGE
  // Hint to the kernel that this region is a good candidate for transparent
  // huge pages. Best-effort; errors are ignored.
  if (slab->length >= 2 * 1024 * 1024) {
    madvise(slab->base_ptr, (size_t)slab->length, MADV_HUGEPAGE);
  }
#endif  // MADV_HUGEPAGE

#ifdef MADV_POPULATE_WRITE
  // MADV_POPULATE_WRITE (kernel 5.14+) faults and zeroes all pages in a
  // single syscall. Much faster than touching each page from userspace.
  populated =
      madvise(slab->base_ptr, (size_t)slab->length, MADV_POPULATE_WRITE) == 0;
#endif  // MADV_POPULATE_WRITE
#endif  // IREE_PLATFORM_LINUX

  if (!populated) {
    // Fallback: touch every page to force allocation and zero-fill.
    // Sequential access pattern for TLB and prefetcher friendliness.
    memset(slab->base_ptr, 0, (size_t)slab->length);
  }

  IREE_TRACE_ZONE_END(z0);
}

// The CPU provider has no cache or freelist — nothing to trim.
static void iree_hal_cpu_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {}

// The CPU provider tracks no statistics beyond what the allocator itself
// provides. Leaf provider — no inner provider to recurse into.
static void iree_hal_cpu_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
}

static const iree_hal_slab_provider_vtable_t iree_hal_cpu_slab_provider_vtable =
    {
        .destroy = iree_hal_cpu_slab_provider_destroy,
        .acquire_slab = iree_hal_cpu_slab_provider_acquire_slab,
        .release_slab = iree_hal_cpu_slab_provider_release_slab,
        .wrap_buffer = iree_hal_cpu_slab_provider_wrap_buffer,
        .prefault = iree_hal_cpu_slab_provider_prefault,
        .trim = iree_hal_cpu_slab_provider_trim,
        .query_stats = iree_hal_cpu_slab_provider_query_stats,
        .query_properties = iree_hal_cpu_slab_provider_query_properties,
};
