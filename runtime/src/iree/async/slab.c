// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/slab.h"

#include "iree/base/internal/memory.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <sys/mman.h>
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

//===----------------------------------------------------------------------===//
// Slab creation helpers
//===----------------------------------------------------------------------===//

// Allocates the slab struct itself (not the buffer memory).
static iree_status_t iree_async_slab_create_struct(
    iree_allocator_t allocator, iree_async_slab_t** out_slab) {
  iree_async_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*slab), (void**)&slab));
  memset(slab, 0, sizeof(*slab));
  iree_atomic_ref_count_init(&slab->ref_count);
  slab->allocator = allocator;
  *out_slab = slab;
  return iree_ok_status();
}

// Validates common slab parameters.
static iree_status_t iree_async_slab_validate_params(
    iree_host_size_t buffer_size, iree_host_size_t buffer_count,
    iree_host_size_t* out_total_size) {
  if (buffer_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size must be non-zero");
  }
  if (buffer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count must be non-zero");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(buffer_size, buffer_count, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer_size %" PRIhsz " * buffer_count %" PRIhsz
                            " overflows",
                            buffer_size, buffer_count);
  }
  *out_total_size = total_size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Slab lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_async_slab_create(iree_async_slab_options_t options,
                                     iree_allocator_t allocator,
                                     iree_async_slab_t** out_slab) {
  IREE_ASSERT_ARGUMENT(out_slab);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_slab = NULL;

  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_slab_validate_params(options.buffer_size,
                                          options.buffer_count, &total_size));

  // Build NUMA allocation options from slab options.
  iree_numa_alloc_options_t numa_options = iree_numa_alloc_options_default();
  if (options.affinity &&
      options.affinity->numa_node != IREE_ASYNC_AFFINITY_NUMA_NODE_ANY) {
    numa_options.node_id = options.affinity->numa_node;
  }
  numa_options.huge_page_size = options.huge_page_size;
  numa_options.use_explicit_huge_pages = options.use_explicit_huge_pages;
  numa_options.hint_transparent_huge_pages =
      options.hint_transparent_huge_pages;

  // Allocate buffer memory.
  void* base_ptr = NULL;
  iree_numa_alloc_info_t alloc_info = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_numa_alloc(total_size, &numa_options, &base_ptr, &alloc_info));

  // Allocate slab struct.
  iree_async_slab_t* slab = NULL;
  iree_status_t status = iree_async_slab_create_struct(allocator, &slab);
  if (iree_status_is_ok(status)) {
    slab->base_ptr = base_ptr;
    slab->buffer_size = options.buffer_size;
    slab->buffer_count = options.buffer_count;
    slab->total_size = total_size;
    slab->source = IREE_ASYNC_SLAB_SOURCE_NUMA_ALLOC;
    slab->source_state.numa.alloc_info = alloc_info;
    *out_slab = slab;
  } else {
    iree_numa_free(base_ptr, &alloc_info);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_async_slab_wrap(void* base_ptr, iree_host_size_t buffer_size,
                                   iree_host_size_t buffer_count,
                                   iree_allocator_t allocator,
                                   iree_async_slab_t** out_slab) {
  IREE_ASSERT_ARGUMENT(base_ptr);
  IREE_ASSERT_ARGUMENT(out_slab);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_slab = NULL;

  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_async_slab_validate_params(buffer_size, buffer_count, &total_size));

  iree_async_slab_t* slab = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_slab_create_struct(allocator, &slab));

  slab->base_ptr = base_ptr;
  slab->buffer_size = buffer_size;
  slab->buffer_count = buffer_count;
  slab->total_size = total_size;
  slab->source = IREE_ASYNC_SLAB_SOURCE_WRAPPED;

  *out_slab = slab;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_async_slab_import_dmabuf(int dmabuf_fd, uint64_t offset,
                                            iree_host_size_t buffer_size,
                                            iree_host_size_t buffer_count,
                                            uint32_t access_flags,
                                            iree_allocator_t allocator,
                                            iree_async_slab_t** out_slab) {
  IREE_ASSERT_ARGUMENT(out_slab);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_slab = NULL;

#if !defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID)
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "dmabuf import requires Linux");
#else

  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_async_slab_validate_params(buffer_size, buffer_count, &total_size));

  if (dmabuf_fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dmabuf_fd must be >= 0 (got %d)", dmabuf_fd);
  }

  // Determine mmap protection flags from access flags.
  // These use the same flag bits as iree_async_buffer_access_flag_bits_e.
  int prot = 0;
  if (access_flags & (1u << 0)) prot |= PROT_READ;   // READ.
  if (access_flags & (1u << 1)) prot |= PROT_WRITE;  // WRITE.

  // mmap requires page-aligned offset. Align down and track the delta.
  iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  uint64_t aligned_offset = offset & ~((uint64_t)page_size - 1);
  iree_host_size_t offset_delta = (iree_host_size_t)(offset - aligned_offset);
  iree_host_size_t aligned_length =
      iree_host_align(offset_delta + total_size, page_size);

  void* mapped_ptr =
      mmap(NULL, aligned_length, prot, MAP_SHARED, dmabuf_fd, aligned_offset);
  if (mapped_ptr == MAP_FAILED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "mmap of dmabuf fd %d failed", dmabuf_fd);
  }

  iree_async_slab_t* slab = NULL;
  iree_status_t status = iree_async_slab_create_struct(allocator, &slab);
  if (iree_status_is_ok(status)) {
    slab->base_ptr = (uint8_t*)mapped_ptr + offset_delta;
    slab->buffer_size = buffer_size;
    slab->buffer_count = buffer_count;
    slab->total_size = total_size;
    slab->source = IREE_ASYNC_SLAB_SOURCE_DMABUF;
    slab->source_state.dmabuf.mapped_ptr = mapped_ptr;
    slab->source_state.dmabuf.mapped_length = aligned_length;
    *out_slab = slab;
  } else {
    munmap(mapped_ptr, aligned_length);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
}

// Destroys the slab, freeing memory according to source.
static void iree_async_slab_destroy(iree_async_slab_t* slab) {
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (slab->source) {
    case IREE_ASYNC_SLAB_SOURCE_NUMA_ALLOC:
      iree_numa_free(slab->base_ptr, &slab->source_state.numa.alloc_info);
      break;
    case IREE_ASYNC_SLAB_SOURCE_WRAPPED:
      // Not owned; caller manages lifetime.
      break;
    case IREE_ASYNC_SLAB_SOURCE_DMABUF:
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
      if (slab->source_state.dmabuf.mapped_ptr) {
        munmap(slab->source_state.dmabuf.mapped_ptr,
               slab->source_state.dmabuf.mapped_length);
      }
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
      break;
  }

  iree_allocator_t allocator = slab->allocator;
  iree_allocator_free(allocator, slab);

  IREE_TRACE_ZONE_END(z0);
}

void iree_async_slab_release(iree_async_slab_t* slab) {
  if (slab && iree_atomic_ref_count_dec(&slab->ref_count) == 1) {
    iree_async_slab_destroy(slab);
  }
}
