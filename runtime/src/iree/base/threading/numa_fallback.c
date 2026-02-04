// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/numa_impl.h"

// Fallback implementation for platforms without NUMA support (macOS,
// Emscripten, generic). Uses standard aligned allocation, reports a single
// NUMA node, and treats bind_memory as a no-op.

#if !defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID) && \
    !defined(IREE_PLATFORM_WINDOWS)

#include <string.h>

// Shared initialization state.
iree_once_flag iree_numa_init_flag = IREE_ONCE_FLAG_INIT;
uint64_t iree_numa_online_nodes_storage[IREE_NUMA_NODE_MASK_WORDS] = {0};
iree_host_size_t iree_numa_online_node_count = 0;

void iree_numa_initialize(void) {
  // Fallback: single node 0.
  iree_bitmap_t bitmap = iree_numa_online_nodes_bitmap();
  iree_bitmap_set(bitmap, 0);
  iree_numa_online_node_count = 1;
}

IREE_API_EXPORT iree_host_size_t iree_numa_node_count(void) {
  iree_call_once(&iree_numa_init_flag, iree_numa_initialize);
  return iree_numa_online_node_count;
}

IREE_API_EXPORT iree_bitmap_t iree_numa_online_nodes(void) {
  iree_call_once(&iree_numa_init_flag, iree_numa_initialize);
  return iree_numa_online_nodes_bitmap();
}

IREE_API_EXPORT iree_numa_node_id_t iree_numa_node_for_current_thread(void) {
  return 0;
}

IREE_API_EXPORT iree_status_t
iree_numa_alloc(iree_host_size_t size, const iree_numa_alloc_options_t* options,
                void** out_ptr, iree_numa_alloc_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_ptr);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_ptr = NULL;
  memset(out_info, 0, sizeof(*out_info));

  // Determine alignment: use requested, or fall back to page size.
  iree_host_size_t alignment = options->alignment;
  if (alignment == 0) {
    alignment = iree_memory_query_info().normal_page_size;
  }

  // Round size up to alignment (with overflow check).
  iree_host_size_t aligned_size = 0;
  if (!iree_host_size_checked_align(size, alignment, &aligned_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "size %" PRIhsz " alignment to %" PRIhsz " overflows", size, alignment);
  }

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_aligned_alloc(alignment, aligned_size, &ptr));

  *out_ptr = ptr;
  out_info->allocated_size = aligned_size;
  out_info->huge_page_size = 0;
  out_info->method = IREE_NUMA_ALLOC_METHOD_STANDARD;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_numa_free(void* ptr,
                                    const iree_numa_alloc_info_t* info) {
  if (!ptr) return;
  iree_aligned_free(ptr);
}

IREE_API_EXPORT iree_status_t iree_numa_bind_memory(
    void* ptr, iree_host_size_t size, iree_numa_node_id_t node_id) {
  // No NUMA support on this platform. Best-effort: succeed silently.
  return iree_ok_status();
}

#endif  // !IREE_PLATFORM_LINUX && !IREE_PLATFORM_ANDROID &&
        // !IREE_PLATFORM_WINDOWS
