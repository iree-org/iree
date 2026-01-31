// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/numa_impl.h"

// Windows NUMA implementation using VirtualAllocExNuma and related APIs.

#if defined(IREE_PLATFORM_WINDOWS)

// clang-format off
#include <windows.h>
// clang-format on
#include <string.h>

//===----------------------------------------------------------------------===//
// NUMA topology queries
//===----------------------------------------------------------------------===//

// Shared initialization state.
iree_once_flag iree_numa_init_flag = IREE_ONCE_FLAG_INIT;
uint64_t iree_numa_online_nodes_storage[IREE_NUMA_NODE_MASK_WORDS] = {0};
iree_host_size_t iree_numa_online_node_count = 0;

void iree_numa_initialize(void) {
  iree_bitmap_t bitmap = iree_numa_online_nodes_bitmap();

  ULONG highest_node = 0;
  if (!GetNumaHighestNodeNumber(&highest_node)) {
    // Fallback: assume single node 0.
    iree_bitmap_set(bitmap, 0);
    iree_numa_online_node_count = 1;
    return;
  }

  // Iterate through nodes 0..highest_node and mark online ones.
  for (ULONG node = 0; node <= highest_node && node < IREE_NUMA_MAX_NODES;
       ++node) {
    // Check if node is online by querying its processor mask.
    GROUP_AFFINITY affinity = {0};
    if (GetNumaNodeProcessorMaskEx((USHORT)node, &affinity)) {
      if (affinity.Mask != 0) {
        iree_bitmap_set(bitmap, (iree_host_size_t)node);
      }
    }
  }

  iree_numa_online_node_count = iree_bitmap_count(bitmap);
  if (iree_numa_online_node_count == 0) {
    // Safety fallback: ensure at least node 0 is present.
    iree_bitmap_set(bitmap, 0);
    iree_numa_online_node_count = 1;
  }
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
  PROCESSOR_NUMBER processor_number;
  GetCurrentProcessorNumberEx(&processor_number);
  USHORT node_number = 0;
  if (!GetNumaProcessorNodeEx(&processor_number, &node_number)) return 0;
  return (iree_numa_node_id_t)node_number;
}

//===----------------------------------------------------------------------===//
// NUMA-aware allocation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_numa_alloc(iree_host_size_t size, const iree_numa_alloc_options_t* options,
                void** out_ptr, iree_numa_alloc_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_ptr);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_ptr = NULL;
  memset(out_info, 0, sizeof(*out_info));

  // Determine allocation flags.
  DWORD alloc_type = MEM_RESERVE | MEM_COMMIT;
  iree_host_size_t alloc_size = size;
  iree_host_size_t huge_page_size = 0;

  // Try large pages if requested.
  if (options->use_explicit_huge_pages) {
    // GetLargePageMinimum() returns the minimum large page size, or 0 if large
    // pages are not supported (requires SeLockMemoryPrivilege).
    SIZE_T large_page_min = GetLargePageMinimum();
    if (large_page_min > 0) {
      huge_page_size = (iree_host_size_t)large_page_min;
      iree_host_size_t aligned_size = 0;
      if (iree_host_size_checked_align(size, huge_page_size, &aligned_size)) {
        alloc_size = aligned_size;
        alloc_type |= MEM_LARGE_PAGES;
      }
      // If alignment overflows, fall through without MEM_LARGE_PAGES.
    }
  }

  // Determine NUMA node.
  DWORD numa_node = (options->node_id == IREE_NUMA_NODE_ANY)
                        ? NUMA_NO_PREFERRED_NODE
                        : (DWORD)options->node_id;

  // Allocate with NUMA placement.
  void* ptr = VirtualAllocExNuma(GetCurrentProcess(), NULL, alloc_size,
                                 alloc_type, PAGE_READWRITE, numa_node);

  // If large pages failed, retry without them.
  if (!ptr && (alloc_type & MEM_LARGE_PAGES)) {
    alloc_type &= ~MEM_LARGE_PAGES;
    alloc_size = size;
    huge_page_size = 0;
    ptr = VirtualAllocExNuma(GetCurrentProcess(), NULL, alloc_size, alloc_type,
                             PAGE_READWRITE, numa_node);
  }

  // If NUMA allocation failed entirely, try without NUMA preference.
  if (!ptr && numa_node != NUMA_NO_PREFERRED_NODE) {
    ptr = VirtualAllocExNuma(GetCurrentProcess(), NULL, alloc_size, alloc_type,
                             PAGE_READWRITE, NUMA_NO_PREFERRED_NODE);
  }

  if (!ptr) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "VirtualAllocExNuma failed for %" PRIhsz " bytes",
                            size);
  }

  *out_ptr = ptr;
  out_info->allocated_size = alloc_size;
  out_info->huge_page_size = huge_page_size;
  if (huge_page_size > 0) {
    out_info->method = IREE_NUMA_ALLOC_METHOD_EXPLICIT_HUGE_PAGES;
  } else {
    out_info->method = IREE_NUMA_ALLOC_METHOD_MMAP;
  }
  return iree_ok_status();
}

IREE_API_EXPORT void iree_numa_free(void* ptr,
                                    const iree_numa_alloc_info_t* info) {
  if (!ptr) return;
  // VirtualFree with MEM_RELEASE frees the entire region regardless of size.
  // The size parameter must be 0 when using MEM_RELEASE.
  VirtualFree(ptr, 0, MEM_RELEASE);
}

IREE_API_EXPORT iree_status_t iree_numa_bind_memory(
    void* ptr, iree_host_size_t size, iree_numa_node_id_t node_id) {
  // Windows does not support post-allocation NUMA binding.
  // Memory placement is determined at VirtualAllocExNuma time.
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
