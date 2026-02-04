// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// NUMA-aware memory allocation
//===----------------------------------------------------------------------===//
//
// Provides a cross-platform API for NUMA-placed memory allocation with optional
// huge page support. Platforms without NUMA support (macOS, Emscripten) get
// transparent fallbacks that use standard allocation paths.
//
// This centralizes the platform-specific mmap/mbind/VirtualAllocExNuma code
// that would otherwise be duplicated in buffer pools, slab allocators, and
// other infrastructure requiring NUMA-aware placement.
//
// ## Usage
//
// Basic allocation with no NUMA preference:
//
//   iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
//   void* ptr = NULL;
//   iree_numa_alloc_info_t info;
//   IREE_RETURN_IF_ERROR(iree_numa_alloc(size, &options, &ptr, &info));
//   // Use ptr...
//   iree_numa_free(ptr, &info);
//
// Allocation on a specific NUMA node:
//
//   iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
//   options.node_id = iree_numa_node_for_current_thread();
//   IREE_RETURN_IF_ERROR(iree_numa_alloc(size, &options, &ptr, &info));
//
// Allocation with transparent huge pages:
//
//   iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
//   options.hint_transparent_huge_pages = true;
//   IREE_RETURN_IF_ERROR(iree_numa_alloc(size, &options, &ptr, &info));
//
// ## Platform support
//
//   - Linux/Android: Full NUMA support via mbind()/get_mempolicy() syscalls.
//     Supports explicit huge pages (MAP_HUGETLB) and transparent huge pages
//     (MADV_HUGEPAGE).
//
//   - Windows: NUMA support via VirtualAllocExNuma(). Supports large pages
//     (MEM_LARGE_PAGES) when SeLockMemoryPrivilege is granted.
//
//   - macOS/Emscripten/Other: Single-node fallback using standard allocation.
//     NUMA functions return sensible defaults (1 node, node 0).
//
// ## Thread safety
//
// All functions are safe to call from any thread.

#ifndef IREE_BASE_THREADING_NUMA_H_
#define IREE_BASE_THREADING_NUMA_H_

#include <stdbool.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Identifies a NUMA node in the system (0-based).
typedef uint32_t iree_numa_node_id_t;

// Sentinel indicating no NUMA placement preference.
#define IREE_NUMA_NODE_ANY ((iree_numa_node_id_t)UINT32_MAX)

// Options controlling NUMA-aware allocation.
// Use iree_numa_alloc_options_default() for a zero-initialized default.
typedef struct iree_numa_alloc_options_t {
  // NUMA node for memory placement. IREE_NUMA_NODE_ANY = no preference.
  iree_numa_node_id_t node_id;

  // Minimum alignment for the standard allocator fallback path. Must be a
  // power of two or 0 (= page-aligned by default). When the allocation uses
  // mmap or VirtualAlloc (the common case), the returned pointer is at least
  // page-aligned regardless of this field. This field only affects the
  // standard allocator fallback, which is used when mmap is unavailable.
  iree_host_size_t alignment;

  // Huge page size: 0 = normal pages, or specific size (2MB/1GB).
  // If non-zero and use_explicit_huge_pages is true, the allocation size will
  // be rounded up to this alignment.
  iree_host_size_t huge_page_size;

  // If true, attempt allocation using explicit huge pages (MAP_HUGETLB on
  // Linux, MEM_LARGE_PAGES on Windows). Falls back gracefully if huge pages
  // are not available.
  bool use_explicit_huge_pages;

  // If true, hint to the kernel that transparent huge pages should be used
  // (MADV_HUGEPAGE on Linux). Best-effort: the kernel may ignore the hint.
  // Also used as a fallback when use_explicit_huge_pages is set but explicit
  // huge page allocation fails.
  bool hint_transparent_huge_pages;
} iree_numa_alloc_options_t;

// Allocation method used (for diagnostics and proper cleanup).
typedef enum iree_numa_alloc_method_e {
  // Memory allocated via explicit huge pages (MAP_HUGETLB / MEM_LARGE_PAGES).
  IREE_NUMA_ALLOC_METHOD_EXPLICIT_HUGE_PAGES = 0,
  // Memory allocated via mmap with MADV_HUGEPAGE hint (THP).
  IREE_NUMA_ALLOC_METHOD_TRANSPARENT_HUGE_PAGES,
  // Memory allocated via mmap (or equivalent) without huge pages.
  IREE_NUMA_ALLOC_METHOD_MMAP,
  // Memory allocated via standard allocator (malloc/aligned_alloc).
  IREE_NUMA_ALLOC_METHOD_STANDARD,
} iree_numa_alloc_method_t;

// Tracks how memory was allocated (needed for correct cleanup).
// Returned by iree_numa_alloc() and must be passed to iree_numa_free().
typedef struct iree_numa_alloc_info_t {
  // Actual allocated size (may be rounded up for alignment/huge pages).
  iree_host_size_t allocated_size;
  // Resolved huge page size (0 if huge pages were not used).
  iree_host_size_t huge_page_size;
  // Which allocation path was used.
  iree_numa_alloc_method_t method;
} iree_numa_alloc_info_t;

//===----------------------------------------------------------------------===//
// Default options
//===----------------------------------------------------------------------===//

// Returns default allocation options (no NUMA preference, normal pages).
static inline iree_numa_alloc_options_t iree_numa_alloc_options_default(void) {
  iree_numa_alloc_options_t options = {0};
  options.node_id = IREE_NUMA_NODE_ANY;
  return options;
}

//===----------------------------------------------------------------------===//
// NUMA topology configuration
//===----------------------------------------------------------------------===//

// Maximum NUMA nodes supported. 128 for platforms with NUMA support (Linux,
// Android, Windows), 1 for fallback platforms (macOS, Emscripten, etc).
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_WINDOWS)
#define IREE_NUMA_MAX_NODES 128
#else
#define IREE_NUMA_MAX_NODES 1
#endif

//===----------------------------------------------------------------------===//
// NUMA topology queries
//===----------------------------------------------------------------------===//

// Returns the total number of online NUMA nodes in the system.
// Returns 1 if NUMA is not available or not supported on this platform.
// Thread-safe: uses iree_call_once for lazy initialization.
IREE_API_EXPORT iree_host_size_t iree_numa_node_count(void);

// Returns a bitmap of online NUMA nodes in the system.
// The bitmap has IREE_NUMA_MAX_NODES bits; bit N is set if node N is online.
// The returned bitmap references static storage valid for the process lifetime.
// Thread-safe: uses iree_call_once for lazy initialization.
IREE_API_EXPORT iree_bitmap_t iree_numa_online_nodes(void);

// Returns the NUMA node ID of the current thread's CPU.
// Returns 0 if NUMA is not available or the query fails.
IREE_API_EXPORT iree_numa_node_id_t iree_numa_node_for_current_thread(void);

//===----------------------------------------------------------------------===//
// NUMA-aware allocation
//===----------------------------------------------------------------------===//

// Allocates memory with NUMA placement and optional huge page backing.
//
// On success, |out_ptr| points to the allocated memory and |out_info| contains
// the information needed for iree_numa_free(). The caller must pass the same
// |out_info| to iree_numa_free() for correct cleanup.
//
// The allocation falls back gracefully:
//  1. Explicit huge pages (if requested)
//  2. mmap + MADV_HUGEPAGE (if THP requested, or as fallback from explicit)
//  3. mmap with NUMA binding (if NUMA requested)
//  4. Standard allocation (final fallback)
IREE_API_EXPORT iree_status_t
iree_numa_alloc(iree_host_size_t size, const iree_numa_alloc_options_t* options,
                void** out_ptr, iree_numa_alloc_info_t* out_info);

// Frees memory allocated with iree_numa_alloc().
// |info| must be the same value returned by the corresponding iree_numa_alloc()
// call. Passing NULL |ptr| is a no-op.
IREE_API_EXPORT void iree_numa_free(void* ptr,
                                    const iree_numa_alloc_info_t* info);

//===----------------------------------------------------------------------===//
// NUMA memory binding
//===----------------------------------------------------------------------===//

// Binds existing memory to a NUMA node. Best-effort: returns OK even if the
// platform does not support post-allocation binding (Windows) or does not have
// NUMA at all (macOS, Emscripten). On Linux, uses mbind() to migrate pages.
//
// Passing IREE_NUMA_NODE_ANY for |node_id| is a no-op (returns OK).
IREE_API_EXPORT iree_status_t iree_numa_bind_memory(
    void* ptr, iree_host_size_t size, iree_numa_node_id_t node_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_NUMA_H_
