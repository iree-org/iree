// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal implementation header for the NUMA abstraction layer.
// Provides platform selection and shared helpers used by the platform-specific
// implementations (numa_linux.c, numa_win32.c, numa_fallback.c).
//
// This file must NOT be included by code outside of the numa_*.c files.

#ifndef IREE_BASE_THREADING_NUMA_IMPL_H_
#define IREE_BASE_THREADING_NUMA_IMPL_H_

// Must be defined before system headers for GNU extensions (mbind, etc).
#define _GNU_SOURCE 1

#include "iree/base/api.h"
#include "iree/base/internal/memory.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/numa.h"

//===----------------------------------------------------------------------===//
// Platform selection
//===----------------------------------------------------------------------===//

// Each platform implementation guards itself with IREE_PLATFORM_* checks.
// Only one implementation will be active per compilation unit:
//   - numa_linux.c: IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
//   - numa_win32.c: IREE_PLATFORM_WINDOWS
//   - numa_fallback.c: everything else (macOS, Emscripten, generic)
//
// The fallback is also used as the final fallback within platform
// implementations when specific features are unavailable.

//===----------------------------------------------------------------------===//
// Shared constants
//===----------------------------------------------------------------------===//

#define IREE_NUMA_HUGE_PAGE_SIZE_2MB ((iree_host_size_t)(2 * 1024 * 1024))
#define IREE_NUMA_HUGE_PAGE_SIZE_1GB ((iree_host_size_t)(1024 * 1024 * 1024))

// Returns 2MB if |requested_size| is 0 (auto-detect), otherwise passes through.
static inline iree_host_size_t iree_numa_resolve_huge_page_size(
    iree_host_size_t requested_size) {
  return requested_size == 0 ? IREE_NUMA_HUGE_PAGE_SIZE_2MB : requested_size;
}

//===----------------------------------------------------------------------===//
// Shared initialization state
//===----------------------------------------------------------------------===//

// Calculate words needed for IREE_NUMA_MAX_NODES bitmap.
#define IREE_NUMA_NODE_MASK_WORDS                          \
  ((IREE_NUMA_MAX_NODES + IREE_BITMAP_BITS_PER_WORD - 1) / \
   IREE_BITMAP_BITS_PER_WORD)

// Shared state for lazy initialization (all zero-initialized).
// Defined in each platform implementation file.
extern iree_once_flag iree_numa_init_flag;
extern uint64_t iree_numa_online_nodes_storage[IREE_NUMA_NODE_MASK_WORDS];
extern iree_host_size_t iree_numa_online_node_count;

// Platform-specific initialization function (called once via iree_call_once).
void iree_numa_initialize(void);

// Helper to get bitmap view of the online nodes storage.
static inline iree_bitmap_t iree_numa_online_nodes_bitmap(void) {
  iree_bitmap_t bitmap;
  bitmap.bit_count = IREE_NUMA_MAX_NODES;
  bitmap.words = iree_numa_online_nodes_storage;
  return bitmap;
}

#endif  // IREE_BASE_THREADING_NUMA_IMPL_H_
