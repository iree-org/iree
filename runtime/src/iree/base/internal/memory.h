// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_MEMORY_H_
#define IREE_BASE_INTERNAL_MEMORY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Memory subsystem information and control
//===----------------------------------------------------------------------===//

enum iree_memory_feature_bits_e {
  IREE_MEMORY_FEATURE_NONE = 0u,

  // Indicates whether executable pages may be allocated within the process.
  // Some platforms or release environments have restrictions on whether
  // executable pages may be allocated from user code (such as iOS).
  IREE_MEMORY_FEATURE_ALLOCATABLE_EXECUTABLE_PAGES = 1u << 0,
};
typedef uint32_t iree_memory_features_t;

// System platform/environment information defining memory parameters.
// These can be used to control application behavior (such as whether to enable
// a JIT if executable pages can be allocated) and allow callers to compute
// memory ranges based on the variable page size of the platform.
typedef struct iree_memory_info_t {
  // The page size and the granularity of page protection and commitment. This
  // is the page size used by the iree_memory_view_t functions.
  iree_host_size_t normal_page_size;

  // The granularity for the starting address at which virtual memory can be
  // allocated.
  iree_host_size_t normal_page_granularity;

  // The minimum page size and granularity for large pages or 0 if unavailable.
  // To use large pages the size and alignment must be a multiple of this value
  // and the IREE_MEMORY_VIEW_FLAG_LARGE_PAGES must be set.
  iree_host_size_t large_page_granularity;

  // Bitfield of features supported by the platform.
  iree_memory_features_t supported_features;
} iree_memory_info_t;

// Queries the system platform/environment memory information.
// Callers should cache the results to avoid repeated queries, such as storing
// the used fields in an allocator upon initialization to reuse during
// allocations made via the allocator.
iree_memory_info_t iree_memory_query_info(void);

// Enter a W^X region where pages will be changed RW->RX or RX->RW and write
// protection should be suspended. Only effects the calling thread and must be
// paired with iree_memory_jit_context_end.
void iree_memory_jit_context_begin(void);

// Exits a W^X region previously entered with iree_memory_jit_context_begin.
void iree_memory_jit_context_end(void);

// Flushes the CPU instruction cache for a given range of bytes.
// May be a no-op depending on architecture, but must be called prior to
// executing code from any pages that have been written during load.
void iree_memory_flush_icache(void* base_address, iree_host_size_t length);

//===----------------------------------------------------------------------===//
// C11 aligned_alloc shim
//===----------------------------------------------------------------------===//

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// WARNING: DO NOT USE THIS FUNCTION FOR NORMAL ALLOCATIONS.
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// This function BYPASSES the iree_allocator_t infrastructure entirely. Using it
// means:
//   - No integration with custom allocators (arena, pool, tracking, etc)
//   - No ability to use iree_allocator_free() - must use iree_aligned_free()
//   - Memory accounting and debugging tools cannot track these allocations
//   - Breaks the allocator abstraction that enables embedded/custom deployments
//
// VALID USES (require explicit approval):
//   - NUMA allocation fallback (numa_*.c) - special physical placement needs
//   - Status storage (status.c internal) - bootstrapping before allocators
//     exist and that allocators use for handling when they fail
//   - Test/benchmark code where allocator integration is irrelevant
//
// FOR ALIGNED ALLOCATIONS IN NORMAL CODE:
//   - Embed aligned data within a larger struct that is normally allocated
//   - Use iree_allocator_malloc with extra padding and manual alignment
//   - Reconsider whether 64-byte alignment is actually necessary
//
// If you think you need this function, you are almost certainly wrong.
// Ask for guidance before using it.

// Allocates |size| bytes of uninitialized storage whose alignment is specified
// by |alignment|. The |size| parameter must be an integral multiple of
// |alignment|. The returned pointer must be freed using iree_aligned_free.
//
// This is a thin wrapper around C11's aligned_alloc when available:
// https://en.cppreference.com/w/c/memory/aligned_alloc
//
// When not available (such as on MSVC) a fallback will be used:
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/posix_memalign.html
iree_status_t iree_aligned_alloc(iree_host_size_t alignment,
                                 iree_host_size_t size, void** out_ptr);
void iree_aligned_free(void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_MEMORY_H_
