// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ELF_PLATFORM_H_
#define IREE_HAL_LOCAL_ELF_PLATFORM_H_

#include "iree/base/api.h"

// TODO(benvanik): move some of this to iree/base/internal/. A lot of this code
// comes from an old partial implementation of memory objects that should be
// finished. When done it will replace the need for all of these platform files.

//==============================================================================
// Alignment utilities
//==============================================================================

// Defines a range of bytes with any arbitrary alignment.
// Most operations will adjust this range by the allocation granularity, meaning
// that a range that stradles a page boundary will be specifying multiple pages
// (such as offset=1, length=4096 with a page size of 4096 indicating 2 pages).
typedef struct iree_byte_range_t {
  iree_host_size_t offset;
  iree_host_size_t length;
} iree_byte_range_t;

static inline uintptr_t iree_page_align_start(uintptr_t addr,
                                              iree_host_size_t page_alignment) {
  return addr & (~(page_alignment - 1));
}

static inline uintptr_t iree_page_align_end(uintptr_t addr,
                                            iree_host_size_t page_alignment) {
  return iree_page_align_start(addr + (page_alignment - 1), page_alignment);
}

// Computes a page-aligned range base and total length from a range.
// This will produce a starting address <= the range offset and a length >=
// the range length.
static inline void iree_page_align_range(void* base_address,
                                         iree_byte_range_t range,
                                         iree_host_size_t page_alignment,
                                         void** out_start_address,
                                         iree_host_size_t* out_aligned_length) {
  void* range_start = (void*)iree_page_align_start(
      (uintptr_t)base_address + range.offset, page_alignment);
  void* range_end = (void*)iree_page_align_end(
      (uintptr_t)base_address + range.offset + range.length, page_alignment);
  *out_start_address = range_start;
  *out_aligned_length =
      (iree_host_size_t)range_end - (iree_host_size_t)range_start;
}

//==============================================================================
// Memory subsystem information and control
//==============================================================================

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

  // Indicates whether executable pages may be allocated within the process.
  // Some platforms or release environments have restrictions on whether
  // executable pages may be allocated from user code (such as iOS).
  bool can_allocate_executable_pages;
} iree_memory_info_t;

// Queries the system platform/environment memory information.
// Callers should cache the results to avoid repeated queries, such as storing
// the used fields in an allocator upon initialization to reuse during
// allocations made via the allocator.
void iree_memory_query_info(iree_memory_info_t* out_info);

// Enter a W^X region where pages will be changed RW->RX or RX->RW and write
// protection should be suspended. Only effects the calling thread and must be
// paired with iree_memory_jit_context_end.
void iree_memory_jit_context_begin(void);

// Exits a W^X region previously entered with iree_memory_jit_context_begin.
void iree_memory_jit_context_end(void);

//==============================================================================
// Virtual address space manipulation
//==============================================================================

// Defines which access operations are allowed on a view of memory.
// Attempts to perform an access not originally allowed when the view was
// defined may result in process termination/exceptions/sadness on platforms
// with real MMUs and are generally not detectable: treat limited access as a
// fail-safe mechanism only.
enum iree_memory_access_bits_t {
  // Pages in the view may be read by the process.
  // Some platforms may not respect this value being unset meaning that reads
  // will still succeed.
  IREE_MEMORY_ACCESS_READ = 1u << 0,
  // Pages in the view may be written by the process.
  // If unset then writes will result in process termination.
  IREE_MEMORY_ACCESS_WRITE = 1u << 1,
  // Pages in the view can be executed as native machine code.
  // Callers must ensure iree_memory_info_t::can_allocate_executable_pages is
  // true prior to requesting executable memory as certain platforms or release
  // environments may not support allocating/using executable pages.
  IREE_MEMORY_ACCESS_EXECUTE = 1u << 2,
};
typedef uint32_t iree_memory_access_t;

// Flags used to control the behavior of allocated memory views.
enum iree_memory_view_flag_bits_t {
  // TODO(benvanik): pull from memory_object.h.
  IREE_MEMORY_VIEW_FLAG_NONE = 0u,

  // Indicates that the memory may be used to execute code.
  // May be used to ask for special privileges (like MAP_JIT on MacOS).
  IREE_MEMORY_VIEW_FLAG_MAY_EXECUTE = 1u << 10,
};
typedef uint32_t iree_memory_view_flags_t;

// Reserves a range of virtual address space in the host process.
// The base alignment will be that of the page granularity as specified
// (normal or large) in |flags| and |total_length| will be adjusted to match.
//
// The resulting range at |out_base_address| will be uncommitted and
// inaccessible on systems with memory protection. Pages within the range must
// first be committed with iree_memory_view_commit_ranges and then may have
// their access permissions changed with iree_memory_view_protect_ranges.
//
// Implemented by VirtualAlloc+MEM_RESERVE/mmap+PROT_NONE.
iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t host_allocator,
                                       void** out_base_address);

// Releases a range of virtual address
void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t host_allocator);

// Commits pages overlapping the byte ranges defined by |byte_ranges|.
// Ranges will be adjusted to the page granularity of the view.
//
// Implemented by VirtualAlloc+MEM_COMMIT/mmap+!PROT_NONE.
iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access);

// Changes the access protection of view byte ranges defined by |byte_ranges|.
// Ranges will be adjusted to the page granularity of the view.
//
// Implemented by VirtualProtect/mprotect:
//  https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualprotect
//  https://man7.org/linux/man-pages/man2/mprotect.2.html
iree_status_t iree_memory_view_protect_ranges(void* base_address,
                                              iree_host_size_t range_count,
                                              const iree_byte_range_t* ranges,
                                              iree_memory_access_t new_access);

// Flushes the CPU instruction cache for a given range of bytes.
// May be a no-op depending on architecture, but must be called prior to
// executing code from any pages that have been written during load.
void iree_memory_view_flush_icache(void* base_address, iree_host_size_t length);

#endif  // IREE_HAL_LOCAL_ELF_PLATFORM_H_
