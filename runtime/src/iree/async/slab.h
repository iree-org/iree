// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ref-counted contiguous memory block for zero-copy I/O.
//
// A slab represents a contiguous region of physical memory (buffer_size *
// buffer_count bytes) that can be registered with one or more proactors for
// zero-copy send and receive operations. Slabs are the foundational memory
// type in the slab/region/pool hierarchy:
//
//   Slab    Physical memory. Ref-counted. No backend knowledge.
//     |
//   Region  Registration handle. Created by proactor. N per slab.
//     |
//   Pool    Lock-free freelist over a region. Send-side acquire/release.
//
// A slab owns its memory and manages its lifetime via reference counting.
// When the last reference is released, the memory is freed according to
// the slab's source (NUMA free, nothing for wrapped memory, munmap for
// dmabuf).
//
// ## Creation paths
//
// Slabs can be created in three ways:
//
//   iree_async_slab_create() -- NUMA-aware allocation with optional huge
//     pages. This is the typical path for send/recv buffer pools.
//
//   iree_async_slab_wrap() -- Wraps existing memory not owned by the slab.
//     The caller is responsible for ensuring the memory outlives the slab.
//     Useful for wrapping pre-allocated buffers or memory from external
//     libraries.
//
//   iree_async_slab_import_dmabuf() -- Maps a dmabuf file descriptor into
//     the process address space. The slab owns the mapping (munmap on
//     release) but not the fd (caller manages fd lifetime).
//
// ## Registration
//
// A slab has no backend knowledge. To use slab memory for zero-copy I/O,
// register it with a proactor via iree_async_proactor_register_slab(),
// which produces an iree_async_region_t with backend-specific handles.
// Multiple registrations per slab are allowed (for multi-backend or
// multi-proactor configurations).
//
// ## Thread safety
//
// Retain/release are thread-safe (atomic ref count). All other fields are
// immutable after creation.

#ifndef IREE_ASYNC_SLAB_H_
#define IREE_ASYNC_SLAB_H_

#include "iree/async/affinity.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/numa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Slab source
//===----------------------------------------------------------------------===//

// How the slab's memory was obtained. Determines teardown behavior.
typedef enum iree_async_slab_source_e {
  // Memory allocated via iree_numa_alloc (NUMA-aware, possibly huge pages).
  // Freed via iree_numa_free on release.
  IREE_ASYNC_SLAB_SOURCE_NUMA_ALLOC = 0,

  // Externally owned memory wrapped by the slab.
  // Not freed on release (caller manages lifetime).
  IREE_ASYNC_SLAB_SOURCE_WRAPPED,

  // Memory obtained by mmapping a dmabuf fd.
  // Freed via munmap on release. The fd is not owned (caller manages).
  IREE_ASYNC_SLAB_SOURCE_DMABUF,
} iree_async_slab_source_t;

//===----------------------------------------------------------------------===//
// Slab options
//===----------------------------------------------------------------------===//

// Configuration for iree_async_slab_create().
typedef struct iree_async_slab_options_t {
  // Size of each buffer in the slab (bytes). Must be non-zero.
  // Typically aligned to page size or cache line for optimal DMA performance.
  iree_host_size_t buffer_size;

  // Number of buffers in the slab. Must be non-zero.
  // For io_uring buffer rings: must be a power of 2 (kernel requirement for
  // provided buffer rings).
  iree_host_size_t buffer_count;

  // Size of huge pages to use when use_explicit_huge_pages is true.
  // Common values:
  //   0 = auto-detect system default (typically 2MB on x86_64)
  //   2097152 = 2MB (standard huge pages, good for slabs 8MB-256MB)
  //   1073741824 = 1GB (requires hugepagesz=1G kernel param, for slabs 256MB+)
  //
  // The slab size is rounded up to this alignment. Using 1GB pages for small
  // slabs wastes significant memory.
  iree_host_size_t huge_page_size;

  // Locality domain for NUMA-aware allocation. NULL means default placement
  // (the system allocator chooses, typically local to the calling thread's
  // NUMA node).
  const iree_async_affinity_t* affinity;

  // If true, attempt to allocate using explicit huge pages (MAP_HUGETLB on
  // Linux). Provides guaranteed huge page backing with lower TLB pressure,
  // beneficial for large slabs (32MB+) in latency-sensitive scenarios.
  //
  // Falls back gracefully to normal pages if:
  //   - Huge pages are not configured on the system
  //   - The requested size is not a multiple of the huge page size
  //   - The system runs out of huge pages
  bool use_explicit_huge_pages;

  // If true and explicit huge pages fail or are not requested, hint to the
  // kernel that transparent huge pages (THP) should be used via MADV_HUGEPAGE.
  // Best-effort: the kernel may or may not honor the hint.
  bool hint_transparent_huge_pages;
} iree_async_slab_options_t;

//===----------------------------------------------------------------------===//
// Slab
//===----------------------------------------------------------------------===//

// Ref-counted contiguous memory block.
// All fields are immutable after creation except the ref count.
typedef struct iree_async_slab_t {
  iree_atomic_ref_count_t ref_count;

  // Allocator used for freeing this struct.
  iree_allocator_t allocator;

  // Start of buffer memory.
  void* base_ptr;

  // Per-buffer configuration.
  iree_host_size_t buffer_size;
  iree_host_size_t buffer_count;

  // Total slab size (buffer_size * buffer_count).
  iree_host_size_t total_size;

  // How the memory was obtained (determines teardown).
  iree_async_slab_source_t source;

  // Source-specific teardown state.
  union {
    struct {
      iree_numa_alloc_info_t alloc_info;
    } numa;
    struct {
      void* mapped_ptr;
      iree_host_size_t mapped_length;
    } dmabuf;
  } source_state;
} iree_async_slab_t;

//===----------------------------------------------------------------------===//
// Slab lifecycle
//===----------------------------------------------------------------------===//

// Returns default slab options. All fields zero except those that require
// non-zero defaults.
static inline iree_async_slab_options_t iree_async_slab_options_default(void) {
  iree_async_slab_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

// Allocates a NUMA-aware slab with optional huge page support.
//
// Allocates (buffer_count * buffer_size) bytes of contiguous memory. If
// options.affinity is non-NULL, the memory is placed on the specified NUMA
// node. Huge page options control explicit or transparent huge page use.
//
// On failure, no resources are leaked and |out_slab| is set to NULL.
iree_status_t iree_async_slab_create(iree_async_slab_options_t options,
                                     iree_allocator_t allocator,
                                     iree_async_slab_t** out_slab);

// Wraps existing memory as a slab without taking ownership.
//
// The caller is responsible for ensuring |base_ptr| remains valid and at a
// stable address for the lifetime of the slab (until the last reference is
// released). The slab does not free the memory on release.
//
// |base_ptr| must point to at least (buffer_size * buffer_count) bytes of
// contiguous memory.
iree_status_t iree_async_slab_wrap(void* base_ptr, iree_host_size_t buffer_size,
                                   iree_host_size_t buffer_count,
                                   iree_allocator_t allocator,
                                   iree_async_slab_t** out_slab);

// Imports a dmabuf file descriptor as a slab.
//
// Maps the dmabuf into the process address space via mmap. The slab owns the
// mapping (unmaps on release) but does NOT own the fd (caller manages fd
// lifetime; fd must remain valid for the slab's lifetime).
//
// |access_flags| determines mmap protection:
//   IREE_ASYNC_BUFFER_ACCESS_FLAG_READ → PROT_READ
//   IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE → PROT_WRITE
//
// The mapped region starts at |offset| within the dmabuf and covers
// (buffer_size * buffer_count) bytes.
iree_status_t iree_async_slab_import_dmabuf(int dmabuf_fd, uint64_t offset,
                                            iree_host_size_t buffer_size,
                                            iree_host_size_t buffer_count,
                                            uint32_t access_flags,
                                            iree_allocator_t allocator,
                                            iree_async_slab_t** out_slab);

// Retains a reference to the slab.
static inline void iree_async_slab_retain(iree_async_slab_t* slab) {
  iree_atomic_ref_count_inc(&slab->ref_count);
}

// Releases a reference to the slab. When the last reference is released,
// frees the memory according to the slab's source and frees the slab struct.
void iree_async_slab_release(iree_async_slab_t* slab);

//===----------------------------------------------------------------------===//
// Slab query
//===----------------------------------------------------------------------===//

// Returns the start of the slab's buffer memory.
static inline void* iree_async_slab_base_ptr(const iree_async_slab_t* slab) {
  return slab->base_ptr;
}

// Returns the per-buffer size in bytes.
static inline iree_host_size_t iree_async_slab_buffer_size(
    const iree_async_slab_t* slab) {
  return slab->buffer_size;
}

// Returns the number of buffers in the slab.
static inline iree_host_size_t iree_async_slab_buffer_count(
    const iree_async_slab_t* slab) {
  return slab->buffer_count;
}

// Returns the total slab size in bytes (buffer_size * buffer_count).
static inline iree_host_size_t iree_async_slab_total_size(
    const iree_async_slab_t* slab) {
  return slab->total_size;
}

// Returns how the slab's memory was obtained.
static inline iree_async_slab_source_t iree_async_slab_source(
    const iree_async_slab_t* slab) {
  return slab->source;
}

// Returns the NUMA allocation info (only meaningful for NUMA_ALLOC source).
// Useful for diagnostics to determine whether huge pages were used.
static inline const iree_numa_alloc_info_t* iree_async_slab_numa_alloc_info(
    const iree_async_slab_t* slab) {
  return &slab->source_state.numa.alloc_info;
}

// Returns a pointer to the start of buffer |index| within the slab.
static inline void* iree_async_slab_buffer_ptr(const iree_async_slab_t* slab,
                                               iree_host_size_t index) {
  return (uint8_t*)slab->base_ptr + index * slab->buffer_size;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_SLAB_H_
