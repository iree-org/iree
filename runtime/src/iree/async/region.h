// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Registered memory region.
//
// An iree_async_region_t represents a contiguous memory range registered with
// a proactor for zero-copy I/O. Registration produces backend-specific handles
// (RDMA memory region keys, io_uring buffer group IDs, dma-buf descriptors)
// that enable kernel-bypass or zero-copy data paths.
//
// Regions are ref-counted. When the last reference is released, the region's
// destroy callback is invoked to deregister from the kernel, release the slab
// reference, and free the region struct.
//
// Regions are created by iree_async_proactor_register_slab() (for slab-backed
// memory) or iree_async_proactor_register_buffer() (for arbitrary host memory).
// Callers interact with regions primarily through iree_async_span_t values.
//
// ## Slab relationship
//
// When a region is backed by a slab, it holds a retained reference to the slab.
// This ensures the physical memory remains valid for the region's entire
// lifetime. The slab is released when the region is destroyed.
//
// ## Backend teardown
//
// Each region has a destroy_fn callback set by the creating proactor backend.
// This callback handles backend-specific deregistration (e.g.,
// IORING_UNREGISTER_BUFFERS, ibv_dereg_mr) before releasing the slab reference
// and freeing the region struct.

#ifndef IREE_ASYNC_REGION_H_
#define IREE_ASYNC_REGION_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_slab_t iree_async_slab_t;
typedef struct iree_async_region_t iree_async_region_t;

//===----------------------------------------------------------------------===//
// Region
//===----------------------------------------------------------------------===//

// Access flags describing how registered memory will be used by I/O operations.
// Backends use these to select optimal registration parameters (e.g., RDMA
// access keys, page pinning options).
enum iree_async_buffer_access_flag_bits_e {
  IREE_ASYNC_BUFFER_ACCESS_FLAG_NONE = 0u,

  // I/O operations will read from this buffer (send, file write).
  IREE_ASYNC_BUFFER_ACCESS_FLAG_READ = 1u << 0,

  // I/O operations will write into this buffer (recv, file read).
  IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE = 1u << 1,

  // Remote peers may read this buffer directly (RDMA read).
  IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ = 1u << 2,

  // Remote peers may write this buffer directly (RDMA write).
  IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE = 1u << 3,
};
typedef uint32_t iree_async_buffer_access_flags_t;

// Callback invoked when a region's ref count reaches zero.
// Must handle all backend-specific deregistration (kernel syscalls, etc.),
// release the slab reference if present, and free the region struct.
// MUST NOT FAIL. If teardown encounters an error, it must abort.
typedef void (*iree_async_region_destroy_fn_t)(iree_async_region_t* region);

typedef void(IREE_API_PTR* iree_async_buffer_recycle_fn_t)(
    void* user_data, uint32_t buffer_index);

// A callback issued to recycle a buffer back to its source (e.g., PBUF_RING).
typedef struct {
  // Callback function pointer. NULL if no recycling is needed.
  iree_async_buffer_recycle_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_async_buffer_recycle_callback_t;

// Returns a no-op buffer recycle callback that implies that no cleanup is
// required.
static inline iree_async_buffer_recycle_callback_t
iree_async_buffer_recycle_callback_null(void) {
  return (iree_async_buffer_recycle_callback_t){NULL, NULL};
}

// Discriminates the backend-specific handles in the region.
//
// Each type corresponds to an active member of the handles union in
// iree_async_region_t. The type determines how the proactor interacts with
// the kernel for I/O operations on this region's memory.
enum iree_async_region_type_e {
  // No backend-specific handles. Used for host buffer registrations where the
  // proactor only needs the base_ptr/length for standard read/write syscalls
  // and for READ-only slab registrations (send path is type-agnostic —
  // iree_async_span_ptr() works without kernel handles).
  IREE_ASYNC_REGION_TYPE_NONE = 0u,

  // RDMA memory registration (ibv_reg_mr or equivalent). Produces local/remote
  // access keys (lkey/rkey) and a memory region handle (ibv_mr*) that enable
  // kernel-bypass data transfer via RDMA verbs. Both the local and remote host
  // must register memory before RDMA read/write operations can target it.
  IREE_ASYNC_REGION_TYPE_RDMA,

  // io_uring kernel registration. For WRITE-access slabs, creates a provided
  // buffer ring (PBUF_RING) with a buffer_group_id for kernel-managed buffer
  // selection during recv. For READ-access slabs, registers into the kernel's
  // fixed buffer table with a base_buffer_index for zero-copy send via
  // IORING_OP_SEND_ZC with IOSQE_FIXED_FILE.
  IREE_ASYNC_REGION_TYPE_IOURING,

  // DMA buffer backed by a file descriptor. Used for GPU-accessible memory
  // shared via dma-buf (Linux DMA buffer sharing framework). The fd and offset
  // identify the region within the exported dma-buf for splice/sendfile-style
  // zero-copy between devices.
  IREE_ASYNC_REGION_TYPE_DMABUF,

  // Emulated registration (metadata-only, no kernel handles). Used by backends
  // that track buffer regions in userspace without kernel-level registration.
  // Buffer selection for recv is proactor-managed (pool acquire) rather than
  // kernel-managed (PBUF_RING).
  IREE_ASYNC_REGION_TYPE_EMULATED,
};
typedef uint8_t iree_async_region_type_t;

// A registered memory region. Created by the proactor during slab or buffer
// registration. Holds backend-specific handles for zero-copy I/O.
typedef struct iree_async_region_t {
  iree_atomic_ref_count_t ref_count;

  // The proactor that owns this registration. Not retained.
  // The proactor MUST outlive the region.
  iree_async_proactor_t* proactor;

  // Optional: slab backing this region. Retained reference.
  // NULL for non-slab regions (e.g., arbitrary host buffer registration).
  iree_async_slab_t* slab;

  // Backend-specific destroy callback. Set by the creating proactor.
  // Called when ref count reaches zero. Handles kernel deregistration,
  // slab release, and region struct free.
  // NULL falls back to legacy behavior (iree_allocator_free via proactor).
  iree_async_region_destroy_fn_t destroy_fn;

  // Optional: buffer recycling callback for recv regions.
  // Set by the proactor during register_slab when WRITE access is requested.
  // Called from lease release to return a buffer to the provided buffer ring.
  // fn is NULL for send-only regions (freelist recycling is handled by pool).
  iree_async_buffer_recycle_callback_t recycle;

  // Discriminates the active member of the handles union.
  iree_async_region_type_t type;

  // Access flags set at registration time.
  iree_async_buffer_access_flags_t access_flags;

  // Registered memory range.
  void* base_ptr;
  iree_host_size_t length;

  // Portable indexed buffer configuration (for buffer pools).
  // Set by proactor during register_slab for indexed buffer regions.
  // 0 if this region is not subdivided into indexed buffers.
  // These are slab properties, independent of backend.
  iree_host_size_t buffer_size;
  uint32_t buffer_count;

  // Backend-specific handles produced by registration.
  union {
    struct {
      uint32_t lkey;  // Local access key.
      uint32_t rkey;  // Remote access key.
      void* mr;       // ibv_mr* or equivalent.
    } rdma;
    struct {
      // For provided buffer ring (recv - kernel selects buffer).
      // -1 if not in a provided buffer ring.
      int16_t buffer_group_id;

      // For fixed buffer table (send - application selects buffer).
      // buf_index = span.offset / buffer_size + base_buffer_index
      // Starting index in kernel's fixed buffer table.
      uint16_t base_buffer_index;
    } iouring;
    struct {
      int fd;
      uint64_t offset;
    } dmabuf;
    uint64_t opaque[4];  // Catch-all for unknown backends.
  } handles;
} iree_async_region_t;

// Destroys the region. Invokes the destroy_fn callback for backend-specific
// teardown, or falls back to freeing via the proactor's allocator.
void iree_async_region_destroy(iree_async_region_t* region);

// Retains a reference to the region.
static inline void iree_async_region_retain(iree_async_region_t* region) {
  iree_atomic_ref_count_inc(&region->ref_count);
}

// Releases a reference to the region. When the last reference is released,
// invokes the destroy callback for backend-specific deregistration and frees
// the region struct. Safe to call with NULL.
static inline void iree_async_region_release(iree_async_region_t* region) {
  if (region && iree_atomic_ref_count_dec(&region->ref_count) == 1) {
    iree_async_region_destroy(region);
  }
}

// Returns the host pointer for a given offset within the region.
static inline void* iree_async_region_ptr(const iree_async_region_t* region,
                                          iree_host_size_t offset) {
  return (uint8_t*)region->base_ptr + offset;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_REGION_H_
