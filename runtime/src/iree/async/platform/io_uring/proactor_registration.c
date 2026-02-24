// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffer and slab registration for io_uring proactor.
//
// This module handles registration of memory regions with the kernel for
// zero-copy I/O operations. It supports:
//   - Simple buffer registration (wraps memory in a region)
//   - DMA-buf registration (mmaps GPU memory for CPU access)
//   - Slab registration (indexed buffers for zero-copy send/recv)

#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/buffer_ring.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/base/internal/memory.h"

//===----------------------------------------------------------------------===//
// Buffer registration types
//===----------------------------------------------------------------------===//

// Combined allocation for registration entry + region.
// This keeps them together in memory and simplifies cleanup.
typedef struct iree_async_io_uring_buffer_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
} iree_async_io_uring_buffer_registration_t;

// Combined allocation for dmabuf registration entry + region.
// Tracks the mmap state for cleanup.
typedef struct iree_async_io_uring_dmabuf_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
  void* mapped_ptr;  // mmap'd address (for munmap on cleanup)
  iree_host_size_t mapped_length;
  int dmabuf_fd;  // Original fd (not owned, stored for region handles)
  // Sparse buffer table slot for kernel-registered zero-copy send.
  // When >= 0, the mmap'd memory is registered in the kernel's fixed buffer
  // table via IORING_REGISTER_BUFFERS_UPDATE. Cleared during destroy.
  // -1 when not registered (pre-5.19 kernel, table full, or WRITE-only).
  int32_t buffer_table_slot;
} iree_async_io_uring_dmabuf_registration_t;

// Destroy callback for buffer registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_io_uring_buffer_registration_destroy(
    iree_async_region_t* region) {
  // Region is embedded at a known offset in the combined allocation.
  // This is safe because we only set this destroy_fn on regions we create
  // with iree_async_io_uring_buffer_registration_t layout.
  iree_async_io_uring_buffer_registration_t* registration =
      (iree_async_io_uring_buffer_registration_t*)((char*)region -
                                                   offsetof(
                                                       iree_async_io_uring_buffer_registration_t,
                                                       region));
  iree_allocator_free(region->proactor->allocator, registration);
}

// Cleanup function for buffer registrations.
// Called when the registration state is cleaned up.
static void iree_async_io_uring_buffer_registration_cleanup(
    void* entry_ptr, void* proactor_ptr) {
  iree_async_io_uring_buffer_registration_t* registration =
      (iree_async_io_uring_buffer_registration_t*)entry_ptr;
  (void)proactor_ptr;
  // Release our reference. If other code retained the region, it stays alive
  // until those references are released. The destroy callback frees the
  // combined allocation when the last reference drops.
  iree_async_region_release(&registration->region);
}

// Forward declaration: shared helper for clearing kernel buffer table slots.
// Defined in the slab registration section below.
static iree_status_t iree_async_io_uring_clear_buffer_slots_locked(
    iree_async_proactor_io_uring_t* proactor, uint16_t base_slot,
    uint16_t count);

// Destroy callback for dmabuf registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_io_uring_dmabuf_registration_destroy(
    iree_async_region_t* region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_io_uring_dmabuf_registration_t* registration =
      (iree_async_io_uring_dmabuf_registration_t*)((char*)region -
                                                   offsetof(
                                                       iree_async_io_uring_dmabuf_registration_t,
                                                       region));

  // Clear the buffer table slot if registered with the kernel for zero-copy.
  // Failure here means the kernel still holds a reference to memory we're about
  // to munmap (I/O still in-flight or a code bug) — hard-abort to prevent
  // DMA-into-freed-pages corruption.
  if (registration->buffer_table_slot >= 0) {
    iree_async_proactor_io_uring_t* proactor =
        iree_async_proactor_io_uring_cast(region->proactor);
    iree_io_uring_sparse_table_lock(proactor->buffer_table);
    IREE_CHECK_OK(iree_async_io_uring_clear_buffer_slots_locked(
        proactor, (uint16_t)registration->buffer_table_slot, 1));
    iree_io_uring_sparse_table_unlock(proactor->buffer_table);
  }

  // Unmap the dmabuf memory.
  if (registration->mapped_ptr) {
    munmap(registration->mapped_ptr, registration->mapped_length);
  }
  iree_allocator_free(region->proactor->allocator, registration);
  IREE_TRACE_ZONE_END(z0);
}

// Cleanup function for dmabuf registrations.
// Called when the registration state is cleaned up.
static void iree_async_io_uring_dmabuf_registration_cleanup(
    void* entry_ptr, void* proactor_ptr) {
  iree_async_io_uring_dmabuf_registration_t* registration =
      (iree_async_io_uring_dmabuf_registration_t*)entry_ptr;
  (void)proactor_ptr;
  // Release our reference. The destroy callback handles munmap and free.
  iree_async_region_release(&registration->region);
}

//===----------------------------------------------------------------------===//
// Buffer registration vtable implementations
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_io_uring_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_entry = NULL;

  // Allocate combined entry + region.
  iree_async_io_uring_buffer_registration_t* registration = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->allocator, sizeof(*registration),
                                (void**)&registration));

  // Initialize the region.
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = proactor;
  region->slab = NULL;  // Not slab-backed.
  region->destroy_fn = iree_async_io_uring_buffer_registration_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_IOURING;
  region->access_flags = access_flags;
  region->base_ptr = (void*)buffer.data;
  region->length = buffer.data_length;
  region->recycle = iree_async_buffer_recycle_callback_null();
  // For io_uring, we don't use IORING_REGISTER_BUFFERS for single buffers
  // because that API requires registering all buffers at once. Instead, we
  // just wrap the memory in a region for span-based access. The actual I/O
  // uses the raw address in the span.
  region->buffer_size = 0;
  region->buffer_count = 0;                      // Not indexed (use address).
  region->handles.iouring.buffer_group_id = -1;  // Not a provided buffer ring.
  region->handles.iouring.base_buffer_index = 0;

  // Initialize the entry.
  iree_async_buffer_registration_entry_t* entry = &registration->entry;
  entry->next = NULL;
  entry->proactor = proactor;
  entry->cleanup_fn = iree_async_io_uring_buffer_registration_cleanup;
  entry->region = region;

  // Link into the caller's state.
  iree_async_buffer_registration_state_add(state, entry);

  *out_entry = entry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_async_proactor_io_uring_register_dmabuf(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_entry = NULL;

  // TODO(benvanik): Implement true devmem TCP zero-copy when available.
  //
  // devmem TCP enables GPU→NIC zero-copy without CPU-side mmap:
  //   - Kernel 6.12+: RX path (SO_DEVMEM_DONTNEED, SCM_DEVMEM_DMABUF cmsg)
  //   - Kernel 6.13+: TX path
  //   - Requires NIC with header-split support (mlx5, ice, etc.)
  //   - Requires hardware flow steering configuration via ethtool
  //   - Requires netlink binding of dmabuf to specific RX/TX queues
  //
  // Current fallback: mmap the dmabuf and use standard I/O paths.
  // This provides coherent access to GPU memory but involves CPU copies.

  // Determine mmap protection flags from access flags.
  int prot = 0;
  if (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ) prot |= PROT_READ;
  if (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE) prot |= PROT_WRITE;

  // mmap requires page-aligned offset. Align offset down to page boundary and
  // adjust length up to cover the full requested range. We track the delta so
  // we can set base_ptr to point to the user's requested data, not the page
  // boundary.
  iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  uint64_t aligned_offset = offset & ~((uint64_t)page_size - 1);
  iree_host_size_t offset_delta = (iree_host_size_t)(offset - aligned_offset);
  iree_host_size_t aligned_length =
      iree_host_align(offset_delta + length, page_size);

  // mmap the dmabuf fd to get a CPU-accessible pointer.
  void* mapped_ptr =
      mmap(NULL, aligned_length, prot, MAP_SHARED, dmabuf_fd, aligned_offset);
  if (mapped_ptr == MAP_FAILED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "mmap of dmabuf fd %d failed", dmabuf_fd);
  }

  // Allocate registration struct.
  iree_async_io_uring_dmabuf_registration_t* registration = NULL;
  iree_status_t status = iree_allocator_malloc(
      base_proactor->allocator, sizeof(*registration), (void**)&registration);
  if (!iree_status_is_ok(status)) {
    munmap(mapped_ptr, aligned_length);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(registration, 0, sizeof(*registration));
  registration->buffer_table_slot = -1;

  // On 5.19+ with a sparse buffer table: register the mmap'd memory as a fixed
  // buffer so the send path can use zero-copy (SEND_ZC with IOSQE_FIXED_FILE).
  // If the sparse table is full or the kernel rejects the registration, fall
  // back to mmap-only access (DMABUF region type) — the mmap still provides
  // correct CPU access, just without zero-copy send optimization.
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  if ((access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ) &&
      proactor->buffer_table != NULL) {
    iree_io_uring_sparse_table_lock(proactor->buffer_table);
    int32_t slot =
        iree_io_uring_sparse_table_acquire(proactor->buffer_table, 1);
    if (slot >= 0) {
      struct iovec iov = {
          .iov_base = (uint8_t*)mapped_ptr + offset_delta,
          .iov_len = length,
      };
      iree_io_uring_rsrc_update2_t update = {
          .offset = (uint32_t)slot,
          .resv = 0,
          .data = (uint64_t)(uintptr_t)&iov,
          .tags = 0,
          .nr = 1,
          .resv2 = 0,
      };
      long ret = 0;
      do {
        ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                      IREE_IORING_REGISTER_BUFFERS_UPDATE, &update,
                      sizeof(update));
      } while (ret < 0 && errno == EINTR);
      if (ret < 0) {
        iree_io_uring_sparse_table_release(proactor->buffer_table,
                                           (uint16_t)slot, 1);
      } else {
        registration->buffer_table_slot = slot;
      }
    }
    iree_io_uring_sparse_table_unlock(proactor->buffer_table);
  }

  // Initialize the region. base_ptr points to the user's requested offset
  // within the mapped range.
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->slab = NULL;  // Not slab-backed.
  region->destroy_fn = iree_async_io_uring_dmabuf_registration_destroy;
  region->base_ptr = (uint8_t*)mapped_ptr + offset_delta;
  region->length = length;
  region->access_flags = access_flags;
  region->recycle = iree_async_buffer_recycle_callback_null();

  if (registration->buffer_table_slot >= 0) {
    // Registered in kernel buffer table — use IOURING type so the send path
    // picks up zero-copy via SEND_ZC with the fixed buffer index.
    region->type = IREE_ASYNC_REGION_TYPE_IOURING;
    region->buffer_size = length;
    region->buffer_count = 1;
    region->handles.iouring.buffer_group_id = -1;
    region->handles.iouring.base_buffer_index =
        (uint16_t)registration->buffer_table_slot;
  } else {
    // No kernel registration — mmap-only fallback.
    region->type = IREE_ASYNC_REGION_TYPE_DMABUF;
    region->buffer_size = 0;
    region->buffer_count = 0;
    region->handles.dmabuf.fd = dmabuf_fd;
    region->handles.dmabuf.offset = offset;
  }

  // Track aligned mmap parameters for cleanup.
  registration->mapped_ptr = mapped_ptr;
  registration->mapped_length = aligned_length;
  registration->dmabuf_fd = dmabuf_fd;

  // Setup entry and cleanup.
  iree_async_buffer_registration_entry_t* entry = &registration->entry;
  entry->next = NULL;
  entry->proactor = base_proactor;
  entry->cleanup_fn = iree_async_io_uring_dmabuf_registration_cleanup;
  entry->region = region;
  iree_async_buffer_registration_state_add(state, entry);

  *out_entry = entry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_proactor_io_uring_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  // Remove from the registration state list.
  iree_async_buffer_registration_state_remove(state, entry);
  // Invoke the cleanup callback to free resources (munmap for dmabuf, etc).
  entry->cleanup_fn(entry, proactor);
}

//===----------------------------------------------------------------------===//
// Slab registration (indexed zero-copy)
//===----------------------------------------------------------------------===//

// Recycle callback for slab regions with provided buffer rings.
// Called from iree_async_buffer_lease_release() to return a buffer to the
// kernel's PBUF_RING for multishot recv operations.
static void iree_async_io_uring_slab_region_recycle(void* context,
                                                    uint32_t index) {
  iree_io_uring_buffer_ring_t* buffer_ring =
      (iree_io_uring_buffer_ring_t*)context;
  iree_io_uring_buffer_ring_recycle(buffer_ring, (uint16_t)index);
}

// Region storage for slab registrations. Heap-allocated, returned to caller.
// Contains the region plus tracking state for cleanup.
typedef struct iree_async_io_uring_slab_region_t {
  iree_async_region_t region;
  // Optional: provided buffer ring for recv operations.
  // Created when access_flags includes WRITE. NULL for send-only.
  iree_io_uring_buffer_ring_t* buffer_ring;
  // True if this region registered with the kernel's fixed buffer table
  // (either via sparse table or legacy IORING_REGISTER_BUFFERS).
  bool registered_fixed_buffers;
  // Starting slot index in the sparse buffer table (for release on cleanup).
  // Only meaningful when registered_fixed_buffers is true and the proactor
  // has a sparse buffer table (5.19+).
  uint16_t fixed_buffer_base;
  // Number of contiguous slots allocated in the sparse buffer table.
  uint16_t fixed_buffer_count;
  // Allocator used for freeing this struct.
  iree_allocator_t allocator;
} iree_async_io_uring_slab_region_t;

// Clears contiguous slots in the kernel's sparse buffer table by updating them
// with empty iovecs, then releases the corresponding bitmap slots. On failure,
// only the successfully cleared slots have their bitmap bits released —
// remaining slots are left allocated to keep kernel and bitmap state
// consistent.
//
// Uses a fixed stack array and loops in batches to avoid heap allocation.
// Caller must hold the sparse table lock.
static iree_status_t iree_async_io_uring_clear_buffer_slots_locked(
    iree_async_proactor_io_uring_t* proactor, uint16_t base_slot,
    uint16_t count) {
  // All iovecs are empty (NULL base, zero length) for clearing. The stack array
  // is reused across batches since the contents are identical each iteration.
  struct iovec empty_iovecs[64];
  memset(empty_iovecs, 0, sizeof(empty_iovecs));

  uint16_t cleared = 0;
  while (cleared < count) {
    uint16_t batch =
        (uint16_t)((count - cleared > 64) ? 64 : (count - cleared));
    iree_io_uring_rsrc_update2_t update = {
        .offset = (uint32_t)(base_slot + cleared),
        .resv = 0,
        .data = (uint64_t)(uintptr_t)empty_iovecs,
        .tags = 0,
        .nr = batch,
        .resv2 = 0,
    };
    long ret = 0;
    do {
      ret =
          syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                  IREE_IORING_REGISTER_BUFFERS_UPDATE, &update, sizeof(update));
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      int saved_errno = errno;
      // Release only the slots we successfully cleared. The remaining slots
      // are left allocated in both the kernel table and the bitmap so they
      // stay consistent.
      if (cleared > 0) {
        iree_io_uring_sparse_table_release(proactor->buffer_table, base_slot,
                                           cleared);
      }
      return iree_make_status(
          iree_status_code_from_errno(saved_errno),
          "IORING_REGISTER_BUFFERS_UPDATE (clear slots %u..%u) failed (%d); "
          "the region was likely released while I/O was in-flight (EBUSY) "
          "or the buffer table has a tracking bug (EINVAL)",
          (unsigned)base_slot, (unsigned)(base_slot + count - 1), saved_errno);
    }
    cleared += batch;
  }

  iree_io_uring_sparse_table_release(proactor->buffer_table, base_slot, count);
  return iree_ok_status();
}

// Unregisters a slab region's fixed buffers from the kernel. Handles both the
// sparse table path (5.19+, IORING_REGISTER_BUFFERS_UPDATE to clear individual
// slots) and the legacy path (pre-5.19, IORING_UNREGISTER_BUFFERS to remove
// the singleton buffer table).
//
// These kernel calls can only fail due to programming errors (releasing a
// region while I/O is still in-flight causes EBUSY, and tracking bugs cause
// EINVAL). They cannot fail for transient or resource-related reasons.
static iree_status_t iree_async_io_uring_slab_region_unregister_fixed_buffers(
    iree_async_io_uring_slab_region_t* slab_region,
    iree_async_proactor_io_uring_t* proactor) {
  if (proactor->buffer_table != NULL) {
    // Sparse table path (5.19+): clear individual kernel slots and release
    // the corresponding bitmap entries.
    iree_io_uring_sparse_table_lock(proactor->buffer_table);
    iree_status_t status = iree_async_io_uring_clear_buffer_slots_locked(
        proactor, slab_region->fixed_buffer_base,
        slab_region->fixed_buffer_count);
    iree_io_uring_sparse_table_unlock(proactor->buffer_table);
    return status;
  }

  // Legacy path (pre-5.19): unregister the entire singleton buffer table.
  long ret = 0;
  int saved_errno = 0;
  do {
    ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                  IREE_IORING_UNREGISTER_BUFFERS, NULL, 0);
    saved_errno = errno;
  } while (ret < 0 && saved_errno == EINTR);
  if (ret < 0) {
    return iree_make_status(
        iree_status_code_from_errno(saved_errno),
        "IORING_UNREGISTER_BUFFERS failed (%d); the region was released "
        "while I/O was in-flight - the kernel holds DMA references to "
        "buffer memory that is about to be freed",
        saved_errno);
  }
  proactor->legacy_registered_buffer_count = 0;
  return iree_ok_status();
}

// Registers slab buffers in the kernel's fixed buffer table for zero-copy send.
// On 5.19+ acquires contiguous slots from the sparse buffer table and populates
// them via IORING_REGISTER_BUFFERS_UPDATE. On pre-5.19 uses the singleton
// IORING_REGISTER_BUFFERS.
//
// On success, populates slab_region->fixed_buffer_base, fixed_buffer_count,
// and sets registered_fixed_buffers to true.
static iree_status_t iree_async_io_uring_slab_region_register_fixed_buffers(
    iree_async_io_uring_slab_region_t* slab_region,
    iree_async_proactor_io_uring_t* proactor, void* base_ptr,
    iree_host_size_t buffer_size, iree_host_size_t buffer_count) {
  // Build iovec array mapping each slab buffer to an iovec entry.
  struct iovec* iovecs = NULL;
  struct iovec stack_iovecs[64];
  bool iovecs_heap_allocated = false;
  if (buffer_count <= 64) {
    iovecs = stack_iovecs;
  } else {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(slab_region->allocator, buffer_count,
                                    sizeof(struct iovec), (void**)&iovecs));
    iovecs_heap_allocated = true;
  }

  for (iree_host_size_t i = 0; i < buffer_count; ++i) {
    iovecs[i].iov_base = (uint8_t*)base_ptr + i * buffer_size;
    iovecs[i].iov_len = buffer_size;
  }

  // Register with the kernel. The sparse table path (5.19+) acquires
  // contiguous slots and populates them individually, allowing multiple
  // independent registrations to coexist. The legacy path (pre-5.19)
  // registers the entire iovec array as a singleton buffer table.
  iree_status_t status = iree_ok_status();
  if (proactor->buffer_table != NULL) {
    iree_io_uring_sparse_table_lock(proactor->buffer_table);
    int32_t base_slot = iree_io_uring_sparse_table_acquire(
        proactor->buffer_table, (uint16_t)buffer_count);
    if (base_slot < 0) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "sparse buffer table full (capacity=%u, requested=%" PRIhsz ")",
          (unsigned)iree_io_uring_sparse_table_capacity(proactor->buffer_table),
          buffer_count);
    }
    if (iree_status_is_ok(status)) {
      iree_io_uring_rsrc_update2_t update = {
          .offset = (uint32_t)base_slot,
          .resv = 0,
          .data = (uint64_t)(uintptr_t)iovecs,
          .tags = 0,
          .nr = (uint32_t)buffer_count,
          .resv2 = 0,
      };
      long ret = 0;
      do {
        ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                      IREE_IORING_REGISTER_BUFFERS_UPDATE, &update,
                      sizeof(update));
      } while (ret < 0 && errno == EINTR);
      if (ret < 0) {
        int saved_errno = errno;
        iree_io_uring_sparse_table_release(proactor->buffer_table,
                                           (uint16_t)base_slot,
                                           (uint16_t)buffer_count);
        status = iree_make_status(iree_status_code_from_errno(saved_errno),
                                  "IORING_REGISTER_BUFFERS_UPDATE failed (%d)",
                                  saved_errno);
      }
    }
    if (iree_status_is_ok(status)) {
      slab_region->fixed_buffer_base = (uint16_t)base_slot;
      slab_region->fixed_buffer_count = (uint16_t)buffer_count;
    }
    iree_io_uring_sparse_table_unlock(proactor->buffer_table);
  } else {
    long ret = 0;
    do {
      ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                    IREE_IORING_REGISTER_BUFFERS, iovecs, buffer_count);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      int saved_errno = errno;
      status =
          iree_make_status(iree_status_code_from_errno(saved_errno),
                           "IORING_REGISTER_BUFFERS failed (%d)", saved_errno);
    } else {
      proactor->legacy_registered_buffer_count = (uint16_t)buffer_count;
      slab_region->fixed_buffer_base = 0;
      slab_region->fixed_buffer_count = (uint16_t)buffer_count;
    }
  }

  if (iovecs_heap_allocated) {
    iree_allocator_free(slab_region->allocator, iovecs);
  }

  if (iree_status_is_ok(status)) {
    slab_region->registered_fixed_buffers = true;
  }

  return status;
}

// Destroy callback for slab regions. Called when region ref count reaches zero.
// Unregisters from kernel, frees buffer ring, releases slab ref, frees struct.
static void iree_async_io_uring_slab_region_destroy(
    iree_async_region_t* region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_io_uring_slab_region_t* slab_region =
      (iree_async_io_uring_slab_region_t*)region;
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(region->proactor);

  // Free the provided buffer ring if present (for recv).
  iree_io_uring_buffer_ring_free(slab_region->buffer_ring);

  // Unregister fixed buffers from the kernel. Failure aborts — the kernel
  // holds DMA references to the slab memory and continuing would cause
  // DMA-into-freed-pages corruption.
  if (slab_region->registered_fixed_buffers) {
    IREE_CHECK_OK(iree_async_io_uring_slab_region_unregister_fixed_buffers(
        slab_region, proactor));
  }

  // Release the slab reference.
  iree_async_slab_release(region->slab);

  // Free the combined allocation.
  iree_allocator_free(slab_region->allocator, slab_region);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_proactor_io_uring_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_region);
  *out_region = NULL;

  iree_host_size_t buffer_size = iree_async_slab_buffer_size(slab);
  iree_host_size_t buffer_count = iree_async_slab_buffer_count(slab);
  void* base_ptr = iree_async_slab_base_ptr(slab);

  // On pre-5.19 kernels (no sparse table), io_uring allows only one buffer
  // table registration at a time. On 5.19+ the sparse table supports multiple
  // independent registrations.
  bool needs_fixed_buffers =
      (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ) != 0;
  if (needs_fixed_buffers && proactor->buffer_table == NULL &&
      proactor->legacy_registered_buffer_count > 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "fixed buffer table already registered with this proactor; pre-5.19 "
        "io_uring allows only one buffer table per ring");
  }

  // Validate buffer count fits in the region handles.
  if (buffer_count > UINT16_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz
                            " exceeds maximum %u for indexed zero-copy",
                            buffer_count, (unsigned)UINT16_MAX);
  }

  // If slab will be used for recv (WRITE access) AND the kernel supports
  // MULTISHOT (5.19+), create a provided buffer ring for kernel-managed buffer
  // selection.
  bool create_recv_ring =
      (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE) != 0 &&
      iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);
  // PBUF_RING requires power-of-2 buffer count.
  if (create_recv_ring && (buffer_count & (buffer_count - 1)) != 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz
                            " must be power of 2 for recv registrations",
                            buffer_count);
  }

  // Validate buffer size fits in the region handles.
  if (buffer_size > UINT32_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size %" PRIhsz
                            " exceeds maximum %u for indexed zero-copy",
                            buffer_size, (unsigned)UINT32_MAX);
  }

  // Reject zero buffer_size - would cause divide-by-zero in index derivation.
  if (buffer_size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size must be > 0 for slab registration");
  }

  // Allocate the slab region struct.
  iree_async_io_uring_slab_region_t* slab_region = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(base_proactor->allocator, sizeof(*slab_region),
                                (void**)&slab_region));
  memset(slab_region, 0, sizeof(*slab_region));
  slab_region->allocator = base_proactor->allocator;

  // Register with kernel's fixed buffer table for send path (READ access).
  if (needs_fixed_buffers) {
    iree_status_t status =
        iree_async_io_uring_slab_region_register_fixed_buffers(
            slab_region, proactor, base_ptr, buffer_size, buffer_count);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(base_proactor->allocator, slab_region);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Create provided buffer ring for recv path (WRITE access).
  iree_io_uring_buffer_ring_t* buffer_ring = NULL;
  int16_t buffer_group_id = -1;
  if (create_recv_ring) {
    // Group IDs are uint16_t and allocated monotonically. Reserve UINT16_MAX
    // as the overflow sentinel — this still allows 65535 concurrent buffer ring
    // registrations, which is unreachable in practice (each requires at least
    // one page of kernel ring metadata plus actual buffer memory).
    if (proactor->next_group_id >= UINT16_MAX) {
      if (slab_region->registered_fixed_buffers) {
        IREE_CHECK_OK(iree_async_io_uring_slab_region_unregister_fixed_buffers(
            slab_region, proactor));
      }
      iree_allocator_free(base_proactor->allocator, slab_region);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "io_uring buffer ring group ID space exhausted "
                              "(maximum 65535 concurrent registrations)");
    }

    iree_io_uring_buffer_ring_options_t ring_options =
        iree_io_uring_buffer_ring_options_default();
    ring_options.buffer_base = base_ptr;
    ring_options.buffer_size = buffer_size;
    ring_options.buffer_count = buffer_count;
    ring_options.group_id = proactor->next_group_id++;

    iree_status_t status = iree_io_uring_buffer_ring_allocate(
        proactor->ring.ring_fd, ring_options, base_proactor->allocator,
        &buffer_ring);
    if (!iree_status_is_ok(status)) {
      // Unregister the buffer table on failure if we registered it.
      // This is a programming error if it fails (we just registered these
      // slots moments ago with no I/O in-flight), so hard-abort.
      if (slab_region->registered_fixed_buffers) {
        IREE_CHECK_OK(iree_async_io_uring_slab_region_unregister_fixed_buffers(
            slab_region, proactor));
      }
      iree_allocator_free(base_proactor->allocator, slab_region);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    buffer_group_id = (int16_t)ring_options.group_id;
  }
  slab_region->buffer_ring = buffer_ring;

  // Initialize the region.
  iree_async_region_t* region = &slab_region->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->slab = slab;
  iree_async_slab_retain(slab);
  region->destroy_fn = iree_async_io_uring_slab_region_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_IOURING;
  region->access_flags = access_flags;
  region->base_ptr = base_ptr;
  region->length = iree_async_slab_total_size(slab);

  // Set recycle callback for recv regions with provided buffer rings.
  if (buffer_ring) {
    region->recycle.fn = iree_async_io_uring_slab_region_recycle;
    region->recycle.user_data = buffer_ring;
  } else {
    region->recycle = iree_async_buffer_recycle_callback_null();
  }

  // Store indexed buffer info for SEND_ZC index derivation and recv.
  region->buffer_size = buffer_size;
  region->buffer_count = (uint32_t)buffer_count;
  region->handles.iouring.buffer_group_id = buffer_group_id;
  region->handles.iouring.base_buffer_index = slab_region->fixed_buffer_base;

  *out_region = region;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
