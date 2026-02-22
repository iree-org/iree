// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/buffer_ring.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/base/internal/memory.h"

//===----------------------------------------------------------------------===//
// Buffer ring structure
//===----------------------------------------------------------------------===//

struct iree_io_uring_buffer_ring_t {
  // Allocator used for this structure and kernel ring memory.
  iree_allocator_t allocator;

  // io_uring ring fd this buffer ring is registered with.
  int ring_fd;

  // Buffer group ID registered with the kernel.
  uint16_t group_id;

  // Number of buffer slots (power of 2).
  uint32_t buffer_count;

  // Mask for ring index wrapping (buffer_count - 1).
  uint32_t index_mask;

  // Size of each buffer in bytes.
  uint32_t buffer_size;

  // Page alignment used for kernel ring memory.
  iree_host_size_t page_alignment;

  // Base address of the caller's buffer slab (not owned).
  uint8_t* buffer_base;

  // Page-aligned kernel ring memory (owned, allocated with aligned alloc).
  // Layout: header with tail index, followed by buffer entry descriptors.
  iree_io_uring_buf_ring_t* kernel_ring;

  // Size of allocated kernel ring memory in bytes.
  iree_host_size_t kernel_ring_size;
};

//===----------------------------------------------------------------------===//
// Buffer ring lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_io_uring_buffer_ring_allocate(
    int ring_fd, iree_io_uring_buffer_ring_options_t options,
    iree_allocator_t allocator, iree_io_uring_buffer_ring_t** out_ring) {
  IREE_ASSERT_ARGUMENT(options.buffer_base);
  IREE_ASSERT_ARGUMENT(out_ring);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_ring = NULL;

  // Validate buffer_count is power of 2.
  iree_host_size_t buffer_count = options.buffer_count;
  if (buffer_count == 0 || (buffer_count & (buffer_count - 1)) != 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count must be a power of 2, got %" PRIhsz,
                            buffer_count);
  }
  if (buffer_count > IREE_IO_URING_MAX_PBUF_RING_ENTRIES) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz " exceeds maximum %u",
                            buffer_count, IREE_IO_URING_MAX_PBUF_RING_ENTRIES);
  }

  // Determine page alignment.
  iree_host_size_t page_alignment = options.page_alignment;
  if (page_alignment == 0) {
    iree_memory_info_t memory_info = iree_memory_query_info();
    page_alignment = memory_info.normal_page_size;
  }

  // Calculate kernel ring memory size with overflow checking.
  iree_host_size_t kernel_ring_size = 0;
  if (!iree_host_size_checked_mul(buffer_count, sizeof(iree_io_uring_buf_t),
                                  &kernel_ring_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel ring size overflow: %" PRIhsz " * %" PRIhsz,
                            buffer_count, sizeof(iree_io_uring_buf_t));
  }

  // Allocate the ring structure.
  iree_io_uring_buffer_ring_t* ring = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*ring), (void**)&ring));

  memset(ring, 0, sizeof(*ring));
  ring->allocator = allocator;
  ring->ring_fd = ring_fd;
  ring->group_id = options.group_id;
  ring->buffer_count = (uint32_t)buffer_count;
  ring->index_mask = (uint32_t)(buffer_count - 1);
  ring->buffer_size = (uint32_t)options.buffer_size;
  ring->page_alignment = page_alignment;
  ring->buffer_base = (uint8_t*)options.buffer_base;
  ring->kernel_ring_size = kernel_ring_size;

  // Allocate page-aligned kernel ring memory.
  iree_status_t status =
      iree_allocator_malloc_aligned(allocator, kernel_ring_size, page_alignment,
                                    /*offset=*/0, (void**)&ring->kernel_ring);

  // Hint for transparent huge pages if requested.
  if (iree_status_is_ok(status) && options.hint_transparent_huge_pages) {
#ifdef MADV_HUGEPAGE
    // Best-effort hint; ignore errors.
    madvise(ring->kernel_ring, kernel_ring_size, MADV_HUGEPAGE);
#endif
  }

  // Register the buffer ring with the kernel.
  if (iree_status_is_ok(status)) {
    iree_io_uring_buf_reg_t reg = {0};
    reg.ring_addr = (uint64_t)(uintptr_t)ring->kernel_ring;
    reg.ring_entries = (uint32_t)buffer_count;
    reg.bgid = options.group_id;

    long ret;
    do {
      ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, ring_fd,
                    IREE_IORING_REGISTER_PBUF_RING, &reg, 1);
    } while (ret < 0 && errno == EINTR);

    if (ret < 0) {
      int err = errno;
      status = iree_make_status(iree_status_code_from_errno(err),
                                "IORING_REGISTER_PBUF_RING failed: %s",
                                strerror(err));
    }
  }

  // Populate all buffer entries as initially available.
  // NOTE: bufs[0].resv aliases the tail field (union overlay in the kernel's
  // io_uring_buf_ring struct). We must NOT write .resv for any entry — the
  // kernel ignores it for buffer selection, and writing it on bufs[0] would
  // clobber the tail counter. This matches liburing's io_uring_buf_ring_add
  // which only writes addr/len/bid.
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_base = ring->buffer_base;
    uint32_t buffer_size = ring->buffer_size;
    for (uint32_t i = 0; i < buffer_count; ++i) {
      ring->kernel_ring->bufs[i].addr =
          (uint64_t)(uintptr_t)(buffer_base +
                                (iree_host_size_t)i * buffer_size);
      ring->kernel_ring->bufs[i].len = buffer_size;
      ring->kernel_ring->bufs[i].bid = (uint16_t)i;
    }
    // Advance tail to make all buffers available.
    iree_io_uring_store_release(&ring->kernel_ring->tail,
                                (uint16_t)buffer_count);
  }

  if (iree_status_is_ok(status)) {
    *out_ring = ring;
  } else {
    iree_allocator_free_aligned(allocator, ring->kernel_ring);
    iree_allocator_free(allocator, ring);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_io_uring_buffer_ring_free(iree_io_uring_buffer_ring_t* ring) {
  if (!ring) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unregister from kernel.
  iree_io_uring_buf_reg_t reg = {0};
  reg.bgid = ring->group_id;
  long ret;
  do {
    ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, ring->ring_fd,
                  IREE_IORING_UNREGISTER_PBUF_RING, &reg, 1);
  } while (ret < 0 && errno == EINTR);
  // PBUF_RING unregister errors are safe to ignore: the ring memory is
  // application-side metadata (buffer index + length pairs) that the kernel
  // stops referencing immediately on unregister. Unlike fixed buffer
  // unregistration, where the kernel holds DMA references to the actual buffer
  // pages, PBUF_RING teardown has no data integrity risk.

  // Free kernel ring memory and structure.
  iree_allocator_t allocator = ring->allocator;
  iree_allocator_free_aligned(allocator, ring->kernel_ring);
  iree_allocator_free(allocator, ring);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Buffer ring operations
//===----------------------------------------------------------------------===//

void iree_io_uring_buffer_ring_recycle(iree_io_uring_buffer_ring_t* ring,
                                       uint16_t buffer_index) {
  // Load current tail. We're the only writer, so relaxed would suffice,
  // but acquire is harmless and matches the store_release pattern.
  uint16_t tail = iree_io_uring_load_acquire(&ring->kernel_ring->tail);

  // Compute slot index (ring wraps at buffer_count).
  uint32_t slot = tail & ring->index_mask;

  // Write buffer entry at the slot. Do NOT write .resv — bufs[0].resv aliases
  // the tail counter via the kernel's union layout. Matches liburing's
  // io_uring_buf_ring_add which only writes addr/len/bid.
  ring->kernel_ring->bufs[slot].addr =
      (uint64_t)(uintptr_t)(ring->buffer_base +
                            (iree_host_size_t)buffer_index * ring->buffer_size);
  ring->kernel_ring->bufs[slot].len = ring->buffer_size;
  ring->kernel_ring->bufs[slot].bid = buffer_index;

  // Advance tail with release semantics (kernel reads this).
  iree_io_uring_store_release(&ring->kernel_ring->tail, (uint16_t)(tail + 1));
}

//===----------------------------------------------------------------------===//
// Buffer ring queries
//===----------------------------------------------------------------------===//

uint16_t iree_io_uring_buffer_ring_group_id(
    const iree_io_uring_buffer_ring_t* ring) {
  return ring->group_id;
}

int iree_io_uring_buffer_ring_fd(const iree_io_uring_buffer_ring_t* ring) {
  return ring->ring_fd;
}

iree_host_size_t iree_io_uring_buffer_ring_page_alignment(
    const iree_io_uring_buffer_ring_t* ring) {
  return ring->page_alignment;
}

iree_host_size_t iree_io_uring_buffer_ring_capacity(
    const iree_io_uring_buffer_ring_t* ring) {
  return ring->buffer_count;
}

iree_host_size_t iree_io_uring_buffer_ring_buffer_size(
    const iree_io_uring_buffer_ring_t* ring) {
  return ring->buffer_size;
}

void* iree_io_uring_buffer_ring_buffer_base(
    const iree_io_uring_buffer_ring_t* ring) {
  return ring->buffer_base;
}
