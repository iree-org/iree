// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/uring.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Syscall wrappers
//===----------------------------------------------------------------------===//

static inline int iree_io_uring_setup(uint32_t entries,
                                      iree_io_uring_params_t* params) {
  return (int)syscall(IREE_IO_URING_SYSCALL_SETUP, entries, params);
}

static inline int iree_io_uring_enter(int ring_fd, uint32_t to_submit,
                                      uint32_t min_complete, uint32_t flags,
                                      void* arg, size_t argsz) {
  return (int)syscall(IREE_IO_URING_SYSCALL_ENTER, ring_fd, to_submit,
                      min_complete, flags, arg, argsz);
}

//===----------------------------------------------------------------------===//
// Ring initialization
//===----------------------------------------------------------------------===//

// Attempts io_uring_setup with progressively fewer flags on failure.
// On success, stores ring fd, features, and fills out_params with kernel data.
static iree_status_t iree_io_uring_ring_try_setup(
    uint32_t entries, uint32_t requested_flags, iree_io_uring_ring_t* ring,
    iree_io_uring_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));

  // Attempt 1: Try with all requested flags.
  out_params->flags = requested_flags;
  int fd = iree_io_uring_setup(entries, out_params);
  if (fd >= 0) {
    ring->ring_fd = fd;
    ring->features = out_params->features;
    return iree_ok_status();
  }

  // All io_uring_setup failures other than EINVAL (which triggers flag
  // fallbacks below) mean io_uring is not usable in this environment.
  // We return UNAVAILABLE with the errno in the message for diagnostics.
  // This matches the API contract in api.h and the industry-standard approach
  // (MariaDB, sled, etc. all treat ENOSYS/EPERM/ENOMEM uniformly as
  // "io_uring not available, fall back").
  if (errno == ENOSYS) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "io_uring not supported by kernel (ENOSYS)");
  }
  if (errno == EPERM) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "io_uring blocked by security policy (EPERM)");
  }

  // EINVAL may mean unsupported flags - try fallbacks.
  if (errno != EINVAL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "io_uring_setup failed (errno %d)", errno);
  }

  // Attempt 2: Remove DEFER_TASKRUN (requires 6.1+).
  if (requested_flags & IREE_IORING_SETUP_DEFER_TASKRUN) {
    memset(out_params, 0, sizeof(*out_params));
    out_params->flags = requested_flags & ~IREE_IORING_SETUP_DEFER_TASKRUN;
    fd = iree_io_uring_setup(entries, out_params);
    if (fd >= 0) {
      ring->ring_fd = fd;
      ring->features = out_params->features;
      return iree_ok_status();
    }
    if (errno != EINVAL) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "io_uring_setup failed (errno %d)", errno);
    }
  }

  // Attempt 3: Remove SINGLE_ISSUER (requires 6.0+).
  if (requested_flags & IREE_IORING_SETUP_SINGLE_ISSUER) {
    memset(out_params, 0, sizeof(*out_params));
    out_params->flags = requested_flags & ~(IREE_IORING_SETUP_SINGLE_ISSUER |
                                            IREE_IORING_SETUP_DEFER_TASKRUN);
    fd = iree_io_uring_setup(entries, out_params);
    if (fd >= 0) {
      ring->ring_fd = fd;
      ring->features = out_params->features;
      return iree_ok_status();
    }
    if (errno != EINVAL) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "io_uring_setup failed (errno %d)", errno);
    }
  }

  // Attempt 4: No optional flags (minimal 5.1 support).
  memset(out_params, 0, sizeof(*out_params));
  out_params->flags = 0;
  fd = iree_io_uring_setup(entries, out_params);
  if (fd >= 0) {
    ring->ring_fd = fd;
    ring->features = out_params->features;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "io_uring_setup failed with no flags (errno %d)",
                          errno);
}

// Validates that an offset + size fits within a mapped region.
// Returns true if valid, false if out of bounds.
static inline bool iree_io_uring_validate_offset(uint32_t offset, size_t size,
                                                 size_t region_size) {
  // Check for overflow in addition and bounds.
  return offset <= region_size && size <= region_size - offset;
}

// Maps the ring buffers using offsets from io_uring_setup params.
static iree_status_t iree_io_uring_ring_map_buffers(
    iree_io_uring_ring_t* ring, const iree_io_uring_params_t* params) {
  // The kernel tells us the sizes in params.sq_entries and params.cq_entries.
  uint32_t sq_entries = params->sq_entries;
  uint32_t cq_entries = params->cq_entries;

  // Compute SQ ring mmap size: we need enough to cover the array at the end.
  // The array is at sq_off.array and has sq_entries uint32_t indices.
  size_t sq_ring_size = params->sq_off.array + sq_entries * sizeof(uint32_t);
  void* sq_ring_ptr =
      mmap(NULL, sq_ring_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_POPULATE, ring->ring_fd, IREE_IORING_OFF_SQ_RING);
  if (sq_ring_ptr == MAP_FAILED) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "SQ ring mmap failed (%d)", errno);
  }
  ring->sq_ring_ptr = sq_ring_ptr;
  ring->sq_ring_size = sq_ring_size;

  // Validate SQ ring offsets before dereferencing.
  // A buggy or mismatched kernel could provide offsets outside the mapped
  // region, causing segfaults.
  if (!iree_io_uring_validate_offset(params->sq_off.head, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.tail, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.ring_mask, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.ring_entries,
                                     sizeof(uint32_t), sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.flags, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(
          params->sq_off.array, sq_entries * sizeof(uint32_t), sq_ring_size)) {
    munmap(sq_ring_ptr, sq_ring_size);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "SQ ring offset out of bounds");
  }

  // Parse SQ ring pointers using kernel-provided offsets.
  ring->sq_head = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.head);
  ring->sq_tail = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.tail);
  ring->sq_ring_mask =
      (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.ring_mask);
  ring->sq_ring_entries =
      (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.ring_entries);
  ring->sq_flags = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.flags);
  ring->sq_array = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.array);

  ring->sq_mask = *ring->sq_ring_mask;
  ring->sq_entries = *ring->sq_ring_entries;

  // Map the SQE array (separate mmap region).
  size_t sqes_size = sq_entries * sizeof(iree_io_uring_sqe_t);
  void* sqes_ptr =
      mmap(NULL, sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
           ring->ring_fd, IREE_IORING_OFF_SQES);
  if (sqes_ptr == MAP_FAILED) {
    munmap(sq_ring_ptr, sq_ring_size);
    ring->sq_ring_ptr = NULL;
    return iree_make_status(iree_status_code_from_errno(errno),
                            "SQE array mmap failed (%d)", errno);
  }
  ring->sqes = (iree_io_uring_sqe_t*)sqes_ptr;
  ring->sqes_size = sqes_size;

  // Compute CQ ring mmap size: we need enough to cover the CQE array.
  // The cqes offset tells us where in the mmap the CQE array starts.
  size_t cq_ring_size =
      params->cq_off.cqes + cq_entries * sizeof(iree_io_uring_cqe_t);
  void* cq_ring_ptr =
      mmap(NULL, cq_ring_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_POPULATE, ring->ring_fd, IREE_IORING_OFF_CQ_RING);
  if (cq_ring_ptr == MAP_FAILED) {
    munmap(sqes_ptr, sqes_size);
    munmap(sq_ring_ptr, sq_ring_size);
    ring->sq_ring_ptr = NULL;
    ring->sqes = NULL;
    return iree_make_status(iree_status_code_from_errno(errno),
                            "CQ ring mmap failed (%d)", errno);
  }
  ring->cq_ring_ptr = cq_ring_ptr;
  ring->cq_ring_size = cq_ring_size;

  // Validate CQ ring offsets before dereferencing.
  if (!iree_io_uring_validate_offset(params->cq_off.head, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.tail, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.ring_mask, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.ring_entries,
                                     sizeof(uint32_t), cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.overflow, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.cqes,
                                     cq_entries * sizeof(iree_io_uring_cqe_t),
                                     cq_ring_size)) {
    munmap(cq_ring_ptr, cq_ring_size);
    munmap(sqes_ptr, sqes_size);
    munmap(sq_ring_ptr, sq_ring_size);
    ring->sq_ring_ptr = NULL;
    ring->sqes = NULL;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CQ ring offset out of bounds");
  }

  // Parse CQ ring pointers using kernel-provided offsets.
  ring->cq_head = (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.head);
  ring->cq_tail = (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.tail);
  ring->cq_ring_mask =
      (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.ring_mask);
  ring->cq_ring_entries =
      (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.ring_entries);
  ring->cq_overflow =
      (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.overflow);
  ring->cqes =
      (iree_io_uring_cqe_t*)((uint8_t*)cq_ring_ptr + params->cq_off.cqes);

  ring->cq_mask = *ring->cq_ring_mask;
  ring->cq_entries = *ring->cq_ring_entries;

  // Initialize local tail to match kernel's view.
  ring->sq_local_tail = *ring->sq_tail;

  return iree_ok_status();
}

iree_status_t iree_io_uring_ring_initialize(
    iree_io_uring_ring_options_t options, iree_io_uring_ring_t* out_ring) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_ring);
  memset(out_ring, 0, sizeof(*out_ring));
  out_ring->ring_fd = -1;

  uint32_t entries = options.sq_entries;
  if (entries == 0) entries = 256;
  // Round up to power of 2 (kernel requirement).
  entries = iree_math_round_up_to_pow2_u32(entries);
  if (entries < 1) entries = 1;
  if (entries > IREE_IORING_MAX_ENTRIES) entries = IREE_IORING_MAX_ENTRIES;

  // Setup fills params with ring sizes and offsets from the kernel.
  iree_io_uring_params_t params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_uring_ring_try_setup(entries, options.setup_flags, out_ring,
                                       &params));

  iree_status_t status = iree_io_uring_ring_map_buffers(out_ring, &params);
  if (!iree_status_is_ok(status)) {
    close(out_ring->ring_fd);
    out_ring->ring_fd = -1;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_io_uring_ring_deinitialize(iree_io_uring_ring_t* ring) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (ring->cq_ring_ptr) {
    munmap(ring->cq_ring_ptr, ring->cq_ring_size);
    ring->cq_ring_ptr = NULL;
  }
  if (ring->sqes) {
    munmap(ring->sqes, ring->sqes_size);
    ring->sqes = NULL;
  }
  if (ring->sq_ring_ptr) {
    munmap(ring->sq_ring_ptr, ring->sq_ring_size);
    ring->sq_ring_ptr = NULL;
  }
  if (ring->ring_fd >= 0) {
    close(ring->ring_fd);
    ring->ring_fd = -1;
  }
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Submission queue operations
//===----------------------------------------------------------------------===//

iree_io_uring_sqe_t* iree_io_uring_ring_get_sqe(iree_io_uring_ring_t* ring) {
  uint32_t head = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                   iree_memory_order_acquire);
  uint32_t tail = ring->sq_local_tail;

  // Check if SQ is full (including uncommitted SQEs).
  if (tail - head >= ring->sq_entries) {
    return NULL;
  }

  uint32_t index = tail & ring->sq_mask;
  iree_io_uring_sqe_t* sqe = &ring->sqes[index];

  // Zero the SQE for the caller.
  memset(sqe, 0, sizeof(*sqe));

  // Update the SQ array to point to this SQE index.
  ring->sq_array[index] = index;

  // Increment local tail only. The kernel-visible *sq_tail is updated
  // during submit(), allowing rollback if encoding fails.
  ring->sq_local_tail = tail + 1;

  return sqe;
}

iree_status_t iree_io_uring_ring_submit(iree_io_uring_ring_t* ring,
                                        uint32_t min_complete, uint32_t flags) {
  // Flush under the SQ lock: read sq_local_tail and advance *sq_tail.
  // The lock ensures no other thread is mid-fill when we compute to_submit.
  iree_io_uring_ring_sq_lock(ring);
  uint32_t to_submit = ring->sq_local_tail - *ring->sq_tail;
  if (to_submit > 0) {
    // Release barrier ensures all SQE field writes complete before kernel sees
    // the new tail.
    iree_atomic_store((iree_atomic_int32_t*)ring->sq_tail, ring->sq_local_tail,
                      iree_memory_order_release);
  }
  iree_io_uring_ring_sq_unlock(ring);

  // Early return only if there's truly nothing to do: no SQEs to submit,
  // not waiting for completions, and not explicitly requesting GETEVENTS.
  // The GETEVENTS check is critical for DEFER_TASKRUN mode where we need to
  // call io_uring_enter to flush deferred completions even with nothing to
  // submit.
  if (to_submit == 0 && min_complete == 0 &&
      !iree_any_bit_set(flags, IREE_IORING_ENTER_GETEVENTS)) {
    return iree_ok_status();
  }

  // io_uring_enter is called OUTSIDE the lock. Only the poll thread may call
  // this (SINGLE_ISSUER constraint). Cross-thread submitters use wake() to
  // trigger the poll thread's io_uring_enter instead.
  //
  // Retry on EINTR - signals can interrupt the syscall but it's always safe
  // to retry. This keeps EINTR handling out of all callers.
  int ret;
  do {
    ret = iree_io_uring_enter(ring->ring_fd, to_submit, min_complete, flags,
                              NULL, 0);
  } while (ret < 0 && errno == EINTR);

  if (ret < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring_enter failed (%d)", errno);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Completion queue operations
//===----------------------------------------------------------------------===//

iree_status_t iree_io_uring_ring_wait_cqe(iree_io_uring_ring_t* ring,
                                          uint32_t min_complete,
                                          bool flush_pending,
                                          iree_duration_t timeout_ns) {
  // Check if we already have enough CQEs.
  if (iree_io_uring_ring_cq_count(ring) >= min_complete) {
    return iree_ok_status();
  }

  // Calculate pending SQEs to submit (if flushing). Lock protects the
  // sq_local_tail read and *sq_tail write against concurrent get_sqe callers.
  uint32_t to_submit = 0;
  if (flush_pending) {
    iree_io_uring_ring_sq_lock(ring);
    to_submit = ring->sq_local_tail - *ring->sq_tail;
    if (to_submit > 0) {
      iree_atomic_store((iree_atomic_int32_t*)ring->sq_tail,
                        ring->sq_local_tail, iree_memory_order_release);
    }
    iree_io_uring_ring_sq_unlock(ring);
  }

  // Compute absolute deadline so EINTR retries use the remaining time rather
  // than restarting the full timeout duration. Without this, frequent signals
  // (profiler sampling, etc.) cause the effective wait to grow unboundedly.
  const bool has_timeout =
      timeout_ns > 0 && timeout_ns != IREE_DURATION_INFINITE;
  iree_time_t deadline = IREE_TIME_INFINITE_FUTURE;
  if (has_timeout) {
    deadline = iree_time_now() + timeout_ns;
  }

  for (;;) {
    // Build flags and args for io_uring_enter.
    uint32_t flags = IREE_IORING_ENTER_GETEVENTS;

    iree_kernel_timespec_t ts;
    iree_io_uring_getevents_arg_t arg;
    void* arg_ptr = NULL;
    size_t arg_sz = 0;

    if (has_timeout) {
      iree_duration_t remaining = deadline - iree_time_now();
      if (remaining <= 0) {
        if (iree_io_uring_ring_cq_count(ring) > 0) {
          return iree_ok_status();
        }
        return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED, "poll timeout");
      }
      flags |= IREE_IORING_ENTER_EXT_ARG;
      ts.tv_sec = remaining / 1000000000LL;
      ts.tv_nsec = remaining % 1000000000LL;
      memset(&arg, 0, sizeof(arg));
      arg.ts = (uint64_t)&ts;
      arg_ptr = &arg;
      arg_sz = sizeof(arg);
    }

    int ret = iree_io_uring_enter(ring->ring_fd, to_submit, min_complete, flags,
                                  arg_ptr, arg_sz);
    // Only flush SQEs on the first iteration; subsequent retries have nothing
    // new to submit.
    to_submit = 0;

    if (ret >= 0) {
      return iree_ok_status();
    }

    if (errno == ETIME) {
      if (iree_io_uring_ring_cq_count(ring) > 0) {
        return iree_ok_status();
      }
      return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED, "poll timeout");
    }
    if (errno == EINTR) {
      if (iree_io_uring_ring_cq_count(ring) > 0) {
        return iree_ok_status();
      }
      // Loop retries with remaining time computed from the deadline.
      continue;
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring_enter failed (%d)", errno);
  }
}
