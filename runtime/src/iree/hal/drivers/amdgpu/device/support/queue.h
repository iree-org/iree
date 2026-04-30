// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side queue manipulation functions and optimized queue access built on
// top of the ABI queue types. Kernel dispatch/work-item helpers live in
// device/support/kernel.h, which is exported below for compatibility.
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

//===----------------------------------------------------------------------===//
// Cached HSA/AMD Queue
//===----------------------------------------------------------------------===//

// A variant of iree_hsa_queue_t/iree_amd_queue_t with all of the hot fields
// hoisted into the struct. Queue metadata is immutable so it is always safe to
// cache the fields and doing so allows us to avoid one indirection on every
// access. Since queue metadata may live in either host or device memory it also
// prevents the device from possibly needing to dereference the host-side
// metadata in order to get the ringbuffer pointer or indices.
//
// NOTE: we try to keep this to the smallest possible set so that it can fit in
// a single cache line. We want one fetch from local memory to get all the info
// needed to start pushing writes.
typedef IREE_AMDGPU_ALIGNAS(
    iree_amdgpu_destructive_interference_size) struct iree_amd_cached_queue_t {
  // Packet storage. Must be accessible on any agents that may operate on it and
  // aligned to at least 64 (the size of an AQL packet).
  void* base_address;

  // Maximum number of packets the queue can hold. Must be a power of 2.
  uint32_t size;

  uint32_t reserved;

  // Signal object used by the application to indicate the ID of a packet that
  // is ready to be processed. The HSA runtime or hardware packet processor
  // manages the doorbell signal. If the application tries to replace or destroy
  // this signal the behavior is undefined.
  iree_hsa_signal_t doorbell_signal;

  // Pointer to the ringbuffer write dispatch ID atomic.
  iree_amdgpu_scoped_atomic_uint64_t* write_dispatch_id;

  // Pointer to the ringbuffer read dispatch ID atomic.
  iree_amdgpu_scoped_atomic_uint64_t* read_dispatch_id;

  // Backing queue that may live in host or device memory.
  iree_amd_queue_t* queue;
} iree_amd_cached_queue_t;

// Returns a cached queue with the hot fields hoisted from the full HSA queue.
static inline iree_amd_cached_queue_t iree_amd_make_cached_queue(
    iree_hsa_queue_t* queue) {
  iree_amd_cached_queue_t result = {
      /*.base_address=*/queue->base_address,
      /*.size=*/queue->size,
      /*.reserved=*/0u,
      /*.doorbell_signal=*/{queue->doorbell_signal.handle},
      /*.write_dispatch_id=*/
      (iree_amdgpu_scoped_atomic_uint64_t*)&((iree_amd_queue_t*)queue)
          ->write_dispatch_id,
      /*.read_dispatch_id=*/
      (iree_amdgpu_scoped_atomic_uint64_t*)&((iree_amd_queue_t*)queue)
          ->read_dispatch_id,
      /*.queue=*/(iree_amd_queue_t*)queue,
  };
  return result;
}

//===----------------------------------------------------------------------===//
// Device Library Functions
//===----------------------------------------------------------------------===//
// These are cloned from llvm-project/amd/device-libs/ockl/src/hsaqs.cl so that
// we don't need to rely on ockl.bc (and don't have to try to find them every
// time we want to see what they do).

#if defined(IREE_AMDGPU_TARGET_DEVICE)

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_load_read_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue,
    iree_amdgpu_memory_order_t memory_order) {
  return iree_amdgpu_scoped_atomic_load(queue->read_dispatch_id, memory_order,
                                        iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hsa_queue_store_read_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amdgpu_scoped_atomic_store(queue->read_dispatch_id, value, memory_order,
                                  iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_load_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue,
    iree_amdgpu_memory_order_t memory_order) {
  return iree_amdgpu_scoped_atomic_load(queue->write_dispatch_id, memory_order,
                                        iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_add_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  return iree_amdgpu_scoped_atomic_fetch_add(queue->write_dispatch_id, value,
                                             memory_order,
                                             iree_amdgpu_memory_scope_system);
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_hsa_queue_cas_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue,
    uint64_t expected, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  uint64_t e = expected;
  iree_amdgpu_scoped_atomic_compare_exchange_strong(
      queue->write_dispatch_id, &e, value, memory_order,
      iree_amdgpu_memory_order_relaxed, iree_amdgpu_memory_scope_system);
  return e;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hsa_queue_store_write_index(
    const iree_amd_cached_queue_t* IREE_AMDGPU_RESTRICT queue, uint64_t value,
    iree_amdgpu_memory_order_t memory_order) {
  iree_amdgpu_scoped_atomic_store(queue->write_dispatch_id, value, memory_order,
                                  iree_amdgpu_memory_scope_system);
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_
