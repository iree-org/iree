// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side queue manipulation functions and optimized queue access built on
// top of the ABI types. For queue struct layouts, AQL packet types, and kernel
// implicit args see abi/queue.h and abi/kernel_args.h (exported below).
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_QUEUE_H_

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/abi/queue.h"        // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/common.h"
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
// OpenCL/HIP Dispatch ABI
//===----------------------------------------------------------------------===//
// These come from llvm-project/amd/device-libs/ockl/src/workitem.cl (the ockl
// functions) and llvm-project/clang/lib/CodeGen/CGBuiltin.cpp (e.g.
// EmitAMDGPUWorkGroupSize). Using either runs a chance of pulling in the
// entire iree_amdgpu_kernel_implicit_args_t struct and we don't want to set
// that. We also don't need it: we aren't requiring OpenCL compatibility and
// have no need for the extra features provided by the implicit args (like
// workgroup offset and device-side enqueue - that's our job).

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns the pointer to the iree_hsa_kernel_dispatch_packet_t being executed.
#define iree_amdgcn_dispatch_ptr()                                 \
  ((const iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT) \
       __builtin_amdgcn_dispatch_ptr())

// __ockl_get_global_id(0) / get_global_id_x using OLD_ABI
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_x(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_x();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_x();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  return (group_id * group_size + local_id);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_y(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_y();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_y();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  return (group_id * group_size + local_id);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_id_z(void) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_z();
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_z();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  return (group_id * group_size + local_id);
}

// __ockl_get_group_id(0)
#define iree_hal_amdgpu_device_group_id_x() __builtin_amdgcn_workgroup_id_x()
#define iree_hal_amdgpu_device_group_id_y() __builtin_amdgcn_workgroup_id_y()
#define iree_hal_amdgpu_device_group_id_z() __builtin_amdgcn_workgroup_id_z()

// __ockl_get_num_groups(0)
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_x(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[0];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  const uint32_t q = grid_size / group_size;
  return q + (grid_size > q * group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_y(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[1];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  const uint32_t q = grid_size / group_size;
  return q + (grid_size > q * group_size);
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_group_count_z(void) {
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[2];
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  const uint32_t q = grid_size / group_size;
  return q + (grid_size > q * group_size);
}

// __ockl_get_local_id(0)
#define iree_hal_amdgpu_device_local_id_x() __builtin_amdgcn_workitem_id_x()
#define iree_hal_amdgpu_device_local_id_y() __builtin_amdgcn_workitem_id_y()
#define iree_hal_amdgpu_device_local_id_z() __builtin_amdgcn_workitem_id_z()

// __ockl_get_local_size(0) / get_local_size_x
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_x(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_x();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[0];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[0];
  const uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_y(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_y();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[1];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[1];
  const uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_workgroup_size_z(void) {
  const uint32_t group_id = __builtin_amdgcn_workgroup_id_z();
  const uint32_t group_size = iree_amdgcn_dispatch_ptr()->workgroup_size[2];
  const uint32_t grid_size = iree_amdgcn_dispatch_ptr()->grid_size[2];
  const uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_1d(void) {
  return iree_hal_amdgpu_device_group_id_x() *
             iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
         iree_hal_amdgpu_device_local_id_x();
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_2d(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[1] +
                      iree_hal_amdgpu_device_local_id_y();
  return id_y * iree_amdgcn_dispatch_ptr()->grid_size[0] + id_x;
}

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE size_t
iree_hal_amdgpu_device_global_linear_id_3d(void) {
  const size_t id_x = iree_hal_amdgpu_device_group_id_x() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[0] +
                      iree_hal_amdgpu_device_local_id_x();
  const size_t id_y = iree_hal_amdgpu_device_group_id_y() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[1] +
                      iree_hal_amdgpu_device_local_id_y();
  const size_t id_z = iree_hal_amdgpu_device_group_id_z() *
                          iree_amdgcn_dispatch_ptr()->workgroup_size[2] +
                      iree_hal_amdgpu_device_local_id_z();
  return (id_z * iree_amdgcn_dispatch_ptr()->grid_size[1] + id_y) *
             iree_amdgcn_dispatch_ptr()->grid_size[0] +
         id_x;
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

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
