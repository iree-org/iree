// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_VMEM_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_VMEM_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// Virtual Memory Utilities
//===----------------------------------------------------------------------===//

// Semantically defines how a vmem allocation can be accessed.
typedef enum iree_hal_amdgpu_vmem_access_mode_e {
  // All agents may produce and consume the memory. Read/write for all agents.
  IREE_HAL_AMDGPU_ACCESS_MODE_SHARED = 0u,
  // Memory is accessed exclusively by the agent it is allocated on.
  // No other agent has access. Read/write for agent only.
  IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE,
  // Memory is consumed exclusively by the agent it is allocated on but may be
  // produced from any agent. This is useful for mailboxes. Read for agent only
  // and write for all agents.
  IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE_CONSUMER,
  // Memory is produced exclusively by the agent it is allocated on but may be
  // consumed from any agent. This is useful for outbound buffers. Write for
  // agent only and read for all agents.
  IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE_PRODUCER,
} iree_hal_amdgpu_vmem_access_mode_t;

// Finds a global memory pool on the |agent| matching any of the specified
// global flags.
iree_status_t iree_hal_amdgpu_find_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_global_flag_t match_flags,
    hsa_amd_memory_pool_t* out_pool);

// Finds a coarse-grained memory pool on the |agent|.
// The returned pool will support allocations and be
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED.
iree_status_t iree_hal_amdgpu_find_coarse_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool);

// Finds a fine-grained memory pool on the |agent|.
// The returned pool will support allocations and be either
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED or
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED.
iree_status_t iree_hal_amdgpu_find_fine_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_vmem_ringbuffer_t
//===----------------------------------------------------------------------===//

// An allocated ringbuffer using virtual memory mapping to present a contiguous
// virtual address range that is backed by a single physical buffer but that
// allows access before and after it.
//
// This presents as a ringbuffer that does not need any special logic for
// wrapping from base offsets used when copying in memory. It follows the
// approach documented in https://lo.calho.st/posts/black-magic-buffer/ and
// https://www.mikeash.com/pyblog/friday-qa-2012-02-17-ring-buffers-and-mirrored-memory-part-ii.html
// of virtual memory mapping the buffer multiple times, example code:
// https://github.com/google/wuffs/blob/main/script/mmap-ring-buffer.c
//
// We use SVM to allocate the physical memory of the ringbuffer and then stitch
// together 3 virtual memory ranges in one contiguous virtual allocation that
// aliases the physical allocation. By treating the middle range as the base
// buffer pointer we are then able to freely dereference both before and after
// the base pointer by up to the ringbuffer size in length.
//   physical: <ringbuffer size> --+------+------+
//                                 v      v      v
//                        virtual: [prev] [base] [next]
//                                 ^      ^
//                                 |      +-- ring_base_ptr
//                                 +--------- va_base_ptr
typedef struct iree_hal_amdgpu_vmem_ringbuffer_t {
  // Capacity of the ringbuffer in bytes.
  // May be larger than the requested size if adjusted to the minimum allocation
  // granule.
  iree_device_size_t capacity;
  // Physical allocation of the pinned ringbuffer memory.
  // This is sized to the requested capacity of the ringbuffer.
  hsa_amd_vmem_alloc_handle_t alloc_handle;
  // Base virtual address pointer of the ringbuffer. This is the start of the
  // reserved address range.
  IREE_AMDGPU_DEVICE_PTR void* va_base_ptr;
  // Base virtual address pointer of the central ringbuffer contents.
  IREE_AMDGPU_DEVICE_PTR void* ring_base_ptr;
} iree_hal_amdgpu_vmem_ringbuffer_t;

// Initializes a ringbuffer by allocating the physical and virtual memory of at
// least the requested |min_capacity| with at least 64 byte alignment.
// |access_descs| will be used to setup accessibility.
iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t min_capacity,
    iree_host_size_t access_desc_count,
    const hsa_amd_memory_access_desc_t* access_descs,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer);

// Initializes a ringbuffer by allocating the physical and virtual memory of at
// least the requested power-of-two |min_capacity| with at least
// least 64 byte alignment. |topology| and |access_mode| will be used to setup
// accessibility.
iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t min_capacity,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_vmem_access_mode_t access_mode,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer);

// Deinitializes a ringbuffer and frees all physical and virtual allocations.
void iree_hal_amdgpu_vmem_ringbuffer_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_vmem_ringbuffer_t* ringbuffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_VMEM_H_
