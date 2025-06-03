// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_options_t
//===----------------------------------------------------------------------===//

// Power-of-two size for the per-device small block pool in bytes.
// Used for command buffer headers and other small data structures.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_SIZE_DEFAULT \
  (32 * 1024)

// Minimum number of small blocks per device allocation.
// Reduces allocation overhead at the cost of underutilizing memory.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT \
  (128)

// Initial capacity in blocks of the per-device small block pool. Block pools
// will grow as needed but accounting is cleaner if we pre-initialize them to a
// (hopefully) sufficient size.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT \
  IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT

// Power-of-two size for the per-device large block pool in bytes.
// Used for command buffer commands and data. Must be large enough to fit inline
// command buffer uploads.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT \
  (256 * 1024)

// Minimum number of large blocks per device allocation.
// Reduces allocation overhead at the cost of underutilizing memory.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT \
  (16)

// Initial capacity in blocks of the per-device large block pool. Block pools
// will grow as needed but accounting is cleaner if we pre-initialize them to a
// (hopefully) sufficient size.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT \
  IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT

// Power-of-two size for the per-device host block pool in bytes.
// Since primarily used for transient submission-specific allocations it need
// not be large.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_HOST_BLOCK_SIZE_DEFAULT (8 * 1024)

// Total number of HAL queues on the physical device.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT (1)

// Options controlling how a physical device is initialized.
typedef struct iree_hal_amdgpu_physical_device_options_t {
  // Size of a block in each device block pool.
  struct {
    // Small device block pool.
    // Used for command buffer headers and other small data structures.
    iree_hal_amdgpu_block_pool_options_t small;
    // Large device block pool.
    // Used for command buffer commands and data. Must be large enough to fit
    // inline command buffer uploads.
    iree_hal_amdgpu_block_pool_options_t large;
  } device_block_pools;

  // Size of the per-device small host block pool.
  // This is primarily used for per-submission resource sets and other transient
  // bookkeeping that should never be _too_ large or live _too_ long.
  iree_host_size_t host_block_pool_size;
  // Initial block count preallocated for the host block pool.
  iree_host_size_t host_block_pool_initial_capacity;

  // Total number of HAL queues on the physical device.
  iree_host_size_t queue_count;
  // Options used to initialize each queue.
  // TODO(benvanik): implement queues.
  // iree_hal_amdgpu_queue_options_t queue_options;
} iree_hal_amdgpu_physical_device_options_t;

// Initializes |out_options| to its default values.
void iree_hal_amdgpu_physical_device_options_initialize(
    iree_hal_amdgpu_physical_device_options_t* out_options);

// Verifies device options to ensure they meet the agent requirements.
iree_status_t iree_hal_amdgpu_physical_device_options_verify(
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): implement iree_hal_amdgpu_physical_device_t.

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
