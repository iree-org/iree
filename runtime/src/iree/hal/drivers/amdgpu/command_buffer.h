// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/affinity.h"

typedef struct iree_hal_amdgpu_block_pools_t iree_hal_amdgpu_block_pools_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_command_buffer_options_t
//===----------------------------------------------------------------------===//

// Determines where and how command buffers are recorded.
typedef enum iree_hal_amdgpu_command_buffer_recording_flags_t {
  IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_NONE = 0u,

  // TODO(benvanik): support lead-physical-device storage. This would need the
  // block pool on the lead device to make its blocks accessible to all devices
  // - today the block pool is device-local only. Produced data is immutable and
  // PCIe atomics/coherency is not required across devices.
  //
  // Allocate embedded data on the lead physical device instead of on each
  // device the command buffer is recorded for. This reduces overall memory
  // consumption and recording time at the cost of cross-device transfers.
  // IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_DATA_ON_LEAD_PHYSICAL_DEVICE
  // = 1u << 0,

  // TODO(benvanik): support compaction. This would require changing the command
  // buffer to use relative offsets for embedded data and a data table for
  // indirecting so that we can move around base pointers. A fixup would be
  // possible as well by launching a kernel that rebased the embedded pointers
  // (though trickier). For now we assume the block pool block size is a big
  // enough lever and most programs only use a handful of command buffers so
  // the waste per command buffer is minimal (compared to a single layer weight
  // in an ML model).
  //
  // Compacts the command buffer when recording ends by reallocating it to the
  // precise size required and reuploads it to each device. This will return any
  // block pool blocks back to their respective pool for reuse and ensure
  // there's no unused device memory - the cost is extra host time to do the
  // reallocation/copies.
  // IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_COMPACT_ON_FINALIZE
  // = 1u << 1,
} iree_hal_amdgpu_command_buffer_recording_flags_t;

// TODO(benvanik): move this someplace common.
//
// Block pools for host memory blocks of various sizes.
typedef struct iree_hal_amdgpu_host_block_pools_t {
  // Used for small allocations of around 1-4KB.
  iree_arena_block_pool_t small;
  // Used for large page-sized allocations of 32-64kB.
  iree_arena_block_pool_t large;
} iree_hal_amdgpu_host_block_pools_t;

// Minimum number of AQL packets in a single command buffer block.
// Any fewer and it's not guaranteed a command buffer can complete execution.
#define IREE_HAL_AMDGPU_COMMAND_BUFFER_MIN_BLOCK_AQL_PACKET_COUNT (16)

// Maximum number of AQL packets in a single command buffer block.
// This is currently limited by the `uint16_t packet_offset` in
// iree_hal_amdgpu_device_cmd_header_t.
//
// TODO(benvanik): currently we also limit this by tracy's outstanding GPU event
// limit. If we made our own timeline (which we really need to for concurrency)
// then we could eliminate this artificial limit.
#define IREE_HAL_AMDGPU_COMMAND_BUFFER_MAX_BLOCK_AQL_PACKET_COUNT            \
  IREE_AMDGPU_MIN(IREE_HAL_AMDGPU_DEVICE_QUERY_RINGBUFFER_CAPACITY,          \
                  (1u << sizeof(((iree_hal_amdgpu_device_cmd_header_t*)NULL) \
                                    ->packet_offset) *                       \
                             8))

// Recording options for a command buffer.
// Referenced data structures such as block pools must remain live for the
// lifetime of the command buffer but the options struct and its storage (such
// as the device block pool list) need not.
typedef struct iree_hal_amdgpu_command_buffer_options_t {
  iree_hal_allocator_t* device_allocator;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t command_categories;
  iree_hal_queue_affinity_t queue_affinity;
  iree_host_size_t binding_capacity;

  // Controls recording behavior (placement, optimization, debugging, etc).
  iree_hal_amdgpu_command_buffer_recording_flags_t recording_flags;

  // Maximum number of AQL packets the command buffer is allowed to issue at
  // a time. Must be at or under the HSA queue capacity of any execution queue
  // the command buffer will be scheduled on. The command buffer may decide to
  // use fewer packets.
  iree_host_size_t block_aql_packet_count;

  // Block pools for host-only (heap) memory blocks of various sizes.
  iree_hal_amdgpu_host_block_pools_t* host_block_pools;

  // Bitmap of physical devices that the command buffer will be recorded for.
  // The command buffer can only be issued on these devices.
  iree_hal_amdgpu_device_affinity_t device_affinity;

  // Compact list of physical device block pools corresponding to the bits set
  // in the device_affinity bitmap. A device affinity of 0b110 would lead to two
  // device block pools in the list at [0] and [1].
  //
  // These pools should be allocated from coarse-grained memory as once we
  // record command buffers we will never change them again and do not need any
  // synchronization.
  iree_hal_amdgpu_block_pools_t* const* device_block_pools /*[device_count]*/;
} iree_hal_amdgpu_command_buffer_options_t;

// Initializes |out_options| to its default values.
void iree_hal_amdgpu_command_buffer_options_initialize(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_amdgpu_command_buffer_options_t* out_options);

// Verifies command buffer options to ensure they meet the requirements of the
// devices the command buffer will be scheduled on.
iree_status_t iree_hal_amdgpu_command_buffer_options_verify(
    const iree_hal_amdgpu_command_buffer_options_t* options);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_command_buffer_t
//===----------------------------------------------------------------------===//

// Creates an AMDGPU command buffer with the given |options| controlling how
// it is recorded and prepared for execution.
//
// Referenced data structures in the options such as block pools must remain
// live for the lifetime of the command buffer.
iree_status_t iree_hal_amdgpu_command_buffer_create(
    const iree_hal_amdgpu_command_buffer_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a AMDGPU command buffer.
bool iree_hal_amdgpu_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Queries the device-side command buffer representation for the GPU device
// agent with |device_ordinal| in the system topology.
// |out_max_kernarg_capacity| will be set to the minimum required kernarg
// reservation used by any block in the command buffer.
iree_status_t iree_hal_amdgpu_command_buffer_query_execution_state(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t device_ordinal,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_command_buffer_t**
        out_device_command_buffer,
    iree_host_size_t* out_max_kernarg_capacity);

#endif  // IREE_HAL_DRIVERS_AMDGPU_COMMAND_BUFFER_H_
