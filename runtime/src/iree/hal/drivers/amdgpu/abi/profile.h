// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_PROFILE_H_

#include "iree/hal/drivers/amdgpu/abi/timestamp.h"

//===----------------------------------------------------------------------===//
// Dispatch event records
//===----------------------------------------------------------------------===//

// Bitfield specifying properties of one AMDGPU dispatch event record.
typedef uint32_t iree_hal_amdgpu_profile_dispatch_event_flags_t;
enum iree_hal_amdgpu_profile_dispatch_event_flag_bits_t {
  IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_NONE = 0u,
  // Dispatch was enqueued through a reusable command buffer.
  IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER = 1u << 0,
  // Workgroup counts were loaded from device memory before dispatch.
  IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS = 1u << 1,
};

// Device-written dispatch execution event.
//
// Host submission writes all static metadata before publishing the harvest
// dispatch. The device-side harvest kernel writes only start_tick/end_tick
// after queue ordering proves the profiled dispatch completed.
typedef struct iree_hal_amdgpu_profile_dispatch_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags describing how the dispatch was produced.
  iree_hal_amdgpu_profile_dispatch_event_flags_t flags;
  // Producer-defined event identifier unique within the dispatch event stream.
  uint64_t event_id;
  // Queue submission epoch containing this dispatch.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 for direct queue dispatch.
  uint64_t command_buffer_id;
  // Session-local executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Command ordinal within a command buffer, or UINT32_MAX for direct dispatch.
  uint32_t command_index;
  // Executable export ordinal dispatched.
  uint32_t export_ordinal;
  // Workgroup counts submitted for each dimension.
  uint32_t workgroup_count[3];
  // Workgroup sizes submitted for each dimension.
  uint32_t workgroup_size[3];
  // Device timestamp captured when dispatch execution started.
  uint64_t start_tick;
  // Device timestamp captured when dispatch execution completed.
  uint64_t end_tick;
} iree_hal_amdgpu_profile_dispatch_event_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_profile_dispatch_event_t) == 88,
    "dispatch event record size is part of the profiling ABI");
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_profile_dispatch_event_t, start_tick) +
            sizeof(iree_hal_amdgpu_timestamp_range_t) ==
        sizeof(iree_hal_amdgpu_profile_dispatch_event_t),
    "dispatch event timestamps must be a trailing timestamp range");

// Returns the timestamp range embedded in |event|.
static inline iree_hal_amdgpu_timestamp_range_t*
iree_hal_amdgpu_profile_dispatch_event_ticks(
    iree_hal_amdgpu_profile_dispatch_event_t* event) {
  return (iree_hal_amdgpu_timestamp_range_t*)&event->start_tick;
}

// Fixed timestamp harvest source used to populate profile dispatch events.
typedef iree_hal_amdgpu_dispatch_timestamp_harvest_source_t
    iree_hal_amdgpu_profile_dispatch_harvest_source_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_profile_dispatch_harvest_source_t) == 16,
    "dispatch harvest source size is part of the profiling ABI");

// Fixed timestamp harvest arguments used to populate profile dispatch events.
typedef iree_hal_amdgpu_dispatch_timestamp_harvest_args_t
    iree_hal_amdgpu_profile_dispatch_harvest_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_profile_dispatch_harvest_args_t) == 16,
    "dispatch harvest args must match the kernel ABI");

//===----------------------------------------------------------------------===//
// Queue device event records
//===----------------------------------------------------------------------===//

// Device-written queue operation event.
//
// Host submission writes all static metadata before publishing the timestamp
// packet. PM4 timestamp packets write only start_tick/end_tick while the
// notification ring epoch continues to own readiness and reclaim.
typedef struct iree_hal_amdgpu_profile_queue_device_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of queue operation represented by this device event.
  uint32_t type;
  // Flags describing queue operation properties.
  uint32_t flags;
  // Reserved for future queue device event fields; must be zero.
  uint32_t reserved0;
  // Producer-defined event identifier unique within the queue device event
  // stream.
  uint64_t event_id;
  // Queue submission epoch containing this device event.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;
  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;
  // Producer-defined stream identifier matching the queue metadata record.
  uint64_t stream_id;
  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
  // Session-local physical device ordinal associated with this operation.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this operation.
  uint32_t queue_ordinal;
  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;
  // Reserved for future queue device event fields; must be zero.
  uint32_t reserved1;
  // Device timestamp captured when queue-visible work started.
  uint64_t start_tick;
  // Device timestamp captured when queue-visible work completed.
  uint64_t end_tick;
} iree_hal_amdgpu_profile_queue_device_event_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_profile_queue_device_event_t) == 96,
    "queue device event record size is part of the profiling ABI");

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_PROFILE_H_
