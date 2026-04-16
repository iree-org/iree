// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_PROFILE_H_

#include "iree/hal/drivers/amdgpu/abi/signal.h"

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
  // Producer-defined event identifier unique within the chunk stream.
  uint64_t event_id;
  // Queue submission epoch containing this dispatch.
  uint64_t submission_id;
  // Process-local command-buffer identifier, or 0 for direct queue dispatch.
  uint64_t command_buffer_id;
  // Process-local executable identifier, or 0 when unavailable.
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

// One device-side timestamp harvest source.
typedef struct iree_hal_amdgpu_profile_dispatch_harvest_source_t {
  // Raw AMD completion signal populated by the CP for the profiled dispatch.
  const iree_amd_signal_t* completion_signal;
  // Profiling event record receiving the harvested timestamps.
  iree_hal_amdgpu_profile_dispatch_event_t* event;
} iree_hal_amdgpu_profile_dispatch_harvest_source_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_profile_dispatch_harvest_source_t) == 16,
    "dispatch harvest source size is part of the profiling ABI");

// Kernel arguments for the dispatch timestamp harvest builtin.
typedef struct iree_hal_amdgpu_profile_dispatch_harvest_args_t {
  // Source table with one entry per profiled dispatch.
  const iree_hal_amdgpu_profile_dispatch_harvest_source_t* sources;
  // Number of entries in |sources|.
  uint32_t source_count;
  // Reserved padding.
  uint32_t reserved0;
} iree_hal_amdgpu_profile_dispatch_harvest_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_profile_dispatch_harvest_args_t) == 16,
    "dispatch harvest args must match the kernel ABI");

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_PROFILE_H_
