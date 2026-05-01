// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_TIMESTAMP_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_TIMESTAMP_H_

#include "iree/hal/drivers/amdgpu/abi/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Timestamp Records
//===----------------------------------------------------------------------===//

enum {
  // Version of the timestamp record ABI defined in this header.
  IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_0 = 0,
};

// Timestamp record types.
typedef enum iree_hal_amdgpu_timestamp_record_type_e {
  IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_NONE = 0,
  IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_COMMAND_BUFFER = 1,
  IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_DISPATCH = 2,
} iree_hal_amdgpu_timestamp_record_type_t;

// Timestamp tick range written by PM4 packets or device harvest kernels.
typedef struct iree_hal_amdgpu_timestamp_range_t {
  // Agent-specific tick captured when the range started.
  iree_amdgpu_device_tick_t start_tick;
  // Agent-specific tick captured when the range completed.
  iree_amdgpu_device_tick_t end_tick;
} iree_hal_amdgpu_timestamp_range_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_timestamp_range_t) == 16,
                          "timestamp range size is part of the ABI");

// Header common to every fixed binary timestamp record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_timestamp_record_header_t {
  // Size of the enclosing record in bytes.
  uint32_t record_length;
  // ABI version from IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_*.
  uint16_t version;
  // Record type from iree_hal_amdgpu_timestamp_record_type_t.
  uint16_t type;
  // Producer-defined ordinal within the record stream for this type.
  uint32_t record_ordinal;
  // Reserved bits that must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_timestamp_record_header_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_timestamp_record_header_t) ==
                              16,
                          "timestamp record header size is part of the ABI");

// Device-written command-buffer execution timestamp record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_command_buffer_timestamp_record_t {
  // Common timestamp record header. Its record ordinal is the command-buffer
  // timestamp record ordinal.
  iree_hal_amdgpu_timestamp_record_header_t header;
  // Producer-defined command-buffer identifier used for correlation.
  uint64_t command_buffer_id;
  // Command-buffer block ordinal when this record describes a block, or
  // UINT32_MAX when this record describes the whole queue execute.
  uint32_t block_ordinal;
  // Reserved bits that must be zero.
  uint32_t reserved0;
  // Device tick range captured for the command-buffer execution.
  iree_hal_amdgpu_timestamp_range_t ticks;
} iree_hal_amdgpu_command_buffer_timestamp_record_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_command_buffer_timestamp_record_t) == 48,
    "command-buffer timestamp record size is part of the ABI");

// Dispatch timestamp record flags.
typedef uint32_t iree_hal_amdgpu_dispatch_timestamp_record_flags_t;
enum iree_hal_amdgpu_dispatch_timestamp_record_flag_bits_t {
  IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_NONE = 0u,
  // Workgroup counts were read from device memory before dispatch.
  IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_INDIRECT_PARAMETERS = 1u << 0,
};

// Device-written dispatch execution timestamp record.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_dispatch_timestamp_record_t {
  // Common timestamp record header. Its record ordinal is the dispatch
  // timestamp record ordinal.
  iree_hal_amdgpu_timestamp_record_header_t header;
  // Producer-defined command-buffer identifier, or 0 for direct dispatch.
  uint64_t command_buffer_id;
  // Producer-defined executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Command-buffer block ordinal containing the dispatch, or UINT32_MAX for a
  // direct dispatch.
  uint32_t block_ordinal;
  // Program-global command index, or UINT32_MAX for a direct dispatch.
  uint32_t command_index;
  // Executable export ordinal dispatched.
  uint32_t export_ordinal;
  // Flags from iree_hal_amdgpu_dispatch_timestamp_record_flag_bits_t.
  iree_hal_amdgpu_dispatch_timestamp_record_flags_t flags;
  // Device tick range captured for the dispatch execution.
  iree_hal_amdgpu_timestamp_range_t ticks;
} iree_hal_amdgpu_dispatch_timestamp_record_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_dispatch_timestamp_record_t) ==
                              64,
                          "dispatch timestamp record size is part of the ABI");

//===----------------------------------------------------------------------===//
// Dispatch Timestamp Harvest ABI
//===----------------------------------------------------------------------===//

// One device-side dispatch timestamp harvest source.
typedef struct iree_hal_amdgpu_dispatch_timestamp_harvest_source_t {
  // Raw AMD completion signal populated by the CP for the timestamped dispatch.
  const iree_amd_signal_t* completion_signal;
  // Timestamp range receiving copied completion-signal ticks.
  iree_hal_amdgpu_timestamp_range_t* ticks;
} iree_hal_amdgpu_dispatch_timestamp_harvest_source_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_dispatch_timestamp_harvest_source_t) == 16,
    "dispatch timestamp harvest source size is part of the ABI");

// Kernel arguments for the dispatch timestamp harvest builtin.
typedef struct iree_hal_amdgpu_dispatch_timestamp_harvest_args_t {
  // Source table with one entry per timestamped dispatch.
  const iree_hal_amdgpu_dispatch_timestamp_harvest_source_t* sources;
  // Number of entries in |sources|.
  uint32_t source_count;
  // Reserved padding that must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_dispatch_timestamp_harvest_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_dispatch_timestamp_harvest_args_t) == 16,
    "dispatch timestamp harvest args size is part of the ABI");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_TIMESTAMP_H_
