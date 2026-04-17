// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_METADATA_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_METADATA_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cursor tracking the profiling metadata records already emitted to one sink.
typedef struct iree_hal_amdgpu_profile_metadata_cursor_t {
  // Number of executable records already emitted.
  iree_host_size_t executable_record_count;
  // Byte length of packed executable export records already emitted.
  iree_host_size_t executable_export_record_data_length;
  // Number of command-buffer records already emitted.
  iree_host_size_t command_buffer_record_count;
  // Number of command-operation records already emitted.
  iree_host_size_t command_operation_record_count;
} iree_hal_amdgpu_profile_metadata_cursor_t;

// Logical-device-owned registry of durable profiling metadata side tables.
typedef struct iree_hal_amdgpu_profile_metadata_registry_t {
  // Host allocator used for registry arrays.
  iree_allocator_t host_allocator;
  // Mutex protecting registry growth and snapshot copies.
  iree_slim_mutex_t mutex;
  // Next non-zero executable id to assign.
  uint64_t next_executable_id;
  // Next non-zero command-buffer id to assign.
  uint64_t next_command_buffer_id;
  // Executable records in id assignment order.
  iree_hal_profile_executable_record_t* executable_records;
  // Number of valid executable records.
  iree_host_size_t executable_record_count;
  // Allocated executable record capacity.
  iree_host_size_t executable_record_capacity;
  // Packed executable export records in executable id assignment order.
  uint8_t* executable_export_record_data;
  // Byte length of valid packed executable export records.
  iree_host_size_t executable_export_record_data_length;
  // Allocated byte capacity for packed executable export records.
  iree_host_size_t executable_export_record_data_capacity;
  // Command-buffer records in id assignment order.
  iree_hal_profile_command_buffer_record_t* command_buffer_records;
  // Number of valid command-buffer records.
  iree_host_size_t command_buffer_record_count;
  // Allocated command-buffer record capacity.
  iree_host_size_t command_buffer_record_capacity;
  // Command-operation records in command-buffer recording order.
  iree_hal_profile_command_operation_record_t* command_operation_records;
  // Number of valid command-operation records.
  iree_host_size_t command_operation_record_count;
  // Allocated command-operation record capacity.
  iree_host_size_t command_operation_record_capacity;
} iree_hal_amdgpu_profile_metadata_registry_t;

// Initializes |out_registry| for logical-device-lifetime metadata.
void iree_hal_amdgpu_profile_metadata_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_metadata_registry_t* out_registry);

// Releases all host allocations owned by |registry|.
void iree_hal_amdgpu_profile_metadata_deinitialize(
    iree_hal_amdgpu_profile_metadata_registry_t* registry);

// Computes the stable 128-bit SipHash-derived profile hash for exact
// code-object bytes.
void iree_hal_amdgpu_profile_metadata_hash_code_object(
    iree_const_byte_span_t code_object_data, uint64_t out_hash[2]);

// Registers immutable executable metadata and assigns |out_executable_id|.
iree_status_t iree_hal_amdgpu_profile_metadata_register_executable(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_host_size_t export_count,
    const iree_hal_executable_export_info_t* export_infos,
    const iree_host_size_t* export_parameter_offsets,
    const uint64_t code_object_hash[2],
    const iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    uint64_t* out_executable_id);

// Registers immutable command-buffer creation metadata and assigns
// |out_command_buffer_id|.
iree_status_t iree_hal_amdgpu_profile_metadata_register_command_buffer(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t physical_device_ordinal, uint64_t* out_command_buffer_id);

// Registers immutable command-buffer operation records.
iree_status_t iree_hal_amdgpu_profile_metadata_register_command_operations(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_host_size_t operation_count,
    const iree_hal_profile_command_operation_record_t* operations);

// Returns true if the registered executable export name matches |pattern|.
bool iree_hal_amdgpu_profile_metadata_export_matches(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_string_view_t pattern);

// Writes metadata records newer than |cursor| and advances |cursor| on success.
iree_status_t iree_hal_amdgpu_profile_metadata_write(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_hal_profile_sink_t* sink, uint64_t session_id, iree_string_view_t name,
    iree_hal_amdgpu_profile_metadata_cursor_t* cursor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_METADATA_H_
