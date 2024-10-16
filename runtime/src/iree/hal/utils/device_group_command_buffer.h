// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_device_group_command_BUFFER_H__
#define IREE_HAL_UTILS_device_group_command_BUFFER_H__

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

typedef struct iree_utils_device_group_command_buffer_interface_vtable_t
    iree_utils_device_group_command_buffer_interface_vtable_t;

typedef struct iree_utils_device_group_command_buffer_interface_t {
  const iree_utils_device_group_command_buffer_interface_vtable_t* vtable;
} iree_utils_device_group_command_buffer_interface_t;

typedef struct iree_utils_device_group_command_buffer_interface_vtable_t {
  void(IREE_API_PTR* destroy)(
      iree_utils_device_group_command_buffer_interface_t* interface);
  iree_status_t(IREE_API_PTR* push_command_buffer_context)(
      iree_utils_device_group_command_buffer_interface_t* interface,
      uint64_t device_idx);
  iree_status_t(IREE_API_PTR* pop_command_buffer_context)(
      iree_utils_device_group_command_buffer_interface_t* interface);
} iree_utils_device_group_command_buffer_interface_vtable_t;

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t deferred record/replay wrapper
//===----------------------------------------------------------------------===//

// Records an command buffer that records into multiple command buffers
// at a time based on the given queue affinity.
//
// After recording the underlying command buffers can be retrieved with
// iree_hal_device_group_command_buffer_get for submission.

IREE_API_EXPORT iree_status_t iree_hal_device_group_command_buffer_create(
    iree_allocator_t host_allocator, uint32_t command_buffer_count,
    iree_hal_command_buffer_t** in_command_buffers,
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_utils_device_group_command_buffer_interface_t* interface,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a multi command buffer.
IREE_API_EXPORT bool iree_hal_device_group_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns a recorded |command_buffer| with the given index
IREE_API_EXPORT iree_status_t iree_hal_device_group_command_buffer_get(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_queue_affinity_t index,
    iree_hal_command_buffer_t** out_command_buffer);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_device_group_command_BUFFER_H__
