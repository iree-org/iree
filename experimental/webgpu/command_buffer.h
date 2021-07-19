// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_WEBGPU_COMMAND_BUFFER_H_

#include "experimental/webgpu/bind_group_cache.h"
#include "experimental/webgpu/builtins.h"
#include "experimental/webgpu/platform/webgpu.h"
#include "experimental/webgpu/staging_buffer.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_webgpu_command_buffer_create(
    iree_hal_device_t* device, WGPUDevice device_handle,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool,
    iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_hal_webgpu_bind_group_cache_t* bind_group_cache,
    iree_hal_webgpu_builtins_t* builtins, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

iree_status_t iree_hal_webgpu_command_buffer_issue(
    iree_hal_command_buffer_t* command_buffer, WGPUQueue queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_COMMAND_BUFFER_H_
