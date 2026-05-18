// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUFFER_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"
#include "iree/hal/drivers/webgpu/webgpu.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_buffer_t
//===----------------------------------------------------------------------===//

// Wraps a WebGPU GPUBuffer (referenced by bridge handle) as an IREE HAL
// buffer. The buffer tracks its bridge handle and maintains a shadow buffer
// in wasm linear memory for host-side mapping operations.
//
// For HOST_LOCAL|DEVICE_VISIBLE buffers (staging write), the GPU buffer is
// created with mappedAtCreation:true and a shadow buffer is allocated
// immediately. The host writes into the shadow, and on unmap the shadow
// contents are copied to the GPU mapped range before the GPU buffer is
// unmapped.
//
// For HOST_VISIBLE|DEVICE_LOCAL buffers (staging read), the GPU buffer must
// be async-mapped via mapAsync before the host can read. On map_range the
// GPU mapped range is copied into a shadow buffer; on unmap_range the shadow
// is freed and the GPU buffer is unmapped.
//
// Device-local buffers (no HOST_VISIBLE) do not support host mapping.

// Creates a WebGPU buffer wrapping a bridge handle.
//
// |buffer_handle| is the bridge handle for the GPUBuffer itself.
// |mapped_at_creation| indicates whether the GPU buffer was created with
// mappedAtCreation:true (staging write path).
//
// If |mapped_at_creation| is true, a shadow buffer is immediately allocated
// in wasm linear memory. The caller can then map the buffer and write into
// the shadow.
iree_status_t iree_hal_webgpu_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_hal_webgpu_handle_t buffer_handle, bool mapped_at_creation,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Creates a buffer wrapper without a GPU buffer (handle = 0).
//
// Used by queue_alloca to return a valid iree_hal_buffer_t* immediately. The
// buffer has no GPU backing yet — its contents are undefined and it must not
// be used in GPU operations until iree_hal_webgpu_buffer_bind is called.
// The buffer's memory_type, allowed_usage, and allocation_size are stored for
// use by bind when creating the actual GPU buffer.
iree_status_t iree_hal_webgpu_buffer_create_stub(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Assigns a GPU buffer to a stub created with
// iree_hal_webgpu_buffer_create_stub.
//
// Creates a GPUBuffer via the bridge using the buffer's stored memory_type,
// allowed_usage, and allocation_size. The buffer transitions from unbound
// (handle = 0) to bound. Must only be called once on a stub buffer.
//
// For future pool allocators, this is where pool acquisition would happen
// instead of fresh buffer creation.
iree_status_t iree_hal_webgpu_buffer_bind(
    iree_hal_buffer_t* buffer, iree_hal_webgpu_handle_t device_handle);

// Detaches and destroys the GPU buffer, setting the handle to 0.
//
// After unbinding, the iree_hal_webgpu_buffer_t C wrapper remains valid but
// has no GPU backing. The GPU memory is freed (or returned to pool in a
// future pooled allocator). Must only be called on a bound buffer.
void iree_hal_webgpu_buffer_unbind(iree_hal_buffer_t* buffer);

// Returns the WebGPU bridge handle for the underlying GPUBuffer.
// Returns 0 for unbound stub buffers.
iree_hal_webgpu_handle_t iree_hal_webgpu_buffer_handle(
    const iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUFFER_H_
