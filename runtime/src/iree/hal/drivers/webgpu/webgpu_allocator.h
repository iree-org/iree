// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_allocator_t
//===----------------------------------------------------------------------===//

// Creates a WebGPU buffer allocator.
//
// |device_handle| is the bridge handle for the GPUDevice used to create
// buffers. The allocator does not retain the device — the caller must ensure
// the device outlives the allocator.
//
// The allocator exposes three memory heaps:
//   - Device-local: STORAGE|COPY_SRC|COPY_DST. Fastest for GPU compute.
//   - Staging write: MAP_WRITE|COPY_SRC. Created with mappedAtCreation:true.
//     Used for uploading data from host to GPU.
//   - Staging read: MAP_READ|COPY_DST. Mapped via mapAsync after GPU writes.
//     Used for downloading results from GPU to host.
iree_status_t iree_hal_webgpu_allocator_create(
    iree_hal_webgpu_handle_t device_handle, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_ALLOCATOR_H_
