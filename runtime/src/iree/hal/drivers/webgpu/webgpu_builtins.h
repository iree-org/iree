// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Built-in WGSL compute shaders for buffer operations that WebGPU's command
// encoder does not natively support (fill, unaligned copy).
//
// These pipelines are created once per device and shared across all command
// buffers. The WGSL source is embedded as C string literals and compiled via
// the bridge at device creation time.

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUILTINS_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUILTINS_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"
#include "iree/hal/drivers/webgpu/webgpu_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Pre-created compute pipelines for buffer operations. Initialized once per
// device, shared across all command buffers recorded on that device.
typedef struct iree_hal_webgpu_builtins_t {
  // fill_buffer: fills a byte range with a repeating 4-byte pattern.
  // Handles arbitrary alignment via per-byte pattern extraction at edges.
  // Bind group layout: binding 0 = target (storage rw), binding 1 = params
  // (uniform, 16 bytes: offset, length, pattern, padding).
  iree_hal_webgpu_handle_t fill_pipeline;
  iree_hal_webgpu_handle_t fill_bind_group_layout;

  // copy_bytes: copies bytes between buffers with arbitrary alignment.
  // Uses per-byte extraction at misaligned edges, fast u32 copy for aligned
  // interior.
  // Bind group layout: binding 0 = source (storage ro), binding 1 = dest
  // (storage rw), binding 2 = params (uniform, 16 bytes: src_offset,
  // dst_offset, length, padding).
  iree_hal_webgpu_handle_t copy_pipeline;
  iree_hal_webgpu_handle_t copy_bind_group_layout;
} iree_hal_webgpu_builtins_t;

// Creates the builtin compute pipelines on |device_handle|. The WGSL source
// is compiled synchronously via the bridge (createComputePipeline is
// synchronous in WebGPU for WGSL input).
iree_status_t iree_hal_webgpu_builtins_initialize(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_builtins_t* out_builtins);

// Releases all pipeline and layout handles.
void iree_hal_webgpu_builtins_deinitialize(
    iree_hal_webgpu_builtins_t* builtins);

// Populates a wire-format builtins descriptor from the builtins struct.
// The descriptor is used by the JS processor to access builtin pipelines.
static inline void iree_hal_webgpu_builtins_get_descriptor(
    const iree_hal_webgpu_builtins_t* builtins,
    iree_hal_webgpu_isa_builtins_descriptor_t* out_descriptor) {
  out_descriptor->fill_pipeline = builtins->fill_pipeline;
  out_descriptor->fill_bind_group_layout = builtins->fill_bind_group_layout;
  out_descriptor->copy_pipeline = builtins->copy_pipeline;
  out_descriptor->copy_bind_group_layout = builtins->copy_bind_group_layout;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_BUILTINS_H_
