// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_BUILTIN_EXECUTABLES_H_
#define IREE_HAL_DRIVERS_METAL_BUILTIN_EXECUTABLES_H_

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/kernel_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_metal_builtin_executable_t {
  iree_allocator_t host_allocator;

  // The number of entry points in this builtin executable.
  iree_host_size_t entry_point_count;
  // THe list of entry points, pointing to the end of the struct allocation.
  iree_hal_metal_kernel_params_t entry_points[];
} iree_hal_metal_builtin_executable_t;
// + Additional inline allocation for holding all entry point kernel parameters.

// Creates builtin executables for polyfill features not directly supported by
// Metal API.
iree_status_t iree_hal_metal_builtin_executable_create(
    id<MTLDevice> device, iree_allocator_t host_allocator,
    iree_hal_metal_builtin_executable_t** out_executable);

void iree_hal_metal_builtin_executable_destroy(
    iree_hal_metal_builtin_executable_t* executable);

// Fills the |target_buffer| at the given |target_offset| of |length| with
// |pattern| using builtin executables dispatched via |encoder|.
//
// Under the hood, this will record all necessary commands to bind kernel
// objects and buffer resources, and the perform dispatch.
iree_status_t iree_hal_metal_builtin_executable_fill_buffer(
    const iree_hal_metal_builtin_executable_t* executable,
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    uint32_t pattern);

// Copies the |source_buffer| at |source_offset| to the |target_buffer| at
// |target_offset| of |length| using builtin executables dispatched via
// |encoder|.
//
// Under the hood, this will record all necessary commands to bind kernel
// objects and buffer resources, and the perform dispatch.
iree_status_t iree_hal_metal_builtin_executable_copy_buffer(
    const iree_hal_metal_builtin_executable_t* executable,
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> source_buffer,
    iree_device_size_t source_offset, id<MTLBuffer> target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_BUILTIN_EXECUTABLES_H_
