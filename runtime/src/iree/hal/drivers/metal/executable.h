// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_METAL_EXECUTABLE_H_

#import <Metal/Metal.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Limitations
//===----------------------------------------------------------------------===//

// The max number of bindings per descriptor set allowed in the Metal HAL
// implementation.
//
// Note that Metal itself is more permissive:
// - Argument buffer tier 1 binding limits:
//   - iOS: 31 buffers (on A11 and later, 96 buffers)
//   - macOS: 64 buffers
// - Argument buffer tier 2 binding limits:
//   - 500,000 buffers or textures
#define IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT 16

// The max number of descriptor sets allowed in the Metal HAL implementation.
//
// This depends on the general descriptor set planning in IREE and should adjust
// with it.
#define IREE_HAL_METAL_MAX_DESCRIPTOR_SET_COUNT 4

// The [[buffer(N)]] index for push constants.
//
// This depends on the general descriptor set planning in IREE and should adjust
// with it. Note that it also needs to be consistent with the compiler side when
// setting up resource location attributes during cross compiling SPIR-V to MSL.
#define IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX \
  (IREE_HAL_METAL_MAX_DESCRIPTOR_SET_COUNT - 1)

// The max number of push constants supported by the Metal HAL implementation.
#define IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT 64

//===----------------------------------------------------------------------===//
// iree_hal_metal_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_metal_source_location_t {
  iree_string_view_t file_name;
  int line;
  iree_string_view_t func_name;
} iree_hal_metal_source_location_t;

// Object and launch parameters for a compute function.
typedef struct iree_hal_metal_pipeline_t {
  id<MTLFunction> function;

  // Cached pipeline used to dispatch the function.
  id<MTLComputePipelineState> pipeline_state;

  // Threadgroup size required during dispatch.
  MTLSize threadgroup_size;

  // Total number of 32-bit constants.
  uint32_t constant_count;
  // Total number of bindings.
  uint32_t binding_count;
  // One bit per binding indicating whether it is read-only.
  uint64_t binding_read_only_bits;

  IREE_TRACE(iree_hal_metal_source_location_t source_location;)
} iree_hal_metal_pipeline_t;

// Creates a Metal kernel library as an IREE executable. The Metal library may
// contain several kernel functions that can be extracted along with the
// associated block size.
//
// Metal represents compute kernels as MTLFunctions. MTLLibrary is just an
// allocation of MTLFunctions. One creates a MTLComputePipelineState from a
// MTLFunction and uses the pipeline state for creating compute pipelines.
// This class bundles all the necessary Metal objects for getting pipeline state
// objects for a compute kernel.
//
// |out_executable| must be released by the caller (see
// iree_hal_executable_release).
iree_status_t iree_hal_metal_executable_create(
    id<MTLDevice> device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the function launch parameters for the given |entry_point|.
iree_status_t iree_hal_metal_executable_lookup_pipeline(
    const iree_hal_executable_t* executable, uint32_t entry_point,
    const iree_hal_metal_pipeline_t** out_pipeline);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_EXECUTABLE_H_
