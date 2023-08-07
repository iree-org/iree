// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_METAL_PIPELINE_LAYOUT_H_
#define IREE_HAL_DRIVERS_METAL_PIPELINE_LAYOUT_H_

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
// iree_hal_metal_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

// Creates a descriptor set layout for the given |bindings|.
//
// |out_descriptor_set_layout| must be released by the caller (see
// iree_hal_descriptor_set_layout_release).
iree_status_t iree_hal_metal_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Returns the total number of bindings in the given descriptor set.
iree_host_size_t iree_hal_metal_descriptor_set_layout_binding_count(
    const iree_hal_descriptor_set_layout_t* descriptor_set_layout);

// Returns the information about a given |binding| in |descriptor_set_layout|.
const iree_hal_descriptor_set_layout_binding_t*
iree_hal_metal_descriptor_set_layout_binding(
    const iree_hal_descriptor_set_layout_t* descriptor_set_layout,
    uint32_t binding);

//===----------------------------------------------------------------------===//
// iree_hal_metal_pipeline_layout_t
//===----------------------------------------------------------------------===//

// Creates a pipeline layout with the given |set_layouts| and
// |push_constant_count|.
//
// |out_pipeline_layout| must be released by the caller (see
// iree_hal_pipeline_layout_release).
iree_status_t iree_hal_metal_pipeline_layout_create(
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count, iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

// Returns the descriptor set layout of the given |set| in |pipeline_layout|.
const iree_hal_descriptor_set_layout_t*
iree_hal_metal_pipeline_layout_descriptor_set_layout(
    const iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set);

// Returns the descriptor set count in the given |pipeline_layout|.
iree_host_size_t iree_hal_metal_pipeline_layout_descriptor_set_count(
    const iree_hal_pipeline_layout_t* pipeline_layout);

// Returns the push constant count in the given |pipeline_layout|.
iree_host_size_t iree_hal_metal_pipeline_layout_push_constant_count(
    const iree_hal_pipeline_layout_t* pipeline_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_METAL_PIPELINE_LAYOUT_H_
