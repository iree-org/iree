// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_CUDA2_PIPELINE_LAYOUT_H_
#define EXPERIMENTAL_CUDA2_PIPELINE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT 64

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cuda2_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Return the binding count for the given descriptor set layout.
iree_host_size_t iree_hal_cuda2_descriptor_set_layout_binding_count(
    iree_hal_descriptor_set_layout_t* descriptor_set_layout);

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_pipeline_layout_t
//===----------------------------------------------------------------------===//

// Creates the kernel arguments.
iree_status_t iree_hal_cuda2_pipeline_layout_create(
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count, iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

// Return the base binding index for the given set.
iree_host_size_t iree_hal_cuda2_base_binding_index(
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set);

// Return the base index for push constant data.
iree_host_size_t iree_hal_cuda2_push_constant_index(
    iree_hal_pipeline_layout_t* base_pipeline_layout);

// Return the number of constants in the pipeline layout.
iree_host_size_t iree_hal_cuda2_pipeline_layout_num_constants(
    iree_hal_pipeline_layout_t* base_pipeline_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_CUDA2_PIPELINE_LAYOUT_H_
