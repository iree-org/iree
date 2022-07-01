// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_PIPELINE_LAYOUT_H_
#define IREE_HAL_LEVEL_ZERO_PIPELINE_LAYOUT_H_

#include "experimental/level_zero/context_wrapper.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_LEVEL_ZERO_MAX_PUSH_CONSTANT_COUNT 64

//===----------------------------------------------------------------------===//
// iree_hal_level_zero_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_level_zero_descriptor_set_layout_create(
    iree_hal_level_zero_context_wrapper_t* context,
    iree_hal_descriptor_set_layout_flags_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Return the binding count for the given descriptor set layout.
iree_host_size_t iree_hal_level_zero_descriptor_set_layout_binding_count(
    iree_hal_descriptor_set_layout_t* descriptor_set_layout);

//===----------------------------------------------------------------------===//
// iree_hal_level_zero_pipeline_layout_t
//===----------------------------------------------------------------------===//

// Creates the kernel arguments.
iree_status_t iree_hal_level_zero_pipeline_layout_create(
    iree_hal_level_zero_context_wrapper_t* context,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

// Return the base binding index for the given set.
iree_host_size_t iree_hal_level_zero_base_binding_index(
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set);

// Return the base index for push constant data.
iree_host_size_t iree_hal_level_zero_push_constant_index(
    iree_hal_pipeline_layout_t* base_pipeline_layout);

// Return the number of constants in the executable layout.
iree_host_size_t iree_hal_level_zero_pipeline_layout_num_constants(
    iree_hal_pipeline_layout_t* base_pipeline_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_PIPELINE_LAYOUT_H_
