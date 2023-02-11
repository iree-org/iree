// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_METAL_PIPELINE_LAYOUT_H_
#define IREE_EXPERIMENTAL_METAL_PIPELINE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_metal_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

// Creates a descriptor set layout for the given |bindings|.
//
// |out_descriptor_set_layout| must be released by the caller (see
// iree_hal_descriptor_set_layout_release).
iree_status_t iree_hal_metal_descriptor_set_layout_create(
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Returns the information about a given |binding| in //
// |base_descriptor_set_layout|.
iree_hal_descriptor_set_layout_binding_t*
iree_hal_metal_descriptor_set_layout_binding(
    iree_hal_descriptor_set_layout_t* base_descriptor_set_layout,
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
    iree_allocator_t host_allocator, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

// Returns the descriptor set layout of the given |set| in
// |base_pipeline_layout|.
iree_hal_descriptor_set_layout_t*
iree_hal_metal_pipeline_layout_descriptor_set_layout(
    iree_hal_pipeline_layout_t* base_pipeline_layout, uint32_t set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_METAL_PIPELINE_LAYOUT_H_
