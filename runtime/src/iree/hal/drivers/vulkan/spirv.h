// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SPIRV_H_
#define IREE_HAL_DRIVERS_VULKAN_SPIRV_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reflected compute entry point inside a SPIR-V module.
typedef struct iree_hal_vulkan_spirv_compute_entry_point_t {
  // NUL-terminated entry point name borrowed from the SPIR-V module words.
  iree_string_view_t name;

  // SPIR-V result id of the entry point function.
  uint32_t id;

  // Static local workgroup size declared by OpExecutionMode LocalSize.
  uint32_t workgroup_size[3];
} iree_hal_vulkan_spirv_compute_entry_point_t;

// Verifies the structural SPIR-V header and instruction stream.
iree_status_t iree_hal_vulkan_spirv_verify_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count);

// Returns true when the module declares PhysicalStorageBuffer64 GLSL450.
iree_status_t iree_hal_vulkan_spirv_uses_physical_storage_buffer64_glsl450(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_uses_memory_model);

// Returns true when the module declares PhysicalStorageBufferAddresses.
iree_status_t
iree_hal_vulkan_spirv_has_physical_storage_buffer_addresses_capability(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_has_capability);

// Returns true when the module contains descriptor set/binding decorations.
iree_status_t iree_hal_vulkan_spirv_has_descriptor_binding_decorations(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_has_descriptor_binding_decorations);

// Counts variables declared in the PushConstant storage class.
iree_status_t iree_hal_vulkan_spirv_count_push_constant_variables(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t* out_push_constant_variable_count);

// Returns true when the module declares descriptor-backed variables.
iree_status_t iree_hal_vulkan_spirv_has_descriptor_storage_class_variables(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    bool* out_has_descriptor_variables);

// Counts compute entry points and their NUL-terminated name storage length.
iree_status_t iree_hal_vulkan_spirv_count_compute_entry_points(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t* out_entry_point_count,
    iree_host_size_t* out_name_storage_size);

// Parses compute entry points and their static local workgroup sizes.
//
// Entry point names are borrowed from |spirv_words| and remain valid only while
// the module words remain live. Duplicate compute entry point names fail load.
iree_status_t iree_hal_vulkan_spirv_parse_compute_entry_points(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t entry_point_capacity,
    iree_hal_vulkan_spirv_compute_entry_point_t* out_entry_points);

// Parses the static local workgroup size for |entry_point|.
//
// |out_entry_point_found| may be NULL. If the entry point exists but has no
// LocalSize execution mode, |out_workgroup_size| is left zeroed.
iree_status_t iree_hal_vulkan_spirv_parse_compute_workgroup_size(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point, bool* out_entry_point_found,
    uint32_t out_workgroup_size[3]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SPIRV_H_
