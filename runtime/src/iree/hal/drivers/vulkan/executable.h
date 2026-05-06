// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_VULKAN_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_pipeline_t
//===----------------------------------------------------------------------===//

// Maps one HAL dispatch binding to one Vulkan descriptor slot.
typedef struct iree_hal_vulkan_descriptor_binding_t {
  // Pipeline-layout set ordinal containing the descriptor.
  uint32_t set_ordinal;

  // Vulkan descriptor binding number within the set.
  uint32_t binding;

  // Array element within the descriptor binding.
  uint32_t array_element;

  // Vulkan descriptor type expected at the slot.
  VkDescriptorType descriptor_type;
} iree_hal_vulkan_descriptor_binding_t;

// Hidden push-constant root for BDA dispatch ABI version 1.
typedef struct iree_hal_vulkan_bda_dispatch_root_v1_t {
  // Device address of the first address64 binding-table entry.
  uint64_t binding_table_address;

  // Device address of the first uint32_t constant-table entry, or zero.
  uint64_t constants_address;

  // First binding table element visible to the shader.
  uint32_t binding_base;

  // First constant table element visible to the shader.
  uint32_t constant_base;

  // ABI-specific dispatch flags.
  uint32_t flags;

  // Reserved field kept zero for ABI v1.
  uint32_t reserved0;
} iree_hal_vulkan_bda_dispatch_root_v1_t;

// Per-binding requirements for BDA dispatch table publication.
typedef struct iree_hal_vulkan_bda_binding_requirement_t {
  // Minimum device address alignment in bytes required by the shader.
  uint32_t minimum_alignment;

  // Minimum buffer range byte length required by the shader.
  uint64_t minimum_length;
} iree_hal_vulkan_bda_binding_requirement_t;

// Prepared Vulkan compute pipeline and HAL export metadata.
typedef struct iree_hal_vulkan_pipeline_t {
  // Vulkan compute pipeline handle owned by the executable.
  VkPipeline handle;

  // Single executable dispatch ABI bit used by this pipeline.
  iree_hal_vulkan_dispatch_abis_t dispatch_abi;

  // Vulkan pipeline layout handle owned by the executable.
  VkPipelineLayout layout;

  // Number of set layout handles in descriptor_set_layouts.
  iree_host_size_t descriptor_set_layout_count;

  // Pipeline-layout ordered descriptor set layout handles.
  VkDescriptorSetLayout* descriptor_set_layouts;

  // Number of HAL dispatch binding mappings in descriptor_bindings.
  iree_host_size_t descriptor_binding_count;

  // HAL dispatch binding mappings in binding ordinal order.
  iree_hal_vulkan_descriptor_binding_t* descriptor_bindings;

  // Buffer-device-address dispatch layout, zeroed for descriptor pipelines.
  struct {
    // Push-constant byte offset of iree_hal_vulkan_bda_dispatch_root_v1_t.
    uint32_t root_push_constant_offset;

    // Push-constant byte length reserved for the hidden BDA root.
    uint32_t root_push_constant_length;

    // Push-constant byte offset of the first HAL inline constant.
    uint32_t constant_push_constant_offset;

    // Byte length of one shader-visible binding table entry. The production
    // BDA v1 path accepts only address64 entries, so this is sizeof(uint64_t).
    uint32_t binding_table_entry_length;

    // Whether binding_count is a verifier-enforced ABI value.
    bool binding_count_known;

    // Number of entries in binding_requirements.
    iree_host_size_t binding_requirement_count;

    // Per-binding validation requirements, or NULL when all bindings are
    // unconstrained beyond device-visible storage-buffer validity.
    iree_hal_vulkan_bda_binding_requirement_t* binding_requirements;
  } bda;

  // Export name stored in executable-owned host memory.
  iree_string_view_t name;

  // Number of 32-bit HAL specialization constants accepted by the export.
  uint16_t constant_count;

  // Number of HAL buffer bindings accepted by the export.
  uint16_t binding_count;

  // Required subgroup size for pipeline creation, or zero for no requirement.
  uint32_t subgroup_size;

  // Static SPIR-V local workgroup size for the entry point, or zero when
  // unavailable.
  uint32_t workgroup_size[3];
} iree_hal_vulkan_pipeline_t;

// Returns the supported Vulkan executable format inferred from
// |executable_data|.
iree_status_t iree_hal_vulkan_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Returns whether a logical device with |enabled_features| can prepare
// |format|.
bool iree_hal_vulkan_executable_format_supported(
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis,
    iree_string_view_t executable_format);

// Creates a prepared Vulkan executable from compiler-produced FlatBuffer data.
iree_status_t iree_hal_vulkan_executable_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features, VkPipelineCache pipeline_cache,
    iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns true if |executable| is a Vulkan executable.
bool iree_hal_vulkan_executable_isa(iree_hal_executable_t* executable);

// Returns the process-local nonzero profiling identifier for |executable|.
uint64_t iree_hal_vulkan_executable_profile_id(
    iree_hal_executable_t* executable);

// Returns the native pipeline metadata for |export_ordinal|.
iree_status_t iree_hal_vulkan_executable_lookup_pipeline(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_vulkan_pipeline_t** out_pipeline);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_EXECUTABLE_H_
