// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_EXTENSIBILITY_UTIL_H_
#define IREE_HAL_DRIVERS_VULKAN_EXTENSIBILITY_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/util/arena.h"

// A list of NUL-terminated strings (so they can be passed directly to Vulkan).
typedef struct iree_hal_vulkan_string_list_t {
  iree_host_size_t count;
  const char** values;
} iree_hal_vulkan_string_list_t;

// Populates |out_enabled_layers| with all layers that are both available in the
// implementation and |required_layers| and |optional_layers| lists.
// |out_enabled_layers| must have capacity at least the sum of
// |required_layers|.count and |optional_layer|.count.
// Returns failure if any |required_layers| are unavailable.
iree_status_t iree_hal_vulkan_match_available_instance_layers(
    const iree::hal::vulkan::DynamicSymbols* syms,
    const iree_hal_vulkan_string_list_t* required_layers,
    const iree_hal_vulkan_string_list_t* optional_layers, iree::Arena* arena,
    iree_hal_vulkan_string_list_t* out_enabled_layers);

// Populates |out_enabled_extensions| with all extensions that are both
// available in the implementation and |required_extensions| and
// |optional_extensions| lists. |out_enabled_extensions| must have capacity at
// least the sum of |required_extensions|.count and |optional_extensions|.count.
// Returns failure if any |required_extensions| are unavailable.
iree_status_t iree_hal_vulkan_match_available_instance_extensions(
    const iree::hal::vulkan::DynamicSymbols* syms,
    const iree_hal_vulkan_string_list_t* required_extensions,
    const iree_hal_vulkan_string_list_t* optional_extensions,
    iree::Arena* arena, iree_hal_vulkan_string_list_t* out_enabled_extensions);

// Populates |out_enabled_extensions| with all extensions that are both
// available in the implementation and |required_extensions| and
// |optional_extensions| lists. |out_enabled_extensions| must have capacity at
// least the sum of |required_extensions|.count and |optional_extensions|.count.
// Returns failure if any |required_extensions| are unavailable.
iree_status_t iree_hal_vulkan_match_available_device_extensions(
    const iree::hal::vulkan::DynamicSymbols* syms,
    VkPhysicalDevice physical_device,
    const iree_hal_vulkan_string_list_t* required_extensions,
    const iree_hal_vulkan_string_list_t* optional_extensions,
    iree::Arena* arena, iree_hal_vulkan_string_list_t* out_enabled_extensions);

// Bits for enabled instance extensions.
// We must use this to query support instead of just detecting symbol names as
// ICDs will resolve the functions sometimes even if they don't support the
// extension (or we didn't ask for it to be enabled).
typedef struct iree_hal_vulkan_instance_extensions_t {
  // VK_EXT_debug_utils is enabled and a debug messenger is registered.
  // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap44.html#VK_EXT_debug_utils
  bool debug_utils : 1;
} iree_hal_vulkan_instance_extensions_t;

// Returns a bitfield with all of the provided extension names.
iree_hal_vulkan_instance_extensions_t
iree_hal_vulkan_populate_enabled_instance_extensions(
    const iree_hal_vulkan_string_list_t* enabled_extension);

// Bits for enabled device extensions.
// We must use this to query support instead of just detecting symbol names as
// ICDs will resolve the functions sometimes even if they don't support the
// extension (or we didn't ask for it to be enabled).
typedef struct iree_hal_vulkan_device_extensions_t {
  // VK_KHR_push_descriptor is enabled and vkCmdPushDescriptorSetKHR is valid.
  bool push_descriptors : 1;
  // VK_KHR_timeline_semaphore is enabled.
  bool timeline_semaphore : 1;
  // VK_EXT_host_query_reset is enabled.
  bool host_query_reset : 1;
  // VK_EXT_calibrated_timestamps is enabled.
  bool calibrated_timestamps : 1;
  // VK_EXT_subgroup_size_control is enabled.
  bool subgroup_size_control : 1;
  // VK_EXT_external_memory_host is enabled.
  bool external_memory_host : 1;
  // VK_KHR_buffer_device_address is enabled.
  bool buffer_device_address : 1;
  // VK_KHR_8bit_storage is enabled.
  bool shader_8bit_storage : 1;
  // VK_KHR_shader_float16_int8 is enabled.
  bool shader_float16_int8 : 1;
  // VK_KHR_cooperative_matrix is enabled.
  bool cooperative_matrix : 1;
} iree_hal_vulkan_device_extensions_t;

// Returns a bitfield with all of the provided extension names.
iree_hal_vulkan_device_extensions_t
iree_hal_vulkan_populate_enabled_device_extensions(
    const iree_hal_vulkan_string_list_t* enabled_extension);

// Returns a bitfield with the extensions that are (likely) available on the
// device symbols. This is less reliable than setting the bits directly when
// the known set of extensions is available.
iree_hal_vulkan_device_extensions_t
iree_hal_vulkan_infer_enabled_device_extensions(
    const iree::hal::vulkan::DynamicSymbols* device_syms);

// Struct for supported device properties.
//
// Note that the fields used here should match the ones used in KernelFeatures
// on the compiler side.
typedef struct iree_hal_vulkan_device_properties_t {
  // Floating-point compute related feature bitfield:
  // * 0b01: f16
  // * 0b10: f64
  // Note that f32 is assumed to always exist and does not appear in this
  // bitfield.
  uint32_t compute_float : 8;
  // Integer compute related feature bitfield:
  // * 0b001: i8
  // * 0b010: i16
  // * 0b100: i64
  // Note that i32 or i1 is assumed to always exist and does not appear in
  // this bitfield.
  uint32_t compute_int : 8;
  // Storage bitwidth requirement bitfiled:
  // * 0b01: 8-bit
  // * 0b10: 16-bit
  uint32_t storage : 8;
  // Subgroup operation requirement bitfield:
  // * 0b01: subgroup shuffle operations
  // * 0b10: subgroup arithmetic operations
  uint32_t subgroup : 8;
  // Dot product operation requirement bitfield:
  // ("dotprod.<input-type>.<output-type>")
  // * 0b01: dotprod.4xi8.i32
  uint32_t dot_product : 8;
  // Cooperative matrix requirement bitfield:
  // ("coopmatrix.<input-element-type>.<output-element-type>.<m>x<n>x<k>")
  // * 0b01: coopmatrix.f16.f16.16x16x16
  uint32_t cooperative_matrix : 8;
  // Addressing more requirement bitfield:
  // ("address.<mode>")
  // * 0b01: address.physical64
  uint32_t address : 8;
} iree_hal_vulkan_iree_hal_vulkan_device_properties_t;

#endif  // IREE_HAL_DRIVERS_VULKAN_EXTENSIBILITY_UTIL_H_
