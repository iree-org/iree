// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/builtin_executables.h"

#include <cstddef>

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/builtin/fill_unaligned_spv.h"
#include "iree/hal/vulkan/native_descriptor_set.h"
#include "iree/hal/vulkan/native_descriptor_set_layout.h"
#include "iree/hal/vulkan/native_executable_layout.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

typedef struct iree_hal_vulkan_builtin_fill_unaligned_constants_t {
  uint32_t fill_pattern;
  uint32_t fill_pattern_width;
  uint32_t fill_offset_bytes;
  uint32_t fill_length_bytes;
} iree_hal_vulkan_builtin_fill_unaligned_constants_t;

}  // namespace

BuiltinExecutables::BuiltinExecutables(VkDeviceHandle* logical_device)
    : logical_device_(logical_device) {}

BuiltinExecutables::~BuiltinExecutables() {
  if (pipeline_ != VK_NULL_HANDLE) {
    logical_device_->syms()->vkDestroyPipeline(*logical_device_, pipeline_,
                                               logical_device_->allocator());
  }

  if (executable_layout_) {
    iree_hal_executable_layout_destroy(executable_layout_);
  }

  for (int i = 0; i < descriptor_set_layouts_.size(); ++i) {
    iree_hal_descriptor_set_layout_release(descriptor_set_layouts_[i]);
  }
}

iree_status_t BuiltinExecutables::InitializeExecutables() {
  IREE_TRACE_SCOPE();

  iree_status_t status = iree_ok_status();

  // Create shader module.
  VkShaderModule fill_unaligned_shader = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    VkShaderModuleCreateInfo shader_create_info;
    shader_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_create_info.pNext = NULL;
    shader_create_info.flags = 0;
    shader_create_info.codeSize = fill_unaligned_spv_create()->size;
    shader_create_info.pCode =
        (const uint32_t*)fill_unaligned_spv_create()->data;
    status = VK_RESULT_TO_STATUS(logical_device_->syms()->vkCreateShaderModule(
        *logical_device_, &shader_create_info, logical_device_->allocator(),
        &fill_unaligned_shader));
  }

  // Create descriptor set layouts for our compute pipeline.
  // The `maxBoundDescriptorSets` limit is 4 on many devices we support, so the
  // compiler should have reserved index 3 for our exclusive use.
  // Even though we're just using set 3, we still need to create layout bindings
  // for sets 0-2.
  descriptor_set_layouts_.reserve(4);
  for (int i = 0; i < 4; ++i) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_descriptor_set_layout_t* layout = NULL;
    iree_hal_descriptor_set_layout_binding_t layout_binding;
    layout_binding.binding = 0;
    layout_binding.type = IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout_binding.access =
        i < 3 ? IREE_HAL_MEMORY_ACCESS_NONE : IREE_HAL_MEMORY_ACCESS_WRITE;
    status = iree_hal_vulkan_native_descriptor_set_layout_create(
        logical_device_,
        i < 3 ? IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE
              : IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_PUSH_ONLY,
        /*binding_count=*/1, &layout_binding, &layout);
    if (iree_status_is_ok(status)) descriptor_set_layouts_.push_back(layout);
  }

  // Create pipeline layout.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_native_executable_layout_create(
        logical_device_,
        /*push_constant_count=*/4, /*set_layout_count=*/4,
        descriptor_set_layouts_.data(), &executable_layout_);
  }

  // Create pipeline.
  if (iree_status_is_ok(status)) {
    VkComputePipelineCreateInfo pipeline_create_info;
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.pNext = NULL;
    pipeline_create_info.flags = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    pipeline_create_info.layout =
        iree_hal_vulkan_native_executable_layout_handle(executable_layout_);
    pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_create_info.basePipelineIndex = 0;
    VkPipelineShaderStageCreateInfo* stage_create_info =
        &pipeline_create_info.stage;
    stage_create_info->sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info->pNext = NULL;
    stage_create_info->flags = 0;
    stage_create_info->stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info->module = fill_unaligned_shader;
    stage_create_info->pName = "main";
    stage_create_info->pSpecializationInfo = NULL;
    status =
        VK_RESULT_TO_STATUS(logical_device_->syms()->vkCreateComputePipelines(
            *logical_device_, /*pipeline_cache=*/VK_NULL_HANDLE,
            /*pipeline_count=*/1, &pipeline_create_info,
            logical_device_->allocator(), &pipeline_));
  }

  // Destroy shader module now that the pipeline is created.
  if (fill_unaligned_shader != VK_NULL_HANDLE) {
    logical_device_->syms()->vkDestroyShaderModule(
        *logical_device_, fill_unaligned_shader, logical_device_->allocator());
  }

  return status;
}

iree_status_t BuiltinExecutables::FillBufferUnaligned(
    VkCommandBuffer command_buffer, DescriptorSetArena* descriptor_set_arena,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, const void* push_constants_to_restore) {
  IREE_TRACE_SCOPE();

  iree_hal_descriptor_set_binding_t binding;
  binding.binding = 0;
  binding.buffer = target_buffer;
  binding.offset = 0;
  binding.length = IREE_WHOLE_BUFFER;
  IREE_RETURN_IF_ERROR(descriptor_set_arena->BindDescriptorSet(
      command_buffer, executable_layout_, /*set=*/3, /*binding_count=*/1,
      &binding));

  logical_device_->syms()->vkCmdBindPipeline(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

  iree_hal_vulkan_builtin_fill_unaligned_constants_t constants;
  switch (pattern_length) {
    case 1:
      constants.fill_pattern = *static_cast<const uint8_t*>(pattern);
      break;
    case 2:
      constants.fill_pattern = *static_cast<const uint16_t*>(pattern);
      break;
    case 4:
      constants.fill_pattern = *static_cast<const uint32_t*>(pattern);
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pattern length (%" PRIhsz ") is not a power of two", pattern_length);
  }
  constants.fill_pattern_width = pattern_length;
  constants.fill_offset_bytes = target_offset;
  constants.fill_length_bytes = length;
  logical_device_->syms()->vkCmdPushConstants(
      command_buffer,
      iree_hal_vulkan_native_executable_layout_handle(executable_layout_),
      VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
      sizeof(iree_hal_vulkan_builtin_fill_unaligned_constants_t), &constants);

  // TODO(scotttodd): insert memory barrier(s)?

  logical_device_->syms()->vkCmdDispatch(command_buffer, 1, 1, 1);

  // Restore push constants.
  logical_device_->syms()->vkCmdPushConstants(
      command_buffer,
      iree_hal_vulkan_native_executable_layout_handle(executable_layout_),
      VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
      sizeof(iree_hal_vulkan_builtin_fill_unaligned_constants_t),
      push_constants_to_restore);

  return iree_ok_status();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
