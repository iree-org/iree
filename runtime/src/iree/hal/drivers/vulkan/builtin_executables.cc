// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/builtin_executables.h"

#include <cstddef>

#include "iree/hal/drivers/vulkan/builtin/builtin_shaders_spv.h"
#include "iree/hal/drivers/vulkan/pipeline_layout.h"
#include "iree/hal/drivers/vulkan/status_util.h"

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

static_assert(sizeof(iree_hal_vulkan_builtin_fill_unaligned_constants_t) ==
                  IREE_HAL_VULKAN_BUILTIN_PUSH_CONSTANTS_SIZE,
              "push constant count must match struct size");

}  // namespace

BuiltinExecutables::BuiltinExecutables(VkDeviceHandle* logical_device)
    : logical_device_(logical_device) {}

BuiltinExecutables::~BuiltinExecutables() {
  if (pipeline_ != VK_NULL_HANDLE) {
    logical_device_->syms()->vkDestroyPipeline(*logical_device_, pipeline_,
                                               logical_device_->allocator());
  }

  if (pipeline_layout_) {
    iree_hal_vulkan_pipeline_layout_release(pipeline_layout_);
  }

  for (size_t i = 0; i < IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET_COUNT; ++i) {
    iree_hal_vulkan_descriptor_set_layout_release(descriptor_set_layouts_[i]);
  }
}

iree_status_t BuiltinExecutables::InitializeExecutables() {
  IREE_TRACE_SCOPE();

  // Create descriptor set layouts for our compute pipeline.
  // Even though we're just using one set, we still need to create dummy set
  // layout (without any bindings) for those preceding this set.
  for (size_t i = 0; i < IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET_COUNT; ++i) {
    iree_hal_vulkan_descriptor_set_layout_t* layout = NULL;
    if (i == IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET) {
      VkDescriptorSetLayoutBinding layout_binding;
      layout_binding.binding = 0;
      layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layout_binding.descriptorCount = 1;
      layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      layout_binding.pImmutableSamplers = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_descriptor_set_layout_create(
          logical_device_, /*flags=*/0,
          /*binding_count=*/1, &layout_binding, &layout));
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_descriptor_set_layout_create(
          logical_device_, /*flags=*/0,
          /*binding_count=*/0, /*bindings=*/nullptr, &layout));
    }
    descriptor_set_layouts_[i] = layout;
  }

  iree_status_t status = iree_ok_status();

  // Create shader module.
  VkShaderModule fill_unaligned_shader = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    VkShaderModuleCreateInfo shader_create_info;
    shader_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_create_info.pNext = NULL;
    shader_create_info.flags = 0;
    shader_create_info.codeSize = builtin_shaders_spv_create()[0].size;
    shader_create_info.pCode =
        (const uint32_t*)builtin_shaders_spv_create()[0].data;
    status = VK_RESULT_TO_STATUS(logical_device_->syms()->vkCreateShaderModule(
        *logical_device_, &shader_create_info, logical_device_->allocator(),
        &fill_unaligned_shader));
  }

  // Create pipeline layout.
  if (iree_status_is_ok(status)) {
    VkPushConstantRange push_constant_ranges[1];
    push_constant_ranges[0].offset = 0;
    push_constant_ranges[0].size = IREE_HAL_VULKAN_BUILTIN_PUSH_CONSTANTS_SIZE;
    push_constant_ranges[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    status = iree_hal_vulkan_pipeline_layout_create(
        logical_device_, IREE_ARRAYSIZE(push_constant_ranges),
        push_constant_ranges, IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET_COUNT,
        descriptor_set_layouts_, &pipeline_layout_);
  }

  // Create pipeline.
  if (iree_status_is_ok(status)) {
    VkComputePipelineCreateInfo pipeline_create_info;
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.pNext = NULL;
    pipeline_create_info.flags = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    pipeline_create_info.layout =
        iree_hal_vulkan_pipeline_layout_handle(pipeline_layout_);
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
    iree_host_size_t pattern_length) {
  IREE_TRACE_SCOPE();

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
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pattern length (%" PRIhsz
                              ") is not a power of two or is too large",
                              pattern_length);
  }

  iree_hal_buffer_ref_t binding;
  binding.reserved = 0;
  binding.buffer = target_buffer;
  binding.offset = 0;
  binding.length = IREE_WHOLE_BUFFER;
  IREE_RETURN_IF_ERROR(descriptor_set_arena->BindDescriptorSet(
      command_buffer, pipeline_layout_, IREE_HAL_VULKAN_BUILTIN_DESCRIPTOR_SET,
      /*binding_count=*/1, &binding));

  logical_device_->syms()->vkCmdBindPipeline(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

  constants.fill_pattern_width = pattern_length;
  constants.fill_offset_bytes = target_offset;
  constants.fill_length_bytes = length;
  logical_device_->syms()->vkCmdPushConstants(
      command_buffer, iree_hal_vulkan_pipeline_layout_handle(pipeline_layout_),
      VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
      sizeof(iree_hal_vulkan_builtin_fill_unaligned_constants_t), &constants);

  // TODO(scotttodd): insert memory barrier if we need to do dispatch<->dispatch
  //   synchronization. The barriers inserted normally by callers would be for
  //   transfer<->dispatch.

  logical_device_->syms()->vkCmdDispatch(command_buffer, 1, 1, 1);

  return iree_ok_status();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
