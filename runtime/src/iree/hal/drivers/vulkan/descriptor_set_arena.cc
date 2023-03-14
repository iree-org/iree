// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/descriptor_set_arena.h"

#include <cstddef>
#include <type_traits>
#include <utility>

#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/extensibility_util.h"
#include "iree/hal/drivers/vulkan/native_pipeline_layout.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/vma_buffer.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

static void PopulateDescriptorSetWriteInfos(
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings, VkDescriptorSet dst_set,
    Arena* arena, iree_host_size_t* out_info_count,
    VkWriteDescriptorSet** out_infos) {
  arena->Reset();
  auto buffer_infos =
      arena->AllocateSpan<VkDescriptorBufferInfo>(binding_count);
  auto write_infos = arena->AllocateSpan<VkWriteDescriptorSet>(binding_count);

  for (int i = 0; i < binding_count; ++i) {
    const auto& binding = bindings[i];

    auto& buffer_info = buffer_infos[i];
    buffer_info.buffer =
        binding.buffer ? iree_hal_vulkan_vma_buffer_handle(
                             iree_hal_buffer_allocated_buffer(binding.buffer))
                       : VK_NULL_HANDLE;
    buffer_info.offset =
        iree_hal_buffer_byte_offset(binding.buffer) + binding.offset;
    if (binding.length == IREE_WHOLE_BUFFER) {
      buffer_info.range = VK_WHOLE_SIZE;
    } else {
      // Round up to a multiple of 32-bit. 32-bit is the most native bitwidth on
      // GPUs; it has the best support compared to other bitwidths. We use VMA
      // to manage GPU memory for us and VMA should already handled proper
      // alignment when performing allocations; here we just need to provide the
      // proper "view" to Vulkan drivers over the allocated memory.
      //
      // Note this is needed because we can see unusal buffers like
      // tensor<3xi8>. Depending on GPU capabilities, this might not always be
      // directly supported by the hardware. Under such circumstances, we need
      // to emulate i8 support with i32. Shader CodeGen takes care of that: the
      // shader will read the buffer as tensor<i32> and perform bit shifts to
      // extract each byte and conduct computations. The extra additional byte
      // is read but not really used by the shader. Here in application we need
      // to match the ABI and provide the buffer as 32-bit aligned, otherwise
      // the whole read by the shader is considered as out of bounds per the
      // Vulkan spec. See
      // https://github.com/openxla/iree/issues/2022#issuecomment-640617234 for
      // more details.
      buffer_info.range = iree_device_align(
          std::min(binding.length, iree_hal_buffer_byte_length(binding.buffer) -
                                       binding.offset),
          4);
    }

    auto& write_info = write_infos[i];
    write_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_info.pNext = nullptr;
    write_info.dstSet = dst_set;
    write_info.dstBinding = binding.binding;
    write_info.dstArrayElement = 0;
    write_info.descriptorCount = 1;
    write_info.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_info.pImageInfo = nullptr;
    write_info.pBufferInfo = &buffer_info;
    write_info.pTexelBufferView = nullptr;
  }

  *out_info_count = write_infos.size();
  *out_infos = write_infos.data();
}

static VkDescriptorSetAllocateInfo PopulateDescriptorSetsAllocateInfo(
    const DescriptorPool& descriptor_pool,
    iree_hal_descriptor_set_layout_t* set_layout) {
  VkDescriptorSetAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.descriptorPool = descriptor_pool.handle;

  VkDescriptorSetLayout set_layout_handle =
      iree_hal_vulkan_native_descriptor_set_layout_handle(set_layout);
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &set_layout_handle;

  return allocate_info;
}

}  // namespace

DescriptorSetArena::DescriptorSetArena(
    DescriptorPoolCache* descriptor_pool_cache)
    : logical_device_(descriptor_pool_cache->logical_device()),
      descriptor_pool_cache_(descriptor_pool_cache) {}

DescriptorSetArena::~DescriptorSetArena() {
  if (!used_descriptor_pools_.empty()) {
    iree_status_ignore(
        descriptor_pool_cache_->ReleaseDescriptorPools(used_descriptor_pools_));
    used_descriptor_pools_.clear();
  }
}

iree_status_t DescriptorSetArena::BindDescriptorSet(
    VkCommandBuffer command_buffer, iree_hal_pipeline_layout_t* pipeline_layout,
    uint32_t set, iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  // Always prefer using push descriptors when available as we can avoid the
  // additional API overhead of updating/resetting pools.
  if (logical_device_->enabled_extensions().push_descriptors) {
    PushDescriptorSet(command_buffer, pipeline_layout, set, binding_count,
                      bindings);
    return iree_ok_status();
  }

  IREE_TRACE_SCOPE0("DescriptorSetArena::BindDescriptorSet");

  auto* set_layout =
      iree_hal_vulkan_native_pipeline_layout_set(pipeline_layout, set);

  // Pick a bucket based on the number of descriptors required.
  // NOTE: right now we are 1:1 with bindings.
  uint32_t required_descriptor_count = static_cast<int>(binding_count * 1);
  uint32_t max_descriptor_count =
      std::max(8u, iree_math_round_up_to_pow2_u32(required_descriptor_count));
  uint32_t bucket =
      iree_math_count_trailing_zeros_u32(max_descriptor_count >> 3);
  if (bucket >= descriptor_pool_buckets_.size()) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many descriptors required: %u (max=%u)",
                            required_descriptor_count,
                            (1 << (descriptor_pool_buckets_.size() + 3)));
  }
  if (descriptor_pool_buckets_[bucket].handle == VK_NULL_HANDLE) {
    // Acquire a pool for this max_descriptor_count bucket.
    IREE_RETURN_IF_ERROR(descriptor_pool_cache_->AcquireDescriptorPool(
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_descriptor_count,
        &descriptor_pool_buckets_[bucket]));
    used_descriptor_pools_.push_back(descriptor_pool_buckets_[bucket]);
  }
  auto& descriptor_pool = descriptor_pool_buckets_[bucket];

  VkDescriptorSetAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.descriptorPool = descriptor_pool.handle;
  VkDescriptorSetLayout set_layout_handle =
      iree_hal_vulkan_native_descriptor_set_layout_handle(set_layout);
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &set_layout_handle;

  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkResult result = syms().vkAllocateDescriptorSets(
      *logical_device_, &allocate_info, &descriptor_set);

  if (result == VK_ERROR_OUT_OF_POOL_MEMORY) {
    // Allocation failed because the pool is either out of descriptors or too
    // fragmented. We'll just allocate another pool.
    IREE_RETURN_IF_ERROR(descriptor_pool_cache_->AcquireDescriptorPool(
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_descriptor_count,
        &descriptor_pool_buckets_[bucket]));
    used_descriptor_pools_.push_back(descriptor_pool_buckets_[bucket]);

    // Allocate descriptor sets.
    VkDescriptorSetAllocateInfo allocate_info;
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.pNext = nullptr;
    allocate_info.descriptorPool = descriptor_pool_buckets_[bucket].handle;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &set_layout_handle;
    descriptor_set = VK_NULL_HANDLE;
    VK_RETURN_IF_ERROR(syms().vkAllocateDescriptorSets(
                           *logical_device_, &allocate_info, &descriptor_set),
                       "vkAllocateDescriptorSets");
  }

  // Get a list of VkWriteDescriptorSet structs with all bound buffers.
  iree_host_size_t write_info_count = 0;
  VkWriteDescriptorSet* write_infos = NULL;
  PopulateDescriptorSetWriteInfos(binding_count, bindings, descriptor_set,
                                  &scratch_arena_, &write_info_count,
                                  &write_infos);

  // This is the reason why push descriptor sets are good.
  // We can't batch these effectively as we don't know prior to recording what
  // descriptor sets we will need and what buffers they will point to (without
  // doing just as much work as actually recording the buffer to try to find
  // out).
  syms().vkUpdateDescriptorSets(*logical_device_,
                                static_cast<uint32_t>(write_info_count),
                                write_infos, 0, nullptr);

  // Bind the descriptor set.
  syms().vkCmdBindDescriptorSets(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      iree_hal_vulkan_native_pipeline_layout_handle(pipeline_layout), set, 1,
      &descriptor_set, 0, nullptr);

  return iree_ok_status();
}

void DescriptorSetArena::PushDescriptorSet(
    VkCommandBuffer command_buffer, iree_hal_pipeline_layout_t* pipeline_layout,
    uint32_t set, iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_TRACE_SCOPE0("DescriptorSetArena::PushDescriptorSet");
  VkPipelineLayout device_pipeline_layout =
      iree_hal_vulkan_native_pipeline_layout_handle(pipeline_layout);

  // Get a list of VkWriteDescriptorSet structs with all bound buffers.
  iree_host_size_t write_info_count = 0;
  VkWriteDescriptorSet* write_infos = NULL;
  PopulateDescriptorSetWriteInfos(binding_count, bindings, VK_NULL_HANDLE,
                                  &scratch_arena_, &write_info_count,
                                  &write_infos);

  // Fast path using push descriptors. These are pooled internally by the
  // command buffer and prevent the need for our own pooling mechanisms.
  syms().vkCmdPushDescriptorSetKHR(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, device_pipeline_layout,
      set, static_cast<uint32_t>(write_info_count), write_infos);
}

DescriptorSetGroup DescriptorSetArena::Flush() {
  IREE_TRACE_SCOPE0("DescriptorSetArena::Flush");

  if (used_descriptor_pools_.empty()) {
    // No resources to free.
    return DescriptorSetGroup{};
  }

  for (auto& bucket : descriptor_pool_buckets_) {
    bucket = {};
  }
  return DescriptorSetGroup(descriptor_pool_cache_,
                            std::move(used_descriptor_pools_));
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
