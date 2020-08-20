// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/vulkan/descriptor_set_arena.h"

#include "iree/base/alignment.h"
#include "iree/base/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_buffer.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

StatusOr<VmaBuffer*> CastBuffer(Buffer* buffer) {
  // TODO(benvanik): assert that the buffer is from the right allocator and
  // that it is compatible with our target queue family.
  return static_cast<VmaBuffer*>(buffer->allocated_buffer());
}

StatusOr<absl::Span<VkWriteDescriptorSet>> PopulateDescriptorSetWriteInfos(
    absl::Span<const DescriptorSet::Binding> bindings, VkDescriptorSet dst_set,
    Arena* arena) {
  arena->Reset();
  auto buffer_infos =
      arena->AllocateSpan<VkDescriptorBufferInfo>(bindings.size());
  auto write_infos = arena->AllocateSpan<VkWriteDescriptorSet>(bindings.size());

  for (int i = 0; i < bindings.size(); ++i) {
    const auto& binding = bindings[i];

    auto& buffer_info = buffer_infos[i];
    IREE_ASSIGN_OR_RETURN(auto buffer, CastBuffer(binding.buffer));
    buffer_info.buffer = buffer->handle();
    // TODO(benvanik): properly subrange (add to BufferBinding).
    buffer_info.offset = binding.buffer->byte_offset();
    // Round up to a multiple of 32-bit. 32-bit is the most native bitwidth on
    // GPUs; it has the best support compared to other bitwidths. We use VMA to
    // manage GPU memory for us and VMA should already handled proper alignment
    // when performing allocations; here we just need to provide the proper
    // "view" to Vulkan drivers over the allocated memory.
    //
    // Note this is needed because we can see unusal buffers like tensor<3xi8>.
    // Depending on GPU capabilities, this might not always be directly
    // supported by the hardware. Under such circumstances, we need to emulate
    // i8 support with i32. Shader CodeGen takes care of that: the shader will
    // read the buffer as tensor<i32> and perform bit shifts to extract each
    // byte and conduct computations. The extra additional byte is read but
    // not really used by the shader. Here in application we need to match the
    // ABI and provide the buffer as 32-bit aligned, otherwise the whole read by
    // the shader is considered as out of bounds per the Vulkan spec.
    // See https://github.com/google/iree/issues/2022#issuecomment-640617234
    // for more details.
    buffer_info.range = iree_align(binding.buffer->byte_length(), 4);

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

  return write_infos;
}

VkDescriptorSetAllocateInfo PopulateDescriptorSetsAllocateInfo(
    const DescriptorPool& descriptor_pool,
    NativeDescriptorSetLayout* set_layout) {
  VkDescriptorSetAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.descriptorPool = descriptor_pool.handle;

  VkDescriptorSetLayout set_layout_handle = set_layout->handle();
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &set_layout_handle;

  return allocate_info;
}

}  // namespace

DescriptorSetArena::DescriptorSetArena(
    ref_ptr<DescriptorPoolCache> descriptor_pool_cache)
    : logical_device_(add_ref(descriptor_pool_cache->logical_device())),
      descriptor_pool_cache_(std::move(descriptor_pool_cache)) {}

DescriptorSetArena::~DescriptorSetArena() {
  if (!used_descriptor_pools_.empty()) {
    descriptor_pool_cache_
        ->ReleaseDescriptorPools(absl::MakeSpan(used_descriptor_pools_))
        .IgnoreError();
    used_descriptor_pools_.clear();
  }
}

Status DescriptorSetArena::BindDescriptorSet(
    VkCommandBuffer command_buffer, PipelineExecutableLayout* executable_layout,
    int32_t set, absl::Span<const DescriptorSet::Binding> bindings) {
  // Always prefer using push descriptors when available as we can avoid the
  // additional API overhead of updating/resetting pools.
  if (logical_device_->enabled_extensions().push_descriptors) {
    return PushDescriptorSet(command_buffer, executable_layout, set, bindings);
  }

  IREE_TRACE_SCOPE0("DescriptorSetArena::BindDescriptorSet");

  auto* set_layout = executable_layout->set_layouts()[set].get();

  // Pick a bucket based on the number of descriptors required.
  // NOTE: right now we are 1:1 with bindings.
  int required_descriptor_count = bindings.size() * 1;
  int max_descriptor_count =
      std::max(8, RoundUpToNearestPow2(required_descriptor_count));
  int bucket = TrailingZeros(max_descriptor_count >> 3);
  if (bucket >= descriptor_pool_buckets_.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Too many descriptors required: " << required_descriptor_count
           << " (max=" << (1 << (descriptor_pool_buckets_.size() + 3)) << ")";
  }
  if (descriptor_pool_buckets_[bucket].handle == VK_NULL_HANDLE) {
    // Acquire a pool for this max_descriptor_count bucket.
    IREE_ASSIGN_OR_RETURN(
        descriptor_pool_buckets_[bucket],
        descriptor_pool_cache_->AcquireDescriptorPool(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_descriptor_count));
    used_descriptor_pools_.push_back(descriptor_pool_buckets_[bucket]);
  }
  auto& descriptor_pool = descriptor_pool_buckets_[bucket];

  VkDescriptorSetAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;
  allocate_info.descriptorPool = descriptor_pool.handle;
  VkDescriptorSetLayout set_layout_handle = set_layout->handle();
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &set_layout_handle;

  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkResult result = syms().vkAllocateDescriptorSets(
      *logical_device_, &allocate_info, &descriptor_set);

  if (result == VK_ERROR_OUT_OF_POOL_MEMORY) {
    // Allocation failed because the pool is either out of descriptors or too
    // fragmented. We'll just allocate another pool.
    IREE_ASSIGN_OR_RETURN(
        descriptor_pool_buckets_[bucket],
        descriptor_pool_cache_->AcquireDescriptorPool(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_descriptor_count));
    used_descriptor_pools_.push_back(descriptor_pool_buckets_[bucket]);

    // Allocate descriptor sets.
    VkDescriptorSetAllocateInfo allocate_info;
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.pNext = nullptr;
    allocate_info.descriptorPool = descriptor_pool_buckets_[bucket].handle;
    VkDescriptorSetLayout set_layout_handle = set_layout->handle();
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &set_layout_handle;
    descriptor_set = VK_NULL_HANDLE;
    VK_RETURN_IF_ERROR(syms().vkAllocateDescriptorSets(
        *logical_device_, &allocate_info, &descriptor_set));
  }

  // Get a list of VkWriteDescriptorSet structs with all bound buffers.
  IREE_ASSIGN_OR_RETURN(auto write_infos,
                        PopulateDescriptorSetWriteInfos(
                            bindings, descriptor_set, &scratch_arena_));

  // This is the reason why push descriptor sets are good.
  // We can't batch these effectively as we don't know prior to recording what
  // descriptor sets we will need and what buffers they will point to (without
  // doing just as much work as actually recording the buffer to try to find
  // out).
  syms().vkUpdateDescriptorSets(*logical_device_, write_infos.size(),
                                write_infos.data(), 0, nullptr);

  // Bind the descriptor set.
  syms().vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 executable_layout->handle(), set, 1,
                                 &descriptor_set, 0, nullptr);

  return OkStatus();
}

Status DescriptorSetArena::PushDescriptorSet(
    VkCommandBuffer command_buffer, PipelineExecutableLayout* executable_layout,
    int32_t set, absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("DescriptorSetArena::PushDescriptorSet");

  // Get a list of VkWriteDescriptorSet structs with all bound buffers.
  IREE_ASSIGN_OR_RETURN(auto write_infos,
                        PopulateDescriptorSetWriteInfos(
                            bindings, VK_NULL_HANDLE, &scratch_arena_));

  // Fast path using push descriptors. These are pooled internally by the
  // command buffer and prevent the need for our own pooling mechanisms.
  syms().vkCmdPushDescriptorSetKHR(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      executable_layout->handle(), set, write_infos.size(), write_infos.data());

  return OkStatus();
}

StatusOr<DescriptorSetGroup> DescriptorSetArena::Flush() {
  IREE_TRACE_SCOPE0("DescriptorSetArena::Flush");

  if (used_descriptor_pools_.empty()) {
    // No resources to free.
    return DescriptorSetGroup{};
  }

  for (auto& bucket : descriptor_pool_buckets_) {
    bucket = {};
  }
  return DescriptorSetGroup(add_ref(descriptor_pool_cache_),
                            std::move(used_descriptor_pools_));
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
