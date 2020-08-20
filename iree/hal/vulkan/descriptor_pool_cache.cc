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

#include "iree/hal/vulkan/descriptor_pool_cache.h"

#include <array>

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// TODO(benvanik): be more conservative with descriptor set count or allow
// chaining in the command buffer when pools run out.
static constexpr int kMaxDescriptorSets = 4096;

}  // namespace

DescriptorSetGroup::~DescriptorSetGroup() {
  CHECK(descriptor_pools_.empty())
      << "DescriptorSetGroup must be reset explicitly";
}

Status DescriptorSetGroup::Reset() {
  IREE_TRACE_SCOPE0("DescriptorSetGroup::Reset");

  if (descriptor_pool_cache_ != nullptr) {
    IREE_RETURN_IF_ERROR(descriptor_pool_cache_->ReleaseDescriptorPools(
        absl::MakeSpan(descriptor_pools_)));
  }
  descriptor_pools_.clear();

  return OkStatus();
}

DescriptorPoolCache::DescriptorPoolCache(ref_ptr<VkDeviceHandle> logical_device)
    : logical_device_(std::move(logical_device)) {}

StatusOr<DescriptorPool> DescriptorPoolCache::AcquireDescriptorPool(
    VkDescriptorType descriptor_type, int max_descriptor_count) {
  IREE_TRACE_SCOPE0("DescriptorPoolCache::AcquireDescriptorPool");

  // TODO(benvanik): lookup in cache.

  VkDescriptorPoolCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.maxSets = kMaxDescriptorSets;
  std::array<VkDescriptorPoolSize, 1> pool_sizes;
  pool_sizes[0].type = descriptor_type;
  pool_sizes[0].descriptorCount = max_descriptor_count;
  create_info.poolSizeCount = pool_sizes.size();
  create_info.pPoolSizes = pool_sizes.data();

  DescriptorPool descriptor_pool;
  descriptor_pool.descriptor_type = descriptor_type;
  descriptor_pool.max_descriptor_count = max_descriptor_count;
  descriptor_pool.handle = VK_NULL_HANDLE;

  VK_RETURN_IF_ERROR(syms().vkCreateDescriptorPool(
      *logical_device_, &create_info, logical_device_->allocator(),
      &descriptor_pool.handle));

  return descriptor_pool;
}

Status DescriptorPoolCache::ReleaseDescriptorPools(
    absl::Span<DescriptorPool> descriptor_pools) {
  IREE_TRACE_SCOPE0("DescriptorPoolCache::ReleaseDescriptorPools");

  for (const auto& descriptor_pool : descriptor_pools) {
    // Always reset immediately. We could do this on allocation instead however
    // this leads to better errors when using the validation layers as we'll
    // throw if there are in-flight command buffers using the sets in the pool.
    VK_RETURN_IF_ERROR(syms().vkResetDescriptorPool(*logical_device_,
                                                    descriptor_pool.handle, 0));

    // TODO(benvanik): release to cache.
    syms().vkDestroyDescriptorPool(*logical_device_, descriptor_pool.handle,
                                   logical_device_->allocator());
  }

  return OkStatus();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
