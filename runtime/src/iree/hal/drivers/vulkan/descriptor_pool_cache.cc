// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/descriptor_pool_cache.h"

#include <array>
#include <cstdint>
#include <ostream>

#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// TODO(benvanik): be more conservative with descriptor set count or allow
// chaining in the command buffer when pools run out.
static constexpr int kMaxDescriptorSets = 4096;

}  // namespace

DescriptorSetGroup::~DescriptorSetGroup() {
  IREE_ASSERT_TRUE(descriptor_pools_.empty(),
                   "DescriptorSetGroup must be reset explicitly");
}

iree_status_t DescriptorSetGroup::Reset() {
  IREE_TRACE_SCOPE0("DescriptorSetGroup::Reset");

  if (descriptor_pool_cache_ != nullptr) {
    IREE_RETURN_IF_ERROR(
        descriptor_pool_cache_->ReleaseDescriptorPools(descriptor_pools_));
  }
  descriptor_pools_.clear();

  return iree_ok_status();
}

DescriptorPoolCache::DescriptorPoolCache(VkDeviceHandle* logical_device)
    : logical_device_(logical_device) {}

iree_status_t DescriptorPoolCache::AcquireDescriptorPool(
    VkDescriptorType descriptor_type, int max_descriptor_count,
    DescriptorPool* out_descriptor_pool) {
  IREE_TRACE_SCOPE0("DescriptorPoolCache::AcquireDescriptorPool");

  // TODO(benvanik): lookup in cache.

  VkDescriptorPoolCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.maxSets = kMaxDescriptorSets;
  std::array<VkDescriptorPoolSize, 1> pool_sizes;
  pool_sizes[0].type = descriptor_type;
  pool_sizes[0].descriptorCount = max_descriptor_count * create_info.maxSets;
  create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  create_info.pPoolSizes = pool_sizes.data();

  DescriptorPool descriptor_pool;
  descriptor_pool.descriptor_type = descriptor_type;
  descriptor_pool.handle = VK_NULL_HANDLE;

  VK_RETURN_IF_ERROR(syms().vkCreateDescriptorPool(
                         *logical_device_, &create_info,
                         logical_device_->allocator(), &descriptor_pool.handle),
                     "vkCreateDescriptorPool");

  *out_descriptor_pool = descriptor_pool;
  return iree_ok_status();
}

iree_status_t DescriptorPoolCache::ReleaseDescriptorPools(
    const std::vector<DescriptorPool>& descriptor_pools) {
  IREE_TRACE_SCOPE0("DescriptorPoolCache::ReleaseDescriptorPools");

  for (const auto& descriptor_pool : descriptor_pools) {
    // Always reset immediately. We could do this on allocation instead however
    // this leads to better errors when using the validation layers as we'll
    // throw if there are in-flight command buffers using the sets in the pool.
    VK_RETURN_IF_ERROR(syms().vkResetDescriptorPool(*logical_device_,
                                                    descriptor_pool.handle, 0),
                       "vkResetDescriptorPool");

    // TODO(benvanik): release to cache.
    syms().vkDestroyDescriptorPool(*logical_device_, descriptor_pool.handle,
                                   logical_device_->allocator());
  }

  return iree_ok_status();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
