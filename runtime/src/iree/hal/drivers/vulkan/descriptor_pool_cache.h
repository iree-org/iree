// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_POOL_CACHE_H_
#define IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_POOL_CACHE_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

class DescriptorPoolCache;

// A descriptor pool with a single descriptor type of some number.
// We only support a single descriptor type for now as we only generate SPIR-V
// that uses a single type.
struct DescriptorPool {
  // Type of the descriptor in the set.
  VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  // Pool handle.
  VkDescriptorPool handle = VK_NULL_HANDLE;
};

// A group of descriptor sets allocated and released together.
// The group must be explicitly reset with Reset() prior to disposing.
class DescriptorSetGroup final {
 public:
  DescriptorSetGroup() = default;
  DescriptorSetGroup(DescriptorPoolCache* descriptor_pool_cache,
                     std::vector<DescriptorPool> descriptor_pools)
      : descriptor_pool_cache_(descriptor_pool_cache),
        descriptor_pools_(std::move(descriptor_pools)) {}
  DescriptorSetGroup(const DescriptorSetGroup&) = delete;
  DescriptorSetGroup& operator=(const DescriptorSetGroup&) = delete;
  DescriptorSetGroup(DescriptorSetGroup&& other) noexcept
      : descriptor_pool_cache_(std::move(other.descriptor_pool_cache_)),
        descriptor_pools_(std::move(other.descriptor_pools_)) {}
  DescriptorSetGroup& operator=(DescriptorSetGroup&& other) {
    std::swap(descriptor_pool_cache_, other.descriptor_pool_cache_);
    std::swap(descriptor_pools_, other.descriptor_pools_);
    return *this;
  }
  ~DescriptorSetGroup();

  iree_status_t Reset();

 private:
  DescriptorPoolCache* descriptor_pool_cache_;
  std::vector<DescriptorPool> descriptor_pools_;
};

// A "cache" (or really, pool) of descriptor pools. These pools are allocated
// as needed to satisfy different descriptor size requirements and are given
// to command buffers during recording to write descriptor updates and bind
// resources. After the descriptors in the pool are no longer used (all
// command buffers using descriptor sets allocated from the pool have retired)
// the pool is returned here to be reused in the future.
class DescriptorPoolCache final {
 public:
  explicit DescriptorPoolCache(VkDeviceHandle* logical_device);

  VkDeviceHandle* logical_device() const { return logical_device_; }
  const DynamicSymbols& syms() const { return *logical_device_->syms(); }

  // Acquires a new descriptor pool for use by the caller.
  // The pool will have been reset and have all descriptor sets available.
  // When all sets allocated from the pool are no longer in use it must be
  // returned to the cache with ReleaseDescriptorPool.
  iree_status_t AcquireDescriptorPool(VkDescriptorType descriptor_type,
                                      int max_descriptor_count,
                                      DescriptorPool* out_descriptor_pool);

  // Releases descriptor pools back to the cache. The pools will be reset
  // immediately and must no longer be in use by any in-flight command.
  iree_status_t ReleaseDescriptorPools(
      const std::vector<DescriptorPool>& descriptor_pools);

 private:
  VkDeviceHandle* logical_device_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_DESCRIPTOR_POOL_CACHE_H_
