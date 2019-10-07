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

#ifndef IREE_HAL_VULKAN_DESCRIPTOR_POOL_CACHE_H_
#define IREE_HAL_VULKAN_DESCRIPTOR_POOL_CACHE_H_

#include "iree/base/ref_ptr.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/handle_util.h"

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
  // Maximum number of descriptors of the given type per allocation.
  int max_descriptor_count = 0;
  // Pool handle.
  VkDescriptorPool handle = VK_NULL_HANDLE;
};

// A group of descriptor sets allocated and released together.
// The group must be explicitly reset with Reset() prior to disposing.
class DescriptorSetGroup final {
 public:
  DescriptorSetGroup() = default;
  DescriptorSetGroup(ref_ptr<DescriptorPoolCache> descriptor_pool_cache,
                     absl::InlinedVector<DescriptorPool, 8> descriptor_pools)
      : descriptor_pool_cache_(std::move(descriptor_pool_cache)),
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

  Status Reset();

 private:
  ref_ptr<DescriptorPoolCache> descriptor_pool_cache_;
  absl::InlinedVector<DescriptorPool, 8> descriptor_pools_;
};

// A "cache" (or really, pool) of descriptor pools. These pools are allocated
// as needed to satisfy different descriptor size requirements and are given
// to command buffers during recording to write descriptor updates and bind
// resources. After the descriptors in the pool are no longer used (all
// command buffers using descriptor sets allocated from the pool have retired)
// the pool is returned here to be reused in the future.
class DescriptorPoolCache final : public RefObject<DescriptorPoolCache> {
 public:
  explicit DescriptorPoolCache(ref_ptr<VkDeviceHandle> logical_device);

  const ref_ptr<VkDeviceHandle>& logical_device() const {
    return logical_device_;
  }
  const DynamicSymbols& syms() const { return *logical_device_->syms(); }

  // Acquires a new descriptor pool for use by the caller.
  // The pool will have been reset and have all descriptor sets available.
  // When all sets allocated from the pool are no longer in use it must be
  // returned to the cache with ReleaseDescriptorPool.
  StatusOr<DescriptorPool> AcquireDescriptorPool(
      VkDescriptorType descriptor_type, int max_descriptor_count);

  // Releases descriptor pools back to the cache. The pools will be reset
  // immediately and must no longer be in use by any in-flight command.
  Status ReleaseDescriptorPools(absl::Span<DescriptorPool> descriptor_pools);

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_DESCRIPTOR_POOL_CACHE_H_
