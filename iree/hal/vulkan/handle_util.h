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

// Helpers for wrapping Vulkan handles that don't require us to wrap every type.
// This keeps our compilation time reasonable (as the vulkancpp library is
// insane) while giving us nice safety around cleanup and ensuring we use
// dynamic symbols and consistent allocators.
//
// Do not add functionality beyond handle management to these types. Keep our
// Vulkan usage mostly functional and C-like to ensure minimal code size and
// readability.

#ifndef IREE_HAL_VULKAN_HANDLE_UTIL_H_
#define IREE_HAL_VULKAN_HANDLE_UTIL_H_

#include <vulkan/vulkan.h>

#include "absl/synchronization/mutex.h"
#include "absl/utility/utility.h"
#include "iree/base/ref_ptr.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"

namespace iree {
namespace hal {
namespace vulkan {

class VkDeviceHandle : public RefObject<VkDeviceHandle> {
 public:
  VkDeviceHandle(const ref_ptr<DynamicSymbols>& syms,
                 DeviceExtensions enabled_extensions,
                 const VkAllocationCallbacks* allocator = nullptr)
      : syms_(add_ref(syms)),
        enabled_extensions_(enabled_extensions),
        allocator_(allocator) {}
  ~VkDeviceHandle() { reset(); }

  VkDeviceHandle(const VkDeviceHandle&) = delete;
  VkDeviceHandle& operator=(const VkDeviceHandle&) = delete;
  VkDeviceHandle(VkDeviceHandle&& other) noexcept
      : value_(absl::exchange(other.value_,
                              static_cast<VkDevice>(VK_NULL_HANDLE))),
        syms_(std::move(other.syms_)),
        enabled_extensions_(other.enabled_extensions_),
        allocator_(other.allocator_) {}

  void reset() {
    if (value_ == VK_NULL_HANDLE) return;
    syms_->vkDestroyDevice(value_, allocator_);
    value_ = VK_NULL_HANDLE;
  }

  VkDevice value() const noexcept { return value_; }
  VkDevice* mutable_value() noexcept { return &value_; }
  operator VkDevice() const noexcept { return value_; }

  const ref_ptr<DynamicSymbols>& syms() const noexcept { return syms_; }
  const VkAllocationCallbacks* allocator() const noexcept { return allocator_; }

  const DeviceExtensions& enabled_extensions() const {
    return enabled_extensions_;
  }

 private:
  VkDevice value_ = VK_NULL_HANDLE;
  ref_ptr<DynamicSymbols> syms_;
  DeviceExtensions enabled_extensions_;
  const VkAllocationCallbacks* allocator_ = nullptr;
};

class VkCommandPoolHandle : public RefObject<VkCommandPoolHandle> {
 public:
  explicit VkCommandPoolHandle(const ref_ptr<VkDeviceHandle>& logical_device)
      : logical_device_(add_ref(logical_device)) {}
  ~VkCommandPoolHandle() { reset(); }

  VkCommandPoolHandle(const VkCommandPoolHandle&) = delete;
  VkCommandPoolHandle& operator=(const VkCommandPoolHandle&) = delete;
  VkCommandPoolHandle(VkCommandPoolHandle&& other) noexcept
      : logical_device_(std::move(other.logical_device_)),
        value_(absl::exchange(other.value_,
                              static_cast<VkCommandPool>(VK_NULL_HANDLE))) {}
  VkCommandPoolHandle& operator=(VkCommandPoolHandle&& other) {
    std::swap(logical_device_, other.logical_device_);
    std::swap(value_, other.value_);
    return *this;
  }

  void reset() {
    if (value_ == VK_NULL_HANDLE) return;
    syms()->vkDestroyCommandPool(*logical_device_, value_, allocator());
    value_ = VK_NULL_HANDLE;
  }

  VkCommandPool value() const noexcept { return value_; }
  VkCommandPool* mutable_value() noexcept { return &value_; }
  operator VkCommandPool() const noexcept { return value_; }

  const ref_ptr<VkDeviceHandle>& logical_device() const noexcept {
    return logical_device_;
  }
  const ref_ptr<DynamicSymbols>& syms() const noexcept {
    return logical_device_->syms();
  }
  const VkAllocationCallbacks* allocator() const noexcept {
    return logical_device_->allocator();
  }

  absl::Mutex* mutex() const { return &mutex_; }

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkCommandPool value_ = VK_NULL_HANDLE;

  // Vulkan command pools are not thread safe and require external
  // synchronization. Since we allow arbitrary threads to allocate and
  // deallocate the HAL command buffers we need to externally synchronize.
  mutable absl::Mutex mutex_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_HANDLE_UTIL_H_
