// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers for wrapping Vulkan handles that don't require us to wrap every type.
// This keeps our compilation time reasonable (as the vulkancpp library is
// insane) while giving us nice safety around cleanup and ensuring we use
// dynamic symbols and consistent allocators.
//
// Do not add functionality beyond handle management to these types. Keep our
// Vulkan usage mostly functional and C-like to ensure minimal code size and
// readability.

#ifndef IREE_HAL_DRIVERS_VULKAN_HANDLE_UTIL_H_
#define IREE_HAL_DRIVERS_VULKAN_HANDLE_UTIL_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"  // IWYU pragma: export
// clang-format on

#include "iree/base/internal/synchronization.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/extensibility_util.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

namespace iree {
namespace hal {
namespace vulkan {

template <class T, class U = T>
constexpr T exchange(T& obj, U&& new_value) {
  T old_value = std::move(obj);
  obj = std::forward<U>(new_value);
  return old_value;
}

class VkDeviceHandle : public RefObject<VkDeviceHandle> {
 public:
  VkDeviceHandle(DynamicSymbols* syms, VkPhysicalDevice physical_device,
                 iree_hal_vulkan_features_t enabled_features,
                 iree_hal_vulkan_device_extensions_t enabled_extensions,
                 iree_hal_vulkan_device_properties_t supported_properties,
                 bool owns_device, iree_allocator_t host_allocator,
                 const VkAllocationCallbacks* allocator = nullptr)
      : physical_device_(physical_device),
        syms_(add_ref(syms)),
        enabled_features_(enabled_features),
        enabled_extensions_(enabled_extensions),
        supported_properties_(supported_properties),
        owns_device_(owns_device),
        allocator_(allocator),
        host_allocator_(host_allocator) {}
  ~VkDeviceHandle() { reset(); }

  VkDeviceHandle(const VkDeviceHandle&) = delete;
  VkDeviceHandle& operator=(const VkDeviceHandle&) = delete;
  VkDeviceHandle(VkDeviceHandle&& other) noexcept
      : physical_device_(
            exchange(other.physical_device_,
                     static_cast<VkPhysicalDevice>(VK_NULL_HANDLE))),
        value_(exchange(other.value_, static_cast<VkDevice>(VK_NULL_HANDLE))),
        syms_(std::move(other.syms_)),
        enabled_extensions_(other.enabled_extensions_),
        supported_properties_(other.supported_properties_),
        owns_device_(other.owns_device_),
        allocator_(other.allocator_),
        host_allocator_(other.host_allocator_) {}

  void reset() {
    if (value_ == VK_NULL_HANDLE) return;
    if (owns_device_) {
      syms_->vkDestroyDevice(value_, allocator_);
    }
    value_ = VK_NULL_HANDLE;
  }

  VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
  operator VkPhysicalDevice() const noexcept { return physical_device_; }

  VkDevice value() const noexcept { return value_; }
  VkDevice* mutable_value() noexcept { return &value_; }
  operator VkDevice() const noexcept { return value_; }

  const ref_ptr<DynamicSymbols>& syms() const noexcept { return syms_; }
  const VkAllocationCallbacks* allocator() const noexcept { return allocator_; }
  iree_allocator_t host_allocator() const noexcept { return host_allocator_; }

  iree_hal_vulkan_features_t enabled_features() const {
    return enabled_features_;
  }

  const iree_hal_vulkan_device_extensions_t& enabled_extensions() const {
    return enabled_extensions_;
  }

  const iree_hal_vulkan_device_properties_t& supported_properties() const {
    return supported_properties_;
  }

 private:
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice value_ = VK_NULL_HANDLE;
  ref_ptr<DynamicSymbols> syms_;
  iree_hal_vulkan_features_t enabled_features_;
  iree_hal_vulkan_device_extensions_t enabled_extensions_;
  iree_hal_vulkan_device_properties_t supported_properties_;
  bool owns_device_;
  const VkAllocationCallbacks* allocator_ = nullptr;
  iree_allocator_t host_allocator_;
};

class VkCommandPoolHandle {
 public:
  explicit VkCommandPoolHandle(VkDeviceHandle* logical_device)
      : logical_device_(logical_device) {
    iree_slim_mutex_initialize(&mutex_);
  }
  ~VkCommandPoolHandle() {
    reset();
    iree_slim_mutex_deinitialize(&mutex_);
  }

  VkCommandPoolHandle(const VkCommandPoolHandle&) = delete;
  VkCommandPoolHandle& operator=(const VkCommandPoolHandle&) = delete;
  VkCommandPoolHandle(VkCommandPoolHandle&& other) noexcept
      : logical_device_(std::move(other.logical_device_)),
        value_(exchange(other.value_,
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

  const VkDeviceHandle* logical_device() const noexcept {
    return logical_device_;
  }
  const ref_ptr<DynamicSymbols>& syms() const noexcept {
    return logical_device_->syms();
  }
  const VkAllocationCallbacks* allocator() const noexcept {
    return logical_device_->allocator();
  }

  iree_status_t Allocate(const VkCommandBufferAllocateInfo* allocate_info,
                         VkCommandBuffer* out_handle) {
    iree_slim_mutex_lock(&mutex_);
    iree_status_t status =
        VK_RESULT_TO_STATUS(syms()->vkAllocateCommandBuffers(
                                *logical_device_, allocate_info, out_handle),
                            "vkAllocateCommandBuffers");
    iree_slim_mutex_unlock(&mutex_);
    return status;
  }

  void Free(VkCommandBuffer handle) {
    iree_slim_mutex_lock(&mutex_);
    syms()->vkFreeCommandBuffers(*logical_device_, value_, 1, &handle);
    iree_slim_mutex_unlock(&mutex_);
  }

 private:
  VkDeviceHandle* logical_device_;
  VkCommandPool value_ = VK_NULL_HANDLE;

  // Vulkan command pools are not thread safe and require external
  // synchronization. Since we allow arbitrary threads to allocate and
  // deallocate the HAL command buffers we need to externally synchronize.
  iree_slim_mutex_t mutex_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_HANDLE_UTIL_H_
