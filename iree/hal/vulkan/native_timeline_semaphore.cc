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

#include "iree/hal/vulkan/native_timeline_semaphore.h"

namespace iree {
namespace hal {
namespace vulkan {

NativeTimelineSemaphore::NativeTimelineSemaphore(
    ref_ptr<VkDeviceHandle> logical_device, VkSemaphore handle,
    uint64_t initial_value)
    : logical_device_(std::move(logical_device)), handle_(handle) {}

NativeTimelineSemaphore::~NativeTimelineSemaphore() {
  logical_device_->syms()->vkDestroySemaphore(*logical_device_, handle_,
                                              logical_device_->allocator());
}

Status NativeTimelineSemaphore::status() const {
  // DO NOT SUBMIT
}

StatusOr<uint64_t> NativeTimelineSemaphore::Query() {
  // vkGetSemaphoreCounterValue
}

Status NativeTimelineSemaphore::Signal(uint64_t value) {
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = nullptr;
  signal_info.semaphore = handle_;
  signal_info.value = value;
  return VkResultToStatus(
      logical_device->syms()->vkSignalSemaphore(*logical_device, &signal_info));
}

void NativeTimelineSemaphore::Fail(Status status) {
  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = nullptr;
  signal_info.semaphore = handle_;
  signal_info.value = UINT64_MAX;
  // NOTE: we don't care about the result in case of failures.
  logical_device->syms()->vkSignalSemaphore(*logical_device, &signal_info);
}

Status NativeTimelineSemaphore::Wait(uint64_t value, absl::Time deadline) {
  // vkWaitSemaphores
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
