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

#include "iree/hal/device_manager.h"

#include <algorithm>

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/heap_buffer.h"

namespace iree {
namespace hal {

DeviceManager::DeviceManager() = default;

DeviceManager::~DeviceManager() {
  IREE_TRACE_SCOPE0("DeviceManager::dtor");
  WaitIdle().IgnoreError();
}

Status DeviceManager::RegisterDevice(ref_ptr<Device> device) {
  IREE_TRACE_SCOPE0("DeviceManager::RegisterDevice");
  absl::MutexLock lock(&device_mutex_);
  if (std::find(devices_.begin(), devices_.end(), device) != devices_.end()) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Device already registered";
  }
  devices_.push_back(std::move(device));
  return OkStatus();
}

Status DeviceManager::UnregisterDevice(Device* device) {
  IREE_TRACE_SCOPE0("DeviceManager::UnregisterDevice");
  absl::MutexLock lock(&device_mutex_);
  auto it = std::find_if(devices_.begin(), devices_.end(),
                         [device](const ref_ptr<Device>& other_device) {
                           return device == other_device.get();
                         });
  if (it == devices_.end()) {
    return NotFoundErrorBuilder(IREE_LOC) << "Device not registered";
  }
  devices_.erase(it);
  return OkStatus();
}

StatusOr<DevicePlacement> DeviceManager::ResolvePlacement(
    const PlacementSpec& placement_spec) const {
  IREE_TRACE_SCOPE0("DeviceManager::ResolvePlacement");
  absl::MutexLock lock(&device_mutex_);
  if (devices_.empty()) {
    return NotFoundErrorBuilder(IREE_LOC) << "No devices registered";
  }

  // TODO(benvanik): multiple devices and placement.
  QCHECK_EQ(devices_.size(), 1)
      << "Multiple devices not yet supported (need placement)";
  DevicePlacement device_placement;
  device_placement.device = devices_.front().get();

  return device_placement;
}

StatusOr<Allocator*> DeviceManager::FindCompatibleAllocator(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    absl::Span<const DevicePlacement> device_placements) const {
  IREE_TRACE_SCOPE0("DeviceManager::FindCompatibleAllocator");
  if (device_placements.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No placements provided";
  }

  // Find the first allocator. As we only return an allocator if all placements
  // are compatible we'll compare allocator[0] against allocator[1,N].
  Allocator* some_allocator = nullptr;
  for (const auto& device_placement : device_placements) {
    auto* allocator = device_placement.device->allocator();
    if (!some_allocator) {
      some_allocator = allocator;
      continue;
    }
    // NOTE: as there can be asymmetry between usage restrictions (A can use B
    // but B cannot use A) we have to compare both directions.
    if (!some_allocator->CanUseBufferLike(allocator, memory_type, buffer_usage,
                                          buffer_usage) ||
        !allocator->CanUseBufferLike(some_allocator, memory_type, buffer_usage,
                                     buffer_usage)) {
      // Allocators are not compatible.
      return NotFoundErrorBuilder(IREE_LOC)
             << "No single allocator found that is compatible with all "
                "placements";
    }
  }
  return some_allocator;
}

StatusOr<ref_ptr<Buffer>> DeviceManager::TryAllocateDeviceVisibleBuffer(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    device_size_t allocation_size,
    absl::Span<const DevicePlacement> device_placements) {
  IREE_TRACE_SCOPE("DeviceManager::TryAllocateDeviceVisibleBuffer:size", int)
  (static_cast<int>(allocation_size));
  if (!AnyBitSet(memory_type & MemoryType::kHostLocal)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Host-local buffers require the kHostLocal bit: "
           << MemoryTypeString(memory_type);
  }

  // Strip kDeviceVisible as we conditionally add it based on support.
  memory_type &= ~MemoryType::kDeviceVisible;

  // Find an allocator that works for device-visible buffers.
  // If this fails we'll fall back to allocation a non-device-visible buffer.
  auto allocator_or =
      FindCompatibleAllocator(memory_type | MemoryType::kDeviceVisible,
                              buffer_usage, device_placements);
  if (allocator_or.ok()) {
    return allocator_or.value()->Allocate(
        memory_type | MemoryType::kDeviceVisible, buffer_usage,
        allocation_size);
  }

  // Fallback to allocating a host-local buffer.
  return HeapBuffer::Allocate(memory_type, buffer_usage, allocation_size);
}

StatusOr<ref_ptr<Buffer>> DeviceManager::AllocateDeviceVisibleBuffer(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    device_size_t allocation_size,
    absl::Span<const DevicePlacement> device_placements) {
  IREE_TRACE_SCOPE("DeviceManager::AllocateDeviceVisibleBuffer:size", int)
  (static_cast<int>(allocation_size));
  if (!AnyBitSet(memory_type & MemoryType::kHostLocal)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Host-local buffers require the kHostLocal bit: "
           << MemoryTypeString(memory_type);
  }

  // Always use device-visible.
  memory_type |= MemoryType::kDeviceVisible;

  // Find an allocator that works for device-visible buffers.
  IREE_ASSIGN_OR_RETURN(
      auto* allocator,
      FindCompatibleAllocator(memory_type, buffer_usage, device_placements));
  return allocator->Allocate(memory_type, buffer_usage, allocation_size);
}

StatusOr<ref_ptr<Buffer>> DeviceManager::AllocateDeviceLocalBuffer(
    MemoryTypeBitfield memory_type, BufferUsageBitfield buffer_usage,
    device_size_t allocation_size,
    absl::Span<const DevicePlacement> device_placements) {
  IREE_TRACE_SCOPE("DeviceManager::AllocateDeviceLocalBuffer:size", int)
  (static_cast<int>(allocation_size));
  if (!AnyBitSet(memory_type & MemoryType::kDeviceLocal)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Device-local buffers require the kDeviceLocal bit: "
           << MemoryTypeString(memory_type);
  }

  // Find an allocator that works for device-local buffers.
  IREE_ASSIGN_OR_RETURN(
      auto* allocator,
      FindCompatibleAllocator(memory_type, buffer_usage, device_placements));
  return allocator->Allocate(memory_type, buffer_usage, allocation_size);
}

Status DeviceManager::Submit(Device* device, CommandQueue* command_queue,
                             absl::Span<const SubmissionBatch> batches,
                             Time deadline_ns) {
  IREE_TRACE_SCOPE0("DeviceManager::Submit");
  return command_queue->Submit(batches);
}

Status DeviceManager::Flush() {
  IREE_TRACE_SCOPE0("DeviceManager::Flush");
  return OkStatus();
}

Status DeviceManager::WaitIdle(Time deadline_ns) {
  IREE_TRACE_SCOPE0("DeviceManager::WaitIdle");
  absl::MutexLock lock(&device_mutex_);
  for (const auto& device : devices_) {
    IREE_RETURN_IF_ERROR(device->WaitIdle(deadline_ns));
  }
  return OkStatus();
}

}  // namespace hal
}  // namespace iree
