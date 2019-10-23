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

#ifndef IREE_HAL_DEVICE_H_
#define IREE_HAL_DEVICE_H_

#include <memory>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "base/status.h"
#include "base/time.h"
#include "hal/allocator.h"
#include "hal/buffer.h"
#include "hal/command_queue.h"
#include "hal/device_info.h"
#include "hal/event.h"
#include "hal/executable_cache.h"
#include "hal/semaphore.h"

namespace iree {
namespace hal {

class Device {
 public:
  virtual ~Device() = default;

  // Information about device capabilities.
  const DeviceInfo& info() const { return device_info_; }

  // TODO(benvanik): status (thermal, power mode, etc).

  // TODO(benvanik): throttling adjustment/power profile.

  // TODO(benvanik): control (suspend/resume, delay, etc).

  // An allocator providing buffers usable by the device.
  // This allocator may be shared with other devices in the same family.
  virtual Allocator* allocator() const = 0;

  // Returns a list of all general-purpose dispatch queues provided by the
  // device. In general these map 1:1 with independent execution contexts,
  // though some devices may hide that and expose only a single queue that is
  // scheduled internally.
  virtual absl::Span<CommandQueue*> dispatch_queues() const = 0;

  // Returns a list of transfer queues provided by the device. These queues may
  // perform transfer operations asynchronously with respect to execution on the
  // dispatch queues. For large sequences of transfer operations always prefer
  // using one of these queues.
  // Note that if the device does not support a dedicated transfer queue this
  // list may be the same as (or a subset of) dispatch_queues.
  virtual absl::Span<CommandQueue*> transfer_queues() const = 0;

  // TODO(b/137153339): accept initial cache data.
  // Creates a device-specific cache for executables prepared for dispatch.
  // The cache manages executable compilation, caching (on disk or in memory),
  // and lifetime. Users can decide to use one or more caches to allow differing
  // lifetimes (such as unloading modules), persistent on disk caching of only
  // specific hot executables, etc.
  //
  // Returns a thread-safe cache that must remain alive until all executables
  // using the cache are no longer in-flight.
  virtual std::shared_ptr<ExecutableCache> CreateExecutableCache() = 0;

  // Creates a command buffer for recording commands to submit to queues owned
  // by this device. The command buffer may come from a pool but will be reset
  // prior to being returned to the caller.
  virtual StatusOr<ref_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBufferModeBitfield mode,
      CommandCategoryBitfield command_categories) = 0;

  // Creates an event for recording into command buffers.
  // The returned event object is only usable with this device and events must
  // only be used to synchronize within the same queue.
  virtual StatusOr<ref_ptr<Event>> CreateEvent() = 0;

  // Creates a binary semaphore that can be used with command queues owned by
  // this device. To use the semaphores with other devices or instances they
  // must first be exported.
  virtual StatusOr<ref_ptr<BinarySemaphore>> CreateBinarySemaphore(
      bool initial_value) = 0;

  // Creates a timeline semaphore that can be used with command queues owned by
  // this device. To use the semaphores with other devices or instances they
  // must first be exported.
  virtual StatusOr<ref_ptr<TimelineSemaphore>> CreateTimelineSemaphore(
      uint64_t initial_value) = 0;

  // Creates a fence that can be used with command queues owned by this device.
  // To use the fences with other devices or instances they must first be
  // exported.
  virtual StatusOr<ref_ptr<Fence>> CreateFence(uint64_t initial_value) = 0;

  // TODO(benvanik): import/export semaphore utilities.
  // TODO(benvanik): import/export fence utilities.
  // TODO(benvanik): fences to wait handles.

  // Blocks the caller until all passed |fences| reach or exceed the specified
  // payload values or the |deadline| elapses. All |fences| must be created from
  // this device (or be imported into it).
  //
  // Returns success if the wait is successful and all fences have been
  // signaled.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline| elapses without all fences
  // having been signaled. Note that a subset of the |fences| may have been
  // signaled and each can be queried to see which ones.
  virtual Status WaitAllFences(absl::Span<const FenceValue> fences,
                               absl::Time deadline) = 0;
  inline Status WaitAllFences(absl::Span<const FenceValue> fences,
                              absl::Duration timeout) {
    return WaitAllFences(fences, RelativeTimeoutToDeadline(timeout));
  }

  // Blocks the caller until at least one of the |fences| reaches or exceeds the
  // specified payload value or the |deadline| elapses. All |fences| must be
  // created from this device (or be imported into it).
  //
  // Returns an arbitrary index into |fences| of a fence that was signaled. Note
  // that more than one fence may have been signaled and all of the other
  // |fences| should be queried or waited on again until waits for them
  // succeed.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline| elapses without any fences
  // having been signaled.
  virtual StatusOr<int> WaitAnyFence(absl::Span<const FenceValue> fences,
                                     absl::Time deadline) = 0;
  inline StatusOr<int> WaitAnyFence(absl::Span<const FenceValue> fences,
                                    absl::Duration timeout) {
    return WaitAnyFence(fences, RelativeTimeoutToDeadline(timeout));
  }

  // Blocks until all outstanding requests on all queues have been
  // completed. This is equivalent to having waited on all outstanding
  // fences.
  virtual Status WaitIdle(absl::Time deadline) = 0;
  inline Status WaitIdle(absl::Duration timeout) {
    return WaitIdle(RelativeTimeoutToDeadline(timeout));
  }
  inline Status WaitIdle() { return WaitIdle(absl::InfiniteFuture()); }

 protected:
  explicit Device(DeviceInfo device_info)
      : device_info_(std::move(device_info)) {}

 private:
  const DeviceInfo device_info_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DEVICE_H_
