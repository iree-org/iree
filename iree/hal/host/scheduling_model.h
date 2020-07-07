// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_HOST_SCHEDULING_MODEL_H_
#define IREE_HAL_HOST_SCHEDULING_MODEL_H_

#include "iree/hal/command_queue.h"

namespace iree {
namespace hal {
namespace host {

// Host-local scheduling interface that device implementations can use to choose
// between various scheduling strategies (such as serial/in-order,
// fiber/out-of-order, etc). The interface models a subset of the Device
// interface relating to the scheduling primitives (such as semaphores) and the
// device-level operations that can be performed on them (such as wait-all).
class SchedulingModel {
 public:
  virtual ~SchedulingModel() = default;

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

  // Creates a command buffer for recording commands to submit to queues owned
  // by this device. The command buffer may come from a pool but will be reset
  // prior to being returned to the caller.
  virtual StatusOr<ref_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBufferModeBitfield mode,
      CommandCategoryBitfield command_categories) = 0;

  // Creates an event for recording into command buffers.
  virtual StatusOr<ref_ptr<Event>> CreateEvent() = 0;

  // Creates a semaphore that can be used with command queues owned by this
  // device. To use the semaphores with other devices or instances they must
  // first be exported.
  virtual StatusOr<ref_ptr<Semaphore>> CreateSemaphore(
      uint64_t initial_value) = 0;

  // Blocks the caller until all passed |semaphores| reach or exceed the
  // specified payload values or the |deadline| elapses. All |semaphores| must
  // be created from this device (or be imported into it).
  //
  // Returns success if the wait is successful and all semaphores have been
  // signaled.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline| elapses without all semaphores
  // having been signaled. Note that a subset of the |semaphores| may have been
  // signaled and each can be queried to see which ones.
  virtual Status WaitAllSemaphores(absl::Span<const SemaphoreValue> semaphores,
                                   absl::Time deadline) = 0;

  // Blocks the caller until at least one of the |semaphores| reaches or exceeds
  // the specified payload value or the |deadline| elapses. All |semaphores|
  // must be created from this device (or be imported into it).
  //
  // Returns an arbitrary index into |semaphores| of a semaphore that was
  // signaled. Note that more than one semaphore may have been signaled and all
  // of the other |semaphores| should be queried or waited on again until waits
  // for them succeed.
  //
  // Returns DEADLINE_EXCEEDED if the |deadline| elapses without any semaphores
  // having been signaled.
  virtual StatusOr<int> WaitAnySemaphore(
      absl::Span<const SemaphoreValue> semaphores, absl::Time deadline) = 0;

  // Blocks until all outstanding requests on all queues have been
  // completed. This is equivalent to having waited on all outstanding
  // semaphores.
  virtual Status WaitIdle(absl::Time deadline) = 0;
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_SCHEDULING_MODEL_H_
