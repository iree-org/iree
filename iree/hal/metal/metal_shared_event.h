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

#ifndef IREE_HAL_METAL_METAL_SHARED_EVENT_H_
#define IREE_HAL_METAL_METAL_SHARED_EVENT_H_

#import <Metal/Metal.h>

#include "absl/synchronization/mutex.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {
namespace metal {

// A semaphore implementation for Metal that directly wraps a MTLSharedEvent.
class MetalSharedEvent final : public Semaphore {
 public:
  // Creates a MetalSharedEvent with the given |initial_value|.
  static StatusOr<ref_ptr<Semaphore>> Create(id<MTLDevice> device,
                                             uint64_t initial_value);

  ~MetalSharedEvent() override;

  id<MTLSharedEvent> handle() const { return metal_handle_; }

  StatusOr<uint64_t> Query() override;

  Status Signal(uint64_t value) override;

  void Fail(Status status) override;

  Status Wait(uint64_t value, Time deadline_ns) override;

 private:
  MetalSharedEvent(id<MTLDevice> device, uint64_t initial_value);

  id<MTLSharedEvent> metal_handle_;

  // NOTE: the MTLSharedEvent is the source of truth. We only need to access
  // this status (and thus take the lock) when we want to either signal failure
  // or query the status in the case of the semaphore being set to UINT64_MAX.
  mutable absl::Mutex status_mutex_;
  Status status_ ABSL_GUARDED_BY(status_mutex_);
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_SHARED_EVENT_H_
