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

#include "iree/hal/metal/metal_shared_event.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/metal/dispatch_time_util.h"

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<ref_ptr<Semaphore>> MetalSharedEvent::Create(id<MTLDevice> device,
                                                      uint64_t initial_value) {
  return assign_ref(new MetalSharedEvent(device, initial_value));
}

MetalSharedEvent::MetalSharedEvent(id<MTLDevice> device, uint64_t initial_value)
    : metal_handle_([device newSharedEvent]) {
  IREE_TRACE_SCOPE0("MetalSharedEvent::ctor");
  metal_handle_.signaledValue = initial_value;
}

MetalSharedEvent::~MetalSharedEvent() {
  IREE_TRACE_SCOPE0("MetalSharedEvent::dtor");
  [metal_handle_ release];
}

StatusOr<uint64_t> MetalSharedEvent::Query() {
  IREE_TRACE_SCOPE0("MetalSharedEvent::Query");
  uint64_t value = metal_handle_.signaledValue;
  if (value == UINT64_MAX) {
    absl::MutexLock lock(&status_mutex_);
    return status_;
  }
  return value;
}

Status MetalSharedEvent::Signal(uint64_t value) {
  IREE_TRACE_SCOPE0("MetalSharedEvent::Signal");
  metal_handle_.signaledValue = value;
  return OkStatus();
}

void MetalSharedEvent::Fail(Status status) {
  IREE_TRACE_SCOPE0("MetalSharedEvent::Fail");
  absl::MutexLock lock(&status_mutex_);
  status_ = std::move(status);
  metal_handle_.signaledValue = UINT64_MAX;
}

Status MetalSharedEvent::Wait(uint64_t value, Time deadline_ns) {
  IREE_TRACE_SCOPE0("MetalSharedEvent::Wait");

  Duration duration_ns = DeadlineToRelativeTimeoutNanos(deadline_ns);
  dispatch_time_t timeout = DurationToDispatchTime(duration_ns);

  // Quick path for impatient waiting to avoid all the overhead of dispatch queues and semaphores.
  if (duration_ns == ZeroDuration()) {
    if (metal_handle_.signaledValue < value) {
      return DeadlineExceededErrorBuilder(IREE_LOC) << "Deadline exceeded waiting for semaphores";
    }
    return OkStatus();
  }

  // Theoretically we don't really need to mark the semaphore handle as __block given that the
  // handle itself is not modified and there is only one block and it will copy the handle.
  // But marking it as __block serves as good documentation purpose.
  __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);

  // Register a listener to the MTLSharedEvent to notify us when the work is done on GPU by
  // signaling a semaphore. The signaling will happen in a new dispatch queue; the current thread
  // will wait on the semaphore.
  dispatch_queue_t wait_notifier = dispatch_queue_create("MetalSharedEvent::Wait Notifier", NULL);
  MTLSharedEventListener* listener =
      [[MTLSharedEventListener alloc] initWithDispatchQueue:wait_notifier];

  [metal_handle_ notifyListener:listener
                        atValue:value
                          block:^(id<MTLSharedEvent>, uint64_t) {
                            dispatch_semaphore_signal(work_done);
                          }];

  long timed_out = dispatch_semaphore_wait(work_done, timeout);

  [listener release];
  dispatch_release(wait_notifier);
  dispatch_release(work_done);

  if (timed_out) {
    return DeadlineExceededErrorBuilder(IREE_LOC)
           << "Deadline exceeded waiting for dispatch_semaphore_t";
  }
  return OkStatus();
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
