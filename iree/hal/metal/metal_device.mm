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

#include "iree/hal/metal/metal_device.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_buffer_validation.h"
#include "iree/hal/metal/apple_time_util.h"
#include "iree/hal/metal/metal_command_buffer.h"
#include "iree/hal/metal/metal_command_queue.h"
#include "iree/hal/metal/metal_shared_event.h"

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<ref_ptr<MetalDevice>> MetalDevice::Create(ref_ptr<Driver> driver,
                                                   const DeviceInfo& device_info) {
  return assign_ref(new MetalDevice(std::move(driver), device_info));
}

MetalDevice::MetalDevice(ref_ptr<Driver> driver, const DeviceInfo& device_info)
    : Device(device_info),
      driver_(std::move(driver)),
      metal_handle_([(__bridge id<MTLDevice>)device_info.device_id() retain]) {
  IREE_TRACE_SCOPE0("MetalDevice::ctor");

  // Grab one queue for dispatch and transfer.
  std::string name = absl::StrCat(device_info.name(), ":queue");
  id<MTLCommandQueue> metal_queue = [metal_handle_ newCommandQueue];  // retained
  command_queue_ = absl::make_unique<MetalCommandQueue>(
      name, CommandCategory::kDispatch | CommandCategory::kTransfer, metal_queue);
  common_queue_ = command_queue_.get();
  // MetalCommandQueue retains by itself. Release here to avoid leaking.
  [metal_queue release];
}

MetalDevice::~MetalDevice() {
  IREE_TRACE_SCOPE0("MetalDevice::dtor");
  [metal_handle_ release];
}

std::string MetalDevice::DebugString() const {
  return absl::StrCat(Device::DebugString(),         //
                      "\n[MetalDevice]",             //
                      "\n    - Dispatch Queues: 1",  //
                      "\n    - Transfer Queues: 1");
}

ref_ptr<ExecutableCache> MetalDevice::CreateExecutableCache() {
  IREE_TRACE_SCOPE0("MetalDevice::CreateExecutableCache");
  return nullptr;
}

StatusOr<ref_ptr<DescriptorSetLayout>> MetalDevice::CreateDescriptorSetLayout(
    DescriptorSetLayout::UsageType usage_type,
    absl::Span<const DescriptorSetLayout::Binding> bindings) {
  IREE_TRACE_SCOPE0("MetalDevice::CreateDescriptorSetLayout");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalDevice::CreateDescriptorSetLayout";
}

StatusOr<ref_ptr<ExecutableLayout>> MetalDevice::CreateExecutableLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants) {
  IREE_TRACE_SCOPE0("MetalDevice::CreateExecutableLayout");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalDevice::CreateExecutableLayout";
}

StatusOr<ref_ptr<DescriptorSet>> MetalDevice::CreateDescriptorSet(
    DescriptorSetLayout* set_layout, absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("MetalDevice::CreateDescriptorSet");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalDevice::CreateDescriptorSet";
}

StatusOr<ref_ptr<CommandBuffer>> MetalDevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode, CommandCategoryBitfield command_categories) {
  IREE_TRACE_SCOPE0("MetalDevice::CreateCommandBuffer");
  @autoreleasepool {
    StatusOr<ref_ptr<CommandBuffer>> command_buffer;
    // We use commandBufferWithUnretainedReferences here to be performant. This is okay becasue
    // IREE tracks the lifetime of various objects with the help from compilers.
    id<MTLCommandBuffer> cmdbuf = [static_cast<MetalCommandQueue*>(common_queue_)->handle()
        commandBufferWithUnretainedReferences];
    command_buffer = MetalCommandBuffer::Create(mode, command_categories, cmdbuf);
    // TODO: WrapCommandBufferWithValidation(allocator(), std::move(impl));
    return command_buffer;
  }
}

StatusOr<ref_ptr<Event>> MetalDevice::CreateEvent() {
  IREE_TRACE_SCOPE0("MetalDevice::CreateEvent");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalDevice::CreateEvent";
}

StatusOr<ref_ptr<Semaphore>> MetalDevice::CreateSemaphore(uint64_t initial_value) {
  IREE_TRACE_SCOPE0("MetalDevice::CreateSemaphore");
  return MetalSharedEvent::Create(metal_handle_, initial_value);
}

Status MetalDevice::WaitAllSemaphores(absl::Span<const SemaphoreValue> semaphores,
                                      Time deadline_ns) {
  IREE_TRACE_SCOPE0("MetalDevice::WaitAllSemaphores");
  // Go through all MetalSharedEvents and wait on each of them given we need all of them to be
  // signaled anyway.
  for (int i = 0; i < semaphores.size(); ++i) {
    auto* semaphore = static_cast<MetalSharedEvent*>(semaphores[i].semaphore);
    IREE_RETURN_IF_ERROR(semaphore->Wait(semaphores[i].value, deadline_ns));
  }
  return OkStatus();
}

StatusOr<int> MetalDevice::WaitAnySemaphore(absl::Span<const SemaphoreValue> semaphores,
                                            Time deadline_ns) {
  IREE_TRACE_SCOPE0("MetalDevice::WaitAnySemaphore");

  if (semaphores.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "expected to have at least one semaphore";
  }

  // If there is just one semaphore, just wait on it.
  if (semaphores.size() == 1) {
    auto* semaphore = static_cast<MetalSharedEvent*>(semaphores[0].semaphore);
    IREE_RETURN_IF_ERROR(semaphore->Wait(semaphores[0].value, deadline_ns));
    return 0;
  }

  // Otherwise, we need to go down a more complicated path by registering listeners to all
  // MTLSharedEvnts to notify us when at least one of them has done the work on GPU by signaling a
  // semaphore. The signaling will happen in a new dispatch queue; the current thread will wait on
  // the semaphore.

  dispatch_time_t timeout = DeadlineToDispatchTime(deadline_ns);

  // Store the handle as a __block variable to allow blocks accessing the same copy for the
  // semaphore handle on heap.
  // Use an initial value of zero so that any semaphore signal will unblock the wait.
  __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);
  // Also create a __block variable to store the index for the signaled semaphore.
  __block int signaled_index = 0;

  dispatch_queue_t wait_notifier =
      dispatch_queue_create("MetalDevice::WaitAnySemaphore Notifier", NULL);
  MTLSharedEventListener* listener =
      [[MTLSharedEventListener alloc] initWithDispatchQueue:wait_notifier];

  // The dispatch queue created in the above is a serial one. So even if multiple semaphores signal,
  // the semaphore signaling should be serialized.
  for (int i = 0; i < semaphores.size(); ++i) {
    auto* semaphore = static_cast<MetalSharedEvent*>(semaphores[i].semaphore);
    [semaphore->handle() notifyListener:listener
                                atValue:semaphores[i].value
                                  block:^(id<MTLSharedEvent>, uint64_t) {
                                    dispatch_semaphore_signal(work_done);
                                    // This should capture the *current* index for each semaphore.
                                    signaled_index = i;
                                  }];
  }

  long timed_out = dispatch_semaphore_wait(work_done, timeout);

  [listener release];
  dispatch_release(wait_notifier);
  dispatch_release(work_done);

  if (timed_out) {
    return DeadlineExceededErrorBuilder(IREE_LOC)
           << "Deadline exceeded waiting for dispatch_semaphore_t";
  }
  return signaled_index;
}

Status MetalDevice::WaitIdle(Time deadline_ns) {
  IREE_TRACE_SCOPE0("MetalDevice::WaitIdle");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalDevice::WaitIdle";
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
