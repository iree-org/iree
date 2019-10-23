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

#include "hal/dawn/dawn_device.h"

#include "absl/memory/memory.h"
#include "base/status.h"
#include "base/tracing.h"
#include "hal/command_queue.h"
#include "hal/executable_cache.h"
#include "hal/fence.h"

namespace iree {
namespace hal {
namespace dawn {

namespace {

// ExecutableCache implementation that compiles but does nothing.
// This will be replaced with something functional soon.
class NoopExecutableCache final : public ExecutableCache {
 public:
  explicit NoopExecutableCache() {}
  ~NoopExecutableCache() override = default;

  bool CanPrepareFormat(ExecutableFormat format) const override {
    return false;
  }

  StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) override {
    return UnimplementedErrorBuilder(IREE_LOC) << "PrepareExecutable NYI";
  }
};

}  // namespace

DawnDevice::DawnDevice(const DeviceInfo& device_info,
                       ::dawn::Device backend_device)
    : Device(device_info), backend_device_(backend_device) {
  IREE_TRACE_SCOPE0("DawnDevice::ctor");

  // TODO(scotttodd): construct command queues, perform other initialization

  // Log some basic device info.
  std::string backend_type_str;
  auto* adapter =
      static_cast<dawn_native::Adapter*>(device_info.driver_handle());
  switch (adapter->GetBackendType()) {
    case dawn_native::BackendType::D3D12:
      backend_type_str = "D3D12";
      break;
    case dawn_native::BackendType::Metal:
      backend_type_str = "Metal";
      break;
    case dawn_native::BackendType::Null:
      backend_type_str = "Null";
      break;
    case dawn_native::BackendType::OpenGL:
      backend_type_str = "OpenGL";
      break;
    case dawn_native::BackendType::Vulkan:
      backend_type_str = "Vulkan";
      break;
  }
  LOG(INFO) << "Created DawnDevice '" << device_info.name() << "' ("
            << backend_type_str << ")";
}

DawnDevice::~DawnDevice() = default;

std::shared_ptr<ExecutableCache> DawnDevice::CreateExecutableCache() {
  return std::make_shared<NoopExecutableCache>();
}

StatusOr<ref_ptr<CommandBuffer>> DawnDevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateCommandBuffer NYI";
}

StatusOr<ref_ptr<Event>> DawnDevice::CreateEvent() {
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateEvent NYI";
}

StatusOr<ref_ptr<BinarySemaphore>> DawnDevice::CreateBinarySemaphore(
    bool initial_value) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateBinarySemaphore");

  return UnimplementedErrorBuilder(IREE_LOC)
         << "Binary semaphores not yet implemented";
}

StatusOr<ref_ptr<TimelineSemaphore>> DawnDevice::CreateTimelineSemaphore(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateTimelineSemaphore");

  return UnimplementedErrorBuilder(IREE_LOC)
         << "Timeline semaphores not yet implemented";
}

StatusOr<ref_ptr<Fence>> DawnDevice::CreateFence(uint64_t initial_value) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateFence");

  return UnimplementedErrorBuilder(IREE_LOC) << "CreateFence NYI";
}

Status DawnDevice::WaitAllFences(absl::Span<const FenceValue> fences,
                                 absl::Time deadline) {
  IREE_TRACE_SCOPE0("DawnDevice::WaitAllFences");

  return UnimplementedErrorBuilder(IREE_LOC) << "WaitAllFences NYI";
}

StatusOr<int> DawnDevice::WaitAnyFence(absl::Span<const FenceValue> fences,
                                       absl::Time deadline) {
  IREE_TRACE_SCOPE0("DawnDevice::WaitAnyFence");

  return UnimplementedErrorBuilder(IREE_LOC) << "WaitAnyFence NYI";
}

Status DawnDevice::WaitIdle(absl::Time deadline) {
  return UnimplementedErrorBuilder(IREE_LOC) << "WaitIdle";
}

}  // namespace dawn
}  // namespace hal
}  // namespace iree
