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

#include "iree/hal/dawn/dawn_driver.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/dawn/dawn_device.h"
#include "iree/hal/device_info.h"
#include "third_party/dawn/src/include/dawn/dawn_proc.h"

namespace iree {
namespace hal {
namespace dawn {

namespace {

// Populates device information from the given dawn_native::Adapter.
StatusOr<DeviceInfo> PopulateDeviceInfo(dawn_native::Adapter* adapter) {
  // TODO(scotttodd): Query these for each backend or implement?
  DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
  // supported_features |= DeviceFeature::kDebugging;
  // supported_features |= DeviceFeature::kCoverage;
  // supported_features |= DeviceFeature::kProfiling;

  // TODO(scotttodd): more clever/sanitized device naming.
  std::string device_name = absl::StrCat("dawn-", adapter->GetPCIInfo().name);

  return DeviceInfo(device_name, supported_features,
                    reinterpret_cast<void*>(adapter));
}

}  // namespace

DawnDriver::DawnDriver() : Driver("dawn") {
  dawn_instance_ = absl::make_unique<dawn_native::Instance>();
}

DawnDriver::~DawnDriver() = default;

StatusOr<std::vector<DeviceInfo>> DawnDriver::EnumerateAvailableDevices() {
  IREE_TRACE_SCOPE0("DawnDriver::EnumerateAvailableDevices");

  if (dawn_backend_adapters_.empty()) {
    // Discover adapters (i.e. devices and their associated backend APIs).
    // Retain the list of adapters so pointers are valid for the lifetime of
    // this object.
    dawn_instance_->DiscoverDefaultAdapters();
    dawn_backend_adapters_ = dawn_instance_->GetAdapters();
  } else {
    // Assume that the list of adapters does not change. This is not guaranteed
    // to be true, but we also don't want to invalidate pointers by requesting
    // a new list each time. If the list of available devices would change,
    // tearing down and creating a new DawnDriver may be your best option.
  }

  // Convert to our HAL structure.
  std::vector<DeviceInfo> device_infos;
  device_infos.reserve(dawn_backend_adapters_.size());
  for (auto& adapter : dawn_backend_adapters_) {
    // TODO(scotttodd): if we fail should we just ignore the device in the list?
    ASSIGN_OR_RETURN(auto device_info, PopulateDeviceInfo(&adapter));
    device_infos.push_back(std::move(device_info));
  }
  return device_infos;
}

StatusOr<ref_ptr<Device>> DawnDriver::CreateDefaultDevice() {
  IREE_TRACE_SCOPE0("DawnDriver::CreateDefaultDevice");

  // Query available devices.
  ASSIGN_OR_RETURN(auto available_devices, EnumerateAvailableDevices());
  if (available_devices.empty()) {
    return NotFoundErrorBuilder(IREE_LOC) << "No devices are available";
  }

  // Create the first non-null device, if any.
  for (const auto& device : available_devices) {
    auto* adapter = static_cast<dawn_native::Adapter*>(device.driver_handle());
    if (adapter->GetBackendType() != dawn_native::BackendType::Null) {
      return CreateDevice(device);
    }
  }

  // Otherwise create the first null device.
  return CreateDevice(available_devices.front());
}

StatusOr<ref_ptr<Device>> DawnDriver::CreateDevice(
    const DeviceInfo& device_info) {
  IREE_TRACE_SCOPE0("DawnDriver::CreateDevice");

  auto* adapter =
      static_cast<dawn_native::Adapter*>(device_info.driver_handle());
  ::WGPUDevice c_backend_device = adapter->CreateDevice();
  if (!c_backend_device) {
    return InternalErrorBuilder(IREE_LOC) << "Failed to create a Dawn device";
  }
  DawnProcTable backend_procs = dawn_native::GetProcs();
  dawnProcSetProcs(&backend_procs);
  ::wgpu::Device backend_device = ::wgpu::Device::Acquire(c_backend_device);

  return make_ref<DawnDevice>(device_info, backend_device);
}

}  // namespace dawn
}  // namespace hal
}  // namespace iree
