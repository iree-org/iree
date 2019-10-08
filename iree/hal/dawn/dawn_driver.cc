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

#include <memory>

#include "iree/base/status.h"
#include "iree/hal/dawn/dawn_device.h"
#include "iree/hal/device_info.h"

namespace iree {
namespace hal {
namespace dawn {

namespace {

DeviceInfo GetDefaultDeviceInfo() {
  DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
  // TODO(scotttodd): implement debugging/profiling features.
  // supported_features |= DeviceFeature::kDebugging;
  // supported_features |= DeviceFeature::kCoverage;
  // supported_features |= DeviceFeature::kProfiling;
  DeviceInfo device_info("dawn", supported_features);
  // TODO(scotttodd): other device info.
  return device_info;
}

}  // namespace

DawnDriver::DawnDriver() : Driver("dawn") {}

DawnDriver::~DawnDriver() = default;

StatusOr<std::vector<DeviceInfo>> DawnDriver::EnumerateAvailableDevices() {
  std::vector<DeviceInfo> device_infos;
  device_infos.push_back(GetDefaultDeviceInfo());
  return device_infos;
}

StatusOr<std::shared_ptr<Device>> DawnDriver::CreateDefaultDevice() {
  return CreateDevice(GetDefaultDeviceInfo());
}

StatusOr<std::shared_ptr<Device>> DawnDriver::CreateDevice(
    const DeviceInfo& device_info) {
  auto device = std::make_shared<DawnDevice>(device_info);
  return device;
}

}  // namespace dawn
}  // namespace hal
}  // namespace iree
