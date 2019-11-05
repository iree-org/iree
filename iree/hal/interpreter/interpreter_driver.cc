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

#include "iree/hal/interpreter/interpreter_driver.h"

#include <memory>

#include "iree/hal/device_info.h"
#include "iree/hal/interpreter/interpreter_device.h"

namespace iree {
namespace hal {

namespace {

DeviceInfo GetDefaultDeviceInfo() {
  DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
  // TODO(benvanik): implement debugging/profiling features.
  // supported_features |= DeviceFeature::kDebugging;
  // supported_features |= DeviceFeature::kCoverage;
  // supported_features |= DeviceFeature::kProfiling;
  DeviceInfo device_info("interpreter", supported_features);
  // TODO(benvanik): device info.
  return device_info;
}

}  // namespace

InterpreterDriver::InterpreterDriver() : Driver("interpreter") {}

InterpreterDriver::~InterpreterDriver() = default;

StatusOr<std::vector<DeviceInfo>>
InterpreterDriver::EnumerateAvailableDevices() {
  std::vector<DeviceInfo> device_infos;
  device_infos.push_back(GetDefaultDeviceInfo());
  return device_infos;
}

StatusOr<ref_ptr<Device>> InterpreterDriver::CreateDefaultDevice() {
  return CreateDevice(GetDefaultDeviceInfo());
}

StatusOr<ref_ptr<Device>> InterpreterDriver::CreateDevice(
    const DeviceInfo& device_info) {
  auto device = make_ref<InterpreterDevice>(device_info);
  return device;
}

}  // namespace hal
}  // namespace iree
