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

#include "iree/hal/llvmjit/llvmjit_driver.h"

#include <memory>

#include "iree/hal/device_info.h"
#include "iree/hal/host/serial/serial_scheduling_model.h"
#include "iree/hal/llvmjit/llvmjit_device.h"

namespace iree {
namespace hal {
namespace llvmjit {
namespace {

DeviceInfo GetDefaultDeviceInfo() {
  DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
  // TODO(benvanik): implement debugging/profiling features.
  // supported_features |= DeviceFeature::kDebugging;
  // supported_features |= DeviceFeature::kCoverage;
  // supported_features |= DeviceFeature::kProfiling;
  DeviceInfo device_info("llvm-ir-jit", "llvm", supported_features);
  // TODO(benvanik): device info.
  return device_info;
}

}  // namespace

LLVMJITDriver::LLVMJITDriver() : Driver("llvmjit") {}

LLVMJITDriver::~LLVMJITDriver() = default;

StatusOr<std::vector<DeviceInfo>> LLVMJITDriver::EnumerateAvailableDevices() {
  std::vector<DeviceInfo> device_infos;
  device_infos.push_back(GetDefaultDeviceInfo());
  return device_infos;
}

StatusOr<ref_ptr<Device>> LLVMJITDriver::CreateDefaultDevice() {
  return CreateDevice(0);
}

StatusOr<ref_ptr<Device>> LLVMJITDriver::CreateDevice(
    DriverDeviceID device_id) {
  auto scheduling_model = std::make_unique<host::SerialSchedulingModel>();
  return make_ref<LLVMJITDevice>(GetDefaultDeviceInfo(),
                                 std::move(scheduling_model));
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
