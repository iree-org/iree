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

#include "iree/hal/vmla/vmla_driver.h"

#include <memory>

#include "iree/base/api_util.h"
#include "iree/base/tracing.h"
#include "iree/hal/device_info.h"
#include "iree/hal/host/serial_scheduling_model.h"
#include "iree/hal/vmla/vmla_device.h"
#include "iree/hal/vmla/vmla_module.h"
#include "iree/vm/module.h"

namespace iree {
namespace hal {
namespace vmla {

namespace {

DeviceInfo GetDefaultDeviceInfo() {
  DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
  // TODO(benvanik): implement debugging/profiling features.
  // supported_features |= DeviceFeature::kDebugging;
  // supported_features |= DeviceFeature::kCoverage;
  // supported_features |= DeviceFeature::kProfiling;
  DeviceInfo device_info("vmla", "vmla", supported_features);
  // TODO(benvanik): device info.
  return device_info;
}

}  // namespace

// static
StatusOr<ref_ptr<Driver>> VMLADriver::Create() {
  IREE_TRACE_SCOPE0("VMLADriver::Create");

  // NOTE: we could use our own allocator here to hide these from any default
  // tracing we have.
  iree_vm_instance_t* instance = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance), IREE_LOC));

  // TODO(benvanik): move to instance-based registration.
  RETURN_IF_ERROR(ModuleRegisterTypes()) << "VMLA type registration failed";

  iree_vm_module_t* vmla_module = nullptr;
  RETURN_IF_ERROR(ModuleCreate(IREE_ALLOCATOR_SYSTEM, &vmla_module))
      << "VMLA shared module creation failed";

  return make_ref<VMLADriver>(instance, vmla_module);
}

VMLADriver::VMLADriver(iree_vm_instance_t* instance,
                       iree_vm_module_t* vmla_module)
    : Driver("vmla"), instance_(instance), vmla_module_(vmla_module) {}

VMLADriver::~VMLADriver() {
  IREE_TRACE_SCOPE0("VMLADriver::dtor");
  iree_vm_module_release(vmla_module_);
  iree_vm_instance_release(instance_);
}

StatusOr<std::vector<DeviceInfo>> VMLADriver::EnumerateAvailableDevices() {
  std::vector<DeviceInfo> device_infos;
  device_infos.push_back(GetDefaultDeviceInfo());
  return device_infos;
}

StatusOr<ref_ptr<Device>> VMLADriver::CreateDefaultDevice() {
  return CreateDevice(0);
}

StatusOr<ref_ptr<Device>> VMLADriver::CreateDevice(DriverDeviceID device_id) {
  auto scheduling_model = std::make_unique<SerialSchedulingModel>();
  auto device =
      make_ref<VMLADevice>(GetDefaultDeviceInfo(), std::move(scheduling_model),
                           instance_, vmla_module_);
  return device;
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
