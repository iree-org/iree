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

#include "iree/hal/vmla/vmla_device.h"

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vmla/vmla_cache.h"

namespace iree {
namespace hal {
namespace vmla {

VMLADevice::VMLADevice(DeviceInfo device_info,
                       std::unique_ptr<host::SchedulingModel> scheduling_model,
                       iree_vm_instance_t* instance,
                       iree_vm_module_t* vmla_module)
    : HostLocalDevice(std::move(device_info), std::move(scheduling_model)),
      instance_(instance),
      vmla_module_(vmla_module) {
  iree_vm_instance_retain(instance_);
  iree_vm_module_retain(vmla_module_);
}

VMLADevice::~VMLADevice() {
  iree_vm_module_release(vmla_module_);
  iree_vm_instance_release(instance_);
}

ref_ptr<ExecutableCache> VMLADevice::CreateExecutableCache() {
  IREE_TRACE_SCOPE0("VMLADevice::CreateExecutableCache");
  return make_ref<VMLACache>(instance_, vmla_module_);
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
