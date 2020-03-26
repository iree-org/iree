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

#ifndef IREE_HAL_LLVMJIT_LLVMJIT_DRIVER_H_
#define IREE_HAL_LLVMJIT_LLVMJIT_DRIVER_H_

#include "iree/hal/driver.h"

namespace iree {
namespace hal {
namespace llvmjit {

class LLVMJITDriver final : public Driver {
 public:
  LLVMJITDriver();
  ~LLVMJITDriver() override;

  StatusOr<std::vector<DeviceInfo>> EnumerateAvailableDevices() override;

  StatusOr<ref_ptr<Device>> CreateDefaultDevice() override;

  StatusOr<ref_ptr<Device>> CreateDevice(DriverDeviceID device_id) override;
};

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_LLVMJIT_LLVMJIT_DRIVER_H_
