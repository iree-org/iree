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

#ifndef IREE_HAL_DEVICE_INFO_H_
#define IREE_HAL_DEVICE_INFO_H_

#include <cstdint>
#include <string>
#include <utility>

#include "base/bitfield.h"

namespace iree {
namespace hal {

// Describes features supported by the device.
// These flags indicate the availability of features that may be enabled at the
// request of the calling application. Note that certain features may disable
// runtime optimizations or require compilation flags to ensure the required
// metadata is present in executables.
enum class DeviceFeature : uint32_t {
  kNone = 0,

  // Device supports executable debugging.
  // When present executables *may* be compiled with
  // ExecutableCachingMode::kEnableDebugging and will have usable debugging
  // related methods. Note that if the input executables do not have embedded
  // debugging information they still may not be able to perform disassembly or
  // fine-grained breakpoint insertion.
  kDebugging = 1 << 0,

  // Device supports executable coverage information.
  // When present executables *may* be compiled with
  // ExecutableCachingMode::kEnableCoverage and will produce coverage buffers
  // during dispatch. Note that input executables must have partial embedded
  // debug information to allow mapping back to source offsets.
  kCoverage = 1 << 1,

  // Device supports executable and command queue profiling.
  // When present executables *may* be compiled with
  // ExecutableCachingMode::kEnableProfiling and will produce profiling buffers
  // during dispatch. Note that input executables must have partial embedded
  // debug information to allow mapping back to source offsets.
  kProfiling = 1 << 2,
};
IREE_BITFIELD(DeviceFeature);
using DeviceFeatureBitfield = DeviceFeature;

// TODO(benvanik): device info (caps, physical mappings, etc).
class DeviceInfo {
 public:
  DeviceInfo(std::string name, DeviceFeatureBitfield supported_features,
             void* driver_handle = nullptr)
      : name_(std::move(name)),
        supported_features_(supported_features),
        driver_handle_(driver_handle) {}

  const std::string& name() const { return name_; }

  // Features supported by the device.
  DeviceFeatureBitfield supported_features() const {
    return supported_features_;
  }

  // Opaque handle used by drivers to correlate this device with their internal
  // listing. This handle will not be valid across driver instances or outside
  // of the current process.
  void* driver_handle() const { return driver_handle_; }

 private:
  const std::string name_;
  const DeviceFeatureBitfield supported_features_;
  void* driver_handle_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DEVICE_INFO_H_
