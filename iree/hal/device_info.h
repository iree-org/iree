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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "iree/hal/api.h"

namespace iree {
namespace hal {

// TODO(benvanik): device info (caps, physical mappings, etc).
class DeviceInfo {
 public:
  DeviceInfo(std::string id, std::string name,
             iree_hal_device_feature_t supported_features,
             iree_hal_device_id_t device_id = 0)
      : id_(std::move(id)),
        name_(std::move(name)),
        supported_features_(supported_features),
        device_id_(device_id) {}

  // Machine-friendly device identifier used to match the device against
  // compiler-generated patterns. This should be consistent with the device IDs
  // emitted by the compiler. For example: `vulkan-v1.1-spec`.
  const std::string& id() const { return id_; }

  // Human-friendly device name.
  const std::string& name() const { return name_; }

  // Features supported by the device.
  iree_hal_device_feature_t supported_features() const {
    return supported_features_;
  }

  // Opaque handle used by drivers to correlate this device with their internal
  // listing. This handle will not be valid across driver instances or outside
  // of the current process.
  iree_hal_device_id_t device_id() const { return device_id_; }

  // Returns a debug string describing the device information.
  std::string DebugString() const {
    std::string features = FormatBitfieldValue(
        supported_features_,
        {
            {IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING, "kDebugging"},
            {IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE, "kCoverage"},
            {IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING, "kProfiling"},
        });

    return absl::StrCat("[DeviceInfo]",                              //
                        "\n  Name: ", name_,                         //
                        "\n  Supported features: [", features, "]",  //
                        "\n  Device ID: ", device_id_);
  }

 private:
  const std::string id_;
  const std::string name_;
  const iree_hal_device_feature_t supported_features_;
  iree_hal_device_id_t device_id_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DEVICE_INFO_H_
