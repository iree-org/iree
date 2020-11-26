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

#include "iree/hal/metal/registration/driver_module.h"

#include "absl/flags/flag.h"
#include "iree/base/flags.h"
#include "iree/base/status.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/metal/metal_driver.h"

ABSL_FLAG(bool, metal_capture, false, "Enables capturing Metal commands.");
ABSL_FLAG(
    std::string, metal_capture_to_file, "",
    "Full path to store the GPU trace file (empty means capture to Xcode)");

namespace iree {
namespace hal {
namespace metal {

StatusOr<ref_ptr<Driver>> CreateMetalDriver() {
  MetalDriverOptions options;
  options.enable_capture = absl::GetFlag(FLAGS_metal_capture);
  options.capture_file = absl::GetFlag(FLAGS_metal_capture_to_file);
  return MetalDriver::Create(options);
}

}  // namespace metal
}  // namespace hal
}  // namespace iree

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_metal_driver_module_register() {
  return ::iree::hal::DriverRegistry::shared_registry()->Register(
      "metal", ::iree::hal::metal::CreateMetalDriver);
}
