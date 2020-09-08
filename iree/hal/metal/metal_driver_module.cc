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

#include <memory>

#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/metal/metal_driver.h"

namespace iree {
namespace hal {
namespace metal {
namespace {

StatusOr<ref_ptr<Driver>> CreateMetalDriver() { return MetalDriver::Create(); }

}  // namespace
}  // namespace metal
}  // namespace hal
}  // namespace iree

IREE_REGISTER_MODULE_INITIALIZER(iree_hal_metal_driver, {
  IREE_QCHECK_OK(::iree::hal::DriverRegistry::shared_registry()->Register(
      "metal", ::iree::hal::metal::CreateMetalDriver));
});
IREE_REGISTER_MODULE_INITIALIZER_SEQUENCE(iree_hal, iree_hal_metal_driver);
