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

#include <memory>

#include "base/init.h"
#include "base/status.h"
#include "base/tracing.h"
#include "hal/dawn/dawn_driver.h"
#include "hal/driver_registry.h"

namespace iree {
namespace hal {
namespace dawn {
namespace {

StatusOr<std::shared_ptr<Driver>> CreateDawnDriver() {
  return std::make_shared<DawnDriver>();
}

}  // namespace
}  // namespace dawn
}  // namespace hal
}  // namespace iree

IREE_REGISTER_MODULE_INITIALIZER(iree_hal_dawn_driver, {
  QCHECK_OK(::iree::hal::DriverRegistry::shared_registry()->Register(
      "dawn", ::iree::hal::dawn::CreateDawnDriver));
});
IREE_REGISTER_MODULE_INITIALIZER_SEQUENCE(iree_hal, iree_hal_dawn_driver);
