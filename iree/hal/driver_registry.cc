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

#include "iree/hal/driver_registry.h"

#include "iree/base/status.h"

namespace iree {
namespace hal {

// static
DriverRegistry* DriverRegistry::shared_registry() {
  static auto* singleton = new DriverRegistry();
  return singleton;
}

DriverRegistry::DriverRegistry() = default;

DriverRegistry::~DriverRegistry() = default;

Status DriverRegistry::Register(std::string driver_name, FactoryFn factory_fn) {
  absl::MutexLock lock(&mutex_);
  for (const auto& pair : driver_factory_fns_) {
    if (pair.first == driver_name) {
      return AlreadyExistsErrorBuilder(IREE_LOC)
             << "Driver already registered: " << driver_name;
    }
  }
  driver_factory_fns_.emplace_back(driver_name, std::move(factory_fn));
  return OkStatus();
}

bool DriverRegistry::HasDriver(absl::string_view driver_name) const {
  absl::MutexLock lock(&mutex_);
  for (const auto& pair : driver_factory_fns_) {
    if (pair.first == driver_name) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> DriverRegistry::EnumerateAvailableDrivers() const {
  absl::MutexLock lock(&mutex_);
  std::vector<std::string> driver_names;
  driver_names.reserve(driver_factory_fns_.size());
  for (const auto& pair : driver_factory_fns_) {
    driver_names.push_back(pair.first);
  }
  return driver_names;
}

StatusOr<std::shared_ptr<Driver>> DriverRegistry::Create(
    absl::string_view driver_name) const {
  FactoryFn factory_fn;
  {
    absl::MutexLock lock(&mutex_);
    for (const auto& pair : driver_factory_fns_) {
      if (pair.first == driver_name) {
        factory_fn = pair.second;
        break;
      }
    }
    if (!factory_fn) {
      return NotFoundErrorBuilder(IREE_LOC)
             << "Driver " << driver_name << " not found";
    }
  }
  return factory_fn();
}

}  // namespace hal
}  // namespace iree

IREE_REGISTER_MODULE_INITIALIZER(
    iree_hal, ::iree::hal::DriverRegistry::shared_registry());
