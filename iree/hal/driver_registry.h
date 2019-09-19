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

#ifndef IREE_HAL_DRIVER_REGISTRY_H_
#define IREE_HAL_DRIVER_REGISTRY_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/hal/driver.h"

namespace iree {
namespace hal {

// Driver registry and factory.
// Factory functions for available drivers are registered with a given name and
// can be invoked with a call to Create. The configuration of the drivers is
// generally contained within the factory function and consumers of the drivers
// don't need to fiddle with things.
//
// This is used for dynamic *safe* link-time driver module registration.
// Roughly: driver_registry provides the shared registry and a way to create
// drivers and *_driver_module.cc files register drivers when linked in.
// Remember to alwayslink=1 on cc_libraries providing modules.
//
// If link-time driver registration is not desired (or possible) it's also
// possible to explicitly register drivers via this registry. This is useful
// when programmatically enabling drivers.
//
// Thread-safe.
class DriverRegistry final {
 public:
  using FactoryFn = std::function<StatusOr<std::shared_ptr<Driver>>()>;

  // The shared driver registry singleton that modules use when linked in.
  static DriverRegistry* shared_registry();

  DriverRegistry();
  ~DriverRegistry();

  // Registers a driver and its factory function.
  // The function will be called to create a new driver whenever it is requested
  // via Create.
  Status Register(std::string driver_name, FactoryFn factory_fn);

  // Returns true if there is a driver registered with the given name.
  bool HasDriver(absl::string_view driver_name) const;

  // Returns a list of registered drivers.
  std::vector<std::string> EnumerateAvailableDrivers() const;

  // TODO(benvanik): flags for enabling debug validation/control/etc.
  // Creates a driver by name.
  StatusOr<std::shared_ptr<Driver>> Create(absl::string_view driver_name) const;

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::pair<std::string, FactoryFn>> driver_factory_fns_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace hal
}  // namespace iree

IREE_DECLARE_MODULE_INITIALIZER(iree_hal);
IREE_REQUIRE_MODULE_LINKED(iree_hal);

#endif  // IREE_HAL_DRIVER_REGISTRY_H_
