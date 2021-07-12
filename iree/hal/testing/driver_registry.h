// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TESTING_DRIVER_REGISTRY_H_
#define IREE_HAL_TESTING_DRIVER_REGISTRY_H_

#include <mutex>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"

namespace iree {
namespace hal {
namespace testing {

// Returns an unordered list of all available drivers in the binary.
// May return empty if there are no available drivers.
static std::vector<std::string> EnumerateAvailableDrivers() {
  // TODO(benvanik): replace with a wrapper fn that uses iree_call_once.
  static std::once_flag register_once;
  std::call_once(register_once, [] {
    IREE_CHECK_OK(iree_hal_register_all_available_drivers(
        iree_hal_driver_registry_default()));
  });
  iree_hal_driver_info_t* driver_infos = NULL;
  iree_host_size_t driver_info_count = 0;
  IREE_CHECK_OK(iree_hal_driver_registry_enumerate(
      iree_hal_driver_registry_default(), iree_allocator_system(),
      &driver_infos, &driver_info_count));
  std::vector<std::string> driver_names(driver_info_count);
  for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
    driver_names[i] = std::string(driver_infos[i].driver_name.data,
                                  driver_infos[i].driver_name.size);
  }
  iree_allocator_free(iree_allocator_system(), driver_infos);
  return driver_names;
}

// Filters out a driver from the given |drivers| set with the given name.
static std::vector<std::string> RemoveDriverByName(
    std::vector<std::string> drivers, const char* filter_name) {
  std::vector<std::string> result;
  result.reserve(drivers.size());
  for (auto& driver : drivers) {
    if (driver != filter_name) result.push_back(driver);
  }
  return result;
}

}  // namespace testing
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_TESTING_DRIVER_REGISTRY_H_
