// Copyright 2021 Google LLC
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

// A example of setting up the the vulkan driver.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/registration/driver_module.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Only register the vulkan HAL driver.
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default()));
  // Create the hal driver from the name.
  iree_hal_driver_t* driver = NULL;
  iree_string_view_t identifier = iree_make_cstring_view("vulkan");
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create_by_name(
      iree_hal_driver_registry_default(), identifier, iree_allocator_system(),
      &driver));
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), device));
  iree_hal_driver_release(driver);
  return iree_ok_status();
}
