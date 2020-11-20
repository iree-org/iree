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

#include <inttypes.h>

#include "absl/flags/flag.h"
#include "iree/base/flags.h"
#include "iree/hal/metal/metal_driver.h"

ABSL_FLAG(bool, metal_capture, false, "Enables capturing Metal commands.");
ABSL_FLAG(
    std::string, metal_capture_to_file, "",
    "Full path to store the GPU trace file (empty means capture to Xcode)");

#define IREE_HAL_METAL_DRIVER_ID_1 0x4D544C31u  // MTL1

static iree_status_t iree_hal_metal_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  // NOTE: we could query supported metal versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      /*driver_id=*/IREE_HAL_METAL_DRIVER_ID,
      /*driver_name=*/iree_make_cstring_view("metal"),
      /*full_name=*/iree_make_cstring_view("Apple Metal GPU"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_METAL_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIXPTR
                            " is provided by this factory",
                            driver_id);
  }
  iree::hal::metal::MetalDriverOptions options;
  options.enable_capture = absl::GetFlag(FLAGS_metal_capture);
  options.capture_file = absl::GetFlag(FLAGS_metal_capture_to_file);
  IREE_ASSIGN_OR_RETURN(auto driver,
                        iree::hal::metal::MetalDriver::Create(options));
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_metal_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      /*self=*/NULL,
      iree_hal_metal_driver_factory_enumerate,
      iree_hal_metal_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
