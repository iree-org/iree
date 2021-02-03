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

#include "iree/hal/vulkan/registration/driver_module.h"

#include <inttypes.h>

#include "absl/flags/flag.h"
#include "iree/base/internal/flags.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/api.h"

#define IREE_HAL_VULKAN_1_X_DRIVER_ID 0x564C4B31u  // VLK1

ABSL_FLAG(bool, vulkan_validation_layers, true,
          "Enables standard Vulkan validation layers.");
ABSL_FLAG(bool, vulkan_debug_utils, true,
          "Enables VK_EXT_debug_utils, records markers, and logs errors.");

ABSL_FLAG(int, vulkan_default_index, 0, "Index of the default Vulkan device.");

ABSL_FLAG(bool, vulkan_force_timeline_semaphore_emulation, false,
          "Uses timeline semaphore emulation even if native support exists.");

static iree_status_t iree_hal_vulkan_create_driver_with_flags(
    iree_string_view_t identifier, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE();

  // Setup driver options from flags. We do this here as we want to enable other
  // consumers that may not be using modules/command line flags to be able to
  // set their options however they want.
  iree_hal_vulkan_driver_options_t driver_options;
  iree_hal_vulkan_driver_options_initialize(&driver_options);

// TODO(benvanik): make this a flag - it's useful for testing the same binary
// against multiple versions of Vulkan.
#if defined(IREE_PLATFORM_ANDROID)
  // TODO(#4494): let's see when we can always enable timeline semaphores.
  driver_options.api_version = VK_API_VERSION_1_1;
#else
  driver_options.api_version = VK_API_VERSION_1_2;
#endif  // IREE_PLATFORM_ANDROID

  if (absl::GetFlag(FLAGS_vulkan_validation_layers)) {
    driver_options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS;
  }
  if (absl::GetFlag(FLAGS_vulkan_debug_utils)) {
    driver_options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS;
  }

  driver_options.default_device_index =
      absl::GetFlag(FLAGS_vulkan_default_index);

  if (absl::GetFlag(FLAGS_vulkan_force_timeline_semaphore_emulation)) {
    driver_options.device_options.flags |=
        IREE_HAL_VULKAN_DEVICE_FORCE_TIMELINE_SEMAPHORE_EMULATION;
  }

  // Load the Vulkan library. This will fail if the library cannot be found or
  // does not have the expected functions.
  iree_hal_vulkan_syms_t* syms = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_syms_create_from_system_loader(allocator, &syms));

  iree_status_t status = iree_hal_vulkan_driver_create(
      identifier, &driver_options, syms, allocator, out_driver);

  iree_hal_vulkan_syms_release(syms);
  return status;
}

static iree_status_t iree_hal_vulkan_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  // NOTE: we could query supported vulkan versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      /*driver_id=*/IREE_HAL_VULKAN_1_X_DRIVER_ID,
      /*driver_name=*/iree_make_cstring_view("vulkan"),
      /*full_name=*/iree_make_cstring_view("Vulkan 1.x (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_VULKAN_1_X_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }

  // When we expose more than one driver (different vulkan versions, etc) we
  // can name them here:
  iree_string_view_t identifier = iree_make_cstring_view("vulkan");

  return iree_hal_vulkan_create_driver_with_flags(identifier, allocator,
                                                  out_driver);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      /*self=*/NULL,
      iree_hal_vulkan_driver_factory_enumerate,
      iree_hal_vulkan_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
