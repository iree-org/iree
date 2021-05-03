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

#include "iree/hal/cuda/registration/driver_module.h"

#include <inttypes.h>

#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/api.h"

#define IREE_HAL_CUDA_DRIVER_ID 0x43554441u  // CUDA

static iree_status_t iree_hal_cuda_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  // NOTE: we could query supported cuda versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_id = IREE_HAL_CUDA_DRIVER_ID,
      .driver_name = iree_string_view_literal("cuda"),
      .full_name = iree_string_view_literal("CUDA (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  if (driver_id != IREE_HAL_CUDA_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  // When we expose more than one driver (different cuda versions, etc) we
  // can name them here:
  iree_string_view_t identifier = iree_make_cstring_view("cuda");

  iree_hal_cuda_driver_options_t driver_options;
  iree_hal_cuda_driver_options_initialize(&driver_options);
  iree_status_t status = iree_hal_cuda_driver_create(
      identifier, &driver_options, allocator, out_driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_cuda_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_cuda_driver_factory_enumerate,
      .try_create = iree_hal_cuda_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
