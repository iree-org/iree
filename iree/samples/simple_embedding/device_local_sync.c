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

// A example of setting up the dylib-sync driver.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/legacy_library_loader.h"
#include "iree/hal/local/sync_driver.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Set paramters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_hal_executable_loader_t* dylib_loader = NULL;
  // TODO(marbre): Use embedded instead of legacy loader.
  IREE_RETURN_IF_ERROR(iree_hal_legacy_library_loader_create(
      iree_allocator_system(), &dylib_loader));
  iree_hal_executable_loader_t* loaders[1] = {dylib_loader};

  iree_string_view_t identifier = iree_make_cstring_view("dylib");

  // Create the synchronous device and release the loader afterwards.
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_create(identifier, &params, IREE_ARRAYSIZE(loaders),
                                  loaders, iree_allocator_system(), device));
  iree_hal_executable_loader_release(dylib_loader);
  return iree_ok_status();
}
