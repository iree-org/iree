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

#ifndef IREE_HAL_LOCAL_SYNC_DEVICE_H_
#define IREE_HAL_LOCAL_SYNC_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parameters configuring an iree_hal_sync_device_t.
// Must be initialized with iree_hal_sync_device_params_initialize prior to use.
typedef struct {
  int reserved;
} iree_hal_sync_device_params_t;

// Initializes |out_params| to default values.
void iree_hal_sync_device_params_initialize(
    iree_hal_sync_device_params_t* out_params);

// Creates a new synchronous local CPU device that performs execution inline
// on threads issuing submissions. |loaders| is the set of executable
// loaders that are available for loading in the device context.
iree_status_t iree_hal_sync_device_create(
    iree_string_view_t identifier, const iree_hal_sync_device_params_t* params,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_SYNC_DEVICE_H_
