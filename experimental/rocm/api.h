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

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_ROCM_API_H_
#define IREE_HAL_ROCM_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_rocm_driver_t
//===----------------------------------------------------------------------===//

// ROCM driver creation options.
typedef struct {
  // Index of the default ROCM device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_rocm_driver_options_t;

IREE_API_EXPORT void iree_hal_rocm_driver_options_initialize(
    iree_hal_rocm_driver_options_t *out_options);

// Creates a ROCM HAL driver that manage its own hipcontext.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t iree_hal_rocm_driver_create(
    iree_string_view_t identifier,
    const iree_hal_rocm_driver_options_t *options,
    iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

// TODO(thomasraoux): Support importing a CUcontext from app.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_API_H_
