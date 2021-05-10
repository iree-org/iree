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

#ifndef IREE_HAL_DRIVERS_INIT_H_
#define IREE_HAL_DRIVERS_INIT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers all drivers that were linked into the current binary based on the
// build configuration. Note that there may be no drivers available.
//
// This only registers IREE core drivers (those under iree/hal/). User-provided
// drivers must be directly registered or directly created, though a user could
// create their own user_register_all_available_drivers() that calls this as
// well as registering their drivers.
IREE_API_EXPORT iree_status_t
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_INIT_H_
