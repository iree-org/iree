// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
