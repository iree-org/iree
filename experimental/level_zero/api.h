// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_LEVEL_ZERO_API_H_
#define IREE_HAL_LEVEL_ZERO_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_level_zero_driver_t
//===----------------------------------------------------------------------===//

// LEVEL_ZERO driver creation options.
typedef struct iree_hal_level_zero_driver_options_t {
  // Index of the default LEVEL_ZERO device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_level_zero_driver_options_t;

IREE_API_EXPORT void iree_hal_level_zero_driver_options_initialize(
    iree_hal_level_zero_driver_options_t *out_options);

// Creates a LEVEL_ZERO HAL driver that manage its own level zero context.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t iree_hal_level_zero_driver_create(
    iree_string_view_t identifier,
    const iree_hal_level_zero_driver_options_t *options,
    iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

// TODO(thomasraoux): Support importing a CUcontext from app.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_API_H_
