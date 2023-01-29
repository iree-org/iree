// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_EXPERIMENTAL_METAL_API_H_
#define IREE_EXPERIMENTAL_METAL_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_metal_driver_t
//===----------------------------------------------------------------------===//

// Creates a Metal HAL driver, from which devices can be created with the given
// |device_params|.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_metal_driver_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_METAL_API_H_
