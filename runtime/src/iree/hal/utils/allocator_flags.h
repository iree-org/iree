// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_ALLOCATOR_FLAGS_H_
#define IREE_HAL_UTILS_ALLOCATOR_FLAGS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// WARNING: including this file will pull in flags code as well as all allocator
// implementations. Only use this if you need the flag and otherwise prefer
// directly instantiating the allocators you want with their structured options
// instead of strings.

// Parses a single flag value and wraps |base_allocator|.
// Flag values are specifications and may include configuration values.
// Examples:
//   some_allocator
//   some_allocator:key=value
//   some_allocator:key=value,key=value
iree_status_t iree_hal_configure_allocator_from_spec(
    iree_string_view_t spec, iree_hal_device_t* device,
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_t** out_wrapped_allocator);

// Configures the |device| allocator based on the --device_allocator= flag.
// This will wrap the underlying device allocator in zero or more configurable
// allocator shims.
//
// WARNING: not thread-safe and must only be called immediately after device
// creation.
iree_status_t iree_hal_configure_allocator_from_flags(
    iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_ALLOCATOR_FLAGS_H_
