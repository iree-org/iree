// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_DEVICE_UTIL_H_
#define IREE_TOOLING_DEVICE_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns a driver registry initialized with all linked driver registered.
iree_hal_driver_registry_t* iree_hal_available_driver_registry(void);

// Returns a device URI for a default device to be used when no other devices
// are specified by the user. It may not work; it's always better to specify
// flags and tools should encourage that.
iree_string_view_t iree_hal_default_device_uri(void);

// TODO(#5724): remove this and replace with an iree_hal_device_set_t.
void iree_hal_get_devices_flag_list(iree_host_size_t* out_count,
                                    const iree_string_view_t** out_list);

// Creates a single device from the --device= flag.
// Uses the |default_device| if no flags were specified.
// Fails if more than one device was specified.
iree_status_t iree_hal_create_device_from_flags(
    iree_hal_driver_registry_t* driver_registry,
    iree_string_view_t default_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

// Equivalent to iree_hal_device_profiling_begin with options sourced from
// command line flags. No-op if profiling is not enabled.
// Must be matched with a call to iree_hal_end_profiling_from_flags.
iree_status_t iree_hal_begin_profiling_from_flags(iree_hal_device_t* device);

// Equivalent to iree_hal_device_profiling_end with options sourced from
// command line flags. No-op if profiling is not enabled.
iree_status_t iree_hal_end_profiling_from_flags(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_DEVICE_UTIL_H_
