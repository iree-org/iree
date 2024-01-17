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

// Returns a reference to the storage of the --device= flag.
// Changes to flags invalidate the storage.
iree_string_view_list_t iree_hal_device_flag_list(void);

// Creates a single device from the --device= flag.
// Uses the |default_device| if no flags were specified.
// Fails if more than one device was specified.
iree_status_t iree_hal_create_device_from_flags(
    iree_hal_driver_registry_t* driver_registry,
    iree_string_view_t default_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

// Creates one or more devices from the repeatable --device= flag.
// Uses the |default_device| if no flags were specified.
iree_status_t iree_hal_create_devices_from_flags(
    iree_hal_driver_registry_t* driver_registry,
    iree_string_view_t default_device, iree_allocator_t host_allocator,
    iree_hal_device_list_t** out_device_list);

// Configures the |device| channel provider based on the current environment.
// Today this simply checks to see if the process is running under MPI and
// initializes that unconditionally.
//
// WARNING: not thread-safe and must only be called immediately after device
// creation.
iree_status_t iree_hal_device_set_default_channel_provider(
    iree_hal_device_t* device);

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
