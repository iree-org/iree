// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A lightweight mock HAL device for testing HAL infrastructure that operates
// on devices without needing real hardware or a full driver stack.
//
// The mock device implements the topology-related vtable methods (id,
// query_capabilities, topology_info, refine_topology_edge,
// assign_topology_info) with configurable behavior. All other vtable methods
// return IREE_STATUS_UNIMPLEMENTED or zero/NULL as appropriate.
//
// This is intended for testing code that coordinates devices (device groups,
// topology construction, multi-device scheduling) rather than code that
// executes work on devices.

#ifndef IREE_HAL_TESTING_MOCK_DEVICE_H_
#define IREE_HAL_TESTING_MOCK_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for creating a mock device.
typedef struct iree_hal_mock_device_options_t {
  // Identifier returned by iree_hal_device_id(). This acts as the driver name
  // for topology edge computation (same-driver pairs get refine calls).
  iree_string_view_t identifier;

  // Capabilities returned by iree_hal_device_query_capabilities().
  // Zero-initialized is valid (no special capabilities).
  iree_hal_device_capabilities_t capabilities;
} iree_hal_mock_device_options_t;

// Initializes |out_options| with safe defaults (empty identifier, zeroed
// capabilities).
void iree_hal_mock_device_options_initialize(
    iree_hal_mock_device_options_t* out_options);

// Creates a mock HAL device with the given |options|.
// The identifier string is copied into the device's own storage.
// |out_device| must be released by the caller (see iree_hal_device_release).
iree_status_t iree_hal_mock_device_create(
    const iree_hal_mock_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TESTING_MOCK_DEVICE_H_
