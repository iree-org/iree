// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_DRIVERS_HEXAGON_API_H_
#define IREE_HAL_DRIVERS_HEXAGON_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_t
//===----------------------------------------------------------------------===//

// Hexagon DSP domain ID as defined by the Hexagon SDK (e.g. CDSP_DOMAIN_ID).
typedef int iree_hal_hexagon_domain_id_t;

// Parameters configuring an iree_hal_hexagon_device_t.
// Must be initialized with iree_hal_hexagon_device_options_initialize prior to
// use.
typedef struct iree_hal_hexagon_device_options_t {
  // Prints diagnostic information (e.g. the DSP session URI) to stderr.
  int verbose;
  // Prints notifications about DSP process domain status changes to stderr.
  int dsp_status_notify;
} iree_hal_hexagon_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_hexagon_device_options_initialize(
    iree_hal_hexagon_device_options_t* out_options);

// Creates a Qualcomm Hexagon HAL device on the DSP with the given |domain_id|
// and |options|. This opens a FastRPC session on the DSP that loads the
// device-side library (libhexagon_dsp_skel.so).
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_hexagon_device_create(
    iree_string_view_t identifier, iree_hal_hexagon_domain_id_t domain_id,
    const iree_hal_hexagon_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_hexagon_driver_t.
// Must be initialized with iree_hal_hexagon_driver_options_initialize prior to
// use.
typedef struct iree_hal_hexagon_driver_options_t {
  // Default device options when none are provided during device creation.
  iree_hal_hexagon_device_options_t default_device_options;
} iree_hal_hexagon_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_hexagon_driver_options_initialize(
    iree_hal_hexagon_driver_options_t* out_options);

// Creates a Qualcomm Hexagon HAL driver with the given |options|, from which
// Hexagon devices can be enumerated and created with specific parameters.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_hexagon_driver_create(
    iree_string_view_t identifier,
    const iree_hal_hexagon_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HEXAGON_API_H_
