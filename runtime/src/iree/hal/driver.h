// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_H_
#define IREE_HAL_DRIVER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Describes a driver providing device enumeration and creation.
// The lifetime of memory referenced by this structure (such as strings) is
// dependent on where it originated.
//
// * When using iree_hal_driver_registry_enumerate the driver info is copied
//   into memory owned by the caller.
// * When queried from a live driver with iree_hal_driver_info the memory is
//   only guaranteed to live for as long as the driver is.
// * When enumerating via factories the information may be valid only while the
//   driver registry lock is held.
typedef struct iree_hal_driver_info_t {
  IREE_API_UNSTABLE

  // Canonical name of the driver as used in command lines, documentation, etc.
  // Examples: 'metal', 'vulkan'
  iree_string_view_t driver_name;

  // Full human-readable name of the driver for display.
  // Examples: 'Vulkan 1.2 (NVIDIA)'.
  iree_string_view_t full_name;

  // TODO(benvanik): version information; useful if wanting to expose multiple
  // versions that may have completely different implementations (like vulkan
  // 1.0, 1.1, and 1.2) but allow a nice sort/selection process.
  // TODO(benvanik): triple, feature flags, etc.
} iree_hal_driver_info_t;

//===----------------------------------------------------------------------===//
// iree_hal_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_driver_t iree_hal_driver_t;

// Retains the given |driver| for the caller.
IREE_API_EXPORT void iree_hal_driver_retain(iree_hal_driver_t* driver);

// Releases the given |driver| from the caller.
IREE_API_EXPORT void iree_hal_driver_release(iree_hal_driver_t* driver);

// Queries available devices and returns them as a list.
// The provided |allocator| will be used to allocate the returned list and after
// the caller is done with it |out_device_infos| must be freed with that same
// allocator by the caller.
IREE_API_EXPORT iree_status_t iree_hal_driver_query_available_devices(
    iree_hal_driver_t* driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos);

// Appends detailed info about a device with the given |device_id| to |builder|.
// Only IDs as queried from the driver may be used.
//
// !!! WARNING !!!
// This information may contain personally identifiable information and should
// only be used for diagnostics: don't mindlessly report this in analytics.
//
// Callers should ensure the builder is at a newline before calling and drivers
// should append a trailing newline if they append any information.
//
// Returns success whether or not any information was added to the builder;
// not all drivers or devices have information.
IREE_API_EXPORT iree_status_t iree_hal_driver_dump_device_info(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder);

// Creates a device with the given |device_ordinal| as enumerated by
// iree_hal_driver_query_available_devices. The ordering of devices is driver
// dependent and may vary across driver instances and processes.
//
// Optionally takes a list of key-value parameters as defined by the device
// implementation. The parameters are only used during creation and need not
// remain live after the function returns.
IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_ordinal(
    iree_hal_driver_t* driver, iree_host_size_t device_ordinal,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Creates a device with the given |device_id| obtained from
// iree_hal_driver_query_available_devices. The |device_id| is an opaque value
// that is stable for the lifetime of the driver but not across driver instances
// or across processes. Only IDs as queried from the driver may be used. If an
// ID of `IREE_HAL_DEVICE_ID_DEFAULT` is provided the driver will select a
// device based on heuristics, flags, the environment, or a die roll.
//
// Optionally takes a list of key-value parameters as defined by the device
// implementation. The parameters are only used during creation and need not
// remain live after the function returns.
IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_id(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Creates a device with the given device path specifier.
// The |device_path| is a driver-dependent string that identifies a particular
// device or type of device. If the path is omitted the driver will select a
// device based on heuristics, flags, the environment, or a die roll.
//
// Optionally takes a list of key-value parameters as defined by the device
// implementation. The parameters are only used during creation and need not
// remain live after the function returns.
IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_path(
    iree_hal_driver_t* driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

// Creates a device with the given `driver://path?params` URI.
// The driver specifier is required in the URI but the path and params are
// optional. The path is a driver-dependent string that identifies a particular
// device or type of device. If the path is omitted the driver will select a
// device based on heuristics, flags, the environment, or a die roll.
//
// This is equivalent to performing URI parsing and calling
// iree_hal_driver_create_device_by_path.
IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_uri(
    iree_hal_driver_t* driver, iree_string_view_t device_uri,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Creates the driver-defined "default" device.
// The driver will select a device based on heuristics, flags, the environment,
// or a die roll.
IREE_API_EXPORT iree_status_t iree_hal_driver_create_default_device(
    iree_hal_driver_t* driver, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_driver_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_driver_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_driver_t* driver);

  iree_status_t(IREE_API_PTR* query_available_devices)(
      iree_hal_driver_t* driver, iree_allocator_t host_allocator,
      iree_host_size_t* out_device_info_count,
      iree_hal_device_info_t** out_device_infos);

  iree_status_t(IREE_API_PTR* dump_device_info)(iree_hal_driver_t* driver,
                                                iree_hal_device_id_t device_id,
                                                iree_string_builder_t* builder);

  iree_status_t(IREE_API_PTR* create_device_by_id)(
      iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
      iree_host_size_t param_count, const iree_string_pair_t* params,
      iree_allocator_t host_allocator, iree_hal_device_t** out_device);
  iree_status_t(IREE_API_PTR* create_device_by_path)(
      iree_hal_driver_t* driver, iree_string_view_t driver_name,
      iree_string_view_t device_path, iree_host_size_t param_count,
      const iree_string_pair_t* params, iree_allocator_t host_allocator,
      iree_hal_device_t** out_device);
} iree_hal_driver_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_driver_vtable_t);

IREE_API_EXPORT void iree_hal_driver_destroy(iree_hal_driver_t* driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_H_
