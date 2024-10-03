// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/driver.h"

#include "iree/hal/drivers/null/api.h"

// TODO(null): if it's possible to have more than one device use real IDs.
// This is a placeholder for this skeleton that just indicates the first and
// only device.
#define IREE_HAL_NULL_DEVICE_ID_DEFAULT 0

typedef struct iree_hal_null_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_null_driver_options_t options;

  // + trailing identifier string storage
} iree_hal_null_driver_t;

static const iree_hal_driver_vtable_t iree_hal_null_driver_vtable;

static iree_hal_null_driver_t* iree_hal_null_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_driver_vtable);
  return (iree_hal_null_driver_t*)base_value;
}

void iree_hal_null_driver_options_initialize(
    iree_hal_null_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));

  // TODO(null): set defaults based on compiler configuration. Flags should not
  // be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.

  iree_hal_null_device_options_initialize(&out_options->default_device_options);
}

static iree_status_t iree_hal_null_driver_options_verify(
    const iree_hal_null_driver_options_t* options) {
  // TODO(null): verify that the parameters are within expected ranges and any
  // requested features are supported.

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_null_driver_create(
    iree_string_view_t identifier,
    const iree_hal_null_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): verify options; this may be moved after any libraries are
  // loaded so the verification can use underlying implementation queries.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_null_driver_options_verify(options));

  iree_hal_null_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_null_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);

  // TODO(null): if there are any string fields then they will need to be
  // retained as well (similar to the identifier they can be tagged on to the
  // end of the driver struct).
  memcpy(&driver->options, options, sizeof(*options));

  // TODO(null): load libraries and query driver support from the system.
  // Devices need not be enumerated here if doing so is expensive; the
  // application may create drivers just to see if they are present but defer
  // device enumeration until the user requests one. Underlying implementations
  // can sometimes do bonkers static init stuff as soon as they are touched and
  // this code may want to do that on-demand instead.
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "driver not implemented");

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_null_driver_t* driver = iree_hal_null_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): if the driver loaded any libraries they should be closed here.

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_null_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  // TODO(null): query available devices and populate the output. Note that
  // unlike most IREE functions this allocates if required in order to allow
  // this to return uncached information. Uncached is preferred as it allows
  // devices that may come and go (power toggles, user visibility toggles, etc)
  // through a process lifetime to appear without needing a full restart.
  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_NULL_DEVICE_ID_DEFAULT,
          .name = iree_string_view_literal("default"),
      },
  };
  *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  return iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(device_infos, sizeof(device_infos)),
      (void**)out_device_infos);
}

static iree_status_t iree_hal_null_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_null_driver_t* driver = iree_hal_null_driver_cast(base_driver);

  // TODO(null): add useful user-level information to the string builder for the
  // given device_id. This is used by the tools in features like
  // `--dump_devices` or may be used by hosting applications for diagnostics.
  (void)driver;

  return iree_ok_status();
}

static iree_status_t iree_hal_null_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_null_driver_t* driver = iree_hal_null_driver_cast(base_driver);

  // TODO(null): use the provided params to overwrite the default options. The
  // format of the params is implementation-defined. The params strings can be
  // directly referenced if needed as the device creation is only allowed to
  // access them during the create call below.
  iree_hal_null_device_options_t options =
      driver->options.default_device_options;

  // TODO(null): implement creation by device_id; this is mostly used as
  // query_available_devices->create_device_by_id to avoid needing to expose
  // device paths (which may not always be 1:1). This skeleton only has a single
  // device so the ID is ignored.
  (void)driver;

  return iree_hal_null_device_create(driver->identifier, &options,
                                     host_allocator, out_device);
}

static iree_status_t iree_hal_null_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_null_driver_t* driver = iree_hal_null_driver_cast(base_driver);

  // TODO(null): use the provided params to overwrite the default options. The
  // format of the params is implementation-defined. The params strings can be
  // directly referenced if needed as the device creation is only allowed to
  // access them during the create call below.
  iree_hal_null_device_options_t options =
      driver->options.default_device_options;

  // TODO(null): support parsing of the device_path. Note that a single driver
  // may respond to multiple driver_name queries. Paths are
  // implementation-specific and there may be multiple formats; for example,
  // device UUID, PCI bus ID, ordinal as used by underlying APIs, etc.
  (void)driver;

  return iree_hal_null_device_create(driver->identifier, &options,
                                     host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_null_driver_vtable = {
    .destroy = iree_hal_null_driver_destroy,
    .query_available_devices = iree_hal_null_driver_query_available_devices,
    .dump_device_info = iree_hal_null_driver_dump_device_info,
    .create_device_by_id = iree_hal_null_driver_create_device_by_id,
    .create_device_by_path = iree_hal_null_driver_create_device_by_path,
};
