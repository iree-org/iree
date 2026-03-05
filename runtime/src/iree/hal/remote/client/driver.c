// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/driver.h"

#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/device.h"
#include "iree/net/transport_factory.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_driver_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_client_driver_options_initialize(
    iree_hal_remote_client_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->transport_factory = NULL;
  iree_hal_remote_client_device_options_initialize(
      &out_options->default_device_options);
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_options_parse(
    iree_hal_remote_client_driver_options_t* options,
    iree_string_pair_list_t params) {
  // Pass all parameters through to device options.
  // Transport name is set by the driver factory from the driver name suffix,
  // not from parsed parameters.
  return iree_hal_remote_client_device_options_parse(
      &options->default_device_options, params);
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_client_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;

  // Default options for devices created from this driver.
  iree_hal_remote_client_driver_options_t options;

  // + trailing identifier string storage
  // + trailing server_address string storage (from default_device_options)
} iree_hal_remote_client_driver_t;

static const iree_hal_driver_vtable_t iree_hal_remote_client_driver_vtable;

static iree_hal_remote_client_driver_t* iree_hal_remote_client_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_driver_vtable);
  return (iree_hal_remote_client_driver_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_driver = NULL;

  // Calculate layout with trailing storage for strings.
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t server_address_offset = 0;
  iree_hal_remote_client_driver_t* driver = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(*driver), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1,
                                        &identifier_offset),
              IREE_STRUCT_FIELD_ALIGNED(
                  options->default_device_options.server_address.size, char, 1,
                  &server_address_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_remote_client_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;

  // Copy options and strings to trailing storage.
  driver->options = *options;
  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char*)driver + identifier_offset);
  // Retain the factory for this driver's lifetime. Device options share the
  // same factory pointer — devices retain their own reference at creation.
  iree_net_transport_factory_retain(options->transport_factory);
  driver->options.default_device_options.transport_factory =
      options->transport_factory;
  iree_string_view_append_to_buffer(
      options->default_device_options.server_address,
      &driver->options.default_device_options.server_address,
      (char*)driver + server_address_offset);

  *out_driver = (iree_hal_driver_t*)driver;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_remote_client_driver_destroy(
    iree_hal_driver_t* base_driver) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_transport_factory_release(driver->options.transport_factory);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_client_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  // Remote driver cannot enumerate devices without connecting.
  // Devices must be created explicitly with a server address.
  *out_device_info_count = 0;
  *out_device_infos = NULL;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  // No device info available without connecting to a server.
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);

  // Parse device-specific options from params.
  iree_hal_remote_client_device_options_t options =
      driver->options.default_device_options;
  iree_string_pair_list_t params_list = {
      .count = param_count,
      .pairs = params,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_device_options_parse(&options, params_list));

  // Device ID is ignored for remote driver; server address determines device.
  return iree_hal_remote_client_device_create(
      driver->identifier, &options, create_params, host_allocator, out_device);
}

static iree_status_t iree_hal_remote_client_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);

  // Parse device-specific options from params.
  iree_hal_remote_client_device_options_t options =
      driver->options.default_device_options;
  iree_string_pair_list_t params_list = {
      .count = param_count,
      .pairs = params,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_device_options_parse(&options, params_list));

  // Use device_path as the server address if provided, otherwise use default.
  if (!iree_string_view_is_empty(device_path)) {
    options.server_address = device_path;
  }

  return iree_hal_remote_client_device_create(
      driver->identifier, &options, create_params, host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_remote_client_driver_vtable = {
    .destroy = iree_hal_remote_client_driver_destroy,
    .query_available_devices =
        iree_hal_remote_client_driver_query_available_devices,
    .dump_device_info = iree_hal_remote_client_driver_dump_device_info,
    .create_device_by_id = iree_hal_remote_client_driver_create_device_by_id,
    .create_device_by_path =
        iree_hal_remote_client_driver_create_device_by_path,
};
