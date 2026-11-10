// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hexagon/driver.h"

#include <string.h>

#include "iree/hal/drivers/hexagon/api.h"
#include "iree/hal/drivers/hexagon/device.h"
#include "iree/hal/drivers/hexagon/utils.h"

// Hexagon SDK headers.
#include "AEEStdErr.h"
#include "remote.h"

// Domain of the device created when no device is specified.
#define IREE_HAL_HEXAGON_DEFAULT_DOMAIN_ID \
  ((iree_hal_hexagon_domain_id_t)CDSP_DOMAIN_ID)

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_driver_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_hexagon_driver_options_initialize(
    iree_hal_hexagon_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  iree_hal_hexagon_device_options_initialize(
      &out_options->default_device_options);
}

static iree_status_t iree_hal_hexagon_driver_options_verify(
    const iree_hal_hexagon_driver_options_t* options) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_hexagon_driver_options_t options;

  // + trailing identifier string storage
} iree_hal_hexagon_driver_t;

static const iree_hal_driver_vtable_t iree_hal_hexagon_driver_vtable;

static iree_hal_hexagon_driver_t* iree_hal_hexagon_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_driver_vtable);
  return (iree_hal_hexagon_driver_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_hexagon_driver_create(
    iree_string_view_t identifier,
    const iree_hal_hexagon_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_driver = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hexagon_driver_options_verify(options));

  iree_hal_hexagon_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_hexagon_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  memcpy(&driver->options, options, sizeof(*options));

  *out_driver = (iree_hal_driver_t*)driver;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hexagon_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_hexagon_driver_t* driver = iree_hal_hexagon_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hexagon_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  *out_device_info_count = 0;
  *out_device_infos = NULL;

  // Report a device for each DSP domain that can run unsigned modules. The
  // device library is not signed, so domains supporting only signed modules
  // are not usable. Querying is not cached so that domains coming and going
  // during the lifetime of the process are reported correctly.
  static const iree_hal_hexagon_domain_id_t all_domain_ids[] = {
      ADSP_DOMAIN_ID, MDSP_DOMAIN_ID,  SDSP_DOMAIN_ID,
      CDSP_DOMAIN_ID, CDSP1_DOMAIN_ID,
  };
  iree_hal_hexagon_domain_id_t domain_ids[IREE_ARRAYSIZE(all_domain_ids)];
  iree_host_size_t domain_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(all_domain_ids); ++i) {
    struct remote_dsp_capability capability = {
        .domain = (uint32_t)all_domain_ids[i],
        .attribute_ID = UNSIGNED_PD_SUPPORT,
        .capability = 0,
    };
    int dsp_err = remote_handle_control(DSPRPC_GET_DSP_INFO, &capability,
                                        sizeof(capability));
    if (dsp_err == AEE_SUCCESS && capability.capability != 0) {
      domain_ids[domain_count++] = all_domain_ids[i];
    }
  }
  if (domain_count == 0) return iree_ok_status();

  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, domain_count * sizeof(*device_infos),
      (void**)&device_infos));
  for (iree_host_size_t i = 0; i < domain_count; ++i) {
    iree_hal_hexagon_fill_device_info(domain_ids[i], &device_infos[i]);
  }
  *out_device_info_count = domain_count;
  *out_device_infos = device_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_hexagon_domain_id_t domain_id =
      iree_hal_hexagon_domain_id_from_device_id(device_id);
  const char* domain_name = "unknown";
  iree_hal_hexagon_get_domain_name(domain_id, &domain_name);
  return iree_string_builder_append_format(
      builder, "\n- DSP domain: %s (ID %d)", domain_name, domain_id);
}

// Parses the device parameters from the device URI into |options|.
// See README.md for the supported parameters.
static iree_status_t iree_hal_hexagon_device_options_parse(
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_hal_hexagon_device_options_t* options) {
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    const iree_string_pair_t* param = &params[i];
    if (iree_string_view_equal_case(param->key, IREE_SV("verbose"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hexagon_parse_bool_from_string(
          param->value, &options->verbose));
    } else if (iree_string_view_equal_case(param->key,
                                           IREE_SV("dsp_status_notify"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hexagon_parse_bool_from_string(
          param->value, &options->dsp_status_notify));
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown hexagon device parameter '%.*s'",
                              (int)param->key.size, param->key.data);
    }
  }
  return iree_ok_status();
}

// Creates a device on the DSP domain with |domain_id|.
static iree_status_t iree_hal_hexagon_driver_create_device(
    iree_hal_hexagon_driver_t* driver, iree_hal_hexagon_domain_id_t domain_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_hexagon_device_options_t options =
      driver->options.default_device_options;
  IREE_RETURN_IF_ERROR(
      iree_hal_hexagon_device_options_parse(param_count, params, &options));
  return iree_hal_hexagon_device_create(driver->identifier, domain_id, &options,
                                        create_params, host_allocator,
                                        out_device);
}

static iree_status_t iree_hal_hexagon_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_hexagon_driver_t* driver = iree_hal_hexagon_driver_cast(base_driver);
  iree_hal_hexagon_domain_id_t domain_id =
      device_id == IREE_HAL_DEVICE_ID_DEFAULT
          ? IREE_HAL_HEXAGON_DEFAULT_DOMAIN_ID
          : iree_hal_hexagon_domain_id_from_device_id(device_id);
  return iree_hal_hexagon_driver_create_device(driver, domain_id, param_count,
                                               params, create_params,
                                               host_allocator, out_device);
}

static iree_status_t iree_hal_hexagon_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_hexagon_driver_t* driver = iree_hal_hexagon_driver_cast(base_driver);

  // The device path is empty (default domain), a domain name, or a domain ID.
  iree_hal_hexagon_domain_id_t domain_id = IREE_HAL_HEXAGON_DEFAULT_DOMAIN_ID;
  uint32_t parsed_domain_id = 0;
  if (iree_string_view_is_empty(device_path)) {
    domain_id = IREE_HAL_HEXAGON_DEFAULT_DOMAIN_ID;
  } else if (iree_string_view_atoi_uint32(device_path, &parsed_domain_id)) {
    domain_id = (iree_hal_hexagon_domain_id_t)parsed_domain_id;
  } else if (!iree_hal_hexagon_get_domain_id(device_path, &domain_id)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown hexagon device path '%.*s'; expected a "
                            "DSP domain name or ID",
                            (int)device_path.size, device_path.data);
  }
  return iree_hal_hexagon_driver_create_device(driver, domain_id, param_count,
                                               params, create_params,
                                               host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_hexagon_driver_vtable = {
    .destroy = iree_hal_hexagon_driver_destroy,
    .query_available_devices = iree_hal_hexagon_driver_query_available_devices,
    .dump_device_info = iree_hal_hexagon_driver_dump_device_info,
    .create_device_by_id = iree_hal_hexagon_driver_create_device_by_id,
    .create_device_by_path = iree_hal_hexagon_driver_create_device_by_path,
};
