// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver.h"

#include <stddef.h>

#include "iree/base/internal/path.h"
#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(driver, method_name) \
  IREE_HAL_VTABLE_DISPATCH(driver, iree_hal_driver, method_name)

IREE_HAL_API_RETAIN_RELEASE(driver);

IREE_API_EXPORT iree_status_t iree_hal_driver_query_available_devices(
    iree_hal_driver_t* driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  *out_device_info_count = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(driver, query_available_devices)(
      driver, host_allocator, out_device_info_count, out_device_infos);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_dump_device_info(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(driver, dump_device_info)(driver, device_id, builder);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_ordinal(
    iree_hal_driver_t* driver, iree_host_size_t device_ordinal,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(!param_count || params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)device_ordinal);

  // Query the devices from the driver.
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_driver_query_available_devices(
              driver, host_allocator, &device_info_count, &device_infos));

  // Get the ID of the Nth device.
  iree_hal_device_id_t device_id = IREE_HAL_DEVICE_ID_DEFAULT;
  iree_status_t status = iree_ok_status();
  if (device_ordinal < device_info_count) {
    device_id = device_infos[device_ordinal].device_id;
  } else {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "device ordinal %" PRIhsz
                              " out of range; driver has %" PRIhsz
                              " devices enumerated",
                              device_ordinal, device_info_count);
  }

  // Drop the memory used for the device enumeration as we only need the ID to
  // proceed.
  iree_allocator_free(host_allocator, device_infos);

  // Create by ID now that we have it.
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_device_by_id(
        driver, device_id, param_count, params, host_allocator, out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_id(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(!param_count || params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)device_id);
  iree_status_t status = _VTABLE_DISPATCH(driver, create_device_by_id)(
      driver, device_id, param_count, params, host_allocator, out_device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_path(
    iree_hal_driver_t* driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(!param_count || params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, driver_name.data, driver_name.size);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, device_path.data, device_path.size);
  iree_status_t status = _VTABLE_DISPATCH(driver, create_device_by_path)(
      driver, driver_name, device_path, param_count, params, host_allocator,
      out_device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_create_device_by_uri(
    iree_hal_driver_t* driver, iree_string_view_t device_uri,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, device_uri.data, device_uri.size);

  iree_string_view_t driver_name, device_path, params_str;
  iree_uri_split(device_uri, &driver_name, &device_path, &params_str);

  // Split the parameter string into a list of key-value pairs.
  // This is a variable length list and we first query to see how much storage
  // is required.
  iree_host_size_t param_capacity = 0;
  iree_host_size_t param_count = 0;
  iree_string_pair_t* params = NULL;
  if (!iree_uri_split_params(params_str, 0, &param_capacity, NULL)) {
    if (param_capacity <= 128) {
      params = iree_alloca(sizeof(*params) * param_capacity);
      iree_uri_split_params(params_str, param_capacity, &param_count, params);
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unreasonably large number of device parameters (%" PRIhsz ") in URI",
          param_capacity);
    }
  }

  // Have the driver create the device.
  iree_status_t status = iree_hal_driver_create_device_by_path(
      driver, driver_name, device_path, param_count, params, host_allocator,
      out_device);
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(status, "creating device '%.*s'",
                                    (int)device_uri.size, device_uri.data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_driver_create_default_device(
    iree_hal_driver_t* driver, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(driver, create_device_by_id)(
      driver, IREE_HAL_DEVICE_ID_DEFAULT, /*param_count=*/0, /*params=*/NULL,
      host_allocator, out_device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
