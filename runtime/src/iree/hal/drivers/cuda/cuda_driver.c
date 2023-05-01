// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/api.h"
#include "iree/hal/drivers/cuda/cuda_device.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"
#include "iree/hal/drivers/cuda/status_util.h"

// Maximum device name length we support.
#define IREE_HAL_CUDA_MAX_DEVICE_NAME_LENGTH 128

#define CUDEVICE_TO_DEVICE_ID(device) (iree_hal_device_id_t)((device) + 1)
#define DEVICE_ID_TO_CUDEVICE(device_id) (CUdevice)((device_id)-1)

typedef struct iree_hal_cuda_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple CUDA versions can be exposed in the
  // same process.
  iree_string_view_t identifier;
  iree_hal_cuda_device_params_t default_params;
  int default_device_index;
  // CUDA symbols.
  iree_hal_cuda_dynamic_symbols_t syms;
} iree_hal_cuda_driver_t;

static const iree_hal_driver_vtable_t iree_hal_cuda_driver_vtable;

static iree_hal_cuda_driver_t* iree_hal_cuda_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_driver_vtable);
  return (iree_hal_cuda_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_cuda_driver_options_initialize(
    iree_hal_cuda_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_cuda_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* default_params,
    const iree_hal_cuda_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_cuda_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_cuda_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  memcpy(&driver->default_params, default_params,
         sizeof(driver->default_params));
  driver->default_device_index = options->default_device_index;

  iree_status_t status =
      iree_hal_cuda_dynamic_symbols_initialize(host_allocator, &driver->syms);

  if (iree_status_is_ok(status)) {
    // Try to load NCCL. This will fail if NCCL is unavailable or incompatible.
    // We only fail on unavailability when the user tries to create a channel
    // and otherwise defer reporting.
    status = iree_hal_cuda_nccl_dynamic_symbols_initialize(host_allocator,
                                                           &driver->syms);
    if (iree_status_is_unavailable(status)) {
      status = iree_status_ignore(status);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

static void iree_hal_cuda_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_dynamic_symbols_deinitialize(&driver->syms);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_cuda_driver_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* default_params,
    const iree_hal_cuda_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(default_params);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_cuda_driver_create_internal(
      identifier, default_params, options, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_init(iree_hal_cuda_driver_t* driver) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, CU_RESULT_TO_STATUS(&driver->syms, cuInit(0), "cuInit"));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Populates device information from the given CUDA physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static iree_status_t iree_hal_cuda_populate_device_info(
    CUdevice device, iree_hal_cuda_dynamic_symbols_t* syms, uint8_t* buffer_ptr,
    uint8_t** out_buffer_ptr, iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  char device_name[IREE_HAL_CUDA_MAX_DEVICE_NAME_LENGTH];
  CUDA_RETURN_IF_ERROR(
      syms, cuDeviceGetName(device_name, sizeof(device_name), device),
      "cuDeviceGetName");
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = CUDEVICE_TO_DEVICE_ID(device);

  // This matches the output of `nvidia-smi -L`.
  CUuuid device_uuid;
  CUDA_RETURN_IF_ERROR(syms, cuDeviceGetUuid(&device_uuid, device),
                       "cuDeviceGetUuid");
  char device_path_str[4 + 36 + 1] = {0};
  snprintf(device_path_str, sizeof(device_path_str),
           "GPU-"
           "%02x%02x%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x%02x%02x%02x%02x",
           (uint8_t)device_uuid.bytes[0], (uint8_t)device_uuid.bytes[1],
           (uint8_t)device_uuid.bytes[2], (uint8_t)device_uuid.bytes[3],
           (uint8_t)device_uuid.bytes[4], (uint8_t)device_uuid.bytes[5],
           (uint8_t)device_uuid.bytes[6], (uint8_t)device_uuid.bytes[7],
           (uint8_t)device_uuid.bytes[8], (uint8_t)device_uuid.bytes[9],
           (uint8_t)device_uuid.bytes[10], (uint8_t)device_uuid.bytes[11],
           (uint8_t)device_uuid.bytes[12], (uint8_t)device_uuid.bytes[13],
           (uint8_t)device_uuid.bytes[14], (uint8_t)device_uuid.bytes[15]);
  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_string_view(device_path_str,
                            IREE_ARRAYSIZE(device_path_str) - 1),
      &out_device_info->path, (char*)buffer_ptr);

  iree_string_view_t device_name_str =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_str, &out_device_info->name, (char*)buffer_ptr);

  *out_buffer_ptr = buffer_ptr;
  return iree_ok_status();
}

// Return true if the device supports all the extension required.
static bool iree_hal_cuda_is_valid_device(iree_hal_cuda_driver_t* driver,
                                          CUdevice device) {
  return true;
}

static iree_status_t iree_hal_cuda_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);

  // Ensure CUDA is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_cuda_init(driver));

  // Query the number of available CUDA devices.
  int device_count = 0;
  CUDA_RETURN_IF_ERROR(&driver->syms, cuDeviceGetCount(&device_count),
                       "cuDeviceGetCount");

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size = device_count * sizeof(iree_hal_device_info_t);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    total_size += IREE_HAL_CUDA_MAX_DEVICE_NAME_LENGTH * sizeof(char);
  }
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);
  int valid_device_count = 0;
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      CUdevice device;
      status = CU_RESULT_TO_STATUS(&driver->syms, cuDeviceGet(&device, i),
                                   "cuDeviceGet");
      if (!iree_status_is_ok(status)) break;
      if (!iree_hal_cuda_is_valid_device(driver, device)) continue;
      status = iree_hal_cuda_populate_device_info(
          device, &driver->syms, buffer_ptr, &buffer_ptr,
          &device_infos[valid_device_count]);
      if (!iree_status_is_ok(status)) break;
      valid_device_count++;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = valid_device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

static iree_status_t iree_hal_cuda_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);
  CUdevice device = (CUdevice)device_id;
  if (!device) return iree_ok_status();
  // TODO: dump detailed device info.
  (void)driver;
  (void)device;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_driver_select_default_device(
    iree_hal_driver_t* base_driver, iree_hal_cuda_dynamic_symbols_t* syms,
    int default_device_index, iree_allocator_t host_allocator,
    CUdevice* out_device) {
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t device_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_driver_query_available_devices(
      base_driver, host_allocator, &device_count, &device_infos));
  iree_status_t status = iree_ok_status();
  if (device_count == 0) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no compatible CUDA devices were found");
  } else if (default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %ld enumerated)",
                              default_device_index, device_count);
  } else {
    *out_device =
        DEVICE_ID_TO_CUDEVICE(device_infos[default_device_index].device_id);
  }
  iree_allocator_free(host_allocator, device_infos);
  return status;
}

static iree_status_t iree_hal_cuda_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure CUDA is initialized before querying it.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_cuda_init(driver));

  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  CUdevice device = 0;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_cuda_driver_select_default_device(
                base_driver, &driver->syms, driver->default_device_index,
                host_allocator, &device));
  } else {
    device = DEVICE_ID_TO_CUDEVICE(device_id);
  }

  iree_string_view_t device_name = iree_make_cstring_view("cuda");

  // Attempt to create the device.
  iree_status_t status = iree_hal_cuda_device_create(
      base_driver, device_name, &driver->default_params, &driver->syms, device,
      host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_driver_create_device_by_uuid(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    const CUuuid* device_uuid, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);

  // Ensure CUDA is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_cuda_init(driver));

  // CUDA doesn't have an API to do this so we need to scan all devices to
  // find the one with the matching UUID.
  int device_count = 0;
  CUDA_RETURN_IF_ERROR(&driver->syms, cuDeviceGetCount(&device_count),
                       "cuDeviceGetCount");
  CUdevice device = 0;
  bool found_device = false;
  for (int i = 0; i < device_count; i++) {
    CUDA_RETURN_IF_ERROR(&driver->syms, cuDeviceGet(&device, i), "cuDeviceGet");
    CUuuid query_uuid;
    CUDA_RETURN_IF_ERROR(&driver->syms, cuDeviceGetUuid(&query_uuid, device),
                         "cuDeviceGetUuid");
    if (memcmp(&device_uuid->bytes[0], &query_uuid.bytes[0],
               sizeof(device_uuid)) == 0) {
      found_device = true;
      break;
    }
  }
  if (!found_device) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "CUDA device with UUID GPU-"
        "%02x%02x%02x%02x-"
        "%02x%02x-"
        "%02x%02x-"
        "%02x%02x-"
        "%02x%02x%02x%02x%02x%02x"
        " not found",
        (uint8_t)device_uuid->bytes[0], (uint8_t)device_uuid->bytes[1],
        (uint8_t)device_uuid->bytes[2], (uint8_t)device_uuid->bytes[3],
        (uint8_t)device_uuid->bytes[4], (uint8_t)device_uuid->bytes[5],
        (uint8_t)device_uuid->bytes[6], (uint8_t)device_uuid->bytes[7],
        (uint8_t)device_uuid->bytes[8], (uint8_t)device_uuid->bytes[9],
        (uint8_t)device_uuid->bytes[10], (uint8_t)device_uuid->bytes[11],
        (uint8_t)device_uuid->bytes[12], (uint8_t)device_uuid->bytes[13],
        (uint8_t)device_uuid->bytes[14], (uint8_t)device_uuid->bytes[15]);
  }

  iree_status_t status = iree_hal_cuda_driver_create_device_by_id(
      base_driver, CUDEVICE_TO_DEVICE_ID(device), param_count, params,
      host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_cuda_driver_create_device_by_index(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    int device_index, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_cuda_driver_t* driver = iree_hal_cuda_driver_cast(base_driver);

  // Ensure CUDA is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_cuda_init(driver));

  CUdevice device = 0;
  CUDA_RETURN_IF_ERROR(&driver->syms, cuDeviceGet(&device, device_index),
                       "cuDeviceGet");

  iree_status_t status = iree_hal_cuda_driver_create_device_by_id(
      base_driver, CUDEVICE_TO_DEVICE_ID(device), param_count, params,
      host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_cuda_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_cuda_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  if (iree_string_view_consume_prefix(&device_path, IREE_SV("GPU-"))) {
    // UUID as returned by cuDeviceGetUuid.
    CUuuid device_uuid;
    if (!iree_string_view_parse_hex_bytes(device_path,
                                          IREE_ARRAYSIZE(device_uuid.bytes),
                                          (uint8_t*)device_uuid.bytes)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid GPU UUID: '%.*s'", (int)device_path.size,
                              device_path.data);
    }
    return iree_hal_cuda_driver_create_device_by_uuid(
        base_driver, driver_name, &device_uuid, param_count, params,
        host_allocator, out_device);
  }

  // Try to parse as a device index.
  int device_index = 0;
  if (iree_string_view_atoi_int32(device_path, &device_index)) {
    return iree_hal_cuda_driver_create_device_by_index(
        base_driver, driver_name, device_index, param_count, params,
        host_allocator, out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_cuda_driver_vtable = {
    .destroy = iree_hal_cuda_driver_destroy,
    .query_available_devices = iree_hal_cuda_driver_query_available_devices,
    .dump_device_info = iree_hal_cuda_driver_dump_device_info,
    .create_device_by_id = iree_hal_cuda_driver_create_device_by_id,
    .create_device_by_path = iree_hal_cuda_driver_create_device_by_path,
};
