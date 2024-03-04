// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/rocm/api.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/rocm_device.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

typedef struct iree_hal_rocm_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple ROCM versions can be exposed in the
  // same process.
  iree_string_view_t identifier;
  int default_device_index;
  // ROCM symbols.
  iree_hal_rocm_dynamic_symbols_t syms;
} iree_hal_rocm_driver_t;

// Maximum device name length supported by the ROCM HAL driver.
#define IREE_MAX_ROCM_DEVICE_NAME_LENGTH 128

// Utility macros to convert between HIPDevice and iree_hal_device_id_t.
#define IREE_HIPDEVICE_TO_DEVICE_ID(device) (iree_hal_device_id_t)((device) + 1)
#define IREE_DEVICE_ID_TO_HIPDEVICE(device_id) (hipDevice_t)((device_id)-1)

static const iree_hal_driver_vtable_t iree_hal_rocm_driver_vtable;

static iree_hal_rocm_driver_t* iree_hal_rocm_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_driver_vtable);
  return (iree_hal_rocm_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_rocm_driver_options_initialize(
    iree_hal_rocm_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_rocm_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_rocm_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
#if defined(IREE_PLATFORM_LINUX)
  // Hack to force device kernel arguments to be preloaded, when available, and
  // improve kernel latency. There doesn't seem to be any API to enable this.
  // This option will become the default in ROCm 6.1.
  // TODO: Remove this after upgrading to ROCm 6.1.
  setenv("HIP_FORCE_DEV_KERNARG", "1", /*replace=*/0);
#endif

  iree_hal_rocm_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_rocm_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  driver->default_device_index = options->default_device_index;
  iree_status_t status =
      iree_hal_rocm_dynamic_symbols_initialize(host_allocator, &driver->syms);
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

static void iree_hal_rocm_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_dynamic_symbols_deinitialize(&driver->syms);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_rocm_driver_create(
    iree_string_view_t identifier,
    const iree_hal_rocm_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_rocm_driver_create_internal(
      identifier, options, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Populates device information from the given ROCM physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static uint8_t* iree_hal_rocm_populate_device_info(
    hipDevice_t device, iree_hal_rocm_dynamic_symbols_t* syms,
    uint8_t* buffer_ptr, iree_hal_device_info_t* out_device_info) {
  char device_name[IREE_MAX_ROCM_DEVICE_NAME_LENGTH];
  hipUUID device_uuid;
  ROCM_IGNORE_ERROR(syms,
                    hipDeviceGetName(device_name, sizeof(device_name), device));
  ROCM_IGNORE_ERROR(syms, hipDeviceGetUuid(&device_uuid, device));
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = IREE_HIPDEVICE_TO_DEVICE_ID(device);

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

  iree_string_view_t device_name_string =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_string, &out_device_info->name, (char*)buffer_ptr);
  return buffer_ptr;
}

static iree_status_t iree_hal_rocm_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  // Query the number of available ROCM devices.
  int device_count = 0;
  ROCM_RETURN_IF_ERROR(&driver->syms, hipGetDeviceCount(&device_count),
                       "hipGetDeviceCount");

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size = device_count * sizeof(iree_hal_device_info_t);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    total_size += IREE_MAX_ROCM_DEVICE_NAME_LENGTH * sizeof(char);
  }
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      hipDevice_t device = 0;
      status = ROCM_RESULT_TO_STATUS(&driver->syms, hipDeviceGet(&device, i),
                                     "hipDeviceGet");
      if (!iree_status_is_ok(status)) break;
      buffer_ptr = iree_hal_rocm_populate_device_info(
          device, &driver->syms, buffer_ptr, &device_infos[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

static iree_status_t iree_hal_rocm_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  hipDevice_t device = IREE_DEVICE_ID_TO_HIPDEVICE(device_id);

  hipDeviceProp_t prop;
  ROCM_RETURN_IF_ERROR(&driver->syms,
                       hipGetDevicePropertiesR0600(&prop, device),
                       "hipGetDevicePropertiesR0600");

  // GPU capabilities and architecture.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-compute-capability: %d.%d", prop.major, prop.minor));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-arch-name: %s", prop.gcnArchName));

  // Launch configuration limits.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- launch-max-block-dims: (%d, %d, %d)", prop.maxThreadsDim[0],
      prop.maxThreadsDim[1], prop.maxThreadsDim[2]));

  int shared_memory_kb = prop.sharedMemPerBlock / 1024;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- block-max-thread-count: %d", prop.maxThreadsPerBlock));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- block-max-32-bit-register-count: %d", prop.regsPerBlock));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- block-max-shared-memory: %d KB", shared_memory_kb));

  // Memory hierarchy related information.
  int const_memory_mb = prop.totalConstMem / 1024 / 1024;
  int global_memory_mb = prop.totalGlobalMem / 1024 / 1024;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-is-integrated-memory: %d", prop.integrated));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-supports-managed-memory: %d", prop.managedMemory));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-total-const-memory-size: %d MB", const_memory_mb));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-total-global-memory-size: %d MB", global_memory_mb));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-l2-cache-size: %d bytes", prop.l2CacheSize));

  // GPU related information.
  int compute_clock_mhz = prop.clockRate / 1000;
  int memory_clock_mhz = prop.memoryClockRate / 1000;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-compute-unit-count: %d", prop.multiProcessorCount));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-compute-max-clock-rate: %d mHz", compute_clock_mhz));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-memory-max-clock-rate: %d mHz", memory_clock_mhz));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-warp-size: %d", prop.warpSize));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_driver_select_default_device(
    iree_hal_rocm_dynamic_symbols_t* syms, int default_device_index,
    iree_allocator_t host_allocator, hipDevice_t* out_device) {
  int device_count = 0;
  ROCM_RETURN_IF_ERROR(syms, hipGetDeviceCount(&device_count),
                       "hipGetDeviceCount");
  iree_status_t status = iree_ok_status();
  if (device_count == 0 || default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %d enumerated)",
                              default_device_index, device_count);
  } else {
    hipDevice_t device;
    ROCM_RETURN_IF_ERROR(syms, hipDeviceGet(&device, default_device_index),
                         "hipDeviceGet");
    *out_device = device;
  }
  return status;
}

static iree_status_t iree_hal_rocm_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(&driver->syms, hipInit(0), "hipInit"));
  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  hipDevice_t device = 0;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_rocm_driver_select_default_device(
                &driver->syms, driver->default_device_index, host_allocator,
                &device));
  } else {
    device = IREE_DEVICE_ID_TO_HIPDEVICE(device_id);
  }

  iree_string_view_t device_name = iree_make_cstring_view("rocm");

  // Attempt to create the device.
  iree_status_t status =
      iree_hal_rocm_device_create(base_driver, device_name, &driver->syms,
                                  device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_rocm_driver_create_device_by_uuid(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    const hipUUID* device_uuid, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query all rocm devices for instance.
  int device_count = 0;
  ROCM_RETURN_IF_ERROR(&driver->syms, hipGetDeviceCount(&device_count),
                       "hipGetDeviceCount");
  hipDevice_t device = 0;
  bool found_device = false;
  for (int i = 0; i < device_count; i++) {
    ROCM_RETURN_IF_ERROR(&driver->syms, hipDeviceGet(&device, i),
                         "hipDeviceGet");
    hipUUID query_uuid;
    ROCM_RETURN_IF_ERROR(&driver->syms, hipDeviceGetUuid(&query_uuid, device),
                         "hipDeviceGetUuid");
    if (memcmp(&device_uuid->bytes[0], &query_uuid.bytes[0],
               sizeof(device_uuid)) == 0) {
      found_device = true;
      break;
    }
  }
  if (!found_device) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "ROCM device with UUID GPU-"
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

  iree_status_t status = iree_hal_rocm_driver_create_device_by_id(
      base_driver, IREE_HIPDEVICE_TO_DEVICE_ID(device), param_count, params,
      host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_rocm_driver_create_device_by_index(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    int device_index, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);

  // Query the number of available HIP devices.
  int device_count = 0;
  ROCM_RETURN_IF_ERROR(&driver->syms, hipGetDeviceCount(&device_count),
                       "hipGetDeviceCount");
  if (device_index >= device_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device %d not found (of %d enumerated)",
                            device_index, device_count);
  }

  hipDevice_t device = 0;
  ROCM_RETURN_IF_ERROR(&driver->syms, hipDeviceGet(&device, device_index),
                       "hipDeviceGet");

  iree_status_t status = iree_hal_rocm_driver_create_device_by_id(
      base_driver, IREE_HIPDEVICE_TO_DEVICE_ID(device), param_count, params,
      host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_rocm_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_rocm_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  if (iree_string_view_consume_prefix(&device_path, IREE_SV("GPU-"))) {
    // UUID as returned by hipDeviceGetUuid.
    hipUUID device_uuid;
    if (!iree_string_view_parse_hex_bytes(device_path,
                                          IREE_ARRAYSIZE(device_uuid.bytes),
                                          (uint8_t*)device_uuid.bytes)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid GPU UUID: '%.*s'", (int)device_path.size,
                              device_path.data);
    }
    return iree_hal_rocm_driver_create_device_by_uuid(
        base_driver, driver_name, &device_uuid, param_count, params,
        host_allocator, out_device);
  }

  // Try to parse as a device index.
  int device_index = 0;
  if (iree_string_view_atoi_int32(device_path, &device_index)) {
    return iree_hal_rocm_driver_create_device_by_index(
        base_driver, driver_name, device_index, param_count, params,
        host_allocator, out_device);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_rocm_driver_vtable = {
    .destroy = iree_hal_rocm_driver_destroy,
    .query_available_devices = iree_hal_rocm_driver_query_available_devices,
    .dump_device_info = iree_hal_rocm_driver_dump_device_info,
    .create_device_by_id = iree_hal_rocm_driver_create_device_by_id,
    .create_device_by_path = iree_hal_rocm_driver_create_device_by_path,
};
