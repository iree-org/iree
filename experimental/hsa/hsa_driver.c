// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "experimental/hsa/api.h"
#include "experimental/hsa/dynamic_symbols.h"
#include "experimental/hsa/hsa_device.h"
#include "experimental/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/base/assert.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

// Maximum device name length supported by the HSA HAL driver.
#define IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH 64

#define IREE_HAL_HSA_MAX_DEVICES 64
#define IREE_HAL_HSA_DEVICE_NOT_FOUND IREE_HAL_HSA_MAX_DEVICES

// Utility macros to convert between hsa_agent_t ID and iree_hal_device_id_t.
#define IREE_DEVICE_ID_TO_HSADEVICE(device_id) (int)((device_id) - 1)

typedef struct iree_hal_hsa_driver_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  // Identifier used for registering the driver in the IREE driver registry.
  iree_string_view_t identifier;
  // HSA driver API dynamic symbols to interact with the HSA system.
  iree_hal_hsa_dynamic_symbols_t hsa_symbols;

  // The default parameters for creating devices using this driver.
  iree_hal_hsa_device_params_t device_params;

  // The index of the default HSA device to use if multiple ones are available.
  int default_device_index;

  // Number of GPU agents
  int num_gpu_agents;

  // IREE device ID to hsa_agent_t
  hsa_agent_t agents[IREE_HAL_HSA_MAX_DEVICES];
} iree_hal_hsa_driver_t;

typedef struct iree_hal_hsa_device_info_t {
} iree_hal_hsa_device_info_t;

// A struct encapsulating common variables we need while communicating with HSA
// callbacks
typedef struct iree_hal_hsa_callback_package_t {
  iree_hal_hsa_driver_t* driver;
  size_t* index;
  void* return_value;
} iree_hal_hsa_callback_package_t;

static const iree_hal_driver_vtable_t iree_hal_hsa_driver_vtable;

static iree_hal_hsa_driver_t* iree_hal_hsa_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_driver_vtable);
  return (iree_hal_hsa_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hsa_driver_options_initialize(
    iree_hal_hsa_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

hsa_status_t iterate_count_gpu_agents_callback(hsa_agent_t agent,
                                               void* base_driver) {
  iree_hal_hsa_callback_package_t* package =
      (iree_hal_hsa_callback_package_t*)(base_driver);
  iree_hal_hsa_driver_t* driver = package->driver;
  int* count_ptr = (int*)package->return_value;
  hsa_device_type_t type;
  hsa_status_t status =
      (&(driver->hsa_symbols))
          ->hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (type == HSA_DEVICE_TYPE_GPU) {
    *count_ptr = *count_ptr + 1;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t iterate_populate_gpu_agents_callback(hsa_agent_t agent,
                                                  void* base_driver) {
  iree_hal_hsa_callback_package_t* package =
      (iree_hal_hsa_callback_package_t*)(base_driver);
  iree_hal_hsa_driver_t* driver = package->driver;
  size_t* index_ptr = package->index;
  hsa_agent_t* agents_ptr = (hsa_agent_t*)package->return_value;

  hsa_device_type_t type;
  hsa_status_t status =
      (&(driver->hsa_symbols))
          ->hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (type == HSA_DEVICE_TYPE_GPU) {
    size_t current_index = *index_ptr;
    agents_ptr[current_index] = agent;
    *index_ptr = current_index + 1;
  }
  return HSA_STATUS_SUCCESS;
}

// Initializes the HSA system.
iree_status_t iree_hal_hsa_init(iree_hal_hsa_driver_t* driver) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HSA_RESULT_TO_STATUS(&driver->hsa_symbols, hsa_init(), "hsa_init");
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Deinitializes the HSA system.
static iree_status_t iree_hal_hsa_shut_down(iree_hal_hsa_driver_t* driver) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_HSA_RESULT_TO_STATUS(
      &driver->hsa_symbols, hsa_shut_down(), "hsa_shut_down");
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_create_internal(
    iree_string_view_t identifier, const iree_hal_hsa_driver_options_t* options,
    const iree_hal_hsa_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_hsa_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_hsa_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  driver->default_device_index = options->default_device_index;

  iree_status_t status = iree_hal_hsa_dynamic_symbols_initialize(
      host_allocator, &driver->hsa_symbols);

  status = iree_hal_hsa_init(driver);

  memcpy(&driver->device_params, device_params, sizeof(driver->device_params));

  driver->num_gpu_agents = 0;

  // Populate HSA agents
  // Query the number of available HSA devices.
  iree_hal_hsa_callback_package_t symbols_and_device_count = {
      .driver = driver, .return_value = &driver->num_gpu_agents};

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, &driver->hsa_symbols,
      hsa_iterate_agents(&iterate_count_gpu_agents_callback,
                         &symbols_and_device_count),
      "hsa_iterate_agents");

  size_t agent_index = 0;
  iree_hal_hsa_callback_package_t symbols_and_agents = {
      .driver = driver, .index = &agent_index, .return_value = driver->agents};

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, &driver->hsa_symbols,
      hsa_iterate_agents(&iterate_populate_gpu_agents_callback,
                         &symbols_and_agents),
      "hsa_iterate_agents");

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_hsa_driver_create(
    iree_string_view_t identifier, const iree_hal_hsa_driver_options_t* options,
    const iree_hal_hsa_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(device_params);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hsa_driver_create_internal(
      identifier, options, device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hsa_driver_destroy(iree_hal_driver_t* base_driver) {
  IREE_ASSERT_ARGUMENT(base_driver);

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // iree_hal_hsa_shut_down(driver);
  // iree_hal_hsa_dynamic_symbols_deinitialize(&driver->hsa_symbols);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

// Methods to translate HSA agents to IREE Device ID
static iree_hal_device_id_t iree_hsadevice_to_device_id(
    iree_hal_hsa_driver_t* driver, hsa_agent_t agent) {
  iree_hal_device_id_t device_id = 0;
  while (device_id != IREE_HAL_HSA_MAX_DEVICES &&
         driver->agents[device_id++].handle != agent.handle);

  return device_id;
}

static hsa_agent_t iree_device_id_to_hsadevice(iree_hal_hsa_driver_t* driver,
                                               iree_hal_device_id_t device_id) {
  return driver->agents[device_id];
}

static iree_status_t get_hsa_agent_uuid(iree_hal_hsa_dynamic_symbols_t* syms,
                                        hsa_agent_t agent,
                                        char* out_device_uuid) {
  // `HSA_AMD_AGENT_INFO_UUID` is part of the `hsa_amd_agent_info_t`
  // However, hsa_agent_get_info expects a hsa_agent_info_t.
  hsa_agent_info_t uuid_info = (int)HSA_AMD_AGENT_INFO_UUID;
  IREE_HSA_RETURN_IF_ERROR(
      syms, hsa_agent_get_info(agent, uuid_info, out_device_uuid),
      "hsa_agent_get_info");

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_populate_device_info(
    iree_hal_hsa_driver_t* driver, hsa_agent_t agent,
    iree_hal_hsa_dynamic_symbols_t* syms, uint8_t* buffer_ptr,
    uint8_t** out_buffer_ptr, iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  char device_name[IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH];

  IREE_HSA_RETURN_IF_ERROR(
      syms, hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, device_name),
      "hsa_agent_get_info");
  memset(out_device_info, 0, sizeof(*out_device_info));

  out_device_info->device_id = iree_hsadevice_to_device_id(driver, agent);

  // Maximum UUID is 21
  char device_uuid[21] = {0};
  get_hsa_agent_uuid(syms, agent, device_uuid);

  // HSA UUID is already prefixed with GPU-
  char device_path_str[4 + 36 + 1] = {0};
  snprintf(device_path_str, sizeof(device_path_str),
           "%c%c%c-"
           "%02x%02x%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x%02x%02x%02x%02x",
           device_uuid[0], device_uuid[1], device_uuid[2],
           (uint8_t)device_uuid[4], (uint8_t)device_uuid[5],
           (uint8_t)device_uuid[6], (uint8_t)device_uuid[7],
           (uint8_t)device_uuid[8], (uint8_t)device_uuid[9],
           (uint8_t)device_uuid[10], (uint8_t)device_uuid[11],
           (uint8_t)device_uuid[12], (uint8_t)device_uuid[13],
           (uint8_t)device_uuid[14], (uint8_t)device_uuid[15],
           (uint8_t)device_uuid[16], (uint8_t)device_uuid[17],
           (uint8_t)device_uuid[18], (uint8_t)device_uuid[19]);

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

static iree_status_t iree_hal_hsa_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HSA is initialized before querying it.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_hsa_init(driver));

  int device_count = driver->num_gpu_agents;

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size =
      device_count * (sizeof(iree_hal_device_info_t) +
                      IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH * sizeof(char));

  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);

  hsa_agent_t* agents = driver->agents;

  int valid_device_count = 0;
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      hsa_agent_t device = agents[i];

      status = iree_hal_hsa_populate_device_info(
          driver, device, &driver->hsa_symbols, buffer_ptr, &buffer_ptr,
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

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_driver_select_default_device(
    iree_hal_driver_t* base_driver, iree_hal_hsa_dynamic_symbols_t* syms,
    int default_device_index, iree_allocator_t host_allocator,
    hsa_agent_t* out_device) {
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t device_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hsa_driver_query_available_devices(
      base_driver, host_allocator, &device_count, &device_infos));

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);

  iree_status_t status = iree_ok_status();
  if (device_count == 0) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no compatible HSA devices were found");
  } else if (default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %" PRIhsz
                              " enumerated)",
                              default_device_index, device_count);
  } else {
    *out_device = iree_device_id_to_hsadevice(driver, default_device_index);
  }
  iree_allocator_free(host_allocator, device_infos);

  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HSA is initialized before querying it.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_hsa_init(driver));

  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  hsa_agent_t agent;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hsa_driver_select_default_device(
                base_driver, &driver->hsa_symbols, driver->default_device_index,
                host_allocator, &agent));
  } else {
    agent = iree_device_id_to_hsadevice(driver,
                                        IREE_DEVICE_ID_TO_HSADEVICE(device_id));
  }

  iree_string_view_t device_name = iree_make_cstring_view("hip");

  // Attempt to create the device now.
  iree_status_t status = iree_hal_hsa_device_create(
      base_driver, device_name, &driver->device_params, &driver->hsa_symbols,
      agent, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_uuid(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    char* device_uuid, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);

  // Ensure HSA is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_hsa_init(driver));
  iree_status_t status;
  // HSA doesn't have an API to do this so we need to scan all devices to
  // find the one with the matching UUID.
  int device_count = driver->num_gpu_agents;

  // Iterate over device info searching for the agent with the right UUID
  bool found_device = false;
  hsa_agent_t device = {0};
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    // Maximum UUID is 21
    char query_uuid[21] = {0};
    status =
        get_hsa_agent_uuid(&driver->hsa_symbols, driver->agents[i], query_uuid);
    char query_uuid_stripped[16] = {0};
    iree_string_view_t query_uuid_sv = iree_make_string_view(query_uuid, 21);
    if (!iree_string_view_parse_hex_bytes(query_uuid_sv,
                                          IREE_ARRAYSIZE(query_uuid_stripped),
                                          (uint8_t*)query_uuid_stripped)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid UUID: '%.*s'", (int)query_uuid_sv.size,
                              query_uuid_sv.data);
    }
    if (!iree_status_is_ok(status)) break;
    if (memcmp(device_uuid, query_uuid_stripped, sizeof(query_uuid_stripped)) ==
        0) {
      found_device = true;
      break;
      device = driver->agents[i];
    }
  }
  if (!found_device) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "HSA device with UUID "
                            "%02x%02x%02x%02x-"
                            "%02x%02x-"
                            "%02x%02x-"
                            "%02x%02x-"
                            "%02x%02x%02x%02x%02x%02x",
                            (uint8_t)device_uuid[4], (uint8_t)device_uuid[5],
                            (uint8_t)device_uuid[6], (uint8_t)device_uuid[7],
                            (uint8_t)device_uuid[8], (uint8_t)device_uuid[9],
                            (uint8_t)device_uuid[10], (uint8_t)device_uuid[11],
                            (uint8_t)device_uuid[12], (uint8_t)device_uuid[13],
                            (uint8_t)device_uuid[14], (uint8_t)device_uuid[15],
                            (uint8_t)device_uuid[16], (uint8_t)device_uuid[17],
                            (uint8_t)device_uuid[18], (uint8_t)device_uuid[19]);
  }

  iree_string_view_t device_name = iree_make_cstring_view("hip");

  // Attempt to create the device now.
  status = iree_hal_hsa_device_create(
      base_driver, device_name, &driver->device_params, &driver->hsa_symbols,
      device, host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_index(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    int device_index, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);

  // Ensure HSA is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_hsa_init(driver));

  // Query the number of available HSA devices.
  int device_count = driver->num_gpu_agents;
  if (device_index >= device_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device %d not found (of %d enumerated)",
                            device_index, device_count);
  }

  hsa_agent_t device = driver->agents[device_index];

  iree_string_view_t device_name = iree_make_cstring_view("hip");

  // Attempt to create the device now.
  iree_status_t status = iree_hal_hsa_device_create(
      base_driver, device_name, &driver->device_params, &driver->hsa_symbols,
      device, host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_hsa_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  bool found = iree_string_view_consume_prefix(&device_path, IREE_SV("GPU-"));

  if (found) {
    char device_uuid[16];
    if (!iree_string_view_parse_hex_bytes(
            device_path, IREE_ARRAYSIZE(device_uuid), (uint8_t*)device_uuid)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid UUID: '%.*s'", (int)device_path.size,
                              device_path.data);
    }
    return iree_hal_hsa_driver_create_device_by_uuid(
        base_driver, driver_name, device_uuid, param_count, params,
        host_allocator, out_device);
  }

  // Try to parse as a device index or device type
  int device_index = -1;

  if (iree_string_view_consume_prefix(&device_path, IREE_SV("GPU")) ||
      iree_string_view_consume_prefix(&device_path, IREE_SV("gpu"))) {
    device_index = 0;
  }

  if (device_index != -1 ||
      iree_string_view_atoi_int32(device_path, &device_index)) {
    return iree_hal_hsa_driver_create_device_by_index(
        base_driver, driver_name, device_index, param_count, params,
        host_allocator, out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_hsa_driver_vtable = {
    .destroy = iree_hal_hsa_driver_destroy,
    .query_available_devices = iree_hal_hsa_driver_query_available_devices,
    .dump_device_info = iree_hal_hsa_driver_dump_device_info,
    .create_device_by_id = iree_hal_hsa_driver_create_device_by_id,
    .create_device_by_path = iree_hal_hsa_driver_create_device_by_path,
};

#undef IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH
#undef IREE_HAL_HSA_MAX_DEVICES
#undef IREE_HAL_HSA_DEVICE_NOT_FOUND
